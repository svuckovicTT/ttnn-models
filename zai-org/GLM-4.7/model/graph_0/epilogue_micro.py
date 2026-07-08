"""Micro-bench the MoE epilogue: scale combine_output[8,16,5120] (per-expert, ROW_MAJOR
from moe_compute) by routing scores and sum over k=8 -> [16,5120]. Current path tilizes
first (~90us). Test whether multiply+reduce can run in ROW_MAJOR to skip the tilize.
Device time via batched enqueue + 1 sync. PCC vs torch reference."""
import time, torch, ttnn
dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

def main():
    dev = ttnn.open_device(device_id=0)
    K, T, H = 8, 16, 5120
    torch.manual_seed(0)
    comb = torch.randn(K, T, H, dtype=torch.bfloat16)
    sc = torch.randn(K, 1, T, 1, dtype=torch.bfloat16)       # scores_k [8,1,16,1]
    gold = (comb.float().unsqueeze(1) * sc.float()).sum(0)   # [1,16,5120]

    comb_rm = ttnn.to_device(ttnn.to_layout(ttnn.from_torch(comb, dtype=ttnn.bfloat16), ttnn.Layout.ROW_MAJOR), dev, dram)
    sc_rm = ttnn.to_device(ttnn.to_layout(ttnn.from_torch(sc, dtype=ttnn.bfloat16), ttnn.Layout.ROW_MAJOR), dev, dram)
    sc_tile = ttnn.to_device(ttnn.to_layout(ttnn.from_torch(sc, dtype=ttnn.bfloat16), ttnn.Layout.TILE), dev, dram)

    def pcc(t):
        a = ttnn.to_torch(ttnn.from_device(t)).float().reshape(-1)
        return torch.corrcoef(torch.stack([a, gold.float().reshape(-1)]))[0,1].item()

    def timed(fn, n=40):
        r=fn(); ttnn.synchronize_device(dev)
        outs=[]; t0=time.time()
        for _ in range(n): outs.append(fn())
        ttnn.synchronize_device(dev); us=(time.time()-t0)/n*1e6
        p=pcc(r)
        for o in outs:
            try: ttnn.deallocate(o, False)
            except Exception: pass
        return us, p

    # A) current: tilize combine -> unsqueeze -> multiply(TILE) -> sum(dim0)
    def pathA():
        ce = ttnn.to_layout(comb_rm, ttnn.Layout.TILE, None, memory_config=dram)
        ce = ttnn.unsqueeze(ce, dim=1)
        scaled = ttnn.multiply(ce, sc_tile, dtype=ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(ce, False)
        s = ttnn.sum(scaled, [0], False, memory_config=dram, compute_kernel_config=None)
        ttnn.deallocate(scaled, False)
        return s
    usA, pA = timed(pathA); print(f"A tilize+mul+sum(TILE)   {usA:7.1f} us  PCC={pA:.5f}")

    # B) RM multiply -> sum(dim0) in RM (if supported)
    def pathB():
        ce = ttnn.unsqueeze(comb_rm, dim=1)
        scaled = ttnn.multiply(ce, sc_rm, dtype=ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(ce, False)
        s = ttnn.sum(scaled, [0], False, memory_config=dram, compute_kernel_config=None)
        ttnn.deallocate(scaled, False)
        return s
    try:
        usB, pB = timed(pathB); print(f"B mul+sum (ROW_MAJOR)    {usB:7.1f} us  PCC={pB:.5f}")
    except Exception as e:
        print(f"B ROW_MAJOR FAILED {str(e)[:120]}")

    # C) RM multiply -> fast_reduce_nc over dim0
    def pathC():
        ce = ttnn.unsqueeze(comb_rm, dim=1)
        scaled = ttnn.multiply(ce, sc_rm, dtype=ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(ce, False)
        s = ttnn.experimental.fast_reduce_nc(scaled, dims=[0], memory_config=dram)
        ttnn.deallocate(scaled, False)
        return s
    try:
        usC, pC = timed(pathC); print(f"C mul(RM)+fast_reduce_nc {usC:7.1f} us  PCC={pC:.5f}")
    except Exception as e:
        print(f"C fast_reduce_nc FAILED {str(e)[:120]}")

    # D) tilize -> multiply -> fast_reduce_nc (keep tilize but faster reduce)
    def pathD():
        ce = ttnn.to_layout(comb_rm, ttnn.Layout.TILE, None, memory_config=dram)
        ce = ttnn.unsqueeze(ce, dim=1)
        scaled = ttnn.multiply(ce, sc_tile, dtype=ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(ce, False)
        s = ttnn.experimental.fast_reduce_nc(scaled, dims=[0], memory_config=dram)
        ttnn.deallocate(scaled, False)
        return s
    try:
        usD, pD = timed(pathD); print(f"D tilize+mul+fast_reduce {usD:7.1f} us  PCC={pD:.5f}")
    except Exception as e:
        print(f"D FAILED {str(e)[:120]}")

    # E) RM mul -> RM sum -> tilize the SMALL [1,16,5120] result (vs tilizing the
    #    big [8,16,5120] combine in A). This is the model-realistic path: summed must
    #    end up TILE for the downstream add(shared)+reduce_scatter.
    def pathE():
        ce = ttnn.unsqueeze(comb_rm, dim=1)
        scaled = ttnn.multiply(ce, sc_rm, dtype=ttnn.bfloat16, memory_config=dram)
        ttnn.deallocate(ce, False)
        s = ttnn.sum(scaled, [0], False, memory_config=dram, compute_kernel_config=None)
        ttnn.deallocate(scaled, False)
        st = ttnn.to_layout(s, ttnn.Layout.TILE, None, memory_config=dram)
        ttnn.deallocate(s, False)
        return st
    try:
        usE, pE = timed(pathE); print(f"E mul+sum(RM)+tilize_sm  {usE:7.1f} us  PCC={pE:.5f}")
    except Exception as e:
        print(f"E FAILED {str(e)[:120]}")

    ttnn.close_device(dev)

if __name__ == "__main__":
    main()
