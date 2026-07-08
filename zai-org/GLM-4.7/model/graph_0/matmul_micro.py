"""Micro-bench decode matmul configs (device time via batched enqueue + 1 sync)."""
import time, torch, ttnn
dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
def g(x,y): return ttnn.CoreCoord(x,y)

def d1(grid, ibw, pcN, osw=1):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid, in0_block_w=ibw, out_subblock_h=1,
        out_subblock_w=osw, per_core_M=1, per_core_N=pcN, fuse_batch=True, mcast_in0=True)

def bench(dev, M, K, N, wdtype, label, cfgs, iters=60):
    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.bfloat16); w = torch.randn(K, N, dtype=torch.bfloat16)
    gold = (a.float() @ w.float())
    at = ttnn.to_device(ttnn.to_layout(ttnn.from_torch(a, dtype=ttnn.bfloat16), ttnn.Layout.TILE), dev, dram)
    wt = ttnn.to_device(ttnn.to_layout(ttnn.from_torch(w, dtype=wdtype), ttnn.Layout.TILE), dev, dram)
    print(f"\n=== {label}: [{M},{K}]x[{K},{N}]  Ktiles={K//32} Ntiles={N//32} ===")
    for cname, pc in cfgs:
        try:
            r = ttnn.matmul(at, wt, memory_config=dram, dtype=ttnn.bfloat16, program_config=pc, compute_kernel_config=None)
            ttnn.synchronize_device(dev)
            rt = ttnn.to_torch(ttnn.from_device(r)).float().reshape(M, N)
            pcc = torch.corrcoef(torch.stack([rt.flatten(), gold.flatten()]))[0,1].item()
            outs=[]; t0=time.time()
            for _ in range(iters):
                outs.append(ttnn.matmul(at, wt, memory_config=dram, dtype=ttnn.bfloat16, program_config=pc, compute_kernel_config=None))
            ttnn.synchronize_device(dev); us=(time.time()-t0)/iters*1e6
            for o in outs: ttnn.deallocate(o, False)
            print(f"  {cname:34s} {us:7.1f} us/op  PCC={pcc:.5f}")
        except Exception as e:
            print(f"  {cname:34s} FAILED {str(e)[:90]}")

def main():
    dev = ttnn.open_device(device_id=0)
    # NOTE: model runs under COL dispatch -> 7-wide worker grid (x=0-6). Use 7-wide.
    # qkv [32,5120,1792]: Ktiles=160, Ntiles=56 -> 7x8=56 cores per_core_N=1
    bench(dev, 32, 5120, 1792, ttnn.bfloat8_b, "qkv", [
        ("default", None),
        ("1D 7x8 ibw8 pcN1", d1(g(7,8),8,1)),
        ("1D 7x8 ibw5 pcN1", d1(g(7,8),5,1)),
        ("1D 7x8 ibw10 pcN1", d1(g(7,8),10,1)),
    ])
    # o_proj [32,1536,5120]: Ktiles=48, Ntiles=160 -> 7x8=56, per_core_N=ceil(160/56)=3
    bench(dev, 32, 1536, 5120, ttnn.bfloat8_b, "o_proj", [
        ("default", None),
        ("1D 7x8 ibw6 pcN3", d1(g(7,8),6,3)),
        ("1D 7x10 ibw6 pcN3", d1(g(7,10),6,3)),
    ])
    # shared gate/up [32,5120,192]: Ktiles=160, Ntiles=6
    bench(dev, 32, 5120, 192, ttnn.bfloat8_b, "shared_gate", [
        ("default", None),
        ("1D 6x1 ibw20 pcN1", d1(g(6,1),20,1)),
        ("1D 7x8 ibw8 pcN1", d1(g(7,8),8,1)),
    ])
    # shared down [32,1536,5120]: Ktiles=48, Ntiles=160
    bench(dev, 32, 1536, 5120, ttnn.bfloat8_b, "shared_down", [
        ("default", None),
        ("1D 7x8 ibw6 pcN3", d1(g(7,8),6,3)),
    ])
    # router gate [32,5120,160] Ktiles=160 Ntiles=5, FP32 out + sigmoid in model
    bench(dev, 32, 5120, 160, ttnn.bfloat8_b, "router_bf16", [
        ("default", None),
        ("1D 5x1 ibw20 pcN1", d1(g(5,1),20,1)),
        ("1D 7x8 ibw8 pcN1", d1(g(7,8),8,1)),
    ])
    ttnn.close_device(dev)

if __name__ == "__main__":
    main()
