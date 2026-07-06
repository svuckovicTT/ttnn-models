"""Standalone micro-benchmark: DRAM rms_norm vs width-sharded rms_norm on the
decode input_layernorm shape [16, 5120]. Verifies numerical match and prints
device time so we can validate the sharded-norm config BEFORE wiring it into the
full model (10-min cycles). Single-device open (rms_norm is a local per-device op)."""
import time
import torch
import ttnn

H = 5120
ROWS = 16
EPS = 9.9999997473787516e-06

hifi4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True,
)
dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def torch_rms(x, g):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
    return (v * g.float()).to(torch.bfloat16)


def main():
    dev = ttnn.open_device(device_id=0)
    torch.manual_seed(0)
    x = torch.randn(ROWS, H, dtype=torch.bfloat16)
    g = torch.randn(H, dtype=torch.bfloat16)
    gold = torch_rms(x, g)

    xt = ttnn.to_device(ttnn.to_layout(ttnn.from_torch(x, dtype=ttnn.bfloat16), ttnn.Layout.TILE), dev, dram)
    gt = ttnn.to_device(ttnn.to_layout(ttnn.from_torch(g, dtype=ttnn.bfloat16), ttnn.Layout.TILE), dev, dram)

    def pcc(t):
        a = ttnn.to_torch(ttnn.from_device(t)).float().reshape(ROWS, H)
        b = gold.float()
        return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()

    def timed(fn, n=20):
        fn(); ttnn.synchronize_device(dev)  # warmup
        t0 = time.time()
        for _ in range(n):
            r = fn()
        ttnn.synchronize_device(dev)
        return (time.time() - t0) / n * 1e6, r  # us

    # --- baseline: DRAM rms_norm ---
    def dram_norm():
        return ttnn.rms_norm(xt, epsilon=EPS, weight=gt, memory_config=dram,
                             program_config=None, compute_kernel_config=hifi4)
    us_dram, r_dram = timed(dram_norm)
    print(f"DRAM       rms_norm: {us_dram:8.1f} us/call  PCC={pcc(r_dram):.6f}")

    # --- width-sharded rms_norm across NC cores, grids that AVOID col x=7
    # (COL dispatch reserves x=7 in the real model). (gx,gy) core block. ---
    # (label, gx, gy) -> NC=gx*gy cores; cols = 160/NC width-tiles per core
    for label, gx, gy in [("4x2", 4, 2), ("4x4", 4, 4), ("7x1", 7, 1), ("4x1", 4, 1)]:
        NC = gx * gy
        if (H // 32) % NC != 0:
            print(f"  {label} (NC={NC}): skip (uneven {H//32}/{NC})"); continue
        cols = (H // 32) // NC
        shard_w = cols * 32
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        shard_spec = ttnn.ShardSpec(grid, [32, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        smem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
        try:
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                subblock_w=1, block_h=1, block_w=cols, inplace=False,
            )
            # Realistic model chain: reshard DRAM->L1 sharded, norm, reshard back to DRAM.
            def chain():
                xs = ttnn.to_memory_config(xt, smem)
                ns = ttnn.rms_norm(xs, epsilon=EPS, weight=gt, memory_config=smem,
                                   program_config=pc, compute_kernel_config=hifi4)
                ttnn.deallocate(xs, False)
                out = ttnn.to_memory_config(ns, dram)
                ttnn.deallocate(ns, False)
                return out
            us_c, r_c = timed(chain)
            print(f"WSHARD {label:4s} NC={NC:2d} chain(reshard+norm+reshard): {us_c:8.1f} us/call  PCC={pcc(r_c):.6f}  (w_tiles/core={cols})")
        except Exception as e:
            print(f"  {label} (NC={NC}): FAILED {str(e)[:160]}")

    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
