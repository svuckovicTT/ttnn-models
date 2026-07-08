"""Micro-bench generic to_layout(TILE) vs deepseek_moe_post_combine_tilize (L1-sharded out)
on the GLM MoE combine [8,16,5120] ROW_MAJOR bf16. Device time via batched enqueue."""
import time, torch, ttnn
dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

def sharded_cfg(shard_w, gx, gy):
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[32, shard_w],
            grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx-1, gy-1))}),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

def main():
    dev = ttnn.open_device(device_id=0)
    torch.manual_seed(0)
    x = torch.randn(8, 16, 5120, dtype=torch.bfloat16)
    gold = x.float()
    xt = ttnn.to_device(ttnn.to_layout(ttnn.from_torch(x, dtype=ttnn.bfloat16), ttnn.Layout.ROW_MAJOR), dev, dram)

    def timed(fn, n=60):
        r = fn(); ttnn.synchronize_device(dev)
        try:
            a = ttnn.to_torch(ttnn.from_device(r)).float().reshape(8,16,5120)
            p = torch.corrcoef(torch.stack([a.flatten(), gold.flatten()]))[0,1].item()
        except Exception as ex:
            p = float("nan")
        outs=[]; t0=time.time()
        for _ in range(n): outs.append(fn())
        ttnn.synchronize_device(dev); us=(time.time()-t0)/n*1e6
        for o in outs:
            try: ttnn.deallocate(o, False)
            except Exception: pass
        return us, p

    usA, pA = timed(lambda: ttnn.to_layout(xt, ttnn.Layout.TILE, None, memory_config=dram))
    print(f"A generic to_layout(TILE)->DRAM         {usA:7.1f} us  PCC={pA:.5f}")

    # 5120 = 160 tiles. shard widths that divide: 5120(1 shard/batch), 2560, 640, 512...
    for label, sw, gx, gy in [("sw5120 8x1", 5120,8,1), ("sw640 8x8", 640,8,8), ("sw2560 8x2", 2560,8,2)]:
        try:
            cfg = sharded_cfg(sw, gx, gy)
            def fn(cfg=cfg):
                return ttnn.experimental.deepseek_moe_post_combine_tilize(xt, output_memory_config=cfg)
            us, p = timed(fn)
            print(f"B post_combine_tilize {label:12s} L1shard {us:7.1f} us  PCC={p:.5f}")
        except Exception as e:
            print(f"B {label:12s} FAILED {str(e).splitlines()[0][:120]}")
    ttnn.close_device(dev)

if __name__ == "__main__":
    main()
