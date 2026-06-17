import ttnn
MESH = (4, 8)
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
cfg = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL)
d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH), l1_small_size=1 << 15,
                          dispatch_core_config=cfg)
try:
    g = d.compute_with_storage_grid_size()
    print("COL compute_with_storage_grid_size:", g.x, g.y)
    mm = d.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.RISCV_0_default)
    xs = [c.x for c in mm]; ys = [c.y for c in mm]
    print("matmul bbox x:[%d,%d] y:[%d,%d]" % (min(xs), max(xs), min(ys), max(ys)))
    print("matmul cores:", sorted((c.x, c.y) for c in mm))
    print("tilize fixed (5-6,8-9) bbox x:[5,6] y:[8,9]")
finally:
    ttnn.close_mesh_device(d)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
