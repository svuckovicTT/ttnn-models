"""DRAM-sharded matmul helpers for attention perf tuning.

Mirrors models/demos/llama3_70b_galaxy/tt/model_config.py (same WH galaxy HW):
width-shard the weight across the 12 DRAM banks and run the matmul with a
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig, feeding an L1-width-sharded
activation. Used to break the DRAM-bandwidth wall on the skinny (M=16) decode
qkv / o_proj matmuls.
"""
import math

import ttnn

TILE = 32
DRAM_BANKS = 12


def _find_grid_k_n(k_tiles, n_tiles):
    max_rows, max_cols = 4, 8
    max_cores = max_rows * max_cols
    possible = sorted(
        [c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0],
        reverse=True,
    )
    for cores in possible:
        for rows in range(1, max_rows + 1):
            if cores % rows == 0 and cores // rows <= max_cols:
                return rows, cores // rows
    raise AssertionError(f"no DRAM-shard grid for k_tiles={k_tiles} n_tiles={n_tiles}")


def compute_grid(k, n):
    """Return (rows, cols, num_cores) compute grid for a (k, n) matmul."""
    rows, cols = _find_grid_k_n(k // TILE, n // TILE)
    return rows, cols, rows * cols


def _core_range_set(rows, cols):
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))}
    )


def dram_weight_grid(device):
    g = device.dram_grid_size()
    return ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))}
    )


def weight_dram_sharded_config(device, k, n):
    """DRAM width-sharded memory config for a [k, n] weight (across 12 banks)."""
    padded = math.ceil(n / (TILE * DRAM_BANKS)) * (TILE * DRAM_BANKS)
    spec = ttnn.ShardSpec(
        dram_weight_grid(device), (k, padded // DRAM_BANKS), ttnn.ShardOrientation.ROW_MAJOR
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec
    )


def in0_l1_width_sharded_config(m, k, rows, cols):
    """L1 width-sharded config for the [m, k] activation across the compute grid."""
    num_cores = rows * cols
    padded_m = math.ceil(m / TILE) * TILE
    spec = ttnn.ShardSpec(
        _core_range_set(rows, cols), (padded_m, k // num_cores), ttnn.ShardOrientation.ROW_MAJOR
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec
    )


def out_l1_width_sharded_config(m, n, rows, cols):
    """L1 width-sharded config for the [m, n] matmul output across the compute grid."""
    num_cores = rows * cols
    padded_m = math.ceil(m / TILE) * TILE
    padded_n = math.ceil(n / (TILE * num_cores)) * (TILE * num_cores)
    spec = ttnn.ShardSpec(
        _core_range_set(rows, cols), (padded_m, padded_n // num_cores), ttnn.ShardOrientation.ROW_MAJOR
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec
    )


def program_config(m, k, n, num_cores):
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=math.ceil(k / (TILE * num_cores)),
        per_core_M=math.ceil(m / TILE),
        per_core_N=math.ceil(n / (TILE * num_cores)),
        fused_activation=None,
    )
