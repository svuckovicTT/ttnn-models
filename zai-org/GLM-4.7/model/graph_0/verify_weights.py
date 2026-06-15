import torch
import ttnn

from params import load_weights_for__main, load_weights_for__main_from_state_dict


def _unique_shards(t):
    """Return the list of distinct per-device shards (as torch tensors) of a
    (possibly mesh-distributed) ttnn tensor, in first-seen device order.

    Plain ttnn.to_torch raises on mesh-distributed tensors, so we convert
    shard-by-shard. Shards are converted lazily and dropped when they duplicate
    an already-seen shard, so replicated tensors (e.g. embed_tokens, which is the
    full weight replicated on all 32 devices) don't blow up host memory."""
    if t.device() is not None:
        t = ttnn.from_device(t)
    uniq = []
    for s in ttnn.get_device_tensors(t):
        p = ttnn.to_torch(s)
        if not any(p.shape == u.shape and torch.equal(p, u) for u in uniq):
            uniq.append(p)
    return uniq


MESH_ROWS, MESH_COLS = 4, 8


def _mesh_reorder_column_major(shards):
    """Reorder shards from physical row-major device order (as returned by
    get_device_tensors over the (4, 8) mesh) into column-major order.

    Weights sharded across the *whole* mesh (the MoE experts, one shard per
    device) are laid out column-major, whereas get_device_tensors yields device
    p = row*MESH_COLS + col in row-major order. Map physical p=(r,c) to logical
    c*MESH_ROWS + r so concatenation reconstructs the original expert order."""
    reordered = [None] * len(shards)
    for p, shard in enumerate(shards):
        r, c = divmod(p, MESH_COLS)
        reordered[c * MESH_ROWS + r] = shard
    return reordered


def _logical_tensor(t, reference):
    """Reconstruct the logical (un-sharded) tensor from a mesh-distributed ttnn
    tensor, matching it to `reference` (the full single-device tensor).

    The disk loader stores weights sharded across the 4x8 mesh with a per-tensor
    layout: replicated (identical on every device), row/col sharded (concat along
    dim 0 or 1), expert-dim sharded, and the expert weights are additionally
    transposed on their last two dims relative to the HF state_dict. We don't know
    each tensor's scheme a priori, so reconstruct by concatenating the distinct
    shards along whichever dim (optionally with a last-two-dim transpose) yields
    the reference shape."""
    shards = _unique_shards(t)
    if len(shards) == 1:
        return shards[0]

    # A tensor with one distinct shard per device is sharded across the whole
    # mesh (the experts); those are laid out column-major, so reorder first.
    if len(shards) == MESH_ROWS * MESH_COLS:
        shards = _mesh_reorder_column_major(shards)

    for dim in range(shards[0].ndim):
        cat = torch.cat(shards, dim=dim)
        if cat.shape == reference.shape:
            return cat
        if cat.ndim >= 2:
            cat_t = cat.transpose(-1, -2).contiguous()
            if cat_t.shape == reference.shape:
                return cat_t

    raise AssertionError(
        f"Could not reconstruct shape {tuple(reference.shape)} from "
        f"{len(shards)} shard(s) of shape {tuple(shards[0].shape)}"
    )


def test_verify_weights():
    disk_weights = load_weights_for__main()
    sd_weights = load_weights_for__main_from_state_dict()

    disk_keys = set(disk_weights.keys())
    sd_keys = set(sd_weights.keys())
    assert disk_keys == sd_keys, (
        f"Key mismatch. In disk but not state_dict: {disk_keys - sd_keys}. "
        f"In state_dict but not disk: {sd_keys - disk_keys}."
    )

    mismatches = []
    for key in sorted(disk_weights.keys()):
        # sd weights are full single-device tensors (layernorms are replicated);
        # collapse any replication to get the reference logical tensor.
        sd_logical = _unique_shards(sd_weights[key])
        assert len(sd_logical) == 1, (
            f"{key}: expected a single logical state_dict tensor, "
            f"got {len(sd_logical)} distinct shards"
        )
        reference = sd_logical[0]

        disk_logical = _logical_tensor(disk_weights[key], reference)

        pt1 = disk_logical.flatten()
        pt2 = reference.flatten()

        if not torch.equal(pt1, pt2):
            max_diff = (pt1.float() - pt2.float()).abs().max().item()
            mismatches.append((key, max_diff))
            print(f"  MISMATCH: {key} (max diff: {max_diff})")
        else:
            print(f"  OK: {key}")

    assert not mismatches, (
        f"{len(mismatches)} / {len(disk_weights)} weights mismatch: "
        + ", ".join(f"{k}={d}" for k, d in mismatches)
    )
    print(f"All {len(disk_weights)} weights match!")
