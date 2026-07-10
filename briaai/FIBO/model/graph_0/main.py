import time
import torch
import ttnn
import utils
import model_pt
import model_ttnn
from utils import calculate_pcc

def build_activations(device):
    # Build the graph inputs from the golden inputs (model_pt.load_input(), the same
    # inputs xla.py feeds the model) instead of reading serialized tensorbins. The
    # FIBO transformer (BriaFiboTransformer2DModel) was captured with these keyword
    # inputs; load_input() returns them flattened in kwargs order, with return_dict
    # (a bool) dropped as a non-tensor.
    raw = model_pt.load_input()
    hidden_states = raw[0]                        # (2, 4096, 48)   noisy latent image
    timestep = raw[1]                             # (2,)            diffusion timestep
    encoder_hidden_states = raw[2]                # (2, 45, 4096)   text prompt embedding
    text_encoder_layers = raw[3]                  # list[46] of (2, 45, 2048): one
                                                  # per-layer text feature per block
    attention_mask = raw[4]["attention_mask"]     # (2, 1, 4141, 4141) joint attn mask
    # raw[5] = return_dict (bool, dropped)
    txt_ids = raw[6]                              # (45, 3)    text RoPE ids
    img_ids = raw[7]                              # (4096, 3)  image RoPE ids

    # The 46 text_encoder_layers flatten into the graph inputs alongside the 6 core
    # tensors (52 total). The compiled forward consumes them in this order (from the
    # input ttir.name attributes in ttnn.mlir):
    inputs = [
        timestep,
        hidden_states,
        text_encoder_layers[0],
        encoder_hidden_states,
        attention_mask,
        img_ids,
        txt_ids,
        *text_encoder_layers[1:],
    ]
    assert len(inputs) == 52, f"expected 52 input tensors, got {len(inputs)}"

    # Every input is *replicated* across the (1, 4) mesh - verified value-exact
    # against the on-disk tensors - so each is placed with ReplicateTensorToMesh
    # (ROW_MAJOR, BFLOAT16, DRAM interleaved), matching the old tensorbin loader.
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )
    activations = []
    for pt_tensor in inputs:
        t = ttnn.from_torch(
            pt_tensor.to(torch.bfloat16),
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        t = ttnn.to_device(t, device, memory_config)
        activations.append(t)
    return activations


def main():
    device = utils.open_device()
    model = model_ttnn.ModelTTNN(device)
    activations = build_activations(device)
    outputs = model(activations)
    return 0


def test_main():
    exact_pcc = 0.999389111995697

    device = utils.open_device()
    model = model_ttnn.ModelTTNN(device)

    activations = build_activations(device)

    host_activations = [ttnn.from_device(act) for act in activations]

    dram_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )
    input_dram_tensors = [
        ttnn.allocate_tensor_on_device(
            act.shape, act.dtype, act.layout, device, dram_mem_config
        )
        for act in activations
    ]

    def copy_inputs():
        for host_act, dram_tensor in zip(host_activations, input_dram_tensors):
            ttnn.copy_host_to_device_tensor(host_act, dram_tensor, cq_id=0)

    # Run 0: compile run to fill the program cache
    copy_inputs()
    start = time.perf_counter()
    outputs = model(input_dram_tensors)
    ttnn.synchronize_device(device)
    end = time.perf_counter()
    compile_elapsed = end - start

    output_host = ttnn.from_device(outputs[0])
    output_shard = ttnn.get_device_tensors(output_host)[0]
    ttnn_output = ttnn.to_torch(output_shard).to(torch.float32)

    golden_output = model_pt.run_pytorch_model().to(torch.float32)

    pcc = calculate_pcc(ttnn_output, golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"

    # Run 1: capture trace
    copy_inputs()
    start = time.perf_counter()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = model(input_dram_tensors)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    end = time.perf_counter()
    capture_elapsed = end - start

    print("\nPerformance:")
    print(f"  Run 0 (compile): {compile_elapsed:.4f}s, FPS: {model_pt.BATCH_SIZE / compile_elapsed:.2f}")
    print(f"  Run 1 (capture): {capture_elapsed:.4f}s, FPS: {model_pt.BATCH_SIZE / capture_elapsed:.2f}")

    # Runs 2-4: execute trace
    for i in range(3):
        copy_inputs()
        start = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        host_output_tensor = output_tensor[0].cpu(blocking=False)
        ttnn.synchronize_device(device)
        end = time.perf_counter()
        elapsed = end - start
        fps = model_pt.BATCH_SIZE / elapsed
        print(f"  Run {i + 2} (trace): {elapsed:.4f}s, FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
