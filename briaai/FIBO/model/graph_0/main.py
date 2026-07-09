import time
import torch
import ttnn
import utils
import model_pt
import model_ttnn
from utils import calculate_pcc

def load_activations_for__main(device):
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg8.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg319.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg325.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg328.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg329.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg330.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg331.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg349.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg373.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg397.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg421.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg445.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg469.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg493.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_14 = utils.load_tensor(
        "./tensors/arg513.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_15 = utils.load_tensor(
        "./tensors/arg529.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_16 = utils.load_tensor(
        "./tensors/arg539.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg549.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg559.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_19 = utils.load_tensor(
        "./tensors/arg569.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_20 = utils.load_tensor(
        "./tensors/arg579.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_21 = utils.load_tensor(
        "./tensors/arg589.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_22 = utils.load_tensor(
        "./tensors/arg599.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_23 = utils.load_tensor(
        "./tensors/arg609.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg619.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_25 = utils.load_tensor(
        "./tensors/arg629.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_26 = utils.load_tensor(
        "./tensors/arg639.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_27 = utils.load_tensor(
        "./tensors/arg649.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_28 = utils.load_tensor(
        "./tensors/arg659.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg669.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg679.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_31 = utils.load_tensor(
        "./tensors/arg689.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_32 = utils.load_tensor(
        "./tensors/arg699.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_33 = utils.load_tensor(
        "./tensors/arg709.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_34 = utils.load_tensor(
        "./tensors/arg719.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_35 = utils.load_tensor(
        "./tensors/arg729.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg739.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_37 = utils.load_tensor(
        "./tensors/arg749.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_38 = utils.load_tensor(
        "./tensors/arg759.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_39 = utils.load_tensor(
        "./tensors/arg769.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_40 = utils.load_tensor(
        "./tensors/arg779.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_41 = utils.load_tensor(
        "./tensors/arg789.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_42 = utils.load_tensor(
        "./tensors/arg799.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_43 = utils.load_tensor(
        "./tensors/arg809.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_44 = utils.load_tensor(
        "./tensors/arg819.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_45 = utils.load_tensor(
        "./tensors/arg829.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_46 = utils.load_tensor(
        "./tensors/arg839.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_47 = utils.load_tensor(
        "./tensors/arg849.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_48 = utils.load_tensor(
        "./tensors/arg859.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_49 = utils.load_tensor(
        "./tensors/arg869.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_50 = utils.load_tensor(
        "./tensors/arg879.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_51 = utils.load_tensor(
        "./tensors/arg889.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
        utils_load_tensor_4,
        utils_load_tensor_5,
        utils_load_tensor_6,
        utils_load_tensor_7,
        utils_load_tensor_8,
        utils_load_tensor_9,
        utils_load_tensor_10,
        utils_load_tensor_11,
        utils_load_tensor_12,
        utils_load_tensor_13,
        utils_load_tensor_14,
        utils_load_tensor_15,
        utils_load_tensor_16,
        utils_load_tensor_17,
        utils_load_tensor_18,
        utils_load_tensor_19,
        utils_load_tensor_20,
        utils_load_tensor_21,
        utils_load_tensor_22,
        utils_load_tensor_23,
        utils_load_tensor_24,
        utils_load_tensor_25,
        utils_load_tensor_26,
        utils_load_tensor_27,
        utils_load_tensor_28,
        utils_load_tensor_29,
        utils_load_tensor_30,
        utils_load_tensor_31,
        utils_load_tensor_32,
        utils_load_tensor_33,
        utils_load_tensor_34,
        utils_load_tensor_35,
        utils_load_tensor_36,
        utils_load_tensor_37,
        utils_load_tensor_38,
        utils_load_tensor_39,
        utils_load_tensor_40,
        utils_load_tensor_41,
        utils_load_tensor_42,
        utils_load_tensor_43,
        utils_load_tensor_44,
        utils_load_tensor_45,
        utils_load_tensor_46,
        utils_load_tensor_47,
        utils_load_tensor_48,
        utils_load_tensor_49,
        utils_load_tensor_50,
        utils_load_tensor_51,
    ]


def main():
    device = utils.open_device()
    model = model_ttnn.ModelTTNN(device)
    activations = load_activations_for__main(device)
    outputs = model(activations)
    return 0


def test_main():
    exact_pcc = 0.999389111995697

    device = utils.open_device()
    model = model_ttnn.ModelTTNN(device)

    activations = load_activations_for__main(device)

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

    for host_act, dram_tensor in zip(host_activations, input_dram_tensors):
        ttnn.copy_host_to_device_tensor(host_act, dram_tensor, cq_id=0)
    outputs = model(input_dram_tensors)
    ttnn.synchronize_device(device)

    output_host = ttnn.from_device(outputs[0])
    output_shard = ttnn.get_device_tensors(output_host)[0]
    ttnn_output = ttnn.to_torch(output_shard).to(torch.float32)

    golden_output = model_pt.run_pytorch_model().to(torch.float32)

    pcc = calculate_pcc(ttnn_output, golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"

    for host_act, dram_tensor in zip(host_activations, input_dram_tensors):
        ttnn.copy_host_to_device_tensor(host_act, dram_tensor, cq_id=0)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = model(input_dram_tensors)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    print("\nPerformance:")
    for i in range(5):
        start = time.perf_counter()
        for host_act, dram_tensor in zip(host_activations, input_dram_tensors):
            ttnn.copy_host_to_device_tensor(host_act, dram_tensor, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        host_output_tensor = output_tensor[0].cpu(blocking=False)
        ttnn.synchronize_device(device)
        end = time.perf_counter()
        elapsed = end - start
        fps = model_pt.BATCH_SIZE / elapsed
        print(f"  Run {i}: {elapsed:.4f}s, FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
