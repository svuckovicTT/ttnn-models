import ttnn
import utils
import ttir_cpu
import torch
import model_pt
from utils import calculate_pcc
from params import load_weights_for__main_from_state_dict


def main_const_eval_0(device):
    ttnn_Tensor_0 = ttnn.Tensor(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        [18],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_Tensor_0]


def main_const_eval_1(arg, device):
    ttnn_to_device_0 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_to_device_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_0, False)
    ttnn_from_device_0 = ttnn.from_device(ttnn_to_layout_0)
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_from_device_0, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_0, False)
    ttnn_to_device_1 = ttnn.to_device(
        ttnn_typecast_0,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    return [ttnn_to_device_1]


def main_const_eval_2(arg, device):
    ttnn_to_device_2 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_to_device_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_2, False)
    ttnn_from_device_1 = ttnn.from_device(ttnn_to_layout_1)
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_from_device_1, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_1, False)
    ttnn_to_device_3 = ttnn.to_device(
        ttnn_typecast_1,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    return [ttnn_to_device_3]


def cpu_hoisted_const_eval_de619321(arg_0, arg_1, arg_2):
    ttnn_to_torch_0 = ttnn.to_torch(arg_0)
    ttnn_to_torch_1 = ttnn.to_torch(arg_1)
    ttnn_to_torch_2 = ttnn.to_torch(arg_2)
    ttir_cpu_permute_0 = ttir_cpu.permute(ttnn_to_torch_0, [1, 0])
    ttir_cpu_permute_1 = ttir_cpu.permute(ttnn_to_torch_1, [1, 0])
    ttir_cpu_permute_2 = ttir_cpu.permute(ttnn_to_torch_2, [1, 0])
    util_create_list_0 = [ttir_cpu_permute_2, ttir_cpu_permute_0, ttir_cpu_permute_1]
    ttir_cpu_concat_0 = ttir_cpu.concat(util_create_list_0, dim=1)
    ttnn_from_torch_0 = ttnn.from_torch(ttir_cpu_concat_0)
    return ttnn_from_torch_0


def main_const_eval_3(arg, device):
    ttnn_typecast_2 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_3 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_4 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_0 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_2, ttnn_typecast_3, ttnn_typecast_4
    )
    ttnn.deallocate(ttnn_typecast_4, False)
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn_typecast_5 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_0, False)
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_typecast_5, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_5, False)
    ttnn_to_device_4 = ttnn.to_device(
        ttnn_to_layout_2,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_slice_0 = ttnn.slice(
        ttnn_to_device_4,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_1 = ttnn.slice(
        ttnn_to_device_4,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_2 = ttnn.slice(
        ttnn_to_device_4,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_4, False)
    ttnn_concat_0 = ttnn.concat(
        [ttnn_slice_1, ttnn_slice_0, ttnn_slice_2],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_2, False)
    ttnn.deallocate(ttnn_slice_1, False)
    ttnn.deallocate(ttnn_slice_0, False)
    ttnn_from_device_2 = ttnn.from_device(ttnn_concat_0)
    ttnn.deallocate(ttnn_concat_0, False)
    ttnn_typecast_6 = ttnn.typecast(
        ttnn_from_device_2, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_2, False)
    ttnn_to_device_5 = ttnn.to_device(
        ttnn_typecast_6,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_6, False)
    return [ttnn_to_device_5]


def main_const_eval_4(arg, device):
    ttnn_typecast_7 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_8 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_9 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_1 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_7, ttnn_typecast_8, ttnn_typecast_9
    )
    ttnn.deallocate(ttnn_typecast_9, False)
    ttnn.deallocate(ttnn_typecast_8, False)
    ttnn.deallocate(ttnn_typecast_7, False)
    ttnn_typecast_10 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_1, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_1, False)
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_typecast_10, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_10, False)
    ttnn_to_device_6 = ttnn.to_device(
        ttnn_to_layout_3,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn_slice_3 = ttnn.slice(
        ttnn_to_device_6,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_4 = ttnn.slice(
        ttnn_to_device_6,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_5 = ttnn.slice(
        ttnn_to_device_6,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_6, False)
    ttnn_concat_1 = ttnn.concat(
        [ttnn_slice_4, ttnn_slice_3, ttnn_slice_5],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_5, False)
    ttnn.deallocate(ttnn_slice_4, False)
    ttnn.deallocate(ttnn_slice_3, False)
    ttnn_from_device_3 = ttnn.from_device(ttnn_concat_1)
    ttnn.deallocate(ttnn_concat_1, False)
    ttnn_typecast_11 = ttnn.typecast(
        ttnn_from_device_3, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_3, False)
    ttnn_to_device_7 = ttnn.to_device(
        ttnn_typecast_11,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_11, False)
    return [ttnn_to_device_7]


def main_const_eval_5(arg, device):
    ttnn_to_device_8 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_4 = ttnn.to_layout(
        ttnn_to_device_8,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_8, False)
    ttnn_from_device_4 = ttnn.from_device(ttnn_to_layout_4)
    ttnn.deallocate(ttnn_to_layout_4, False)
    ttnn_typecast_12 = ttnn.typecast(
        ttnn_from_device_4, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_4, False)
    ttnn_to_device_9 = ttnn.to_device(
        ttnn_typecast_12,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_12, False)
    return [ttnn_to_device_9]


def main_const_eval_6(arg, device):
    ttnn_to_device_10 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_to_device_10,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_10, False)
    ttnn_from_device_5 = ttnn.from_device(ttnn_to_layout_5)
    ttnn.deallocate(ttnn_to_layout_5, False)
    ttnn_typecast_13 = ttnn.typecast(
        ttnn_from_device_5, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_5, False)
    ttnn_to_device_11 = ttnn.to_device(
        ttnn_typecast_13,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_13, False)
    return [ttnn_to_device_11]


def main_const_eval_7(arg, device):
    ttnn_to_device_12 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_6 = ttnn.to_layout(
        ttnn_to_device_12,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_12, False)
    ttnn_from_device_6 = ttnn.from_device(ttnn_to_layout_6)
    ttnn.deallocate(ttnn_to_layout_6, False)
    ttnn_typecast_14 = ttnn.typecast(
        ttnn_from_device_6, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_6, False)
    ttnn_to_device_13 = ttnn.to_device(
        ttnn_typecast_14,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_14, False)
    return [ttnn_to_device_13]


def main_const_eval_8(device):
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=18,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_full_0]


def main_const_eval_9(arg, device):
    ttnn_to_device_14 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_7 = ttnn.to_layout(
        ttnn_to_device_14,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_14, False)
    ttnn_from_device_7 = ttnn.from_device(ttnn_to_layout_7)
    ttnn.deallocate(ttnn_to_layout_7, False)
    ttnn_typecast_15 = ttnn.typecast(
        ttnn_from_device_7, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_7, False)
    ttnn_to_device_15 = ttnn.to_device(
        ttnn_typecast_15,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_15, False)
    return [ttnn_to_device_15]


def main_const_eval_10(arg, device):
    ttnn_to_device_16 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_8 = ttnn.to_layout(
        ttnn_to_device_16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_16, False)
    ttnn_from_device_8 = ttnn.from_device(ttnn_to_layout_8)
    ttnn.deallocate(ttnn_to_layout_8, False)
    ttnn_typecast_16 = ttnn.typecast(
        ttnn_from_device_8, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_8, False)
    ttnn_to_device_17 = ttnn.to_device(
        ttnn_typecast_16,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_16, False)
    return [ttnn_to_device_17]


def main_const_eval_11(arg, device):
    ttnn_to_device_18 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_9 = ttnn.to_layout(
        ttnn_to_device_18,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_18, False)
    ttnn_from_device_9 = ttnn.from_device(ttnn_to_layout_9)
    ttnn.deallocate(ttnn_to_layout_9, False)
    ttnn_typecast_17 = ttnn.typecast(
        ttnn_from_device_9, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_9, False)
    ttnn_to_device_19 = ttnn.to_device(
        ttnn_typecast_17,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_17, False)
    return [ttnn_to_device_19]


def main_const_eval_12(arg, device):
    ttnn_to_device_20 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_10 = ttnn.to_layout(
        ttnn_to_device_20,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_20, False)
    ttnn_from_device_10 = ttnn.from_device(ttnn_to_layout_10)
    ttnn.deallocate(ttnn_to_layout_10, False)
    ttnn_typecast_18 = ttnn.typecast(
        ttnn_from_device_10, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_10, False)
    ttnn_to_device_21 = ttnn.to_device(
        ttnn_typecast_18,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_18, False)
    return [ttnn_to_device_21]


def main_const_eval_13(arg, device):
    ttnn_to_device_22 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_11 = ttnn.to_layout(
        ttnn_to_device_22,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_22, False)
    ttnn_from_device_11 = ttnn.from_device(ttnn_to_layout_11)
    ttnn.deallocate(ttnn_to_layout_11, False)
    ttnn_typecast_19 = ttnn.typecast(
        ttnn_from_device_11, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_11, False)
    ttnn_to_device_23 = ttnn.to_device(
        ttnn_typecast_19,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_19, False)
    return [ttnn_to_device_23]


def main_const_eval_14(arg, device):
    ttnn_to_device_24 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_12 = ttnn.to_layout(
        ttnn_to_device_24,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_24, False)
    ttnn_from_device_12 = ttnn.from_device(ttnn_to_layout_12)
    ttnn.deallocate(ttnn_to_layout_12, False)
    ttnn_typecast_20 = ttnn.typecast(
        ttnn_from_device_12, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_12, False)
    ttnn_to_device_25 = ttnn.to_device(
        ttnn_typecast_20,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_20, False)
    return [ttnn_to_device_25]


def main_const_eval_15(arg, device):
    ttnn_typecast_21 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_22 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_23 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_2 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_21, ttnn_typecast_22, ttnn_typecast_23
    )
    ttnn.deallocate(ttnn_typecast_23, False)
    ttnn.deallocate(ttnn_typecast_22, False)
    ttnn.deallocate(ttnn_typecast_21, False)
    ttnn_typecast_24 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_2, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_2, False)
    ttnn_to_layout_13 = ttnn.to_layout(
        ttnn_typecast_24, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_24, False)
    ttnn_to_device_26 = ttnn.to_device(
        ttnn_to_layout_13,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_13, False)
    ttnn_slice_6 = ttnn.slice(
        ttnn_to_device_26,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_7 = ttnn.slice(
        ttnn_to_device_26,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_8 = ttnn.slice(
        ttnn_to_device_26,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_26, False)
    ttnn_concat_2 = ttnn.concat(
        [ttnn_slice_7, ttnn_slice_6, ttnn_slice_8],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_8, False)
    ttnn.deallocate(ttnn_slice_7, False)
    ttnn.deallocate(ttnn_slice_6, False)
    ttnn_from_device_13 = ttnn.from_device(ttnn_concat_2)
    ttnn.deallocate(ttnn_concat_2, False)
    ttnn_typecast_25 = ttnn.typecast(
        ttnn_from_device_13, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_13, False)
    ttnn_to_device_27 = ttnn.to_device(
        ttnn_typecast_25,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_25, False)
    return [ttnn_to_device_27]


def main_const_eval_16(arg, device):
    ttnn_typecast_26 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_27 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_28 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_3 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_26, ttnn_typecast_27, ttnn_typecast_28
    )
    ttnn.deallocate(ttnn_typecast_28, False)
    ttnn.deallocate(ttnn_typecast_27, False)
    ttnn.deallocate(ttnn_typecast_26, False)
    ttnn_typecast_29 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_3, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_3, False)
    ttnn_to_layout_14 = ttnn.to_layout(
        ttnn_typecast_29, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_29, False)
    ttnn_to_device_28 = ttnn.to_device(
        ttnn_to_layout_14,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_14, False)
    ttnn_slice_9 = ttnn.slice(
        ttnn_to_device_28,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_10 = ttnn.slice(
        ttnn_to_device_28,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_11 = ttnn.slice(
        ttnn_to_device_28,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_28, False)
    ttnn_concat_3 = ttnn.concat(
        [ttnn_slice_10, ttnn_slice_9, ttnn_slice_11],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_11, False)
    ttnn.deallocate(ttnn_slice_10, False)
    ttnn.deallocate(ttnn_slice_9, False)
    ttnn_from_device_14 = ttnn.from_device(ttnn_concat_3)
    ttnn.deallocate(ttnn_concat_3, False)
    ttnn_typecast_30 = ttnn.typecast(
        ttnn_from_device_14, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_14, False)
    ttnn_to_device_29 = ttnn.to_device(
        ttnn_typecast_30,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_30, False)
    return [ttnn_to_device_29]


def main_const_eval_17(arg, device):
    ttnn_typecast_31 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_32 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_33 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_4 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_31, ttnn_typecast_32, ttnn_typecast_33
    )
    ttnn.deallocate(ttnn_typecast_33, False)
    ttnn.deallocate(ttnn_typecast_32, False)
    ttnn.deallocate(ttnn_typecast_31, False)
    ttnn_typecast_34 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_4, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_4, False)
    ttnn_to_layout_15 = ttnn.to_layout(
        ttnn_typecast_34, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_34, False)
    ttnn_to_device_30 = ttnn.to_device(
        ttnn_to_layout_15,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_15, False)
    ttnn_slice_12 = ttnn.slice(
        ttnn_to_device_30,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_13 = ttnn.slice(
        ttnn_to_device_30,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_14 = ttnn.slice(
        ttnn_to_device_30,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_30, False)
    ttnn_concat_4 = ttnn.concat(
        [ttnn_slice_13, ttnn_slice_12, ttnn_slice_14],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_14, False)
    ttnn.deallocate(ttnn_slice_13, False)
    ttnn.deallocate(ttnn_slice_12, False)
    ttnn_from_device_15 = ttnn.from_device(ttnn_concat_4)
    ttnn.deallocate(ttnn_concat_4, False)
    ttnn_typecast_35 = ttnn.typecast(
        ttnn_from_device_15, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_15, False)
    ttnn_to_device_31 = ttnn.to_device(
        ttnn_typecast_35,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_35, False)
    return [ttnn_to_device_31]


def main_const_eval_18(arg, device):
    ttnn_to_device_32 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_16 = ttnn.to_layout(
        ttnn_to_device_32,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_32, False)
    ttnn_from_device_16 = ttnn.from_device(ttnn_to_layout_16)
    ttnn.deallocate(ttnn_to_layout_16, False)
    ttnn_typecast_36 = ttnn.typecast(
        ttnn_from_device_16, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_16, False)
    ttnn_to_device_33 = ttnn.to_device(
        ttnn_typecast_36,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_36, False)
    return [ttnn_to_device_33]


def main_const_eval_19(arg, device):
    ttnn_to_device_34 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_17 = ttnn.to_layout(
        ttnn_to_device_34,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_34, False)
    ttnn_from_device_17 = ttnn.from_device(ttnn_to_layout_17)
    ttnn.deallocate(ttnn_to_layout_17, False)
    ttnn_typecast_37 = ttnn.typecast(
        ttnn_from_device_17, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_17, False)
    ttnn_to_device_35 = ttnn.to_device(
        ttnn_typecast_37,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_37, False)
    return [ttnn_to_device_35]


def main_const_eval_20(arg, device):
    ttnn_to_device_36 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_18 = ttnn.to_layout(
        ttnn_to_device_36,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_36, False)
    ttnn_from_device_18 = ttnn.from_device(ttnn_to_layout_18)
    ttnn.deallocate(ttnn_to_layout_18, False)
    ttnn_typecast_38 = ttnn.typecast(
        ttnn_from_device_18, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_18, False)
    ttnn_to_device_37 = ttnn.to_device(
        ttnn_typecast_38,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_38, False)
    return [ttnn_to_device_37]


def main_const_eval_21(arg, device):
    ttnn_to_device_38 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_19 = ttnn.to_layout(
        ttnn_to_device_38,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_38, False)
    ttnn_from_device_19 = ttnn.from_device(ttnn_to_layout_19)
    ttnn.deallocate(ttnn_to_layout_19, False)
    ttnn_typecast_39 = ttnn.typecast(
        ttnn_from_device_19, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_19, False)
    ttnn_to_device_39 = ttnn.to_device(
        ttnn_typecast_39,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_39, False)
    return [ttnn_to_device_39]


def main_const_eval_22(arg, device):
    ttnn_to_device_40 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_20 = ttnn.to_layout(
        ttnn_to_device_40,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_40, False)
    ttnn_from_device_20 = ttnn.from_device(ttnn_to_layout_20)
    ttnn.deallocate(ttnn_to_layout_20, False)
    ttnn_typecast_40 = ttnn.typecast(
        ttnn_from_device_20, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_20, False)
    ttnn_to_device_41 = ttnn.to_device(
        ttnn_typecast_40,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_40, False)
    return [ttnn_to_device_41]


def main_const_eval_23(arg, device):
    ttnn_to_device_42 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_21 = ttnn.to_layout(
        ttnn_to_device_42,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_42, False)
    ttnn_from_device_21 = ttnn.from_device(ttnn_to_layout_21)
    ttnn.deallocate(ttnn_to_layout_21, False)
    ttnn_typecast_41 = ttnn.typecast(
        ttnn_from_device_21, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_21, False)
    ttnn_to_device_43 = ttnn.to_device(
        ttnn_typecast_41,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_41, False)
    return [ttnn_to_device_43]


def main_const_eval_24(arg, device):
    ttnn_typecast_42 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_43 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_44 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_5 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_42, ttnn_typecast_43, ttnn_typecast_44
    )
    ttnn.deallocate(ttnn_typecast_44, False)
    ttnn.deallocate(ttnn_typecast_43, False)
    ttnn.deallocate(ttnn_typecast_42, False)
    ttnn_typecast_45 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_5, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_5, False)
    ttnn_to_layout_22 = ttnn.to_layout(
        ttnn_typecast_45, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_45, False)
    ttnn_to_device_44 = ttnn.to_device(
        ttnn_to_layout_22,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_22, False)
    ttnn_slice_15 = ttnn.slice(
        ttnn_to_device_44,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_16 = ttnn.slice(
        ttnn_to_device_44,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_17 = ttnn.slice(
        ttnn_to_device_44,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_44, False)
    ttnn_concat_5 = ttnn.concat(
        [ttnn_slice_16, ttnn_slice_15, ttnn_slice_17],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_17, False)
    ttnn.deallocate(ttnn_slice_16, False)
    ttnn.deallocate(ttnn_slice_15, False)
    ttnn_from_device_22 = ttnn.from_device(ttnn_concat_5)
    ttnn.deallocate(ttnn_concat_5, False)
    ttnn_typecast_46 = ttnn.typecast(
        ttnn_from_device_22, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_22, False)
    ttnn_to_device_45 = ttnn.to_device(
        ttnn_typecast_46,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_46, False)
    return [ttnn_to_device_45]


def cpu_hoisted_const_eval_32c82fce(arg):
    ttnn_to_torch_3 = ttnn.to_torch(arg)
    ttir_cpu_reshape_0 = ttir_cpu.reshape(ttnn_to_torch_3, [1, 32, 1])
    ttnn_from_torch_1 = ttnn.from_torch(ttir_cpu_reshape_0)
    return ttnn_from_torch_1


def main_const_eval_25(arg, device):
    ttnn_typecast_47 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_32c82fce_0 = cpu_hoisted_const_eval_32c82fce(
        ttnn_typecast_47
    )
    ttnn.deallocate(ttnn_typecast_47, False)
    ttnn_to_layout_23 = ttnn.to_layout(
        cpu_hoisted_const_eval_32c82fce_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_32c82fce_0, False)
    ttnn_to_device_46 = ttnn.to_device(
        ttnn_to_layout_23,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_23, False)
    return [ttnn_to_device_46]


def main_const_eval_26(arg, device):
    ttnn_typecast_48 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_49 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_50 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_6 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_48, ttnn_typecast_49, ttnn_typecast_50
    )
    ttnn.deallocate(ttnn_typecast_50, False)
    ttnn.deallocate(ttnn_typecast_49, False)
    ttnn.deallocate(ttnn_typecast_48, False)
    ttnn_typecast_51 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_6, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_6, False)
    ttnn_to_layout_24 = ttnn.to_layout(
        ttnn_typecast_51, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_51, False)
    ttnn_to_device_47 = ttnn.to_device(
        ttnn_to_layout_24,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_24, False)
    ttnn_slice_18 = ttnn.slice(
        ttnn_to_device_47,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_19 = ttnn.slice(
        ttnn_to_device_47,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_20 = ttnn.slice(
        ttnn_to_device_47,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_47, False)
    ttnn_concat_6 = ttnn.concat(
        [ttnn_slice_19, ttnn_slice_18, ttnn_slice_20],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_20, False)
    ttnn.deallocate(ttnn_slice_19, False)
    ttnn.deallocate(ttnn_slice_18, False)
    ttnn_from_device_23 = ttnn.from_device(ttnn_concat_6)
    ttnn.deallocate(ttnn_concat_6, False)
    ttnn_typecast_52 = ttnn.typecast(
        ttnn_from_device_23, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_23, False)
    ttnn_to_device_48 = ttnn.to_device(
        ttnn_typecast_52,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_52, False)
    return [ttnn_to_device_48]


def main_const_eval_27(arg, device):
    ttnn_typecast_53 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_54 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_55 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_7 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_53, ttnn_typecast_54, ttnn_typecast_55
    )
    ttnn.deallocate(ttnn_typecast_55, False)
    ttnn.deallocate(ttnn_typecast_54, False)
    ttnn.deallocate(ttnn_typecast_53, False)
    ttnn_typecast_56 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_7, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_7, False)
    ttnn_to_layout_25 = ttnn.to_layout(
        ttnn_typecast_56, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_56, False)
    ttnn_to_device_49 = ttnn.to_device(
        ttnn_to_layout_25,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_25, False)
    ttnn_slice_21 = ttnn.slice(
        ttnn_to_device_49,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_22 = ttnn.slice(
        ttnn_to_device_49,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_23 = ttnn.slice(
        ttnn_to_device_49,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_49, False)
    ttnn_concat_7 = ttnn.concat(
        [ttnn_slice_22, ttnn_slice_21, ttnn_slice_23],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_23, False)
    ttnn.deallocate(ttnn_slice_22, False)
    ttnn.deallocate(ttnn_slice_21, False)
    ttnn_from_device_24 = ttnn.from_device(ttnn_concat_7)
    ttnn.deallocate(ttnn_concat_7, False)
    ttnn_typecast_57 = ttnn.typecast(
        ttnn_from_device_24, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_24, False)
    ttnn_to_device_50 = ttnn.to_device(
        ttnn_typecast_57,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_57, False)
    return [ttnn_to_device_50]


def main_const_eval_28(arg, device):
    ttnn_to_device_51 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_26 = ttnn.to_layout(
        ttnn_to_device_51,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_51, False)
    ttnn_from_device_25 = ttnn.from_device(ttnn_to_layout_26)
    ttnn.deallocate(ttnn_to_layout_26, False)
    ttnn_typecast_58 = ttnn.typecast(
        ttnn_from_device_25, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_25, False)
    ttnn_to_device_52 = ttnn.to_device(
        ttnn_typecast_58,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_58, False)
    return [ttnn_to_device_52]


def main_const_eval_29(arg, device):
    ttnn_to_device_53 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_27 = ttnn.to_layout(
        ttnn_to_device_53,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_53, False)
    ttnn_from_device_26 = ttnn.from_device(ttnn_to_layout_27)
    ttnn.deallocate(ttnn_to_layout_27, False)
    ttnn_typecast_59 = ttnn.typecast(
        ttnn_from_device_26, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_26, False)
    ttnn_to_device_54 = ttnn.to_device(
        ttnn_typecast_59,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_59, False)
    return [ttnn_to_device_54]


def main_const_eval_30(arg, device):
    ttnn_to_device_55 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_28 = ttnn.to_layout(
        ttnn_to_device_55,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_55, False)
    ttnn_from_device_27 = ttnn.from_device(ttnn_to_layout_28)
    ttnn.deallocate(ttnn_to_layout_28, False)
    ttnn_typecast_60 = ttnn.typecast(
        ttnn_from_device_27, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_27, False)
    ttnn_to_device_56 = ttnn.to_device(
        ttnn_typecast_60,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_60, False)
    return [ttnn_to_device_56]


def main_const_eval_31(arg, device):
    ttnn_to_device_57 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_29 = ttnn.to_layout(
        ttnn_to_device_57,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_57, False)
    ttnn_from_device_28 = ttnn.from_device(ttnn_to_layout_29)
    ttnn.deallocate(ttnn_to_layout_29, False)
    ttnn_typecast_61 = ttnn.typecast(
        ttnn_from_device_28, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_28, False)
    ttnn_to_device_58 = ttnn.to_device(
        ttnn_typecast_61,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_61, False)
    return [ttnn_to_device_58]


def main_const_eval_32(arg, device):
    ttnn_to_device_59 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_30 = ttnn.to_layout(
        ttnn_to_device_59,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_59, False)
    ttnn_from_device_29 = ttnn.from_device(ttnn_to_layout_30)
    ttnn.deallocate(ttnn_to_layout_30, False)
    ttnn_typecast_62 = ttnn.typecast(
        ttnn_from_device_29, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_29, False)
    ttnn_to_device_60 = ttnn.to_device(
        ttnn_typecast_62,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_62, False)
    return [ttnn_to_device_60]


def main_const_eval_33(arg, device):
    ttnn_to_device_61 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_31 = ttnn.to_layout(
        ttnn_to_device_61,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_61, False)
    ttnn_from_device_30 = ttnn.from_device(ttnn_to_layout_31)
    ttnn.deallocate(ttnn_to_layout_31, False)
    ttnn_typecast_63 = ttnn.typecast(
        ttnn_from_device_30, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_30, False)
    ttnn_to_device_62 = ttnn.to_device(
        ttnn_typecast_63,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_63, False)
    return [ttnn_to_device_62]


def main_const_eval_34(arg, device):
    ttnn_typecast_64 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_65 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_66 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_8 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_64, ttnn_typecast_65, ttnn_typecast_66
    )
    ttnn.deallocate(ttnn_typecast_66, False)
    ttnn.deallocate(ttnn_typecast_65, False)
    ttnn.deallocate(ttnn_typecast_64, False)
    ttnn_typecast_67 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_8, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_8, False)
    ttnn_to_layout_32 = ttnn.to_layout(
        ttnn_typecast_67, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_67, False)
    ttnn_to_device_63 = ttnn.to_device(
        ttnn_to_layout_32,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_32, False)
    ttnn_slice_24 = ttnn.slice(
        ttnn_to_device_63,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_25 = ttnn.slice(
        ttnn_to_device_63,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_26 = ttnn.slice(
        ttnn_to_device_63,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_63, False)
    ttnn_concat_8 = ttnn.concat(
        [ttnn_slice_25, ttnn_slice_24, ttnn_slice_26],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_26, False)
    ttnn.deallocate(ttnn_slice_25, False)
    ttnn.deallocate(ttnn_slice_24, False)
    ttnn_from_device_31 = ttnn.from_device(ttnn_concat_8)
    ttnn.deallocate(ttnn_concat_8, False)
    ttnn_typecast_68 = ttnn.typecast(
        ttnn_from_device_31, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_31, False)
    ttnn_to_device_64 = ttnn.to_device(
        ttnn_typecast_68,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_68, False)
    return [ttnn_to_device_64]


def main_const_eval_35(arg, device):
    ttnn_to_device_65 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_33 = ttnn.to_layout(
        ttnn_to_device_65,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_65, False)
    ttnn_from_device_32 = ttnn.from_device(ttnn_to_layout_33)
    ttnn.deallocate(ttnn_to_layout_33, False)
    ttnn_typecast_69 = ttnn.typecast(
        ttnn_from_device_32, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_32, False)
    ttnn_to_device_66 = ttnn.to_device(
        ttnn_typecast_69,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_69, False)
    return [ttnn_to_device_66]


def main_const_eval_36(arg, device):
    ttnn_to_device_67 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_34 = ttnn.to_layout(
        ttnn_to_device_67,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_67, False)
    ttnn_from_device_33 = ttnn.from_device(ttnn_to_layout_34)
    ttnn.deallocate(ttnn_to_layout_34, False)
    ttnn_typecast_70 = ttnn.typecast(
        ttnn_from_device_33, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_33, False)
    ttnn_to_device_68 = ttnn.to_device(
        ttnn_typecast_70,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_70, False)
    return [ttnn_to_device_68]


def main_const_eval_37(arg, device):
    ttnn_typecast_71 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_72 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_73 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_9 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_71, ttnn_typecast_72, ttnn_typecast_73
    )
    ttnn.deallocate(ttnn_typecast_73, False)
    ttnn.deallocate(ttnn_typecast_72, False)
    ttnn.deallocate(ttnn_typecast_71, False)
    ttnn_typecast_74 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_9, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_9, False)
    ttnn_to_layout_35 = ttnn.to_layout(
        ttnn_typecast_74, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_74, False)
    ttnn_to_device_69 = ttnn.to_device(
        ttnn_to_layout_35,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_35, False)
    ttnn_slice_27 = ttnn.slice(
        ttnn_to_device_69,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_28 = ttnn.slice(
        ttnn_to_device_69,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_29 = ttnn.slice(
        ttnn_to_device_69,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_69, False)
    ttnn_concat_9 = ttnn.concat(
        [ttnn_slice_28, ttnn_slice_27, ttnn_slice_29],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_29, False)
    ttnn.deallocate(ttnn_slice_28, False)
    ttnn.deallocate(ttnn_slice_27, False)
    ttnn_from_device_34 = ttnn.from_device(ttnn_concat_9)
    ttnn.deallocate(ttnn_concat_9, False)
    ttnn_typecast_75 = ttnn.typecast(
        ttnn_from_device_34, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_34, False)
    ttnn_to_device_70 = ttnn.to_device(
        ttnn_typecast_75,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_75, False)
    return [ttnn_to_device_70]


def main_const_eval_38(arg, device):
    ttnn_to_device_71 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_36 = ttnn.to_layout(
        ttnn_to_device_71,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_71, False)
    ttnn_from_device_35 = ttnn.from_device(ttnn_to_layout_36)
    ttnn.deallocate(ttnn_to_layout_36, False)
    ttnn_typecast_76 = ttnn.typecast(
        ttnn_from_device_35, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_35, False)
    ttnn_to_device_72 = ttnn.to_device(
        ttnn_typecast_76,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_76, False)
    return [ttnn_to_device_72]


def main_const_eval_39(arg, device):
    ttnn_to_device_73 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_37 = ttnn.to_layout(
        ttnn_to_device_73,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_73, False)
    ttnn_from_device_36 = ttnn.from_device(ttnn_to_layout_37)
    ttnn.deallocate(ttnn_to_layout_37, False)
    ttnn_typecast_77 = ttnn.typecast(
        ttnn_from_device_36, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_36, False)
    ttnn_to_device_74 = ttnn.to_device(
        ttnn_typecast_77,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_77, False)
    return [ttnn_to_device_74]


def main_const_eval_40(arg, device):
    ttnn_to_device_75 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_38 = ttnn.to_layout(
        ttnn_to_device_75,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_75, False)
    ttnn_from_device_37 = ttnn.from_device(ttnn_to_layout_38)
    ttnn.deallocate(ttnn_to_layout_38, False)
    ttnn_typecast_78 = ttnn.typecast(
        ttnn_from_device_37, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_37, False)
    ttnn_to_device_76 = ttnn.to_device(
        ttnn_typecast_78,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_78, False)
    return [ttnn_to_device_76]


def main_const_eval_41(arg, device):
    ttnn_to_device_77 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_39 = ttnn.to_layout(
        ttnn_to_device_77,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_77, False)
    ttnn_from_device_38 = ttnn.from_device(ttnn_to_layout_39)
    ttnn.deallocate(ttnn_to_layout_39, False)
    ttnn_typecast_79 = ttnn.typecast(
        ttnn_from_device_38, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_38, False)
    ttnn_to_device_78 = ttnn.to_device(
        ttnn_typecast_79,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_79, False)
    return [ttnn_to_device_78]


def main_const_eval_42(arg, device):
    ttnn_to_device_79 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_40 = ttnn.to_layout(
        ttnn_to_device_79,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_79, False)
    ttnn_from_device_39 = ttnn.from_device(ttnn_to_layout_40)
    ttnn.deallocate(ttnn_to_layout_40, False)
    ttnn_typecast_80 = ttnn.typecast(
        ttnn_from_device_39, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_39, False)
    ttnn_to_device_80 = ttnn.to_device(
        ttnn_typecast_80,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_80, False)
    return [ttnn_to_device_80]


def main_const_eval_43(arg, device):
    ttnn_to_device_81 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_41 = ttnn.to_layout(
        ttnn_to_device_81,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_81, False)
    ttnn_from_device_40 = ttnn.from_device(ttnn_to_layout_41)
    ttnn.deallocate(ttnn_to_layout_41, False)
    ttnn_typecast_81 = ttnn.typecast(
        ttnn_from_device_40, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_40, False)
    ttnn_to_device_82 = ttnn.to_device(
        ttnn_typecast_81,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_81, False)
    return [ttnn_to_device_82]


def main_const_eval_44(arg, device):
    ttnn_to_device_83 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_42 = ttnn.to_layout(
        ttnn_to_device_83,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_83, False)
    ttnn_from_device_41 = ttnn.from_device(ttnn_to_layout_42)
    ttnn.deallocate(ttnn_to_layout_42, False)
    ttnn_typecast_82 = ttnn.typecast(
        ttnn_from_device_41, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_41, False)
    ttnn_to_device_84 = ttnn.to_device(
        ttnn_typecast_82,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_82, False)
    return [ttnn_to_device_84]


def main_const_eval_45(arg, device):
    ttnn_to_device_85 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_43 = ttnn.to_layout(
        ttnn_to_device_85,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_85, False)
    ttnn_from_device_42 = ttnn.from_device(ttnn_to_layout_43)
    ttnn.deallocate(ttnn_to_layout_43, False)
    ttnn_typecast_83 = ttnn.typecast(
        ttnn_from_device_42, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_42, False)
    ttnn_to_device_86 = ttnn.to_device(
        ttnn_typecast_83,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_83, False)
    return [ttnn_to_device_86]


def main_const_eval_46(arg, device):
    ttnn_to_device_87 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_44 = ttnn.to_layout(
        ttnn_to_device_87,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_87, False)
    ttnn_from_device_43 = ttnn.from_device(ttnn_to_layout_44)
    ttnn.deallocate(ttnn_to_layout_44, False)
    ttnn_typecast_84 = ttnn.typecast(
        ttnn_from_device_43, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_43, False)
    ttnn_to_device_88 = ttnn.to_device(
        ttnn_typecast_84,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_84, False)
    return [ttnn_to_device_88]


def main_const_eval_47(arg, device):
    ttnn_typecast_85 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_86 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_87 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_10 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_85, ttnn_typecast_86, ttnn_typecast_87
    )
    ttnn.deallocate(ttnn_typecast_87, False)
    ttnn.deallocate(ttnn_typecast_86, False)
    ttnn.deallocate(ttnn_typecast_85, False)
    ttnn_typecast_88 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_10, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_10, False)
    ttnn_to_layout_45 = ttnn.to_layout(
        ttnn_typecast_88, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_88, False)
    ttnn_to_device_89 = ttnn.to_device(
        ttnn_to_layout_45,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_45, False)
    ttnn_slice_30 = ttnn.slice(
        ttnn_to_device_89,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_31 = ttnn.slice(
        ttnn_to_device_89,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_32 = ttnn.slice(
        ttnn_to_device_89,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_89, False)
    ttnn_concat_10 = ttnn.concat(
        [ttnn_slice_31, ttnn_slice_30, ttnn_slice_32],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_32, False)
    ttnn.deallocate(ttnn_slice_31, False)
    ttnn.deallocate(ttnn_slice_30, False)
    ttnn_from_device_44 = ttnn.from_device(ttnn_concat_10)
    ttnn.deallocate(ttnn_concat_10, False)
    ttnn_typecast_89 = ttnn.typecast(
        ttnn_from_device_44, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_44, False)
    ttnn_to_device_90 = ttnn.to_device(
        ttnn_typecast_89,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_89, False)
    return [ttnn_to_device_90]


def main_const_eval_48(arg, device):
    ttnn_typecast_90 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_91 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_92 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_11 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_90, ttnn_typecast_91, ttnn_typecast_92
    )
    ttnn.deallocate(ttnn_typecast_92, False)
    ttnn.deallocate(ttnn_typecast_91, False)
    ttnn.deallocate(ttnn_typecast_90, False)
    ttnn_typecast_93 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_11, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_11, False)
    ttnn_to_layout_46 = ttnn.to_layout(
        ttnn_typecast_93, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_93, False)
    ttnn_to_device_91 = ttnn.to_device(
        ttnn_to_layout_46,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_46, False)
    ttnn_slice_33 = ttnn.slice(
        ttnn_to_device_91,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_34 = ttnn.slice(
        ttnn_to_device_91,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_35 = ttnn.slice(
        ttnn_to_device_91,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_91, False)
    ttnn_concat_11 = ttnn.concat(
        [ttnn_slice_34, ttnn_slice_33, ttnn_slice_35],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_35, False)
    ttnn.deallocate(ttnn_slice_34, False)
    ttnn.deallocate(ttnn_slice_33, False)
    ttnn_from_device_45 = ttnn.from_device(ttnn_concat_11)
    ttnn.deallocate(ttnn_concat_11, False)
    ttnn_typecast_94 = ttnn.typecast(
        ttnn_from_device_45, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_45, False)
    ttnn_to_device_92 = ttnn.to_device(
        ttnn_typecast_94,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_94, False)
    return [ttnn_to_device_92]


def main_const_eval_49(arg, device):
    ttnn_to_device_93 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_47 = ttnn.to_layout(
        ttnn_to_device_93,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_93, False)
    ttnn_from_device_46 = ttnn.from_device(ttnn_to_layout_47)
    ttnn.deallocate(ttnn_to_layout_47, False)
    ttnn_typecast_95 = ttnn.typecast(
        ttnn_from_device_46, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_46, False)
    ttnn_to_device_94 = ttnn.to_device(
        ttnn_typecast_95,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_95, False)
    return [ttnn_to_device_94]


def main_const_eval_50(device):
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_full_1]


def main_const_eval_51(arg, device):
    ttnn_to_device_95 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_48 = ttnn.to_layout(
        ttnn_to_device_95,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_95, False)
    ttnn_from_device_47 = ttnn.from_device(ttnn_to_layout_48)
    ttnn.deallocate(ttnn_to_layout_48, False)
    ttnn_typecast_96 = ttnn.typecast(
        ttnn_from_device_47, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_47, False)
    ttnn_to_device_96 = ttnn.to_device(
        ttnn_typecast_96,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_96, False)
    return [ttnn_to_device_96]


def main_const_eval_52(arg, device):
    ttnn_to_device_97 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_49 = ttnn.to_layout(
        ttnn_to_device_97,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_97, False)
    ttnn_from_device_48 = ttnn.from_device(ttnn_to_layout_49)
    ttnn.deallocate(ttnn_to_layout_49, False)
    ttnn_typecast_97 = ttnn.typecast(
        ttnn_from_device_48, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_48, False)
    ttnn_to_device_98 = ttnn.to_device(
        ttnn_typecast_97,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_97, False)
    return [ttnn_to_device_98]


def main_const_eval_53(arg, device):
    ttnn_to_device_99 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_50 = ttnn.to_layout(
        ttnn_to_device_99,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_99, False)
    ttnn_from_device_49 = ttnn.from_device(ttnn_to_layout_50)
    ttnn.deallocate(ttnn_to_layout_50, False)
    ttnn_typecast_98 = ttnn.typecast(
        ttnn_from_device_49, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_49, False)
    ttnn_to_device_100 = ttnn.to_device(
        ttnn_typecast_98,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_98, False)
    return [ttnn_to_device_100]


def main_const_eval_54(arg, device):
    ttnn_to_device_101 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_51 = ttnn.to_layout(
        ttnn_to_device_101,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_101, False)
    ttnn_from_device_50 = ttnn.from_device(ttnn_to_layout_51)
    ttnn.deallocate(ttnn_to_layout_51, False)
    ttnn_typecast_99 = ttnn.typecast(
        ttnn_from_device_50, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_50, False)
    ttnn_to_device_102 = ttnn.to_device(
        ttnn_typecast_99,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_99, False)
    return [ttnn_to_device_102]


def main_const_eval_55(arg, device):
    ttnn_to_device_103 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_52 = ttnn.to_layout(
        ttnn_to_device_103,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_103, False)
    ttnn_from_device_51 = ttnn.from_device(ttnn_to_layout_52)
    ttnn.deallocate(ttnn_to_layout_52, False)
    ttnn_typecast_100 = ttnn.typecast(
        ttnn_from_device_51, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_51, False)
    ttnn_to_device_104 = ttnn.to_device(
        ttnn_typecast_100,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_100, False)
    return [ttnn_to_device_104]


def main_const_eval_56(arg, device):
    ttnn_to_device_105 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_53 = ttnn.to_layout(
        ttnn_to_device_105,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_105, False)
    ttnn_from_device_52 = ttnn.from_device(ttnn_to_layout_53)
    ttnn.deallocate(ttnn_to_layout_53, False)
    ttnn_typecast_101 = ttnn.typecast(
        ttnn_from_device_52, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_52, False)
    ttnn_to_device_106 = ttnn.to_device(
        ttnn_typecast_101,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_101, False)
    return [ttnn_to_device_106]


def main_const_eval_57(arg, device):
    ttnn_to_device_107 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_54 = ttnn.to_layout(
        ttnn_to_device_107,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_107, False)
    ttnn_from_device_53 = ttnn.from_device(ttnn_to_layout_54)
    ttnn.deallocate(ttnn_to_layout_54, False)
    ttnn_typecast_102 = ttnn.typecast(
        ttnn_from_device_53, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_53, False)
    ttnn_to_device_108 = ttnn.to_device(
        ttnn_typecast_102,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_102, False)
    return [ttnn_to_device_108]


def main_const_eval_58(arg, device):
    ttnn_to_device_109 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_55 = ttnn.to_layout(
        ttnn_to_device_109,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_109, False)
    ttnn_from_device_54 = ttnn.from_device(ttnn_to_layout_55)
    ttnn.deallocate(ttnn_to_layout_55, False)
    ttnn_typecast_103 = ttnn.typecast(
        ttnn_from_device_54, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_54, False)
    ttnn_to_device_110 = ttnn.to_device(
        ttnn_typecast_103,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_103, False)
    return [ttnn_to_device_110]


def main_const_eval_59(arg, device):
    ttnn_to_device_111 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_56 = ttnn.to_layout(
        ttnn_to_device_111,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_111, False)
    ttnn_from_device_55 = ttnn.from_device(ttnn_to_layout_56)
    ttnn.deallocate(ttnn_to_layout_56, False)
    ttnn_typecast_104 = ttnn.typecast(
        ttnn_from_device_55, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_55, False)
    ttnn_to_device_112 = ttnn.to_device(
        ttnn_typecast_104,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_104, False)
    return [ttnn_to_device_112]


def main_const_eval_60(arg, device):
    ttnn_to_device_113 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_57 = ttnn.to_layout(
        ttnn_to_device_113,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_113, False)
    ttnn_from_device_56 = ttnn.from_device(ttnn_to_layout_57)
    ttnn.deallocate(ttnn_to_layout_57, False)
    ttnn_typecast_105 = ttnn.typecast(
        ttnn_from_device_56, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_56, False)
    ttnn_to_device_114 = ttnn.to_device(
        ttnn_typecast_105,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_105, False)
    return [ttnn_to_device_114]


def main_const_eval_61(device):
    ttnn_Tensor_1 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
        ],
        [1, 1, 1, 128],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_Tensor_1]


def main_const_eval_62(arg, device):
    ttnn_to_device_115 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_58 = ttnn.to_layout(
        ttnn_to_device_115,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_115, False)
    ttnn_from_device_57 = ttnn.from_device(ttnn_to_layout_58)
    ttnn.deallocate(ttnn_to_layout_58, False)
    ttnn_typecast_106 = ttnn.typecast(
        ttnn_from_device_57, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_57, False)
    ttnn_to_device_116 = ttnn.to_device(
        ttnn_typecast_106,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_106, False)
    return [ttnn_to_device_116]


def main_const_eval_63(arg, device):
    ttnn_to_device_117 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_59 = ttnn.to_layout(
        ttnn_to_device_117,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_117, False)
    ttnn_from_device_58 = ttnn.from_device(ttnn_to_layout_59)
    ttnn.deallocate(ttnn_to_layout_59, False)
    ttnn_typecast_107 = ttnn.typecast(
        ttnn_from_device_58, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_58, False)
    ttnn_to_device_118 = ttnn.to_device(
        ttnn_typecast_107,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_107, False)
    return [ttnn_to_device_118]


def main_const_eval_64(arg, device):
    ttnn_to_device_119 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_60 = ttnn.to_layout(
        ttnn_to_device_119,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_119, False)
    ttnn_from_device_59 = ttnn.from_device(ttnn_to_layout_60)
    ttnn.deallocate(ttnn_to_layout_60, False)
    ttnn_typecast_108 = ttnn.typecast(
        ttnn_from_device_59, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_59, False)
    ttnn_to_device_120 = ttnn.to_device(
        ttnn_typecast_108,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_108, False)
    return [ttnn_to_device_120]


def main_const_eval_65(arg, device):
    ttnn_to_device_121 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_to_device_121]


def main_const_eval_66(arg, device):
    ttnn_to_device_122 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_61 = ttnn.to_layout(
        ttnn_to_device_122,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_122, False)
    ttnn_from_device_60 = ttnn.from_device(ttnn_to_layout_61)
    ttnn.deallocate(ttnn_to_layout_61, False)
    ttnn_typecast_109 = ttnn.typecast(
        ttnn_from_device_60, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_60, False)
    ttnn_to_device_123 = ttnn.to_device(
        ttnn_typecast_109,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_109, False)
    return [ttnn_to_device_123]


def main_const_eval_67(arg, device):
    ttnn_to_device_124 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_62 = ttnn.to_layout(
        ttnn_to_device_124,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_124, False)
    ttnn_from_device_61 = ttnn.from_device(ttnn_to_layout_62)
    ttnn.deallocate(ttnn_to_layout_62, False)
    ttnn_typecast_110 = ttnn.typecast(
        ttnn_from_device_61, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_61, False)
    ttnn_to_device_125 = ttnn.to_device(
        ttnn_typecast_110,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_110, False)
    return [ttnn_to_device_125]


def main_const_eval_68(arg, device):
    ttnn_typecast_111 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_112 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_113 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_12 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_111, ttnn_typecast_112, ttnn_typecast_113
    )
    ttnn.deallocate(ttnn_typecast_113, False)
    ttnn.deallocate(ttnn_typecast_112, False)
    ttnn.deallocate(ttnn_typecast_111, False)
    ttnn_typecast_114 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_12, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_12, False)
    ttnn_to_layout_63 = ttnn.to_layout(
        ttnn_typecast_114, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_114, False)
    ttnn_to_device_126 = ttnn.to_device(
        ttnn_to_layout_63,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_63, False)
    ttnn_slice_36 = ttnn.slice(
        ttnn_to_device_126,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_37 = ttnn.slice(
        ttnn_to_device_126,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_38 = ttnn.slice(
        ttnn_to_device_126,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_126, False)
    ttnn_concat_12 = ttnn.concat(
        [ttnn_slice_37, ttnn_slice_36, ttnn_slice_38],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_38, False)
    ttnn.deallocate(ttnn_slice_37, False)
    ttnn.deallocate(ttnn_slice_36, False)
    ttnn_from_device_62 = ttnn.from_device(ttnn_concat_12)
    ttnn.deallocate(ttnn_concat_12, False)
    ttnn_typecast_115 = ttnn.typecast(
        ttnn_from_device_62, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_62, False)
    ttnn_to_device_127 = ttnn.to_device(
        ttnn_typecast_115,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_115, False)
    return [ttnn_to_device_127]


def main_const_eval_69(arg, device):
    ttnn_typecast_116 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_117 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_118 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_13 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_116, ttnn_typecast_117, ttnn_typecast_118
    )
    ttnn.deallocate(ttnn_typecast_118, False)
    ttnn.deallocate(ttnn_typecast_117, False)
    ttnn.deallocate(ttnn_typecast_116, False)
    ttnn_typecast_119 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_13, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_13, False)
    ttnn_to_layout_64 = ttnn.to_layout(
        ttnn_typecast_119, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_119, False)
    ttnn_to_device_128 = ttnn.to_device(
        ttnn_to_layout_64,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_64, False)
    ttnn_slice_39 = ttnn.slice(
        ttnn_to_device_128,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_40 = ttnn.slice(
        ttnn_to_device_128,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_41 = ttnn.slice(
        ttnn_to_device_128,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_128, False)
    ttnn_concat_13 = ttnn.concat(
        [ttnn_slice_40, ttnn_slice_39, ttnn_slice_41],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_41, False)
    ttnn.deallocate(ttnn_slice_40, False)
    ttnn.deallocate(ttnn_slice_39, False)
    ttnn_from_device_63 = ttnn.from_device(ttnn_concat_13)
    ttnn.deallocate(ttnn_concat_13, False)
    ttnn_typecast_120 = ttnn.typecast(
        ttnn_from_device_63, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_63, False)
    ttnn_to_device_129 = ttnn.to_device(
        ttnn_typecast_120,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_120, False)
    return [ttnn_to_device_129]


def main_const_eval_70(arg, device):
    ttnn_to_device_130 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_65 = ttnn.to_layout(
        ttnn_to_device_130,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_130, False)
    ttnn_from_device_64 = ttnn.from_device(ttnn_to_layout_65)
    ttnn.deallocate(ttnn_to_layout_65, False)
    ttnn_typecast_121 = ttnn.typecast(
        ttnn_from_device_64, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_64, False)
    ttnn_to_device_131 = ttnn.to_device(
        ttnn_typecast_121,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_121, False)
    return [ttnn_to_device_131]


def main_const_eval_71(arg, device):
    ttnn_to_device_132 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_66 = ttnn.to_layout(
        ttnn_to_device_132,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_132, False)
    ttnn_from_device_65 = ttnn.from_device(ttnn_to_layout_66)
    ttnn.deallocate(ttnn_to_layout_66, False)
    ttnn_typecast_122 = ttnn.typecast(
        ttnn_from_device_65, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_65, False)
    ttnn_to_device_133 = ttnn.to_device(
        ttnn_typecast_122,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_122, False)
    return [ttnn_to_device_133]


def main_const_eval_72(arg, device):
    ttnn_to_device_134 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_67 = ttnn.to_layout(
        ttnn_to_device_134,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_134, False)
    ttnn_from_device_66 = ttnn.from_device(ttnn_to_layout_67)
    ttnn.deallocate(ttnn_to_layout_67, False)
    ttnn_typecast_123 = ttnn.typecast(
        ttnn_from_device_66, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_66, False)
    ttnn_to_device_135 = ttnn.to_device(
        ttnn_typecast_123,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_123, False)
    return [ttnn_to_device_135]


def main_const_eval_73(arg, device):
    ttnn_to_device_136 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_68 = ttnn.to_layout(
        ttnn_to_device_136,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_136, False)
    ttnn_from_device_67 = ttnn.from_device(ttnn_to_layout_68)
    ttnn.deallocate(ttnn_to_layout_68, False)
    ttnn_typecast_124 = ttnn.typecast(
        ttnn_from_device_67, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_67, False)
    ttnn_to_device_137 = ttnn.to_device(
        ttnn_typecast_124,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_124, False)
    return [ttnn_to_device_137]


def main_const_eval_74(arg, device):
    ttnn_typecast_125 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_126 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_127 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_14 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_125, ttnn_typecast_126, ttnn_typecast_127
    )
    ttnn.deallocate(ttnn_typecast_127, False)
    ttnn.deallocate(ttnn_typecast_126, False)
    ttnn.deallocate(ttnn_typecast_125, False)
    ttnn_typecast_128 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_14, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_14, False)
    ttnn_to_layout_69 = ttnn.to_layout(
        ttnn_typecast_128, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_128, False)
    ttnn_to_device_138 = ttnn.to_device(
        ttnn_to_layout_69,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_69, False)
    ttnn_slice_42 = ttnn.slice(
        ttnn_to_device_138,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_43 = ttnn.slice(
        ttnn_to_device_138,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_44 = ttnn.slice(
        ttnn_to_device_138,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_138, False)
    ttnn_concat_14 = ttnn.concat(
        [ttnn_slice_43, ttnn_slice_42, ttnn_slice_44],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_44, False)
    ttnn.deallocate(ttnn_slice_43, False)
    ttnn.deallocate(ttnn_slice_42, False)
    ttnn_from_device_68 = ttnn.from_device(ttnn_concat_14)
    ttnn.deallocate(ttnn_concat_14, False)
    ttnn_typecast_129 = ttnn.typecast(
        ttnn_from_device_68, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_68, False)
    ttnn_to_device_139 = ttnn.to_device(
        ttnn_typecast_129,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_129, False)
    return [ttnn_to_device_139]


def main_const_eval_75(arg, device):
    ttnn_to_device_140 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_70 = ttnn.to_layout(
        ttnn_to_device_140,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_140, False)
    ttnn_from_device_69 = ttnn.from_device(ttnn_to_layout_70)
    ttnn.deallocate(ttnn_to_layout_70, False)
    ttnn_typecast_130 = ttnn.typecast(
        ttnn_from_device_69, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_69, False)
    ttnn_to_device_141 = ttnn.to_device(
        ttnn_typecast_130,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_130, False)
    return [ttnn_to_device_141]


def main_const_eval_76(arg, device):
    ttnn_to_device_142 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_71 = ttnn.to_layout(
        ttnn_to_device_142,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_142, False)
    ttnn_from_device_70 = ttnn.from_device(ttnn_to_layout_71)
    ttnn.deallocate(ttnn_to_layout_71, False)
    ttnn_typecast_131 = ttnn.typecast(
        ttnn_from_device_70, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_70, False)
    ttnn_to_device_143 = ttnn.to_device(
        ttnn_typecast_131,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_131, False)
    return [ttnn_to_device_143]


def main_const_eval_77(arg, device):
    ttnn_to_device_144 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_72 = ttnn.to_layout(
        ttnn_to_device_144,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_144, False)
    ttnn_from_device_71 = ttnn.from_device(ttnn_to_layout_72)
    ttnn.deallocate(ttnn_to_layout_72, False)
    ttnn_typecast_132 = ttnn.typecast(
        ttnn_from_device_71, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_71, False)
    ttnn_to_device_145 = ttnn.to_device(
        ttnn_typecast_132,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_132, False)
    return [ttnn_to_device_145]


def main_const_eval_78(arg, device):
    ttnn_to_device_146 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_73 = ttnn.to_layout(
        ttnn_to_device_146,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_146, False)
    ttnn_from_device_72 = ttnn.from_device(ttnn_to_layout_73)
    ttnn.deallocate(ttnn_to_layout_73, False)
    ttnn_typecast_133 = ttnn.typecast(
        ttnn_from_device_72, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_72, False)
    ttnn_to_device_147 = ttnn.to_device(
        ttnn_typecast_133,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_133, False)
    return [ttnn_to_device_147]


def main_const_eval_79(arg, device):
    ttnn_to_device_148 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_74 = ttnn.to_layout(
        ttnn_to_device_148,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_148, False)
    ttnn_from_device_73 = ttnn.from_device(ttnn_to_layout_74)
    ttnn.deallocate(ttnn_to_layout_74, False)
    ttnn_typecast_134 = ttnn.typecast(
        ttnn_from_device_73, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_73, False)
    ttnn_to_device_149 = ttnn.to_device(
        ttnn_typecast_134,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_134, False)
    return [ttnn_to_device_149]


def main_const_eval_80(arg, device):
    ttnn_to_device_150 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_75 = ttnn.to_layout(
        ttnn_to_device_150,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_150, False)
    ttnn_from_device_74 = ttnn.from_device(ttnn_to_layout_75)
    ttnn.deallocate(ttnn_to_layout_75, False)
    ttnn_typecast_135 = ttnn.typecast(
        ttnn_from_device_74, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_74, False)
    ttnn_to_device_151 = ttnn.to_device(
        ttnn_typecast_135,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_135, False)
    return [ttnn_to_device_151]


def main_const_eval_81(arg, device):
    ttnn_typecast_136 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_137 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_138 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_de619321_15 = cpu_hoisted_const_eval_de619321(
        ttnn_typecast_136, ttnn_typecast_137, ttnn_typecast_138
    )
    ttnn.deallocate(ttnn_typecast_138, False)
    ttnn.deallocate(ttnn_typecast_137, False)
    ttnn.deallocate(ttnn_typecast_136, False)
    ttnn_typecast_139 = ttnn.typecast(
        cpu_hoisted_const_eval_de619321_15, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_de619321_15, False)
    ttnn_to_layout_76 = ttnn.to_layout(
        ttnn_typecast_139, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_139, False)
    ttnn_to_device_152 = ttnn.to_device(
        ttnn_to_layout_76,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_76, False)
    ttnn_slice_45 = ttnn.slice(
        ttnn_to_device_152,
        [0, 0],
        [2048, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_46 = ttnn.slice(
        ttnn_to_device_152,
        [0, 512],
        [2048, 2560],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_47 = ttnn.slice(
        ttnn_to_device_152,
        [0, 2560],
        [2048, 3072],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_152, False)
    ttnn_concat_15 = ttnn.concat(
        [ttnn_slice_46, ttnn_slice_45, ttnn_slice_47],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_47, False)
    ttnn.deallocate(ttnn_slice_46, False)
    ttnn.deallocate(ttnn_slice_45, False)
    ttnn_from_device_75 = ttnn.from_device(ttnn_concat_15)
    ttnn.deallocate(ttnn_concat_15, False)
    ttnn_typecast_140 = ttnn.typecast(
        ttnn_from_device_75, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_75, False)
    ttnn_to_device_153 = ttnn.to_device(
        ttnn_typecast_140,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_140, False)
    return [ttnn_to_device_153]


def main_const_eval_82(arg, device):
    ttnn_to_device_154 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_77 = ttnn.to_layout(
        ttnn_to_device_154,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_154, False)
    ttnn_from_device_76 = ttnn.from_device(ttnn_to_layout_77)
    ttnn.deallocate(ttnn_to_layout_77, False)
    ttnn_typecast_141 = ttnn.typecast(
        ttnn_from_device_76, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_76, False)
    ttnn_to_device_155 = ttnn.to_device(
        ttnn_typecast_141,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_141, False)
    return [ttnn_to_device_155]


def main_const_eval_83(device):
    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_full_2]


def main_const_eval_84(arg, device):
    ttnn_to_device_156 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_78 = ttnn.to_layout(
        ttnn_to_device_156,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_156, False)
    ttnn_from_device_77 = ttnn.from_device(ttnn_to_layout_78)
    ttnn.deallocate(ttnn_to_layout_78, False)
    ttnn_typecast_142 = ttnn.typecast(
        ttnn_from_device_77, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_77, False)
    ttnn_to_device_157 = ttnn.to_device(
        ttnn_typecast_142,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_142, False)
    return [ttnn_to_device_157]


def main_const_eval_85(arg, device):
    ttnn_to_device_158 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_79 = ttnn.to_layout(
        ttnn_to_device_158,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_158, False)
    ttnn_from_device_78 = ttnn.from_device(ttnn_to_layout_79)
    ttnn.deallocate(ttnn_to_layout_79, False)
    ttnn_typecast_143 = ttnn.typecast(
        ttnn_from_device_78, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_78, False)
    ttnn_to_device_159 = ttnn.to_device(
        ttnn_typecast_143,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_143, False)
    return [ttnn_to_device_159]


def main_const_eval_86(arg, device):
    ttnn_to_device_160 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_80 = ttnn.to_layout(
        ttnn_to_device_160,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_160, False)
    ttnn_from_device_79 = ttnn.from_device(ttnn_to_layout_80)
    ttnn.deallocate(ttnn_to_layout_80, False)
    ttnn_typecast_144 = ttnn.typecast(
        ttnn_from_device_79, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_79, False)
    ttnn_to_device_161 = ttnn.to_device(
        ttnn_typecast_144,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_144, False)
    return [ttnn_to_device_161]


def main_const_eval_87(arg, device):
    ttnn_to_device_162 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_81 = ttnn.to_layout(
        ttnn_to_device_162,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_162, False)
    ttnn_from_device_80 = ttnn.from_device(ttnn_to_layout_81)
    ttnn.deallocate(ttnn_to_layout_81, False)
    ttnn_typecast_145 = ttnn.typecast(
        ttnn_from_device_80, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_80, False)
    ttnn_to_device_163 = ttnn.to_device(
        ttnn_typecast_145,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_145, False)
    return [ttnn_to_device_163]


ce_cache__main = {}


def _main(activations, weights, device):
    global ce_cache__main
    ce_cache__main = consteval__main(ce_cache__main, weights, device)
    args_1 = activations[0]
    args_0 = activations[1]
    args_2 = activations[2]
    args_3 = activations[3]
    args_4 = activations[4]
    args_5 = activations[5]
    args_6 = activations[6]
    args_7 = activations[7]
    args_8 = activations[8]
    args_9 = activations[9]
    args_10 = activations[10]
    args_11 = activations[11]
    args_12 = activations[12]
    args_13 = activations[13]
    args_14 = activations[14]
    args_15 = activations[15]
    args_16 = activations[16]
    args_17 = activations[17]
    args_18 = activations[18]
    args_19 = activations[19]
    args_20 = activations[20]
    args_21 = activations[21]
    args_22 = activations[22]
    args_23 = activations[23]
    args_24 = activations[24]
    args_25 = activations[25]
    args_26 = activations[26]
    args_27 = activations[27]
    args_28 = activations[28]
    args_29 = activations[29]
    args_30 = activations[30]
    args_31 = activations[31]
    args_32 = activations[32]
    args_33 = activations[33]
    args_34 = activations[34]
    args_35 = activations[35]
    args_36 = activations[36]
    args_37 = activations[37]
    args_38 = activations[38]
    args_39 = activations[39]
    args_40 = activations[40]
    args_41 = activations[41]
    args_42 = activations[42]
    args_43 = activations[43]
    args_44 = activations[44]
    args_45 = activations[45]
    args_46 = activations[46]
    args_47 = activations[47]
    args_48 = activations[48]
    var_0 = ce_cache__main["main_const_eval_8"]
    ttnn_to_memory_config_0 = ttnn.to_memory_config(
        ce_cache__main["main_const_eval_0"],
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_0 = ttnn.add(
        args_1,
        ttnn_to_memory_config_0,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_0, False)
    ttnn_typecast_146 = ttnn.typecast(
        args_0,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(args_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_typecast_146,
        [576],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_146, False)
    ttnn_to_memory_config_1 = ttnn.to_memory_config(
        ttnn_reshape_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_embedding_0 = ttnn.embedding(
        ttnn_to_memory_config_1,
        ce_cache__main["main_const_eval_65"],
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_1, False)
    ttnn_to_memory_config_2 = ttnn.to_memory_config(
        ttnn_embedding_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_rms_norm_0 = ttnn.rms_norm(
        ttnn_to_memory_config_2,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.0.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_2, False)
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_rms_norm_0,
        ce_cache__main["main_const_eval_47"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_matmul_0,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    v_1, v_2, v_3 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_1,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_to_memory_config_3 = ttnn.to_memory_config(
        ttnn_add_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_typecast_147 = ttnn.typecast(
        ttnn_to_memory_config_3,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_3, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_typecast_147,
        [1, 1, 18],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_147, False)
    ttnn_matmul_1 = ttnn.matmul(
        ce_cache__main["main_const_eval_25"],
        ttnn_reshape_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.FLOAT32,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=0,
            untilize_out=False,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_matmul_1,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_matmul_1, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_permute_0,
        [1, 1, 18, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_concat_16 = ttnn.concat(
        [ttnn_reshape_3, ttnn_reshape_3],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_3, False)
    ttnn_to_memory_config_4 = ttnn.to_memory_config(
        ttnn_concat_16,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_cos_0 = ttnn.cos(
        ttnn_to_memory_config_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_4, False)
    ttnn_typecast_148 = ttnn.typecast(
        ttnn_cos_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_cos_0, False)
    ttnn_to_memory_config_5 = ttnn.to_memory_config(
        ttnn_concat_16,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_concat_16, False)
    ttnn_sin_0 = ttnn.sin(
        ttnn_to_memory_config_5,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_5, False)
    ttnn_typecast_149 = ttnn.typecast(
        ttnn_sin_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_sin_0, False)
    ttnn_to_memory_config_6 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_7 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_0 = ttnn.experimental.rotary_embedding(
        v_2,
        ttnn_to_memory_config_7,
        ttnn_to_memory_config_6,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_7, False)
    ttnn.deallocate(ttnn_to_memory_config_6, False)
    ttnn.deallocate(v_2, False)
    ttnn_slice_48 = ttnn.slice(
        ttnn_experimental_rotary_embedding_0,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_0, False)
    ttnn_slice_49 = ttnn.slice(
        ttnn_slice_48,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_49, 0)
    ttnn.deallocate(ttnn_slice_49, False)
    ttnn_slice_50 = ttnn.slice(
        ttnn_slice_48,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_50, 1)
    ttnn.deallocate(ttnn_slice_50, False)
    ttnn_slice_51 = ttnn.slice(
        ttnn_slice_48,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_51, 2)
    ttnn.deallocate(ttnn_slice_51, False)
    ttnn_slice_52 = ttnn.slice(
        ttnn_slice_48,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_52, 3)
    ttnn.deallocate(ttnn_slice_52, False)
    ttnn_slice_53 = ttnn.slice(
        ttnn_slice_48,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_53, 4)
    ttnn.deallocate(ttnn_slice_53, False)
    ttnn_slice_54 = ttnn.slice(
        ttnn_slice_48,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_54, 5)
    ttnn.deallocate(ttnn_slice_54, False)
    ttnn_slice_55 = ttnn.slice(
        ttnn_slice_48,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_55, 6)
    ttnn.deallocate(ttnn_slice_55, False)
    ttnn_slice_56 = ttnn.slice(
        ttnn_slice_48,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_56, 7)
    ttnn.deallocate(ttnn_slice_56, False)
    ttnn_slice_57 = ttnn.slice(
        ttnn_slice_48,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_57, 8)
    ttnn.deallocate(ttnn_slice_57, False)
    ttnn_slice_58 = ttnn.slice(
        ttnn_slice_48,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_58, 9)
    ttnn.deallocate(ttnn_slice_58, False)
    ttnn_slice_59 = ttnn.slice(
        ttnn_slice_48,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_59, 10)
    ttnn.deallocate(ttnn_slice_59, False)
    ttnn_slice_60 = ttnn.slice(
        ttnn_slice_48,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_60, 11)
    ttnn.deallocate(ttnn_slice_60, False)
    ttnn_slice_61 = ttnn.slice(
        ttnn_slice_48,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_61, 12)
    ttnn.deallocate(ttnn_slice_61, False)
    ttnn_slice_62 = ttnn.slice(
        ttnn_slice_48,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_62, 13)
    ttnn.deallocate(ttnn_slice_62, False)
    ttnn_slice_63 = ttnn.slice(
        ttnn_slice_48,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_63, 14)
    ttnn.deallocate(ttnn_slice_63, False)
    ttnn_slice_64 = ttnn.slice(
        ttnn_slice_48,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_64, 15)
    ttnn.deallocate(ttnn_slice_64, False)
    ttnn_slice_65 = ttnn.slice(
        ttnn_slice_48,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_65, 16)
    ttnn.deallocate(ttnn_slice_65, False)
    ttnn_slice_66 = ttnn.slice(
        ttnn_slice_48,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_66, 17)
    ttnn.deallocate(ttnn_slice_66, False)
    ttnn_slice_67 = ttnn.slice(
        ttnn_slice_48,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_67, 18)
    ttnn.deallocate(ttnn_slice_67, False)
    ttnn_slice_68 = ttnn.slice(
        ttnn_slice_48,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_68, 19)
    ttnn.deallocate(ttnn_slice_68, False)
    ttnn_slice_69 = ttnn.slice(
        ttnn_slice_48,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_69, 20)
    ttnn.deallocate(ttnn_slice_69, False)
    ttnn_slice_70 = ttnn.slice(
        ttnn_slice_48,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_70, 21)
    ttnn.deallocate(ttnn_slice_70, False)
    ttnn_slice_71 = ttnn.slice(
        ttnn_slice_48,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_71, 22)
    ttnn.deallocate(ttnn_slice_71, False)
    ttnn_slice_72 = ttnn.slice(
        ttnn_slice_48,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_72, 23)
    ttnn.deallocate(ttnn_slice_72, False)
    ttnn_slice_73 = ttnn.slice(
        ttnn_slice_48,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_73, 24)
    ttnn.deallocate(ttnn_slice_73, False)
    ttnn_slice_74 = ttnn.slice(
        ttnn_slice_48,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_74, 25)
    ttnn.deallocate(ttnn_slice_74, False)
    ttnn_slice_75 = ttnn.slice(
        ttnn_slice_48,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_75, 26)
    ttnn.deallocate(ttnn_slice_75, False)
    ttnn_slice_76 = ttnn.slice(
        ttnn_slice_48,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_76, 27)
    ttnn.deallocate(ttnn_slice_76, False)
    ttnn_slice_77 = ttnn.slice(
        ttnn_slice_48,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_77, 28)
    ttnn.deallocate(ttnn_slice_77, False)
    ttnn_slice_78 = ttnn.slice(
        ttnn_slice_48,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_78, 29)
    ttnn.deallocate(ttnn_slice_78, False)
    ttnn_slice_79 = ttnn.slice(
        ttnn_slice_48,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_2, ttnn_slice_79, 30)
    ttnn.deallocate(ttnn_slice_79, False)
    ttnn_slice_80 = ttnn.slice(
        ttnn_slice_48,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_48, False)
    ttnn.fill_cache(args_2, ttnn_slice_80, 31)
    ttnn.deallocate(ttnn_slice_80, False)
    ttnn_slice_81 = ttnn.slice(
        v_3,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_81, 0)
    ttnn.deallocate(ttnn_slice_81, False)
    ttnn_slice_82 = ttnn.slice(
        v_3,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_82, 1)
    ttnn.deallocate(ttnn_slice_82, False)
    ttnn_slice_83 = ttnn.slice(
        v_3,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_83, 2)
    ttnn.deallocate(ttnn_slice_83, False)
    ttnn_slice_84 = ttnn.slice(
        v_3,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_84, 3)
    ttnn.deallocate(ttnn_slice_84, False)
    ttnn_slice_85 = ttnn.slice(
        v_3,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_85, 4)
    ttnn.deallocate(ttnn_slice_85, False)
    ttnn_slice_86 = ttnn.slice(
        v_3,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_86, 5)
    ttnn.deallocate(ttnn_slice_86, False)
    ttnn_slice_87 = ttnn.slice(
        v_3,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_87, 6)
    ttnn.deallocate(ttnn_slice_87, False)
    ttnn_slice_88 = ttnn.slice(
        v_3,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_88, 7)
    ttnn.deallocate(ttnn_slice_88, False)
    ttnn_slice_89 = ttnn.slice(
        v_3,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_89, 8)
    ttnn.deallocate(ttnn_slice_89, False)
    ttnn_slice_90 = ttnn.slice(
        v_3,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_90, 9)
    ttnn.deallocate(ttnn_slice_90, False)
    ttnn_slice_91 = ttnn.slice(
        v_3,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_91, 10)
    ttnn.deallocate(ttnn_slice_91, False)
    ttnn_slice_92 = ttnn.slice(
        v_3,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_92, 11)
    ttnn.deallocate(ttnn_slice_92, False)
    ttnn_slice_93 = ttnn.slice(
        v_3,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_93, 12)
    ttnn.deallocate(ttnn_slice_93, False)
    ttnn_slice_94 = ttnn.slice(
        v_3,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_94, 13)
    ttnn.deallocate(ttnn_slice_94, False)
    ttnn_slice_95 = ttnn.slice(
        v_3,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_95, 14)
    ttnn.deallocate(ttnn_slice_95, False)
    ttnn_slice_96 = ttnn.slice(
        v_3,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_96, 15)
    ttnn.deallocate(ttnn_slice_96, False)
    ttnn_slice_97 = ttnn.slice(
        v_3,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_97, 16)
    ttnn.deallocate(ttnn_slice_97, False)
    ttnn_slice_98 = ttnn.slice(
        v_3,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_98, 17)
    ttnn.deallocate(ttnn_slice_98, False)
    ttnn_slice_99 = ttnn.slice(
        v_3,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_99, 18)
    ttnn.deallocate(ttnn_slice_99, False)
    ttnn_slice_100 = ttnn.slice(
        v_3,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_100, 19)
    ttnn.deallocate(ttnn_slice_100, False)
    ttnn_slice_101 = ttnn.slice(
        v_3,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_101, 20)
    ttnn.deallocate(ttnn_slice_101, False)
    ttnn_slice_102 = ttnn.slice(
        v_3,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_102, 21)
    ttnn.deallocate(ttnn_slice_102, False)
    ttnn_slice_103 = ttnn.slice(
        v_3,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_103, 22)
    ttnn.deallocate(ttnn_slice_103, False)
    ttnn_slice_104 = ttnn.slice(
        v_3,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_104, 23)
    ttnn.deallocate(ttnn_slice_104, False)
    ttnn_slice_105 = ttnn.slice(
        v_3,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_105, 24)
    ttnn.deallocate(ttnn_slice_105, False)
    ttnn_slice_106 = ttnn.slice(
        v_3,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_106, 25)
    ttnn.deallocate(ttnn_slice_106, False)
    ttnn_slice_107 = ttnn.slice(
        v_3,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_107, 26)
    ttnn.deallocate(ttnn_slice_107, False)
    ttnn_slice_108 = ttnn.slice(
        v_3,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_108, 27)
    ttnn.deallocate(ttnn_slice_108, False)
    ttnn_slice_109 = ttnn.slice(
        v_3,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_109, 28)
    ttnn.deallocate(ttnn_slice_109, False)
    ttnn_slice_110 = ttnn.slice(
        v_3,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_110, 29)
    ttnn.deallocate(ttnn_slice_110, False)
    ttnn_slice_111 = ttnn.slice(
        v_3,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_3, ttnn_slice_111, 30)
    ttnn.deallocate(ttnn_slice_111, False)
    ttnn_slice_112 = ttnn.slice(
        v_3,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_3, False)
    ttnn.fill_cache(args_3, ttnn_slice_112, 31)
    ttnn.deallocate(ttnn_slice_112, False)
    ttnn_to_memory_config_8 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_1 = ttnn.add(
        args_1,
        ttnn_to_memory_config_8,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_8, False)
    ttnn.deallocate(args_1, False)
    ttnn_to_memory_config_9 = ttnn.to_memory_config(
        ttnn_add_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_1, False)
    ttnn_to_memory_config_10 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_11 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_1 = ttnn.experimental.rotary_embedding(
        v_1,
        ttnn_to_memory_config_11,
        ttnn_to_memory_config_10,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_11, False)
    ttnn.deallocate(ttnn_to_memory_config_10, False)
    ttnn.deallocate(v_1, False)
    ttnn_slice_113 = ttnn.slice(
        ttnn_experimental_rotary_embedding_1,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_1, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_add_0,
        [1, 1, 18, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_to_memory_config_12 = ttnn.to_memory_config(
        ttnn_reshape_4,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_4, False)
    ttnn_to_memory_config_13 = ttnn.to_memory_config(
        ce_cache__main["main_const_eval_61"],
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_ge_0 = ttnn.ge(
        ttnn_to_memory_config_12,
        ttnn_to_memory_config_13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_13, False)
    ttnn.deallocate(ttnn_to_memory_config_12, False)
    ttnn_to_memory_config_14 = ttnn.to_memory_config(
        ce_cache__main["main_const_eval_50"],
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_15 = ttnn.to_memory_config(
        ce_cache__main["main_const_eval_83"],
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_where_0 = ttnn.where(
        ttnn_ge_0,
        ttnn_to_memory_config_15,
        ttnn_to_memory_config_14,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_15, False)
    ttnn.deallocate(ttnn_to_memory_config_14, False)
    ttnn.deallocate(ttnn_ge_0, False)
    ttnn_to_memory_config_16 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_0 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_113,
            args_2,
            args_3,
            attn_mask=ttnn_to_memory_config_16,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_16, False)
    ttnn.deallocate(ttnn_slice_113, False)
    ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_0, False)
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_0,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_0, False)
    ttnn_matmul_2 = ttnn.matmul(
        ttnn_reshape_5,
        ce_cache__main["main_const_eval_18"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_5, False)
    ttnn_add_2 = ttnn.add(
        ttnn_matmul_2,
        ttnn_embedding_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_2, False)
    ttnn.deallocate(ttnn_embedding_0, False)
    ttnn_rms_norm_1 = ttnn.rms_norm(
        ttnn_add_2,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.0.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_3 = ttnn.matmul(
        ttnn_rms_norm_1,
        ce_cache__main["main_const_eval_30"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_4 = ttnn.matmul(
        ttnn_rms_norm_1,
        ce_cache__main["main_const_eval_40"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_1, False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_matmul_3,
        ttnn_matmul_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_4, False)
    ttnn.deallocate(ttnn_matmul_3, False)
    ttnn_matmul_5 = ttnn.matmul(
        ttnn_multiply_0,
        ce_cache__main["main_const_eval_13"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_0, False)
    ttnn_add_3 = ttnn.add(
        ttnn_matmul_5,
        ttnn_add_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_5, False)
    ttnn.deallocate(ttnn_add_2, False)
    ttnn_rms_norm_2 = ttnn.rms_norm(
        ttnn_add_3,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.1.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_6 = ttnn.matmul(
        ttnn_rms_norm_2,
        ce_cache__main["main_const_eval_15"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_2, False)
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_matmul_6,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_6, False)
    v_4, v_5, v_6 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_6,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_6, False)
    ttnn_to_memory_config_17 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_18 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_2 = ttnn.experimental.rotary_embedding(
        v_5,
        ttnn_to_memory_config_18,
        ttnn_to_memory_config_17,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_18, False)
    ttnn.deallocate(ttnn_to_memory_config_17, False)
    ttnn.deallocate(v_5, False)
    ttnn_slice_114 = ttnn.slice(
        ttnn_experimental_rotary_embedding_2,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_2, False)
    ttnn_slice_115 = ttnn.slice(
        ttnn_slice_114,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_115, 0)
    ttnn.deallocate(ttnn_slice_115, False)
    ttnn_slice_116 = ttnn.slice(
        ttnn_slice_114,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_116, 1)
    ttnn.deallocate(ttnn_slice_116, False)
    ttnn_slice_117 = ttnn.slice(
        ttnn_slice_114,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_117, 2)
    ttnn.deallocate(ttnn_slice_117, False)
    ttnn_slice_118 = ttnn.slice(
        ttnn_slice_114,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_118, 3)
    ttnn.deallocate(ttnn_slice_118, False)
    ttnn_slice_119 = ttnn.slice(
        ttnn_slice_114,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_119, 4)
    ttnn.deallocate(ttnn_slice_119, False)
    ttnn_slice_120 = ttnn.slice(
        ttnn_slice_114,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_120, 5)
    ttnn.deallocate(ttnn_slice_120, False)
    ttnn_slice_121 = ttnn.slice(
        ttnn_slice_114,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_121, 6)
    ttnn.deallocate(ttnn_slice_121, False)
    ttnn_slice_122 = ttnn.slice(
        ttnn_slice_114,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_122, 7)
    ttnn.deallocate(ttnn_slice_122, False)
    ttnn_slice_123 = ttnn.slice(
        ttnn_slice_114,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_123, 8)
    ttnn.deallocate(ttnn_slice_123, False)
    ttnn_slice_124 = ttnn.slice(
        ttnn_slice_114,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_124, 9)
    ttnn.deallocate(ttnn_slice_124, False)
    ttnn_slice_125 = ttnn.slice(
        ttnn_slice_114,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_125, 10)
    ttnn.deallocate(ttnn_slice_125, False)
    ttnn_slice_126 = ttnn.slice(
        ttnn_slice_114,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_126, 11)
    ttnn.deallocate(ttnn_slice_126, False)
    ttnn_slice_127 = ttnn.slice(
        ttnn_slice_114,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_127, 12)
    ttnn.deallocate(ttnn_slice_127, False)
    ttnn_slice_128 = ttnn.slice(
        ttnn_slice_114,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_128, 13)
    ttnn.deallocate(ttnn_slice_128, False)
    ttnn_slice_129 = ttnn.slice(
        ttnn_slice_114,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_129, 14)
    ttnn.deallocate(ttnn_slice_129, False)
    ttnn_slice_130 = ttnn.slice(
        ttnn_slice_114,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_130, 15)
    ttnn.deallocate(ttnn_slice_130, False)
    ttnn_slice_131 = ttnn.slice(
        ttnn_slice_114,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_131, 16)
    ttnn.deallocate(ttnn_slice_131, False)
    ttnn_slice_132 = ttnn.slice(
        ttnn_slice_114,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_132, 17)
    ttnn.deallocate(ttnn_slice_132, False)
    ttnn_slice_133 = ttnn.slice(
        ttnn_slice_114,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_133, 18)
    ttnn.deallocate(ttnn_slice_133, False)
    ttnn_slice_134 = ttnn.slice(
        ttnn_slice_114,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_134, 19)
    ttnn.deallocate(ttnn_slice_134, False)
    ttnn_slice_135 = ttnn.slice(
        ttnn_slice_114,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_135, 20)
    ttnn.deallocate(ttnn_slice_135, False)
    ttnn_slice_136 = ttnn.slice(
        ttnn_slice_114,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_136, 21)
    ttnn.deallocate(ttnn_slice_136, False)
    ttnn_slice_137 = ttnn.slice(
        ttnn_slice_114,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_137, 22)
    ttnn.deallocate(ttnn_slice_137, False)
    ttnn_slice_138 = ttnn.slice(
        ttnn_slice_114,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_138, 23)
    ttnn.deallocate(ttnn_slice_138, False)
    ttnn_slice_139 = ttnn.slice(
        ttnn_slice_114,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_139, 24)
    ttnn.deallocate(ttnn_slice_139, False)
    ttnn_slice_140 = ttnn.slice(
        ttnn_slice_114,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_140, 25)
    ttnn.deallocate(ttnn_slice_140, False)
    ttnn_slice_141 = ttnn.slice(
        ttnn_slice_114,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_141, 26)
    ttnn.deallocate(ttnn_slice_141, False)
    ttnn_slice_142 = ttnn.slice(
        ttnn_slice_114,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_142, 27)
    ttnn.deallocate(ttnn_slice_142, False)
    ttnn_slice_143 = ttnn.slice(
        ttnn_slice_114,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_143, 28)
    ttnn.deallocate(ttnn_slice_143, False)
    ttnn_slice_144 = ttnn.slice(
        ttnn_slice_114,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_144, 29)
    ttnn.deallocate(ttnn_slice_144, False)
    ttnn_slice_145 = ttnn.slice(
        ttnn_slice_114,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_5, ttnn_slice_145, 30)
    ttnn.deallocate(ttnn_slice_145, False)
    ttnn_slice_146 = ttnn.slice(
        ttnn_slice_114,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_114, False)
    ttnn.fill_cache(args_5, ttnn_slice_146, 31)
    ttnn.deallocate(ttnn_slice_146, False)
    ttnn_slice_147 = ttnn.slice(
        v_6,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_147, 0)
    ttnn.deallocate(ttnn_slice_147, False)
    ttnn_slice_148 = ttnn.slice(
        v_6,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_148, 1)
    ttnn.deallocate(ttnn_slice_148, False)
    ttnn_slice_149 = ttnn.slice(
        v_6,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_149, 2)
    ttnn.deallocate(ttnn_slice_149, False)
    ttnn_slice_150 = ttnn.slice(
        v_6,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_150, 3)
    ttnn.deallocate(ttnn_slice_150, False)
    ttnn_slice_151 = ttnn.slice(
        v_6,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_151, 4)
    ttnn.deallocate(ttnn_slice_151, False)
    ttnn_slice_152 = ttnn.slice(
        v_6,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_152, 5)
    ttnn.deallocate(ttnn_slice_152, False)
    ttnn_slice_153 = ttnn.slice(
        v_6,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_153, 6)
    ttnn.deallocate(ttnn_slice_153, False)
    ttnn_slice_154 = ttnn.slice(
        v_6,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_154, 7)
    ttnn.deallocate(ttnn_slice_154, False)
    ttnn_slice_155 = ttnn.slice(
        v_6,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_155, 8)
    ttnn.deallocate(ttnn_slice_155, False)
    ttnn_slice_156 = ttnn.slice(
        v_6,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_156, 9)
    ttnn.deallocate(ttnn_slice_156, False)
    ttnn_slice_157 = ttnn.slice(
        v_6,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_157, 10)
    ttnn.deallocate(ttnn_slice_157, False)
    ttnn_slice_158 = ttnn.slice(
        v_6,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_158, 11)
    ttnn.deallocate(ttnn_slice_158, False)
    ttnn_slice_159 = ttnn.slice(
        v_6,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_159, 12)
    ttnn.deallocate(ttnn_slice_159, False)
    ttnn_slice_160 = ttnn.slice(
        v_6,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_160, 13)
    ttnn.deallocate(ttnn_slice_160, False)
    ttnn_slice_161 = ttnn.slice(
        v_6,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_161, 14)
    ttnn.deallocate(ttnn_slice_161, False)
    ttnn_slice_162 = ttnn.slice(
        v_6,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_162, 15)
    ttnn.deallocate(ttnn_slice_162, False)
    ttnn_slice_163 = ttnn.slice(
        v_6,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_163, 16)
    ttnn.deallocate(ttnn_slice_163, False)
    ttnn_slice_164 = ttnn.slice(
        v_6,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_164, 17)
    ttnn.deallocate(ttnn_slice_164, False)
    ttnn_slice_165 = ttnn.slice(
        v_6,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_165, 18)
    ttnn.deallocate(ttnn_slice_165, False)
    ttnn_slice_166 = ttnn.slice(
        v_6,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_166, 19)
    ttnn.deallocate(ttnn_slice_166, False)
    ttnn_slice_167 = ttnn.slice(
        v_6,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_167, 20)
    ttnn.deallocate(ttnn_slice_167, False)
    ttnn_slice_168 = ttnn.slice(
        v_6,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_168, 21)
    ttnn.deallocate(ttnn_slice_168, False)
    ttnn_slice_169 = ttnn.slice(
        v_6,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_169, 22)
    ttnn.deallocate(ttnn_slice_169, False)
    ttnn_slice_170 = ttnn.slice(
        v_6,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_170, 23)
    ttnn.deallocate(ttnn_slice_170, False)
    ttnn_slice_171 = ttnn.slice(
        v_6,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_171, 24)
    ttnn.deallocate(ttnn_slice_171, False)
    ttnn_slice_172 = ttnn.slice(
        v_6,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_172, 25)
    ttnn.deallocate(ttnn_slice_172, False)
    ttnn_slice_173 = ttnn.slice(
        v_6,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_173, 26)
    ttnn.deallocate(ttnn_slice_173, False)
    ttnn_slice_174 = ttnn.slice(
        v_6,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_174, 27)
    ttnn.deallocate(ttnn_slice_174, False)
    ttnn_slice_175 = ttnn.slice(
        v_6,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_175, 28)
    ttnn.deallocate(ttnn_slice_175, False)
    ttnn_slice_176 = ttnn.slice(
        v_6,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_176, 29)
    ttnn.deallocate(ttnn_slice_176, False)
    ttnn_slice_177 = ttnn.slice(
        v_6,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_6, ttnn_slice_177, 30)
    ttnn.deallocate(ttnn_slice_177, False)
    ttnn_slice_178 = ttnn.slice(
        v_6,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_6, False)
    ttnn.fill_cache(args_6, ttnn_slice_178, 31)
    ttnn.deallocate(ttnn_slice_178, False)
    ttnn_to_memory_config_19 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_4 = ttnn.add(
        args_4,
        ttnn_to_memory_config_19,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_19, False)
    ttnn.deallocate(args_4, False)
    ttnn_to_memory_config_20 = ttnn.to_memory_config(
        ttnn_add_4,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_4, False)
    ttnn_to_memory_config_21 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_22 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_3 = ttnn.experimental.rotary_embedding(
        v_4,
        ttnn_to_memory_config_22,
        ttnn_to_memory_config_21,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_22, False)
    ttnn.deallocate(ttnn_to_memory_config_21, False)
    ttnn.deallocate(v_4, False)
    ttnn_slice_179 = ttnn.slice(
        ttnn_experimental_rotary_embedding_3,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_3, False)
    ttnn_to_memory_config_23 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_1 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_179,
            args_5,
            args_6,
            attn_mask=ttnn_to_memory_config_23,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_23, False)
    ttnn.deallocate(ttnn_slice_179, False)
    ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_1, False)
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_1,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_1, False)
    ttnn_matmul_7 = ttnn.matmul(
        ttnn_reshape_7,
        ce_cache__main["main_const_eval_78"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_7, False)
    ttnn_add_5 = ttnn.add(
        ttnn_matmul_7,
        ttnn_add_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_7, False)
    ttnn.deallocate(ttnn_add_3, False)
    ttnn_rms_norm_3 = ttnn.rms_norm(
        ttnn_add_5,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.1.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_8 = ttnn.matmul(
        ttnn_rms_norm_3,
        ce_cache__main["main_const_eval_80"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_9 = ttnn.matmul(
        ttnn_rms_norm_3,
        ce_cache__main["main_const_eval_20"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_3, False)
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_matmul_8,
        ttnn_matmul_9,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_9, False)
    ttnn.deallocate(ttnn_matmul_8, False)
    ttnn_matmul_10 = ttnn.matmul(
        ttnn_multiply_1,
        ce_cache__main["main_const_eval_33"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_1, False)
    ttnn_add_6 = ttnn.add(
        ttnn_matmul_10,
        ttnn_add_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_10, False)
    ttnn.deallocate(ttnn_add_5, False)
    ttnn_rms_norm_4 = ttnn.rms_norm(
        ttnn_add_6,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.2.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_11 = ttnn.matmul(
        ttnn_rms_norm_4,
        ce_cache__main["main_const_eval_34"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_4, False)
    ttnn_reshape_8 = ttnn.reshape(
        ttnn_matmul_11,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_11, False)
    v_7, v_8, v_9 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_8,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_8, False)
    ttnn_to_memory_config_24 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_25 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_4 = ttnn.experimental.rotary_embedding(
        v_8,
        ttnn_to_memory_config_25,
        ttnn_to_memory_config_24,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_25, False)
    ttnn.deallocate(ttnn_to_memory_config_24, False)
    ttnn.deallocate(v_8, False)
    ttnn_slice_180 = ttnn.slice(
        ttnn_experimental_rotary_embedding_4,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_4, False)
    ttnn_slice_181 = ttnn.slice(
        ttnn_slice_180,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_181, 0)
    ttnn.deallocate(ttnn_slice_181, False)
    ttnn_slice_182 = ttnn.slice(
        ttnn_slice_180,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_182, 1)
    ttnn.deallocate(ttnn_slice_182, False)
    ttnn_slice_183 = ttnn.slice(
        ttnn_slice_180,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_183, 2)
    ttnn.deallocate(ttnn_slice_183, False)
    ttnn_slice_184 = ttnn.slice(
        ttnn_slice_180,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_184, 3)
    ttnn.deallocate(ttnn_slice_184, False)
    ttnn_slice_185 = ttnn.slice(
        ttnn_slice_180,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_185, 4)
    ttnn.deallocate(ttnn_slice_185, False)
    ttnn_slice_186 = ttnn.slice(
        ttnn_slice_180,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_186, 5)
    ttnn.deallocate(ttnn_slice_186, False)
    ttnn_slice_187 = ttnn.slice(
        ttnn_slice_180,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_187, 6)
    ttnn.deallocate(ttnn_slice_187, False)
    ttnn_slice_188 = ttnn.slice(
        ttnn_slice_180,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_188, 7)
    ttnn.deallocate(ttnn_slice_188, False)
    ttnn_slice_189 = ttnn.slice(
        ttnn_slice_180,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_189, 8)
    ttnn.deallocate(ttnn_slice_189, False)
    ttnn_slice_190 = ttnn.slice(
        ttnn_slice_180,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_190, 9)
    ttnn.deallocate(ttnn_slice_190, False)
    ttnn_slice_191 = ttnn.slice(
        ttnn_slice_180,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_191, 10)
    ttnn.deallocate(ttnn_slice_191, False)
    ttnn_slice_192 = ttnn.slice(
        ttnn_slice_180,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_192, 11)
    ttnn.deallocate(ttnn_slice_192, False)
    ttnn_slice_193 = ttnn.slice(
        ttnn_slice_180,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_193, 12)
    ttnn.deallocate(ttnn_slice_193, False)
    ttnn_slice_194 = ttnn.slice(
        ttnn_slice_180,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_194, 13)
    ttnn.deallocate(ttnn_slice_194, False)
    ttnn_slice_195 = ttnn.slice(
        ttnn_slice_180,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_195, 14)
    ttnn.deallocate(ttnn_slice_195, False)
    ttnn_slice_196 = ttnn.slice(
        ttnn_slice_180,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_196, 15)
    ttnn.deallocate(ttnn_slice_196, False)
    ttnn_slice_197 = ttnn.slice(
        ttnn_slice_180,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_197, 16)
    ttnn.deallocate(ttnn_slice_197, False)
    ttnn_slice_198 = ttnn.slice(
        ttnn_slice_180,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_198, 17)
    ttnn.deallocate(ttnn_slice_198, False)
    ttnn_slice_199 = ttnn.slice(
        ttnn_slice_180,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_199, 18)
    ttnn.deallocate(ttnn_slice_199, False)
    ttnn_slice_200 = ttnn.slice(
        ttnn_slice_180,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_200, 19)
    ttnn.deallocate(ttnn_slice_200, False)
    ttnn_slice_201 = ttnn.slice(
        ttnn_slice_180,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_201, 20)
    ttnn.deallocate(ttnn_slice_201, False)
    ttnn_slice_202 = ttnn.slice(
        ttnn_slice_180,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_202, 21)
    ttnn.deallocate(ttnn_slice_202, False)
    ttnn_slice_203 = ttnn.slice(
        ttnn_slice_180,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_203, 22)
    ttnn.deallocate(ttnn_slice_203, False)
    ttnn_slice_204 = ttnn.slice(
        ttnn_slice_180,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_204, 23)
    ttnn.deallocate(ttnn_slice_204, False)
    ttnn_slice_205 = ttnn.slice(
        ttnn_slice_180,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_205, 24)
    ttnn.deallocate(ttnn_slice_205, False)
    ttnn_slice_206 = ttnn.slice(
        ttnn_slice_180,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_206, 25)
    ttnn.deallocate(ttnn_slice_206, False)
    ttnn_slice_207 = ttnn.slice(
        ttnn_slice_180,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_207, 26)
    ttnn.deallocate(ttnn_slice_207, False)
    ttnn_slice_208 = ttnn.slice(
        ttnn_slice_180,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_208, 27)
    ttnn.deallocate(ttnn_slice_208, False)
    ttnn_slice_209 = ttnn.slice(
        ttnn_slice_180,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_209, 28)
    ttnn.deallocate(ttnn_slice_209, False)
    ttnn_slice_210 = ttnn.slice(
        ttnn_slice_180,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_210, 29)
    ttnn.deallocate(ttnn_slice_210, False)
    ttnn_slice_211 = ttnn.slice(
        ttnn_slice_180,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_8, ttnn_slice_211, 30)
    ttnn.deallocate(ttnn_slice_211, False)
    ttnn_slice_212 = ttnn.slice(
        ttnn_slice_180,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_180, False)
    ttnn.fill_cache(args_8, ttnn_slice_212, 31)
    ttnn.deallocate(ttnn_slice_212, False)
    ttnn_slice_213 = ttnn.slice(
        v_9,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_213, 0)
    ttnn.deallocate(ttnn_slice_213, False)
    ttnn_slice_214 = ttnn.slice(
        v_9,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_214, 1)
    ttnn.deallocate(ttnn_slice_214, False)
    ttnn_slice_215 = ttnn.slice(
        v_9,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_215, 2)
    ttnn.deallocate(ttnn_slice_215, False)
    ttnn_slice_216 = ttnn.slice(
        v_9,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_216, 3)
    ttnn.deallocate(ttnn_slice_216, False)
    ttnn_slice_217 = ttnn.slice(
        v_9,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_217, 4)
    ttnn.deallocate(ttnn_slice_217, False)
    ttnn_slice_218 = ttnn.slice(
        v_9,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_218, 5)
    ttnn.deallocate(ttnn_slice_218, False)
    ttnn_slice_219 = ttnn.slice(
        v_9,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_219, 6)
    ttnn.deallocate(ttnn_slice_219, False)
    ttnn_slice_220 = ttnn.slice(
        v_9,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_220, 7)
    ttnn.deallocate(ttnn_slice_220, False)
    ttnn_slice_221 = ttnn.slice(
        v_9,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_221, 8)
    ttnn.deallocate(ttnn_slice_221, False)
    ttnn_slice_222 = ttnn.slice(
        v_9,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_222, 9)
    ttnn.deallocate(ttnn_slice_222, False)
    ttnn_slice_223 = ttnn.slice(
        v_9,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_223, 10)
    ttnn.deallocate(ttnn_slice_223, False)
    ttnn_slice_224 = ttnn.slice(
        v_9,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_224, 11)
    ttnn.deallocate(ttnn_slice_224, False)
    ttnn_slice_225 = ttnn.slice(
        v_9,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_225, 12)
    ttnn.deallocate(ttnn_slice_225, False)
    ttnn_slice_226 = ttnn.slice(
        v_9,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_226, 13)
    ttnn.deallocate(ttnn_slice_226, False)
    ttnn_slice_227 = ttnn.slice(
        v_9,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_227, 14)
    ttnn.deallocate(ttnn_slice_227, False)
    ttnn_slice_228 = ttnn.slice(
        v_9,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_228, 15)
    ttnn.deallocate(ttnn_slice_228, False)
    ttnn_slice_229 = ttnn.slice(
        v_9,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_229, 16)
    ttnn.deallocate(ttnn_slice_229, False)
    ttnn_slice_230 = ttnn.slice(
        v_9,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_230, 17)
    ttnn.deallocate(ttnn_slice_230, False)
    ttnn_slice_231 = ttnn.slice(
        v_9,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_231, 18)
    ttnn.deallocate(ttnn_slice_231, False)
    ttnn_slice_232 = ttnn.slice(
        v_9,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_232, 19)
    ttnn.deallocate(ttnn_slice_232, False)
    ttnn_slice_233 = ttnn.slice(
        v_9,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_233, 20)
    ttnn.deallocate(ttnn_slice_233, False)
    ttnn_slice_234 = ttnn.slice(
        v_9,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_234, 21)
    ttnn.deallocate(ttnn_slice_234, False)
    ttnn_slice_235 = ttnn.slice(
        v_9,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_235, 22)
    ttnn.deallocate(ttnn_slice_235, False)
    ttnn_slice_236 = ttnn.slice(
        v_9,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_236, 23)
    ttnn.deallocate(ttnn_slice_236, False)
    ttnn_slice_237 = ttnn.slice(
        v_9,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_237, 24)
    ttnn.deallocate(ttnn_slice_237, False)
    ttnn_slice_238 = ttnn.slice(
        v_9,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_238, 25)
    ttnn.deallocate(ttnn_slice_238, False)
    ttnn_slice_239 = ttnn.slice(
        v_9,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_239, 26)
    ttnn.deallocate(ttnn_slice_239, False)
    ttnn_slice_240 = ttnn.slice(
        v_9,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_240, 27)
    ttnn.deallocate(ttnn_slice_240, False)
    ttnn_slice_241 = ttnn.slice(
        v_9,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_241, 28)
    ttnn.deallocate(ttnn_slice_241, False)
    ttnn_slice_242 = ttnn.slice(
        v_9,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_242, 29)
    ttnn.deallocate(ttnn_slice_242, False)
    ttnn_slice_243 = ttnn.slice(
        v_9,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_9, ttnn_slice_243, 30)
    ttnn.deallocate(ttnn_slice_243, False)
    ttnn_slice_244 = ttnn.slice(
        v_9,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_9, False)
    ttnn.fill_cache(args_9, ttnn_slice_244, 31)
    ttnn.deallocate(ttnn_slice_244, False)
    ttnn_to_memory_config_26 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_7 = ttnn.add(
        args_7,
        ttnn_to_memory_config_26,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_26, False)
    ttnn.deallocate(args_7, False)
    ttnn_to_memory_config_27 = ttnn.to_memory_config(
        ttnn_add_7,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_7, False)
    ttnn_to_memory_config_28 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_29 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_5 = ttnn.experimental.rotary_embedding(
        v_7,
        ttnn_to_memory_config_29,
        ttnn_to_memory_config_28,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_29, False)
    ttnn.deallocate(ttnn_to_memory_config_28, False)
    ttnn.deallocate(v_7, False)
    ttnn_slice_245 = ttnn.slice(
        ttnn_experimental_rotary_embedding_5,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_5, False)
    ttnn_to_memory_config_30 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_2 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_245,
            args_8,
            args_9,
            attn_mask=ttnn_to_memory_config_30,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_30, False)
    ttnn.deallocate(ttnn_slice_245, False)
    ttnn_transformer_concatenate_heads_2 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_2, False)
    ttnn_reshape_9 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_2,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_2, False)
    ttnn_matmul_12 = ttnn.matmul(
        ttnn_reshape_9,
        ce_cache__main["main_const_eval_58"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_9, False)
    ttnn_add_8 = ttnn.add(
        ttnn_matmul_12,
        ttnn_add_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_12, False)
    ttnn.deallocate(ttnn_add_6, False)
    ttnn_rms_norm_5 = ttnn.rms_norm(
        ttnn_add_8,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.2.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_13 = ttnn.matmul(
        ttnn_rms_norm_5,
        ce_cache__main["main_const_eval_6"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_14 = ttnn.matmul(
        ttnn_rms_norm_5,
        ce_cache__main["main_const_eval_84"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_5, False)
    ttnn_multiply_2 = ttnn.multiply(
        ttnn_matmul_13,
        ttnn_matmul_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_14, False)
    ttnn.deallocate(ttnn_matmul_13, False)
    ttnn_matmul_15 = ttnn.matmul(
        ttnn_multiply_2,
        ce_cache__main["main_const_eval_51"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_2, False)
    ttnn_add_9 = ttnn.add(
        ttnn_matmul_15,
        ttnn_add_8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_15, False)
    ttnn.deallocate(ttnn_add_8, False)
    ttnn_rms_norm_6 = ttnn.rms_norm(
        ttnn_add_9,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.3.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_16 = ttnn.matmul(
        ttnn_rms_norm_6,
        ce_cache__main["main_const_eval_26"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_6, False)
    ttnn_reshape_10 = ttnn.reshape(
        ttnn_matmul_16,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_16, False)
    v_10, v_11, v_12 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_10,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_10, False)
    ttnn_to_memory_config_31 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_32 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_6 = ttnn.experimental.rotary_embedding(
        v_11,
        ttnn_to_memory_config_32,
        ttnn_to_memory_config_31,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_32, False)
    ttnn.deallocate(ttnn_to_memory_config_31, False)
    ttnn.deallocate(v_11, False)
    ttnn_slice_246 = ttnn.slice(
        ttnn_experimental_rotary_embedding_6,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_6, False)
    ttnn_slice_247 = ttnn.slice(
        ttnn_slice_246,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_247, 0)
    ttnn.deallocate(ttnn_slice_247, False)
    ttnn_slice_248 = ttnn.slice(
        ttnn_slice_246,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_248, 1)
    ttnn.deallocate(ttnn_slice_248, False)
    ttnn_slice_249 = ttnn.slice(
        ttnn_slice_246,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_249, 2)
    ttnn.deallocate(ttnn_slice_249, False)
    ttnn_slice_250 = ttnn.slice(
        ttnn_slice_246,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_250, 3)
    ttnn.deallocate(ttnn_slice_250, False)
    ttnn_slice_251 = ttnn.slice(
        ttnn_slice_246,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_251, 4)
    ttnn.deallocate(ttnn_slice_251, False)
    ttnn_slice_252 = ttnn.slice(
        ttnn_slice_246,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_252, 5)
    ttnn.deallocate(ttnn_slice_252, False)
    ttnn_slice_253 = ttnn.slice(
        ttnn_slice_246,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_253, 6)
    ttnn.deallocate(ttnn_slice_253, False)
    ttnn_slice_254 = ttnn.slice(
        ttnn_slice_246,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_254, 7)
    ttnn.deallocate(ttnn_slice_254, False)
    ttnn_slice_255 = ttnn.slice(
        ttnn_slice_246,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_255, 8)
    ttnn.deallocate(ttnn_slice_255, False)
    ttnn_slice_256 = ttnn.slice(
        ttnn_slice_246,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_256, 9)
    ttnn.deallocate(ttnn_slice_256, False)
    ttnn_slice_257 = ttnn.slice(
        ttnn_slice_246,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_257, 10)
    ttnn.deallocate(ttnn_slice_257, False)
    ttnn_slice_258 = ttnn.slice(
        ttnn_slice_246,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_258, 11)
    ttnn.deallocate(ttnn_slice_258, False)
    ttnn_slice_259 = ttnn.slice(
        ttnn_slice_246,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_259, 12)
    ttnn.deallocate(ttnn_slice_259, False)
    ttnn_slice_260 = ttnn.slice(
        ttnn_slice_246,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_260, 13)
    ttnn.deallocate(ttnn_slice_260, False)
    ttnn_slice_261 = ttnn.slice(
        ttnn_slice_246,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_261, 14)
    ttnn.deallocate(ttnn_slice_261, False)
    ttnn_slice_262 = ttnn.slice(
        ttnn_slice_246,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_262, 15)
    ttnn.deallocate(ttnn_slice_262, False)
    ttnn_slice_263 = ttnn.slice(
        ttnn_slice_246,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_263, 16)
    ttnn.deallocate(ttnn_slice_263, False)
    ttnn_slice_264 = ttnn.slice(
        ttnn_slice_246,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_264, 17)
    ttnn.deallocate(ttnn_slice_264, False)
    ttnn_slice_265 = ttnn.slice(
        ttnn_slice_246,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_265, 18)
    ttnn.deallocate(ttnn_slice_265, False)
    ttnn_slice_266 = ttnn.slice(
        ttnn_slice_246,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_266, 19)
    ttnn.deallocate(ttnn_slice_266, False)
    ttnn_slice_267 = ttnn.slice(
        ttnn_slice_246,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_267, 20)
    ttnn.deallocate(ttnn_slice_267, False)
    ttnn_slice_268 = ttnn.slice(
        ttnn_slice_246,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_268, 21)
    ttnn.deallocate(ttnn_slice_268, False)
    ttnn_slice_269 = ttnn.slice(
        ttnn_slice_246,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_269, 22)
    ttnn.deallocate(ttnn_slice_269, False)
    ttnn_slice_270 = ttnn.slice(
        ttnn_slice_246,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_270, 23)
    ttnn.deallocate(ttnn_slice_270, False)
    ttnn_slice_271 = ttnn.slice(
        ttnn_slice_246,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_271, 24)
    ttnn.deallocate(ttnn_slice_271, False)
    ttnn_slice_272 = ttnn.slice(
        ttnn_slice_246,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_272, 25)
    ttnn.deallocate(ttnn_slice_272, False)
    ttnn_slice_273 = ttnn.slice(
        ttnn_slice_246,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_273, 26)
    ttnn.deallocate(ttnn_slice_273, False)
    ttnn_slice_274 = ttnn.slice(
        ttnn_slice_246,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_274, 27)
    ttnn.deallocate(ttnn_slice_274, False)
    ttnn_slice_275 = ttnn.slice(
        ttnn_slice_246,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_275, 28)
    ttnn.deallocate(ttnn_slice_275, False)
    ttnn_slice_276 = ttnn.slice(
        ttnn_slice_246,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_276, 29)
    ttnn.deallocate(ttnn_slice_276, False)
    ttnn_slice_277 = ttnn.slice(
        ttnn_slice_246,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_11, ttnn_slice_277, 30)
    ttnn.deallocate(ttnn_slice_277, False)
    ttnn_slice_278 = ttnn.slice(
        ttnn_slice_246,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_246, False)
    ttnn.fill_cache(args_11, ttnn_slice_278, 31)
    ttnn.deallocate(ttnn_slice_278, False)
    ttnn_slice_279 = ttnn.slice(
        v_12,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_279, 0)
    ttnn.deallocate(ttnn_slice_279, False)
    ttnn_slice_280 = ttnn.slice(
        v_12,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_280, 1)
    ttnn.deallocate(ttnn_slice_280, False)
    ttnn_slice_281 = ttnn.slice(
        v_12,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_281, 2)
    ttnn.deallocate(ttnn_slice_281, False)
    ttnn_slice_282 = ttnn.slice(
        v_12,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_282, 3)
    ttnn.deallocate(ttnn_slice_282, False)
    ttnn_slice_283 = ttnn.slice(
        v_12,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_283, 4)
    ttnn.deallocate(ttnn_slice_283, False)
    ttnn_slice_284 = ttnn.slice(
        v_12,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_284, 5)
    ttnn.deallocate(ttnn_slice_284, False)
    ttnn_slice_285 = ttnn.slice(
        v_12,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_285, 6)
    ttnn.deallocate(ttnn_slice_285, False)
    ttnn_slice_286 = ttnn.slice(
        v_12,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_286, 7)
    ttnn.deallocate(ttnn_slice_286, False)
    ttnn_slice_287 = ttnn.slice(
        v_12,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_287, 8)
    ttnn.deallocate(ttnn_slice_287, False)
    ttnn_slice_288 = ttnn.slice(
        v_12,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_288, 9)
    ttnn.deallocate(ttnn_slice_288, False)
    ttnn_slice_289 = ttnn.slice(
        v_12,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_289, 10)
    ttnn.deallocate(ttnn_slice_289, False)
    ttnn_slice_290 = ttnn.slice(
        v_12,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_290, 11)
    ttnn.deallocate(ttnn_slice_290, False)
    ttnn_slice_291 = ttnn.slice(
        v_12,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_291, 12)
    ttnn.deallocate(ttnn_slice_291, False)
    ttnn_slice_292 = ttnn.slice(
        v_12,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_292, 13)
    ttnn.deallocate(ttnn_slice_292, False)
    ttnn_slice_293 = ttnn.slice(
        v_12,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_293, 14)
    ttnn.deallocate(ttnn_slice_293, False)
    ttnn_slice_294 = ttnn.slice(
        v_12,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_294, 15)
    ttnn.deallocate(ttnn_slice_294, False)
    ttnn_slice_295 = ttnn.slice(
        v_12,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_295, 16)
    ttnn.deallocate(ttnn_slice_295, False)
    ttnn_slice_296 = ttnn.slice(
        v_12,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_296, 17)
    ttnn.deallocate(ttnn_slice_296, False)
    ttnn_slice_297 = ttnn.slice(
        v_12,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_297, 18)
    ttnn.deallocate(ttnn_slice_297, False)
    ttnn_slice_298 = ttnn.slice(
        v_12,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_298, 19)
    ttnn.deallocate(ttnn_slice_298, False)
    ttnn_slice_299 = ttnn.slice(
        v_12,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_299, 20)
    ttnn.deallocate(ttnn_slice_299, False)
    ttnn_slice_300 = ttnn.slice(
        v_12,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_300, 21)
    ttnn.deallocate(ttnn_slice_300, False)
    ttnn_slice_301 = ttnn.slice(
        v_12,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_301, 22)
    ttnn.deallocate(ttnn_slice_301, False)
    ttnn_slice_302 = ttnn.slice(
        v_12,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_302, 23)
    ttnn.deallocate(ttnn_slice_302, False)
    ttnn_slice_303 = ttnn.slice(
        v_12,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_303, 24)
    ttnn.deallocate(ttnn_slice_303, False)
    ttnn_slice_304 = ttnn.slice(
        v_12,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_304, 25)
    ttnn.deallocate(ttnn_slice_304, False)
    ttnn_slice_305 = ttnn.slice(
        v_12,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_305, 26)
    ttnn.deallocate(ttnn_slice_305, False)
    ttnn_slice_306 = ttnn.slice(
        v_12,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_306, 27)
    ttnn.deallocate(ttnn_slice_306, False)
    ttnn_slice_307 = ttnn.slice(
        v_12,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_307, 28)
    ttnn.deallocate(ttnn_slice_307, False)
    ttnn_slice_308 = ttnn.slice(
        v_12,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_308, 29)
    ttnn.deallocate(ttnn_slice_308, False)
    ttnn_slice_309 = ttnn.slice(
        v_12,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_12, ttnn_slice_309, 30)
    ttnn.deallocate(ttnn_slice_309, False)
    ttnn_slice_310 = ttnn.slice(
        v_12,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_12, False)
    ttnn.fill_cache(args_12, ttnn_slice_310, 31)
    ttnn.deallocate(ttnn_slice_310, False)
    ttnn_to_memory_config_33 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_10 = ttnn.add(
        args_10,
        ttnn_to_memory_config_33,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_33, False)
    ttnn.deallocate(args_10, False)
    ttnn_to_memory_config_34 = ttnn.to_memory_config(
        ttnn_add_10,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_10, False)
    ttnn_to_memory_config_35 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_36 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_7 = ttnn.experimental.rotary_embedding(
        v_10,
        ttnn_to_memory_config_36,
        ttnn_to_memory_config_35,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_36, False)
    ttnn.deallocate(ttnn_to_memory_config_35, False)
    ttnn.deallocate(v_10, False)
    ttnn_slice_311 = ttnn.slice(
        ttnn_experimental_rotary_embedding_7,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_7, False)
    ttnn_to_memory_config_37 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_3 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_311,
            args_11,
            args_12,
            attn_mask=ttnn_to_memory_config_37,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_37, False)
    ttnn.deallocate(ttnn_slice_311, False)
    ttnn_transformer_concatenate_heads_3 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_3, False)
    ttnn_reshape_11 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_3,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_3, False)
    ttnn_matmul_17 = ttnn.matmul(
        ttnn_reshape_11,
        ce_cache__main["main_const_eval_1"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_11, False)
    ttnn_add_11 = ttnn.add(
        ttnn_matmul_17,
        ttnn_add_9,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_17, False)
    ttnn.deallocate(ttnn_add_9, False)
    ttnn_rms_norm_7 = ttnn.rms_norm(
        ttnn_add_11,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.3.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_18 = ttnn.matmul(
        ttnn_rms_norm_7,
        ce_cache__main["main_const_eval_31"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_19 = ttnn.matmul(
        ttnn_rms_norm_7,
        ce_cache__main["main_const_eval_62"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_7, False)
    ttnn_multiply_3 = ttnn.multiply(
        ttnn_matmul_18,
        ttnn_matmul_19,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_19, False)
    ttnn.deallocate(ttnn_matmul_18, False)
    ttnn_matmul_20 = ttnn.matmul(
        ttnn_multiply_3,
        ce_cache__main["main_const_eval_66"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_3, False)
    ttnn_add_12 = ttnn.add(
        ttnn_matmul_20,
        ttnn_add_11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_20, False)
    ttnn.deallocate(ttnn_add_11, False)
    ttnn_rms_norm_8 = ttnn.rms_norm(
        ttnn_add_12,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.4.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_21 = ttnn.matmul(
        ttnn_rms_norm_8,
        ce_cache__main["main_const_eval_4"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_8, False)
    ttnn_reshape_12 = ttnn.reshape(
        ttnn_matmul_21,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_21, False)
    v_13, v_14, v_15 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_12,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_12, False)
    ttnn_to_memory_config_38 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_39 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_8 = ttnn.experimental.rotary_embedding(
        v_14,
        ttnn_to_memory_config_39,
        ttnn_to_memory_config_38,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_39, False)
    ttnn.deallocate(ttnn_to_memory_config_38, False)
    ttnn.deallocate(v_14, False)
    ttnn_slice_312 = ttnn.slice(
        ttnn_experimental_rotary_embedding_8,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_8, False)
    ttnn_slice_313 = ttnn.slice(
        ttnn_slice_312,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_313, 0)
    ttnn.deallocate(ttnn_slice_313, False)
    ttnn_slice_314 = ttnn.slice(
        ttnn_slice_312,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_314, 1)
    ttnn.deallocate(ttnn_slice_314, False)
    ttnn_slice_315 = ttnn.slice(
        ttnn_slice_312,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_315, 2)
    ttnn.deallocate(ttnn_slice_315, False)
    ttnn_slice_316 = ttnn.slice(
        ttnn_slice_312,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_316, 3)
    ttnn.deallocate(ttnn_slice_316, False)
    ttnn_slice_317 = ttnn.slice(
        ttnn_slice_312,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_317, 4)
    ttnn.deallocate(ttnn_slice_317, False)
    ttnn_slice_318 = ttnn.slice(
        ttnn_slice_312,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_318, 5)
    ttnn.deallocate(ttnn_slice_318, False)
    ttnn_slice_319 = ttnn.slice(
        ttnn_slice_312,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_319, 6)
    ttnn.deallocate(ttnn_slice_319, False)
    ttnn_slice_320 = ttnn.slice(
        ttnn_slice_312,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_320, 7)
    ttnn.deallocate(ttnn_slice_320, False)
    ttnn_slice_321 = ttnn.slice(
        ttnn_slice_312,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_321, 8)
    ttnn.deallocate(ttnn_slice_321, False)
    ttnn_slice_322 = ttnn.slice(
        ttnn_slice_312,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_322, 9)
    ttnn.deallocate(ttnn_slice_322, False)
    ttnn_slice_323 = ttnn.slice(
        ttnn_slice_312,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_323, 10)
    ttnn.deallocate(ttnn_slice_323, False)
    ttnn_slice_324 = ttnn.slice(
        ttnn_slice_312,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_324, 11)
    ttnn.deallocate(ttnn_slice_324, False)
    ttnn_slice_325 = ttnn.slice(
        ttnn_slice_312,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_325, 12)
    ttnn.deallocate(ttnn_slice_325, False)
    ttnn_slice_326 = ttnn.slice(
        ttnn_slice_312,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_326, 13)
    ttnn.deallocate(ttnn_slice_326, False)
    ttnn_slice_327 = ttnn.slice(
        ttnn_slice_312,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_327, 14)
    ttnn.deallocate(ttnn_slice_327, False)
    ttnn_slice_328 = ttnn.slice(
        ttnn_slice_312,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_328, 15)
    ttnn.deallocate(ttnn_slice_328, False)
    ttnn_slice_329 = ttnn.slice(
        ttnn_slice_312,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_329, 16)
    ttnn.deallocate(ttnn_slice_329, False)
    ttnn_slice_330 = ttnn.slice(
        ttnn_slice_312,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_330, 17)
    ttnn.deallocate(ttnn_slice_330, False)
    ttnn_slice_331 = ttnn.slice(
        ttnn_slice_312,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_331, 18)
    ttnn.deallocate(ttnn_slice_331, False)
    ttnn_slice_332 = ttnn.slice(
        ttnn_slice_312,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_332, 19)
    ttnn.deallocate(ttnn_slice_332, False)
    ttnn_slice_333 = ttnn.slice(
        ttnn_slice_312,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_333, 20)
    ttnn.deallocate(ttnn_slice_333, False)
    ttnn_slice_334 = ttnn.slice(
        ttnn_slice_312,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_334, 21)
    ttnn.deallocate(ttnn_slice_334, False)
    ttnn_slice_335 = ttnn.slice(
        ttnn_slice_312,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_335, 22)
    ttnn.deallocate(ttnn_slice_335, False)
    ttnn_slice_336 = ttnn.slice(
        ttnn_slice_312,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_336, 23)
    ttnn.deallocate(ttnn_slice_336, False)
    ttnn_slice_337 = ttnn.slice(
        ttnn_slice_312,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_337, 24)
    ttnn.deallocate(ttnn_slice_337, False)
    ttnn_slice_338 = ttnn.slice(
        ttnn_slice_312,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_338, 25)
    ttnn.deallocate(ttnn_slice_338, False)
    ttnn_slice_339 = ttnn.slice(
        ttnn_slice_312,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_339, 26)
    ttnn.deallocate(ttnn_slice_339, False)
    ttnn_slice_340 = ttnn.slice(
        ttnn_slice_312,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_340, 27)
    ttnn.deallocate(ttnn_slice_340, False)
    ttnn_slice_341 = ttnn.slice(
        ttnn_slice_312,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_341, 28)
    ttnn.deallocate(ttnn_slice_341, False)
    ttnn_slice_342 = ttnn.slice(
        ttnn_slice_312,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_342, 29)
    ttnn.deallocate(ttnn_slice_342, False)
    ttnn_slice_343 = ttnn.slice(
        ttnn_slice_312,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_14, ttnn_slice_343, 30)
    ttnn.deallocate(ttnn_slice_343, False)
    ttnn_slice_344 = ttnn.slice(
        ttnn_slice_312,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_312, False)
    ttnn.fill_cache(args_14, ttnn_slice_344, 31)
    ttnn.deallocate(ttnn_slice_344, False)
    ttnn_slice_345 = ttnn.slice(
        v_15,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_345, 0)
    ttnn.deallocate(ttnn_slice_345, False)
    ttnn_slice_346 = ttnn.slice(
        v_15,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_346, 1)
    ttnn.deallocate(ttnn_slice_346, False)
    ttnn_slice_347 = ttnn.slice(
        v_15,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_347, 2)
    ttnn.deallocate(ttnn_slice_347, False)
    ttnn_slice_348 = ttnn.slice(
        v_15,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_348, 3)
    ttnn.deallocate(ttnn_slice_348, False)
    ttnn_slice_349 = ttnn.slice(
        v_15,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_349, 4)
    ttnn.deallocate(ttnn_slice_349, False)
    ttnn_slice_350 = ttnn.slice(
        v_15,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_350, 5)
    ttnn.deallocate(ttnn_slice_350, False)
    ttnn_slice_351 = ttnn.slice(
        v_15,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_351, 6)
    ttnn.deallocate(ttnn_slice_351, False)
    ttnn_slice_352 = ttnn.slice(
        v_15,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_352, 7)
    ttnn.deallocate(ttnn_slice_352, False)
    ttnn_slice_353 = ttnn.slice(
        v_15,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_353, 8)
    ttnn.deallocate(ttnn_slice_353, False)
    ttnn_slice_354 = ttnn.slice(
        v_15,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_354, 9)
    ttnn.deallocate(ttnn_slice_354, False)
    ttnn_slice_355 = ttnn.slice(
        v_15,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_355, 10)
    ttnn.deallocate(ttnn_slice_355, False)
    ttnn_slice_356 = ttnn.slice(
        v_15,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_356, 11)
    ttnn.deallocate(ttnn_slice_356, False)
    ttnn_slice_357 = ttnn.slice(
        v_15,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_357, 12)
    ttnn.deallocate(ttnn_slice_357, False)
    ttnn_slice_358 = ttnn.slice(
        v_15,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_358, 13)
    ttnn.deallocate(ttnn_slice_358, False)
    ttnn_slice_359 = ttnn.slice(
        v_15,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_359, 14)
    ttnn.deallocate(ttnn_slice_359, False)
    ttnn_slice_360 = ttnn.slice(
        v_15,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_360, 15)
    ttnn.deallocate(ttnn_slice_360, False)
    ttnn_slice_361 = ttnn.slice(
        v_15,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_361, 16)
    ttnn.deallocate(ttnn_slice_361, False)
    ttnn_slice_362 = ttnn.slice(
        v_15,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_362, 17)
    ttnn.deallocate(ttnn_slice_362, False)
    ttnn_slice_363 = ttnn.slice(
        v_15,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_363, 18)
    ttnn.deallocate(ttnn_slice_363, False)
    ttnn_slice_364 = ttnn.slice(
        v_15,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_364, 19)
    ttnn.deallocate(ttnn_slice_364, False)
    ttnn_slice_365 = ttnn.slice(
        v_15,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_365, 20)
    ttnn.deallocate(ttnn_slice_365, False)
    ttnn_slice_366 = ttnn.slice(
        v_15,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_366, 21)
    ttnn.deallocate(ttnn_slice_366, False)
    ttnn_slice_367 = ttnn.slice(
        v_15,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_367, 22)
    ttnn.deallocate(ttnn_slice_367, False)
    ttnn_slice_368 = ttnn.slice(
        v_15,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_368, 23)
    ttnn.deallocate(ttnn_slice_368, False)
    ttnn_slice_369 = ttnn.slice(
        v_15,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_369, 24)
    ttnn.deallocate(ttnn_slice_369, False)
    ttnn_slice_370 = ttnn.slice(
        v_15,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_370, 25)
    ttnn.deallocate(ttnn_slice_370, False)
    ttnn_slice_371 = ttnn.slice(
        v_15,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_371, 26)
    ttnn.deallocate(ttnn_slice_371, False)
    ttnn_slice_372 = ttnn.slice(
        v_15,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_372, 27)
    ttnn.deallocate(ttnn_slice_372, False)
    ttnn_slice_373 = ttnn.slice(
        v_15,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_373, 28)
    ttnn.deallocate(ttnn_slice_373, False)
    ttnn_slice_374 = ttnn.slice(
        v_15,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_374, 29)
    ttnn.deallocate(ttnn_slice_374, False)
    ttnn_slice_375 = ttnn.slice(
        v_15,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_15, ttnn_slice_375, 30)
    ttnn.deallocate(ttnn_slice_375, False)
    ttnn_slice_376 = ttnn.slice(
        v_15,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_15, False)
    ttnn.fill_cache(args_15, ttnn_slice_376, 31)
    ttnn.deallocate(ttnn_slice_376, False)
    ttnn_to_memory_config_40 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_13 = ttnn.add(
        args_13,
        ttnn_to_memory_config_40,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_40, False)
    ttnn.deallocate(args_13, False)
    ttnn_to_memory_config_41 = ttnn.to_memory_config(
        ttnn_add_13,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_13, False)
    ttnn_to_memory_config_42 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_43 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_9 = ttnn.experimental.rotary_embedding(
        v_13,
        ttnn_to_memory_config_43,
        ttnn_to_memory_config_42,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_43, False)
    ttnn.deallocate(ttnn_to_memory_config_42, False)
    ttnn.deallocate(v_13, False)
    ttnn_slice_377 = ttnn.slice(
        ttnn_experimental_rotary_embedding_9,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_9, False)
    ttnn_to_memory_config_44 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_4 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_377,
            args_14,
            args_15,
            attn_mask=ttnn_to_memory_config_44,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_44, False)
    ttnn.deallocate(ttnn_slice_377, False)
    ttnn_transformer_concatenate_heads_4 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_4, False)
    ttnn_reshape_13 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_4,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_4, False)
    ttnn_matmul_22 = ttnn.matmul(
        ttnn_reshape_13,
        ce_cache__main["main_const_eval_64"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_13, False)
    ttnn_add_14 = ttnn.add(
        ttnn_matmul_22,
        ttnn_add_12,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_22, False)
    ttnn.deallocate(ttnn_add_12, False)
    ttnn_rms_norm_9 = ttnn.rms_norm(
        ttnn_add_14,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.4.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_23 = ttnn.matmul(
        ttnn_rms_norm_9,
        ce_cache__main["main_const_eval_52"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_24 = ttnn.matmul(
        ttnn_rms_norm_9,
        ce_cache__main["main_const_eval_41"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_9, False)
    ttnn_multiply_4 = ttnn.multiply(
        ttnn_matmul_23,
        ttnn_matmul_24,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_24, False)
    ttnn.deallocate(ttnn_matmul_23, False)
    ttnn_matmul_25 = ttnn.matmul(
        ttnn_multiply_4,
        ce_cache__main["main_const_eval_7"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_4, False)
    ttnn_add_15 = ttnn.add(
        ttnn_matmul_25,
        ttnn_add_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_25, False)
    ttnn.deallocate(ttnn_add_14, False)
    ttnn_rms_norm_10 = ttnn.rms_norm(
        ttnn_add_15,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.5.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_26 = ttnn.matmul(
        ttnn_rms_norm_10,
        ce_cache__main["main_const_eval_68"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_10, False)
    ttnn_reshape_14 = ttnn.reshape(
        ttnn_matmul_26,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_26, False)
    v_16, v_17, v_18 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_14,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_14, False)
    ttnn_to_memory_config_45 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_46 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_10 = ttnn.experimental.rotary_embedding(
        v_17,
        ttnn_to_memory_config_46,
        ttnn_to_memory_config_45,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_46, False)
    ttnn.deallocate(ttnn_to_memory_config_45, False)
    ttnn.deallocate(v_17, False)
    ttnn_slice_378 = ttnn.slice(
        ttnn_experimental_rotary_embedding_10,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_10, False)
    ttnn_slice_379 = ttnn.slice(
        ttnn_slice_378,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_379, 0)
    ttnn.deallocate(ttnn_slice_379, False)
    ttnn_slice_380 = ttnn.slice(
        ttnn_slice_378,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_380, 1)
    ttnn.deallocate(ttnn_slice_380, False)
    ttnn_slice_381 = ttnn.slice(
        ttnn_slice_378,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_381, 2)
    ttnn.deallocate(ttnn_slice_381, False)
    ttnn_slice_382 = ttnn.slice(
        ttnn_slice_378,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_382, 3)
    ttnn.deallocate(ttnn_slice_382, False)
    ttnn_slice_383 = ttnn.slice(
        ttnn_slice_378,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_383, 4)
    ttnn.deallocate(ttnn_slice_383, False)
    ttnn_slice_384 = ttnn.slice(
        ttnn_slice_378,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_384, 5)
    ttnn.deallocate(ttnn_slice_384, False)
    ttnn_slice_385 = ttnn.slice(
        ttnn_slice_378,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_385, 6)
    ttnn.deallocate(ttnn_slice_385, False)
    ttnn_slice_386 = ttnn.slice(
        ttnn_slice_378,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_386, 7)
    ttnn.deallocate(ttnn_slice_386, False)
    ttnn_slice_387 = ttnn.slice(
        ttnn_slice_378,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_387, 8)
    ttnn.deallocate(ttnn_slice_387, False)
    ttnn_slice_388 = ttnn.slice(
        ttnn_slice_378,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_388, 9)
    ttnn.deallocate(ttnn_slice_388, False)
    ttnn_slice_389 = ttnn.slice(
        ttnn_slice_378,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_389, 10)
    ttnn.deallocate(ttnn_slice_389, False)
    ttnn_slice_390 = ttnn.slice(
        ttnn_slice_378,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_390, 11)
    ttnn.deallocate(ttnn_slice_390, False)
    ttnn_slice_391 = ttnn.slice(
        ttnn_slice_378,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_391, 12)
    ttnn.deallocate(ttnn_slice_391, False)
    ttnn_slice_392 = ttnn.slice(
        ttnn_slice_378,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_392, 13)
    ttnn.deallocate(ttnn_slice_392, False)
    ttnn_slice_393 = ttnn.slice(
        ttnn_slice_378,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_393, 14)
    ttnn.deallocate(ttnn_slice_393, False)
    ttnn_slice_394 = ttnn.slice(
        ttnn_slice_378,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_394, 15)
    ttnn.deallocate(ttnn_slice_394, False)
    ttnn_slice_395 = ttnn.slice(
        ttnn_slice_378,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_395, 16)
    ttnn.deallocate(ttnn_slice_395, False)
    ttnn_slice_396 = ttnn.slice(
        ttnn_slice_378,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_396, 17)
    ttnn.deallocate(ttnn_slice_396, False)
    ttnn_slice_397 = ttnn.slice(
        ttnn_slice_378,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_397, 18)
    ttnn.deallocate(ttnn_slice_397, False)
    ttnn_slice_398 = ttnn.slice(
        ttnn_slice_378,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_398, 19)
    ttnn.deallocate(ttnn_slice_398, False)
    ttnn_slice_399 = ttnn.slice(
        ttnn_slice_378,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_399, 20)
    ttnn.deallocate(ttnn_slice_399, False)
    ttnn_slice_400 = ttnn.slice(
        ttnn_slice_378,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_400, 21)
    ttnn.deallocate(ttnn_slice_400, False)
    ttnn_slice_401 = ttnn.slice(
        ttnn_slice_378,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_401, 22)
    ttnn.deallocate(ttnn_slice_401, False)
    ttnn_slice_402 = ttnn.slice(
        ttnn_slice_378,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_402, 23)
    ttnn.deallocate(ttnn_slice_402, False)
    ttnn_slice_403 = ttnn.slice(
        ttnn_slice_378,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_403, 24)
    ttnn.deallocate(ttnn_slice_403, False)
    ttnn_slice_404 = ttnn.slice(
        ttnn_slice_378,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_404, 25)
    ttnn.deallocate(ttnn_slice_404, False)
    ttnn_slice_405 = ttnn.slice(
        ttnn_slice_378,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_405, 26)
    ttnn.deallocate(ttnn_slice_405, False)
    ttnn_slice_406 = ttnn.slice(
        ttnn_slice_378,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_406, 27)
    ttnn.deallocate(ttnn_slice_406, False)
    ttnn_slice_407 = ttnn.slice(
        ttnn_slice_378,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_407, 28)
    ttnn.deallocate(ttnn_slice_407, False)
    ttnn_slice_408 = ttnn.slice(
        ttnn_slice_378,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_408, 29)
    ttnn.deallocate(ttnn_slice_408, False)
    ttnn_slice_409 = ttnn.slice(
        ttnn_slice_378,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_17, ttnn_slice_409, 30)
    ttnn.deallocate(ttnn_slice_409, False)
    ttnn_slice_410 = ttnn.slice(
        ttnn_slice_378,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_378, False)
    ttnn.fill_cache(args_17, ttnn_slice_410, 31)
    ttnn.deallocate(ttnn_slice_410, False)
    ttnn_slice_411 = ttnn.slice(
        v_18,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_411, 0)
    ttnn.deallocate(ttnn_slice_411, False)
    ttnn_slice_412 = ttnn.slice(
        v_18,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_412, 1)
    ttnn.deallocate(ttnn_slice_412, False)
    ttnn_slice_413 = ttnn.slice(
        v_18,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_413, 2)
    ttnn.deallocate(ttnn_slice_413, False)
    ttnn_slice_414 = ttnn.slice(
        v_18,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_414, 3)
    ttnn.deallocate(ttnn_slice_414, False)
    ttnn_slice_415 = ttnn.slice(
        v_18,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_415, 4)
    ttnn.deallocate(ttnn_slice_415, False)
    ttnn_slice_416 = ttnn.slice(
        v_18,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_416, 5)
    ttnn.deallocate(ttnn_slice_416, False)
    ttnn_slice_417 = ttnn.slice(
        v_18,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_417, 6)
    ttnn.deallocate(ttnn_slice_417, False)
    ttnn_slice_418 = ttnn.slice(
        v_18,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_418, 7)
    ttnn.deallocate(ttnn_slice_418, False)
    ttnn_slice_419 = ttnn.slice(
        v_18,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_419, 8)
    ttnn.deallocate(ttnn_slice_419, False)
    ttnn_slice_420 = ttnn.slice(
        v_18,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_420, 9)
    ttnn.deallocate(ttnn_slice_420, False)
    ttnn_slice_421 = ttnn.slice(
        v_18,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_421, 10)
    ttnn.deallocate(ttnn_slice_421, False)
    ttnn_slice_422 = ttnn.slice(
        v_18,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_422, 11)
    ttnn.deallocate(ttnn_slice_422, False)
    ttnn_slice_423 = ttnn.slice(
        v_18,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_423, 12)
    ttnn.deallocate(ttnn_slice_423, False)
    ttnn_slice_424 = ttnn.slice(
        v_18,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_424, 13)
    ttnn.deallocate(ttnn_slice_424, False)
    ttnn_slice_425 = ttnn.slice(
        v_18,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_425, 14)
    ttnn.deallocate(ttnn_slice_425, False)
    ttnn_slice_426 = ttnn.slice(
        v_18,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_426, 15)
    ttnn.deallocate(ttnn_slice_426, False)
    ttnn_slice_427 = ttnn.slice(
        v_18,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_427, 16)
    ttnn.deallocate(ttnn_slice_427, False)
    ttnn_slice_428 = ttnn.slice(
        v_18,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_428, 17)
    ttnn.deallocate(ttnn_slice_428, False)
    ttnn_slice_429 = ttnn.slice(
        v_18,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_429, 18)
    ttnn.deallocate(ttnn_slice_429, False)
    ttnn_slice_430 = ttnn.slice(
        v_18,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_430, 19)
    ttnn.deallocate(ttnn_slice_430, False)
    ttnn_slice_431 = ttnn.slice(
        v_18,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_431, 20)
    ttnn.deallocate(ttnn_slice_431, False)
    ttnn_slice_432 = ttnn.slice(
        v_18,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_432, 21)
    ttnn.deallocate(ttnn_slice_432, False)
    ttnn_slice_433 = ttnn.slice(
        v_18,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_433, 22)
    ttnn.deallocate(ttnn_slice_433, False)
    ttnn_slice_434 = ttnn.slice(
        v_18,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_434, 23)
    ttnn.deallocate(ttnn_slice_434, False)
    ttnn_slice_435 = ttnn.slice(
        v_18,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_435, 24)
    ttnn.deallocate(ttnn_slice_435, False)
    ttnn_slice_436 = ttnn.slice(
        v_18,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_436, 25)
    ttnn.deallocate(ttnn_slice_436, False)
    ttnn_slice_437 = ttnn.slice(
        v_18,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_437, 26)
    ttnn.deallocate(ttnn_slice_437, False)
    ttnn_slice_438 = ttnn.slice(
        v_18,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_438, 27)
    ttnn.deallocate(ttnn_slice_438, False)
    ttnn_slice_439 = ttnn.slice(
        v_18,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_439, 28)
    ttnn.deallocate(ttnn_slice_439, False)
    ttnn_slice_440 = ttnn.slice(
        v_18,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_440, 29)
    ttnn.deallocate(ttnn_slice_440, False)
    ttnn_slice_441 = ttnn.slice(
        v_18,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_18, ttnn_slice_441, 30)
    ttnn.deallocate(ttnn_slice_441, False)
    ttnn_slice_442 = ttnn.slice(
        v_18,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_18, False)
    ttnn.fill_cache(args_18, ttnn_slice_442, 31)
    ttnn.deallocate(ttnn_slice_442, False)
    ttnn_to_memory_config_47 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_16 = ttnn.add(
        args_16,
        ttnn_to_memory_config_47,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_47, False)
    ttnn.deallocate(args_16, False)
    ttnn_to_memory_config_48 = ttnn.to_memory_config(
        ttnn_add_16,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_16, False)
    ttnn_to_memory_config_49 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_50 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_11 = ttnn.experimental.rotary_embedding(
        v_16,
        ttnn_to_memory_config_50,
        ttnn_to_memory_config_49,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_50, False)
    ttnn.deallocate(ttnn_to_memory_config_49, False)
    ttnn.deallocate(v_16, False)
    ttnn_slice_443 = ttnn.slice(
        ttnn_experimental_rotary_embedding_11,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_11, False)
    ttnn_to_memory_config_51 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_5 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_443,
            args_17,
            args_18,
            attn_mask=ttnn_to_memory_config_51,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_51, False)
    ttnn.deallocate(ttnn_slice_443, False)
    ttnn_transformer_concatenate_heads_5 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_5,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_5, False)
    ttnn_reshape_15 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_5,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_5, False)
    ttnn_matmul_27 = ttnn.matmul(
        ttnn_reshape_15,
        ce_cache__main["main_const_eval_86"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_15, False)
    ttnn_add_17 = ttnn.add(
        ttnn_matmul_27,
        ttnn_add_15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_27, False)
    ttnn.deallocate(ttnn_add_15, False)
    ttnn_rms_norm_11 = ttnn.rms_norm(
        ttnn_add_17,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.5.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_28 = ttnn.matmul(
        ttnn_rms_norm_11,
        ce_cache__main["main_const_eval_70"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_29 = ttnn.matmul(
        ttnn_rms_norm_11,
        ce_cache__main["main_const_eval_21"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_11, False)
    ttnn_multiply_5 = ttnn.multiply(
        ttnn_matmul_28,
        ttnn_matmul_29,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_29, False)
    ttnn.deallocate(ttnn_matmul_28, False)
    ttnn_matmul_30 = ttnn.matmul(
        ttnn_multiply_5,
        ce_cache__main["main_const_eval_35"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_5, False)
    ttnn_add_18 = ttnn.add(
        ttnn_matmul_30,
        ttnn_add_17,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_30, False)
    ttnn.deallocate(ttnn_add_17, False)
    ttnn_rms_norm_12 = ttnn.rms_norm(
        ttnn_add_18,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.6.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_31 = ttnn.matmul(
        ttnn_rms_norm_12,
        ce_cache__main["main_const_eval_37"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_12, False)
    ttnn_reshape_16 = ttnn.reshape(
        ttnn_matmul_31,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_31, False)
    v_19, v_20, v_21 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_16,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_16, False)
    ttnn_to_memory_config_52 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_53 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_12 = ttnn.experimental.rotary_embedding(
        v_20,
        ttnn_to_memory_config_53,
        ttnn_to_memory_config_52,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_53, False)
    ttnn.deallocate(ttnn_to_memory_config_52, False)
    ttnn.deallocate(v_20, False)
    ttnn_slice_444 = ttnn.slice(
        ttnn_experimental_rotary_embedding_12,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_12, False)
    ttnn_slice_445 = ttnn.slice(
        ttnn_slice_444,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_445, 0)
    ttnn.deallocate(ttnn_slice_445, False)
    ttnn_slice_446 = ttnn.slice(
        ttnn_slice_444,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_446, 1)
    ttnn.deallocate(ttnn_slice_446, False)
    ttnn_slice_447 = ttnn.slice(
        ttnn_slice_444,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_447, 2)
    ttnn.deallocate(ttnn_slice_447, False)
    ttnn_slice_448 = ttnn.slice(
        ttnn_slice_444,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_448, 3)
    ttnn.deallocate(ttnn_slice_448, False)
    ttnn_slice_449 = ttnn.slice(
        ttnn_slice_444,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_449, 4)
    ttnn.deallocate(ttnn_slice_449, False)
    ttnn_slice_450 = ttnn.slice(
        ttnn_slice_444,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_450, 5)
    ttnn.deallocate(ttnn_slice_450, False)
    ttnn_slice_451 = ttnn.slice(
        ttnn_slice_444,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_451, 6)
    ttnn.deallocate(ttnn_slice_451, False)
    ttnn_slice_452 = ttnn.slice(
        ttnn_slice_444,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_452, 7)
    ttnn.deallocate(ttnn_slice_452, False)
    ttnn_slice_453 = ttnn.slice(
        ttnn_slice_444,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_453, 8)
    ttnn.deallocate(ttnn_slice_453, False)
    ttnn_slice_454 = ttnn.slice(
        ttnn_slice_444,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_454, 9)
    ttnn.deallocate(ttnn_slice_454, False)
    ttnn_slice_455 = ttnn.slice(
        ttnn_slice_444,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_455, 10)
    ttnn.deallocate(ttnn_slice_455, False)
    ttnn_slice_456 = ttnn.slice(
        ttnn_slice_444,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_456, 11)
    ttnn.deallocate(ttnn_slice_456, False)
    ttnn_slice_457 = ttnn.slice(
        ttnn_slice_444,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_457, 12)
    ttnn.deallocate(ttnn_slice_457, False)
    ttnn_slice_458 = ttnn.slice(
        ttnn_slice_444,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_458, 13)
    ttnn.deallocate(ttnn_slice_458, False)
    ttnn_slice_459 = ttnn.slice(
        ttnn_slice_444,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_459, 14)
    ttnn.deallocate(ttnn_slice_459, False)
    ttnn_slice_460 = ttnn.slice(
        ttnn_slice_444,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_460, 15)
    ttnn.deallocate(ttnn_slice_460, False)
    ttnn_slice_461 = ttnn.slice(
        ttnn_slice_444,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_461, 16)
    ttnn.deallocate(ttnn_slice_461, False)
    ttnn_slice_462 = ttnn.slice(
        ttnn_slice_444,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_462, 17)
    ttnn.deallocate(ttnn_slice_462, False)
    ttnn_slice_463 = ttnn.slice(
        ttnn_slice_444,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_463, 18)
    ttnn.deallocate(ttnn_slice_463, False)
    ttnn_slice_464 = ttnn.slice(
        ttnn_slice_444,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_464, 19)
    ttnn.deallocate(ttnn_slice_464, False)
    ttnn_slice_465 = ttnn.slice(
        ttnn_slice_444,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_465, 20)
    ttnn.deallocate(ttnn_slice_465, False)
    ttnn_slice_466 = ttnn.slice(
        ttnn_slice_444,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_466, 21)
    ttnn.deallocate(ttnn_slice_466, False)
    ttnn_slice_467 = ttnn.slice(
        ttnn_slice_444,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_467, 22)
    ttnn.deallocate(ttnn_slice_467, False)
    ttnn_slice_468 = ttnn.slice(
        ttnn_slice_444,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_468, 23)
    ttnn.deallocate(ttnn_slice_468, False)
    ttnn_slice_469 = ttnn.slice(
        ttnn_slice_444,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_469, 24)
    ttnn.deallocate(ttnn_slice_469, False)
    ttnn_slice_470 = ttnn.slice(
        ttnn_slice_444,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_470, 25)
    ttnn.deallocate(ttnn_slice_470, False)
    ttnn_slice_471 = ttnn.slice(
        ttnn_slice_444,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_471, 26)
    ttnn.deallocate(ttnn_slice_471, False)
    ttnn_slice_472 = ttnn.slice(
        ttnn_slice_444,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_472, 27)
    ttnn.deallocate(ttnn_slice_472, False)
    ttnn_slice_473 = ttnn.slice(
        ttnn_slice_444,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_473, 28)
    ttnn.deallocate(ttnn_slice_473, False)
    ttnn_slice_474 = ttnn.slice(
        ttnn_slice_444,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_474, 29)
    ttnn.deallocate(ttnn_slice_474, False)
    ttnn_slice_475 = ttnn.slice(
        ttnn_slice_444,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_20, ttnn_slice_475, 30)
    ttnn.deallocate(ttnn_slice_475, False)
    ttnn_slice_476 = ttnn.slice(
        ttnn_slice_444,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_444, False)
    ttnn.fill_cache(args_20, ttnn_slice_476, 31)
    ttnn.deallocate(ttnn_slice_476, False)
    ttnn_slice_477 = ttnn.slice(
        v_21,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_477, 0)
    ttnn.deallocate(ttnn_slice_477, False)
    ttnn_slice_478 = ttnn.slice(
        v_21,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_478, 1)
    ttnn.deallocate(ttnn_slice_478, False)
    ttnn_slice_479 = ttnn.slice(
        v_21,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_479, 2)
    ttnn.deallocate(ttnn_slice_479, False)
    ttnn_slice_480 = ttnn.slice(
        v_21,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_480, 3)
    ttnn.deallocate(ttnn_slice_480, False)
    ttnn_slice_481 = ttnn.slice(
        v_21,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_481, 4)
    ttnn.deallocate(ttnn_slice_481, False)
    ttnn_slice_482 = ttnn.slice(
        v_21,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_482, 5)
    ttnn.deallocate(ttnn_slice_482, False)
    ttnn_slice_483 = ttnn.slice(
        v_21,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_483, 6)
    ttnn.deallocate(ttnn_slice_483, False)
    ttnn_slice_484 = ttnn.slice(
        v_21,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_484, 7)
    ttnn.deallocate(ttnn_slice_484, False)
    ttnn_slice_485 = ttnn.slice(
        v_21,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_485, 8)
    ttnn.deallocate(ttnn_slice_485, False)
    ttnn_slice_486 = ttnn.slice(
        v_21,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_486, 9)
    ttnn.deallocate(ttnn_slice_486, False)
    ttnn_slice_487 = ttnn.slice(
        v_21,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_487, 10)
    ttnn.deallocate(ttnn_slice_487, False)
    ttnn_slice_488 = ttnn.slice(
        v_21,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_488, 11)
    ttnn.deallocate(ttnn_slice_488, False)
    ttnn_slice_489 = ttnn.slice(
        v_21,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_489, 12)
    ttnn.deallocate(ttnn_slice_489, False)
    ttnn_slice_490 = ttnn.slice(
        v_21,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_490, 13)
    ttnn.deallocate(ttnn_slice_490, False)
    ttnn_slice_491 = ttnn.slice(
        v_21,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_491, 14)
    ttnn.deallocate(ttnn_slice_491, False)
    ttnn_slice_492 = ttnn.slice(
        v_21,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_492, 15)
    ttnn.deallocate(ttnn_slice_492, False)
    ttnn_slice_493 = ttnn.slice(
        v_21,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_493, 16)
    ttnn.deallocate(ttnn_slice_493, False)
    ttnn_slice_494 = ttnn.slice(
        v_21,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_494, 17)
    ttnn.deallocate(ttnn_slice_494, False)
    ttnn_slice_495 = ttnn.slice(
        v_21,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_495, 18)
    ttnn.deallocate(ttnn_slice_495, False)
    ttnn_slice_496 = ttnn.slice(
        v_21,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_496, 19)
    ttnn.deallocate(ttnn_slice_496, False)
    ttnn_slice_497 = ttnn.slice(
        v_21,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_497, 20)
    ttnn.deallocate(ttnn_slice_497, False)
    ttnn_slice_498 = ttnn.slice(
        v_21,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_498, 21)
    ttnn.deallocate(ttnn_slice_498, False)
    ttnn_slice_499 = ttnn.slice(
        v_21,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_499, 22)
    ttnn.deallocate(ttnn_slice_499, False)
    ttnn_slice_500 = ttnn.slice(
        v_21,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_500, 23)
    ttnn.deallocate(ttnn_slice_500, False)
    ttnn_slice_501 = ttnn.slice(
        v_21,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_501, 24)
    ttnn.deallocate(ttnn_slice_501, False)
    ttnn_slice_502 = ttnn.slice(
        v_21,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_502, 25)
    ttnn.deallocate(ttnn_slice_502, False)
    ttnn_slice_503 = ttnn.slice(
        v_21,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_503, 26)
    ttnn.deallocate(ttnn_slice_503, False)
    ttnn_slice_504 = ttnn.slice(
        v_21,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_504, 27)
    ttnn.deallocate(ttnn_slice_504, False)
    ttnn_slice_505 = ttnn.slice(
        v_21,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_505, 28)
    ttnn.deallocate(ttnn_slice_505, False)
    ttnn_slice_506 = ttnn.slice(
        v_21,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_506, 29)
    ttnn.deallocate(ttnn_slice_506, False)
    ttnn_slice_507 = ttnn.slice(
        v_21,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_21, ttnn_slice_507, 30)
    ttnn.deallocate(ttnn_slice_507, False)
    ttnn_slice_508 = ttnn.slice(
        v_21,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_21, False)
    ttnn.fill_cache(args_21, ttnn_slice_508, 31)
    ttnn.deallocate(ttnn_slice_508, False)
    ttnn_to_memory_config_54 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_19 = ttnn.add(
        args_19,
        ttnn_to_memory_config_54,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_54, False)
    ttnn.deallocate(args_19, False)
    ttnn_to_memory_config_55 = ttnn.to_memory_config(
        ttnn_add_19,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_19, False)
    ttnn_to_memory_config_56 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_57 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_13 = ttnn.experimental.rotary_embedding(
        v_19,
        ttnn_to_memory_config_57,
        ttnn_to_memory_config_56,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_57, False)
    ttnn.deallocate(ttnn_to_memory_config_56, False)
    ttnn.deallocate(v_19, False)
    ttnn_slice_509 = ttnn.slice(
        ttnn_experimental_rotary_embedding_13,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_13, False)
    ttnn_to_memory_config_58 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_6 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_509,
            args_20,
            args_21,
            attn_mask=ttnn_to_memory_config_58,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_58, False)
    ttnn.deallocate(ttnn_slice_509, False)
    ttnn_transformer_concatenate_heads_6 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_6, False)
    ttnn_reshape_17 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_6,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_6, False)
    ttnn_matmul_32 = ttnn.matmul(
        ttnn_reshape_17,
        ce_cache__main["main_const_eval_12"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_17, False)
    ttnn_add_20 = ttnn.add(
        ttnn_matmul_32,
        ttnn_add_18,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_32, False)
    ttnn.deallocate(ttnn_add_18, False)
    ttnn_rms_norm_13 = ttnn.rms_norm(
        ttnn_add_20,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.6.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_33 = ttnn.matmul(
        ttnn_rms_norm_13,
        ce_cache__main["main_const_eval_11"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_34 = ttnn.matmul(
        ttnn_rms_norm_13,
        ce_cache__main["main_const_eval_77"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_13, False)
    ttnn_multiply_6 = ttnn.multiply(
        ttnn_matmul_33,
        ttnn_matmul_34,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_34, False)
    ttnn.deallocate(ttnn_matmul_33, False)
    ttnn_matmul_35 = ttnn.matmul(
        ttnn_multiply_6,
        ce_cache__main["main_const_eval_55"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_6, False)
    ttnn_add_21 = ttnn.add(
        ttnn_matmul_35,
        ttnn_add_20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_35, False)
    ttnn.deallocate(ttnn_add_20, False)
    ttnn_rms_norm_14 = ttnn.rms_norm(
        ttnn_add_21,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.7.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_36 = ttnn.matmul(
        ttnn_rms_norm_14,
        ce_cache__main["main_const_eval_16"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_14, False)
    ttnn_reshape_18 = ttnn.reshape(
        ttnn_matmul_36,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_36, False)
    v_22, v_23, v_24 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_18,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_18, False)
    ttnn_to_memory_config_59 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_60 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_14 = ttnn.experimental.rotary_embedding(
        v_23,
        ttnn_to_memory_config_60,
        ttnn_to_memory_config_59,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_60, False)
    ttnn.deallocate(ttnn_to_memory_config_59, False)
    ttnn.deallocate(v_23, False)
    ttnn_slice_510 = ttnn.slice(
        ttnn_experimental_rotary_embedding_14,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_14, False)
    ttnn_slice_511 = ttnn.slice(
        ttnn_slice_510,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_511, 0)
    ttnn.deallocate(ttnn_slice_511, False)
    ttnn_slice_512 = ttnn.slice(
        ttnn_slice_510,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_512, 1)
    ttnn.deallocate(ttnn_slice_512, False)
    ttnn_slice_513 = ttnn.slice(
        ttnn_slice_510,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_513, 2)
    ttnn.deallocate(ttnn_slice_513, False)
    ttnn_slice_514 = ttnn.slice(
        ttnn_slice_510,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_514, 3)
    ttnn.deallocate(ttnn_slice_514, False)
    ttnn_slice_515 = ttnn.slice(
        ttnn_slice_510,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_515, 4)
    ttnn.deallocate(ttnn_slice_515, False)
    ttnn_slice_516 = ttnn.slice(
        ttnn_slice_510,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_516, 5)
    ttnn.deallocate(ttnn_slice_516, False)
    ttnn_slice_517 = ttnn.slice(
        ttnn_slice_510,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_517, 6)
    ttnn.deallocate(ttnn_slice_517, False)
    ttnn_slice_518 = ttnn.slice(
        ttnn_slice_510,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_518, 7)
    ttnn.deallocate(ttnn_slice_518, False)
    ttnn_slice_519 = ttnn.slice(
        ttnn_slice_510,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_519, 8)
    ttnn.deallocate(ttnn_slice_519, False)
    ttnn_slice_520 = ttnn.slice(
        ttnn_slice_510,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_520, 9)
    ttnn.deallocate(ttnn_slice_520, False)
    ttnn_slice_521 = ttnn.slice(
        ttnn_slice_510,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_521, 10)
    ttnn.deallocate(ttnn_slice_521, False)
    ttnn_slice_522 = ttnn.slice(
        ttnn_slice_510,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_522, 11)
    ttnn.deallocate(ttnn_slice_522, False)
    ttnn_slice_523 = ttnn.slice(
        ttnn_slice_510,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_523, 12)
    ttnn.deallocate(ttnn_slice_523, False)
    ttnn_slice_524 = ttnn.slice(
        ttnn_slice_510,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_524, 13)
    ttnn.deallocate(ttnn_slice_524, False)
    ttnn_slice_525 = ttnn.slice(
        ttnn_slice_510,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_525, 14)
    ttnn.deallocate(ttnn_slice_525, False)
    ttnn_slice_526 = ttnn.slice(
        ttnn_slice_510,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_526, 15)
    ttnn.deallocate(ttnn_slice_526, False)
    ttnn_slice_527 = ttnn.slice(
        ttnn_slice_510,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_527, 16)
    ttnn.deallocate(ttnn_slice_527, False)
    ttnn_slice_528 = ttnn.slice(
        ttnn_slice_510,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_528, 17)
    ttnn.deallocate(ttnn_slice_528, False)
    ttnn_slice_529 = ttnn.slice(
        ttnn_slice_510,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_529, 18)
    ttnn.deallocate(ttnn_slice_529, False)
    ttnn_slice_530 = ttnn.slice(
        ttnn_slice_510,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_530, 19)
    ttnn.deallocate(ttnn_slice_530, False)
    ttnn_slice_531 = ttnn.slice(
        ttnn_slice_510,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_531, 20)
    ttnn.deallocate(ttnn_slice_531, False)
    ttnn_slice_532 = ttnn.slice(
        ttnn_slice_510,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_532, 21)
    ttnn.deallocate(ttnn_slice_532, False)
    ttnn_slice_533 = ttnn.slice(
        ttnn_slice_510,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_533, 22)
    ttnn.deallocate(ttnn_slice_533, False)
    ttnn_slice_534 = ttnn.slice(
        ttnn_slice_510,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_534, 23)
    ttnn.deallocate(ttnn_slice_534, False)
    ttnn_slice_535 = ttnn.slice(
        ttnn_slice_510,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_535, 24)
    ttnn.deallocate(ttnn_slice_535, False)
    ttnn_slice_536 = ttnn.slice(
        ttnn_slice_510,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_536, 25)
    ttnn.deallocate(ttnn_slice_536, False)
    ttnn_slice_537 = ttnn.slice(
        ttnn_slice_510,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_537, 26)
    ttnn.deallocate(ttnn_slice_537, False)
    ttnn_slice_538 = ttnn.slice(
        ttnn_slice_510,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_538, 27)
    ttnn.deallocate(ttnn_slice_538, False)
    ttnn_slice_539 = ttnn.slice(
        ttnn_slice_510,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_539, 28)
    ttnn.deallocate(ttnn_slice_539, False)
    ttnn_slice_540 = ttnn.slice(
        ttnn_slice_510,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_540, 29)
    ttnn.deallocate(ttnn_slice_540, False)
    ttnn_slice_541 = ttnn.slice(
        ttnn_slice_510,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_23, ttnn_slice_541, 30)
    ttnn.deallocate(ttnn_slice_541, False)
    ttnn_slice_542 = ttnn.slice(
        ttnn_slice_510,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_510, False)
    ttnn.fill_cache(args_23, ttnn_slice_542, 31)
    ttnn.deallocate(ttnn_slice_542, False)
    ttnn_slice_543 = ttnn.slice(
        v_24,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_543, 0)
    ttnn.deallocate(ttnn_slice_543, False)
    ttnn_slice_544 = ttnn.slice(
        v_24,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_544, 1)
    ttnn.deallocate(ttnn_slice_544, False)
    ttnn_slice_545 = ttnn.slice(
        v_24,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_545, 2)
    ttnn.deallocate(ttnn_slice_545, False)
    ttnn_slice_546 = ttnn.slice(
        v_24,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_546, 3)
    ttnn.deallocate(ttnn_slice_546, False)
    ttnn_slice_547 = ttnn.slice(
        v_24,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_547, 4)
    ttnn.deallocate(ttnn_slice_547, False)
    ttnn_slice_548 = ttnn.slice(
        v_24,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_548, 5)
    ttnn.deallocate(ttnn_slice_548, False)
    ttnn_slice_549 = ttnn.slice(
        v_24,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_549, 6)
    ttnn.deallocate(ttnn_slice_549, False)
    ttnn_slice_550 = ttnn.slice(
        v_24,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_550, 7)
    ttnn.deallocate(ttnn_slice_550, False)
    ttnn_slice_551 = ttnn.slice(
        v_24,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_551, 8)
    ttnn.deallocate(ttnn_slice_551, False)
    ttnn_slice_552 = ttnn.slice(
        v_24,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_552, 9)
    ttnn.deallocate(ttnn_slice_552, False)
    ttnn_slice_553 = ttnn.slice(
        v_24,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_553, 10)
    ttnn.deallocate(ttnn_slice_553, False)
    ttnn_slice_554 = ttnn.slice(
        v_24,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_554, 11)
    ttnn.deallocate(ttnn_slice_554, False)
    ttnn_slice_555 = ttnn.slice(
        v_24,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_555, 12)
    ttnn.deallocate(ttnn_slice_555, False)
    ttnn_slice_556 = ttnn.slice(
        v_24,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_556, 13)
    ttnn.deallocate(ttnn_slice_556, False)
    ttnn_slice_557 = ttnn.slice(
        v_24,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_557, 14)
    ttnn.deallocate(ttnn_slice_557, False)
    ttnn_slice_558 = ttnn.slice(
        v_24,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_558, 15)
    ttnn.deallocate(ttnn_slice_558, False)
    ttnn_slice_559 = ttnn.slice(
        v_24,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_559, 16)
    ttnn.deallocate(ttnn_slice_559, False)
    ttnn_slice_560 = ttnn.slice(
        v_24,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_560, 17)
    ttnn.deallocate(ttnn_slice_560, False)
    ttnn_slice_561 = ttnn.slice(
        v_24,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_561, 18)
    ttnn.deallocate(ttnn_slice_561, False)
    ttnn_slice_562 = ttnn.slice(
        v_24,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_562, 19)
    ttnn.deallocate(ttnn_slice_562, False)
    ttnn_slice_563 = ttnn.slice(
        v_24,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_563, 20)
    ttnn.deallocate(ttnn_slice_563, False)
    ttnn_slice_564 = ttnn.slice(
        v_24,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_564, 21)
    ttnn.deallocate(ttnn_slice_564, False)
    ttnn_slice_565 = ttnn.slice(
        v_24,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_565, 22)
    ttnn.deallocate(ttnn_slice_565, False)
    ttnn_slice_566 = ttnn.slice(
        v_24,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_566, 23)
    ttnn.deallocate(ttnn_slice_566, False)
    ttnn_slice_567 = ttnn.slice(
        v_24,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_567, 24)
    ttnn.deallocate(ttnn_slice_567, False)
    ttnn_slice_568 = ttnn.slice(
        v_24,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_568, 25)
    ttnn.deallocate(ttnn_slice_568, False)
    ttnn_slice_569 = ttnn.slice(
        v_24,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_569, 26)
    ttnn.deallocate(ttnn_slice_569, False)
    ttnn_slice_570 = ttnn.slice(
        v_24,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_570, 27)
    ttnn.deallocate(ttnn_slice_570, False)
    ttnn_slice_571 = ttnn.slice(
        v_24,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_571, 28)
    ttnn.deallocate(ttnn_slice_571, False)
    ttnn_slice_572 = ttnn.slice(
        v_24,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_572, 29)
    ttnn.deallocate(ttnn_slice_572, False)
    ttnn_slice_573 = ttnn.slice(
        v_24,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_24, ttnn_slice_573, 30)
    ttnn.deallocate(ttnn_slice_573, False)
    ttnn_slice_574 = ttnn.slice(
        v_24,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_24, False)
    ttnn.fill_cache(args_24, ttnn_slice_574, 31)
    ttnn.deallocate(ttnn_slice_574, False)
    ttnn_to_memory_config_61 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_22 = ttnn.add(
        args_22,
        ttnn_to_memory_config_61,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_61, False)
    ttnn.deallocate(args_22, False)
    ttnn_to_memory_config_62 = ttnn.to_memory_config(
        ttnn_add_22,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_22, False)
    ttnn_to_memory_config_63 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_64 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_15 = ttnn.experimental.rotary_embedding(
        v_22,
        ttnn_to_memory_config_64,
        ttnn_to_memory_config_63,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_64, False)
    ttnn.deallocate(ttnn_to_memory_config_63, False)
    ttnn.deallocate(v_22, False)
    ttnn_slice_575 = ttnn.slice(
        ttnn_experimental_rotary_embedding_15,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_15, False)
    ttnn_to_memory_config_65 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_7 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_575,
            args_23,
            args_24,
            attn_mask=ttnn_to_memory_config_65,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_65, False)
    ttnn.deallocate(ttnn_slice_575, False)
    ttnn_transformer_concatenate_heads_7 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_7, False)
    ttnn_reshape_19 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_7,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_7, False)
    ttnn_matmul_37 = ttnn.matmul(
        ttnn_reshape_19,
        ce_cache__main["main_const_eval_38"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_19, False)
    ttnn_add_23 = ttnn.add(
        ttnn_matmul_37,
        ttnn_add_21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_37, False)
    ttnn.deallocate(ttnn_add_21, False)
    ttnn_rms_norm_15 = ttnn.rms_norm(
        ttnn_add_23,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.7.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_38 = ttnn.matmul(
        ttnn_rms_norm_15,
        ce_cache__main["main_const_eval_53"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_39 = ttnn.matmul(
        ttnn_rms_norm_15,
        ce_cache__main["main_const_eval_57"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_15, False)
    ttnn_multiply_7 = ttnn.multiply(
        ttnn_matmul_38,
        ttnn_matmul_39,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_39, False)
    ttnn.deallocate(ttnn_matmul_38, False)
    ttnn_matmul_40 = ttnn.matmul(
        ttnn_multiply_7,
        ce_cache__main["main_const_eval_5"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_7, False)
    ttnn_add_24 = ttnn.add(
        ttnn_matmul_40,
        ttnn_add_23,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_40, False)
    ttnn.deallocate(ttnn_add_23, False)
    ttnn_rms_norm_16 = ttnn.rms_norm(
        ttnn_add_24,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.8.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_41 = ttnn.matmul(
        ttnn_rms_norm_16,
        ce_cache__main["main_const_eval_69"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_16, False)
    ttnn_reshape_20 = ttnn.reshape(
        ttnn_matmul_41,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_41, False)
    v_25, v_26, v_27 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_20,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_20, False)
    ttnn_to_memory_config_66 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_67 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_16 = ttnn.experimental.rotary_embedding(
        v_26,
        ttnn_to_memory_config_67,
        ttnn_to_memory_config_66,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_67, False)
    ttnn.deallocate(ttnn_to_memory_config_66, False)
    ttnn.deallocate(v_26, False)
    ttnn_slice_576 = ttnn.slice(
        ttnn_experimental_rotary_embedding_16,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_16, False)
    ttnn_slice_577 = ttnn.slice(
        ttnn_slice_576,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_577, 0)
    ttnn.deallocate(ttnn_slice_577, False)
    ttnn_slice_578 = ttnn.slice(
        ttnn_slice_576,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_578, 1)
    ttnn.deallocate(ttnn_slice_578, False)
    ttnn_slice_579 = ttnn.slice(
        ttnn_slice_576,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_579, 2)
    ttnn.deallocate(ttnn_slice_579, False)
    ttnn_slice_580 = ttnn.slice(
        ttnn_slice_576,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_580, 3)
    ttnn.deallocate(ttnn_slice_580, False)
    ttnn_slice_581 = ttnn.slice(
        ttnn_slice_576,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_581, 4)
    ttnn.deallocate(ttnn_slice_581, False)
    ttnn_slice_582 = ttnn.slice(
        ttnn_slice_576,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_582, 5)
    ttnn.deallocate(ttnn_slice_582, False)
    ttnn_slice_583 = ttnn.slice(
        ttnn_slice_576,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_583, 6)
    ttnn.deallocate(ttnn_slice_583, False)
    ttnn_slice_584 = ttnn.slice(
        ttnn_slice_576,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_584, 7)
    ttnn.deallocate(ttnn_slice_584, False)
    ttnn_slice_585 = ttnn.slice(
        ttnn_slice_576,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_585, 8)
    ttnn.deallocate(ttnn_slice_585, False)
    ttnn_slice_586 = ttnn.slice(
        ttnn_slice_576,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_586, 9)
    ttnn.deallocate(ttnn_slice_586, False)
    ttnn_slice_587 = ttnn.slice(
        ttnn_slice_576,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_587, 10)
    ttnn.deallocate(ttnn_slice_587, False)
    ttnn_slice_588 = ttnn.slice(
        ttnn_slice_576,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_588, 11)
    ttnn.deallocate(ttnn_slice_588, False)
    ttnn_slice_589 = ttnn.slice(
        ttnn_slice_576,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_589, 12)
    ttnn.deallocate(ttnn_slice_589, False)
    ttnn_slice_590 = ttnn.slice(
        ttnn_slice_576,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_590, 13)
    ttnn.deallocate(ttnn_slice_590, False)
    ttnn_slice_591 = ttnn.slice(
        ttnn_slice_576,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_591, 14)
    ttnn.deallocate(ttnn_slice_591, False)
    ttnn_slice_592 = ttnn.slice(
        ttnn_slice_576,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_592, 15)
    ttnn.deallocate(ttnn_slice_592, False)
    ttnn_slice_593 = ttnn.slice(
        ttnn_slice_576,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_593, 16)
    ttnn.deallocate(ttnn_slice_593, False)
    ttnn_slice_594 = ttnn.slice(
        ttnn_slice_576,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_594, 17)
    ttnn.deallocate(ttnn_slice_594, False)
    ttnn_slice_595 = ttnn.slice(
        ttnn_slice_576,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_595, 18)
    ttnn.deallocate(ttnn_slice_595, False)
    ttnn_slice_596 = ttnn.slice(
        ttnn_slice_576,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_596, 19)
    ttnn.deallocate(ttnn_slice_596, False)
    ttnn_slice_597 = ttnn.slice(
        ttnn_slice_576,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_597, 20)
    ttnn.deallocate(ttnn_slice_597, False)
    ttnn_slice_598 = ttnn.slice(
        ttnn_slice_576,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_598, 21)
    ttnn.deallocate(ttnn_slice_598, False)
    ttnn_slice_599 = ttnn.slice(
        ttnn_slice_576,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_599, 22)
    ttnn.deallocate(ttnn_slice_599, False)
    ttnn_slice_600 = ttnn.slice(
        ttnn_slice_576,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_600, 23)
    ttnn.deallocate(ttnn_slice_600, False)
    ttnn_slice_601 = ttnn.slice(
        ttnn_slice_576,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_601, 24)
    ttnn.deallocate(ttnn_slice_601, False)
    ttnn_slice_602 = ttnn.slice(
        ttnn_slice_576,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_602, 25)
    ttnn.deallocate(ttnn_slice_602, False)
    ttnn_slice_603 = ttnn.slice(
        ttnn_slice_576,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_603, 26)
    ttnn.deallocate(ttnn_slice_603, False)
    ttnn_slice_604 = ttnn.slice(
        ttnn_slice_576,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_604, 27)
    ttnn.deallocate(ttnn_slice_604, False)
    ttnn_slice_605 = ttnn.slice(
        ttnn_slice_576,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_605, 28)
    ttnn.deallocate(ttnn_slice_605, False)
    ttnn_slice_606 = ttnn.slice(
        ttnn_slice_576,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_606, 29)
    ttnn.deallocate(ttnn_slice_606, False)
    ttnn_slice_607 = ttnn.slice(
        ttnn_slice_576,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_26, ttnn_slice_607, 30)
    ttnn.deallocate(ttnn_slice_607, False)
    ttnn_slice_608 = ttnn.slice(
        ttnn_slice_576,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_576, False)
    ttnn.fill_cache(args_26, ttnn_slice_608, 31)
    ttnn.deallocate(ttnn_slice_608, False)
    ttnn_slice_609 = ttnn.slice(
        v_27,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_609, 0)
    ttnn.deallocate(ttnn_slice_609, False)
    ttnn_slice_610 = ttnn.slice(
        v_27,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_610, 1)
    ttnn.deallocate(ttnn_slice_610, False)
    ttnn_slice_611 = ttnn.slice(
        v_27,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_611, 2)
    ttnn.deallocate(ttnn_slice_611, False)
    ttnn_slice_612 = ttnn.slice(
        v_27,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_612, 3)
    ttnn.deallocate(ttnn_slice_612, False)
    ttnn_slice_613 = ttnn.slice(
        v_27,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_613, 4)
    ttnn.deallocate(ttnn_slice_613, False)
    ttnn_slice_614 = ttnn.slice(
        v_27,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_614, 5)
    ttnn.deallocate(ttnn_slice_614, False)
    ttnn_slice_615 = ttnn.slice(
        v_27,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_615, 6)
    ttnn.deallocate(ttnn_slice_615, False)
    ttnn_slice_616 = ttnn.slice(
        v_27,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_616, 7)
    ttnn.deallocate(ttnn_slice_616, False)
    ttnn_slice_617 = ttnn.slice(
        v_27,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_617, 8)
    ttnn.deallocate(ttnn_slice_617, False)
    ttnn_slice_618 = ttnn.slice(
        v_27,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_618, 9)
    ttnn.deallocate(ttnn_slice_618, False)
    ttnn_slice_619 = ttnn.slice(
        v_27,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_619, 10)
    ttnn.deallocate(ttnn_slice_619, False)
    ttnn_slice_620 = ttnn.slice(
        v_27,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_620, 11)
    ttnn.deallocate(ttnn_slice_620, False)
    ttnn_slice_621 = ttnn.slice(
        v_27,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_621, 12)
    ttnn.deallocate(ttnn_slice_621, False)
    ttnn_slice_622 = ttnn.slice(
        v_27,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_622, 13)
    ttnn.deallocate(ttnn_slice_622, False)
    ttnn_slice_623 = ttnn.slice(
        v_27,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_623, 14)
    ttnn.deallocate(ttnn_slice_623, False)
    ttnn_slice_624 = ttnn.slice(
        v_27,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_624, 15)
    ttnn.deallocate(ttnn_slice_624, False)
    ttnn_slice_625 = ttnn.slice(
        v_27,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_625, 16)
    ttnn.deallocate(ttnn_slice_625, False)
    ttnn_slice_626 = ttnn.slice(
        v_27,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_626, 17)
    ttnn.deallocate(ttnn_slice_626, False)
    ttnn_slice_627 = ttnn.slice(
        v_27,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_627, 18)
    ttnn.deallocate(ttnn_slice_627, False)
    ttnn_slice_628 = ttnn.slice(
        v_27,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_628, 19)
    ttnn.deallocate(ttnn_slice_628, False)
    ttnn_slice_629 = ttnn.slice(
        v_27,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_629, 20)
    ttnn.deallocate(ttnn_slice_629, False)
    ttnn_slice_630 = ttnn.slice(
        v_27,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_630, 21)
    ttnn.deallocate(ttnn_slice_630, False)
    ttnn_slice_631 = ttnn.slice(
        v_27,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_631, 22)
    ttnn.deallocate(ttnn_slice_631, False)
    ttnn_slice_632 = ttnn.slice(
        v_27,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_632, 23)
    ttnn.deallocate(ttnn_slice_632, False)
    ttnn_slice_633 = ttnn.slice(
        v_27,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_633, 24)
    ttnn.deallocate(ttnn_slice_633, False)
    ttnn_slice_634 = ttnn.slice(
        v_27,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_634, 25)
    ttnn.deallocate(ttnn_slice_634, False)
    ttnn_slice_635 = ttnn.slice(
        v_27,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_635, 26)
    ttnn.deallocate(ttnn_slice_635, False)
    ttnn_slice_636 = ttnn.slice(
        v_27,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_636, 27)
    ttnn.deallocate(ttnn_slice_636, False)
    ttnn_slice_637 = ttnn.slice(
        v_27,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_637, 28)
    ttnn.deallocate(ttnn_slice_637, False)
    ttnn_slice_638 = ttnn.slice(
        v_27,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_638, 29)
    ttnn.deallocate(ttnn_slice_638, False)
    ttnn_slice_639 = ttnn.slice(
        v_27,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_27, ttnn_slice_639, 30)
    ttnn.deallocate(ttnn_slice_639, False)
    ttnn_slice_640 = ttnn.slice(
        v_27,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_27, False)
    ttnn.fill_cache(args_27, ttnn_slice_640, 31)
    ttnn.deallocate(ttnn_slice_640, False)
    ttnn_to_memory_config_68 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_25 = ttnn.add(
        args_25,
        ttnn_to_memory_config_68,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_68, False)
    ttnn.deallocate(args_25, False)
    ttnn_to_memory_config_69 = ttnn.to_memory_config(
        ttnn_add_25,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_25, False)
    ttnn_to_memory_config_70 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_71 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_17 = ttnn.experimental.rotary_embedding(
        v_25,
        ttnn_to_memory_config_71,
        ttnn_to_memory_config_70,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_71, False)
    ttnn.deallocate(ttnn_to_memory_config_70, False)
    ttnn.deallocate(v_25, False)
    ttnn_slice_641 = ttnn.slice(
        ttnn_experimental_rotary_embedding_17,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_17, False)
    ttnn_to_memory_config_72 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_8 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_641,
            args_26,
            args_27,
            attn_mask=ttnn_to_memory_config_72,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_72, False)
    ttnn.deallocate(ttnn_slice_641, False)
    ttnn_transformer_concatenate_heads_8 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_8,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_8, False)
    ttnn_reshape_21 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_8,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_8, False)
    ttnn_matmul_42 = ttnn.matmul(
        ttnn_reshape_21,
        ce_cache__main["main_const_eval_59"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_21, False)
    ttnn_add_26 = ttnn.add(
        ttnn_matmul_42,
        ttnn_add_24,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_42, False)
    ttnn.deallocate(ttnn_add_24, False)
    ttnn_rms_norm_17 = ttnn.rms_norm(
        ttnn_add_26,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.8.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_43 = ttnn.matmul(
        ttnn_rms_norm_17,
        ce_cache__main["main_const_eval_29"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_44 = ttnn.matmul(
        ttnn_rms_norm_17,
        ce_cache__main["main_const_eval_42"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_17, False)
    ttnn_multiply_8 = ttnn.multiply(
        ttnn_matmul_43,
        ttnn_matmul_44,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_44, False)
    ttnn.deallocate(ttnn_matmul_43, False)
    ttnn_matmul_45 = ttnn.matmul(
        ttnn_multiply_8,
        ce_cache__main["main_const_eval_22"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_8, False)
    ttnn_add_27 = ttnn.add(
        ttnn_matmul_45,
        ttnn_add_26,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_45, False)
    ttnn.deallocate(ttnn_add_26, False)
    ttnn_rms_norm_18 = ttnn.rms_norm(
        ttnn_add_27,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.9.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_46 = ttnn.matmul(
        ttnn_rms_norm_18,
        ce_cache__main["main_const_eval_48"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_18, False)
    ttnn_reshape_22 = ttnn.reshape(
        ttnn_matmul_46,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_46, False)
    v_28, v_29, v_30 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_22,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_22, False)
    ttnn_to_memory_config_73 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_74 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_18 = ttnn.experimental.rotary_embedding(
        v_29,
        ttnn_to_memory_config_74,
        ttnn_to_memory_config_73,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_74, False)
    ttnn.deallocate(ttnn_to_memory_config_73, False)
    ttnn.deallocate(v_29, False)
    ttnn_slice_642 = ttnn.slice(
        ttnn_experimental_rotary_embedding_18,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_18, False)
    ttnn_slice_643 = ttnn.slice(
        ttnn_slice_642,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_643, 0)
    ttnn.deallocate(ttnn_slice_643, False)
    ttnn_slice_644 = ttnn.slice(
        ttnn_slice_642,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_644, 1)
    ttnn.deallocate(ttnn_slice_644, False)
    ttnn_slice_645 = ttnn.slice(
        ttnn_slice_642,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_645, 2)
    ttnn.deallocate(ttnn_slice_645, False)
    ttnn_slice_646 = ttnn.slice(
        ttnn_slice_642,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_646, 3)
    ttnn.deallocate(ttnn_slice_646, False)
    ttnn_slice_647 = ttnn.slice(
        ttnn_slice_642,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_647, 4)
    ttnn.deallocate(ttnn_slice_647, False)
    ttnn_slice_648 = ttnn.slice(
        ttnn_slice_642,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_648, 5)
    ttnn.deallocate(ttnn_slice_648, False)
    ttnn_slice_649 = ttnn.slice(
        ttnn_slice_642,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_649, 6)
    ttnn.deallocate(ttnn_slice_649, False)
    ttnn_slice_650 = ttnn.slice(
        ttnn_slice_642,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_650, 7)
    ttnn.deallocate(ttnn_slice_650, False)
    ttnn_slice_651 = ttnn.slice(
        ttnn_slice_642,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_651, 8)
    ttnn.deallocate(ttnn_slice_651, False)
    ttnn_slice_652 = ttnn.slice(
        ttnn_slice_642,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_652, 9)
    ttnn.deallocate(ttnn_slice_652, False)
    ttnn_slice_653 = ttnn.slice(
        ttnn_slice_642,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_653, 10)
    ttnn.deallocate(ttnn_slice_653, False)
    ttnn_slice_654 = ttnn.slice(
        ttnn_slice_642,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_654, 11)
    ttnn.deallocate(ttnn_slice_654, False)
    ttnn_slice_655 = ttnn.slice(
        ttnn_slice_642,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_655, 12)
    ttnn.deallocate(ttnn_slice_655, False)
    ttnn_slice_656 = ttnn.slice(
        ttnn_slice_642,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_656, 13)
    ttnn.deallocate(ttnn_slice_656, False)
    ttnn_slice_657 = ttnn.slice(
        ttnn_slice_642,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_657, 14)
    ttnn.deallocate(ttnn_slice_657, False)
    ttnn_slice_658 = ttnn.slice(
        ttnn_slice_642,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_658, 15)
    ttnn.deallocate(ttnn_slice_658, False)
    ttnn_slice_659 = ttnn.slice(
        ttnn_slice_642,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_659, 16)
    ttnn.deallocate(ttnn_slice_659, False)
    ttnn_slice_660 = ttnn.slice(
        ttnn_slice_642,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_660, 17)
    ttnn.deallocate(ttnn_slice_660, False)
    ttnn_slice_661 = ttnn.slice(
        ttnn_slice_642,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_661, 18)
    ttnn.deallocate(ttnn_slice_661, False)
    ttnn_slice_662 = ttnn.slice(
        ttnn_slice_642,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_662, 19)
    ttnn.deallocate(ttnn_slice_662, False)
    ttnn_slice_663 = ttnn.slice(
        ttnn_slice_642,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_663, 20)
    ttnn.deallocate(ttnn_slice_663, False)
    ttnn_slice_664 = ttnn.slice(
        ttnn_slice_642,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_664, 21)
    ttnn.deallocate(ttnn_slice_664, False)
    ttnn_slice_665 = ttnn.slice(
        ttnn_slice_642,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_665, 22)
    ttnn.deallocate(ttnn_slice_665, False)
    ttnn_slice_666 = ttnn.slice(
        ttnn_slice_642,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_666, 23)
    ttnn.deallocate(ttnn_slice_666, False)
    ttnn_slice_667 = ttnn.slice(
        ttnn_slice_642,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_667, 24)
    ttnn.deallocate(ttnn_slice_667, False)
    ttnn_slice_668 = ttnn.slice(
        ttnn_slice_642,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_668, 25)
    ttnn.deallocate(ttnn_slice_668, False)
    ttnn_slice_669 = ttnn.slice(
        ttnn_slice_642,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_669, 26)
    ttnn.deallocate(ttnn_slice_669, False)
    ttnn_slice_670 = ttnn.slice(
        ttnn_slice_642,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_670, 27)
    ttnn.deallocate(ttnn_slice_670, False)
    ttnn_slice_671 = ttnn.slice(
        ttnn_slice_642,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_671, 28)
    ttnn.deallocate(ttnn_slice_671, False)
    ttnn_slice_672 = ttnn.slice(
        ttnn_slice_642,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_672, 29)
    ttnn.deallocate(ttnn_slice_672, False)
    ttnn_slice_673 = ttnn.slice(
        ttnn_slice_642,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_29, ttnn_slice_673, 30)
    ttnn.deallocate(ttnn_slice_673, False)
    ttnn_slice_674 = ttnn.slice(
        ttnn_slice_642,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_642, False)
    ttnn.fill_cache(args_29, ttnn_slice_674, 31)
    ttnn.deallocate(ttnn_slice_674, False)
    ttnn_slice_675 = ttnn.slice(
        v_30,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_675, 0)
    ttnn.deallocate(ttnn_slice_675, False)
    ttnn_slice_676 = ttnn.slice(
        v_30,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_676, 1)
    ttnn.deallocate(ttnn_slice_676, False)
    ttnn_slice_677 = ttnn.slice(
        v_30,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_677, 2)
    ttnn.deallocate(ttnn_slice_677, False)
    ttnn_slice_678 = ttnn.slice(
        v_30,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_678, 3)
    ttnn.deallocate(ttnn_slice_678, False)
    ttnn_slice_679 = ttnn.slice(
        v_30,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_679, 4)
    ttnn.deallocate(ttnn_slice_679, False)
    ttnn_slice_680 = ttnn.slice(
        v_30,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_680, 5)
    ttnn.deallocate(ttnn_slice_680, False)
    ttnn_slice_681 = ttnn.slice(
        v_30,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_681, 6)
    ttnn.deallocate(ttnn_slice_681, False)
    ttnn_slice_682 = ttnn.slice(
        v_30,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_682, 7)
    ttnn.deallocate(ttnn_slice_682, False)
    ttnn_slice_683 = ttnn.slice(
        v_30,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_683, 8)
    ttnn.deallocate(ttnn_slice_683, False)
    ttnn_slice_684 = ttnn.slice(
        v_30,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_684, 9)
    ttnn.deallocate(ttnn_slice_684, False)
    ttnn_slice_685 = ttnn.slice(
        v_30,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_685, 10)
    ttnn.deallocate(ttnn_slice_685, False)
    ttnn_slice_686 = ttnn.slice(
        v_30,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_686, 11)
    ttnn.deallocate(ttnn_slice_686, False)
    ttnn_slice_687 = ttnn.slice(
        v_30,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_687, 12)
    ttnn.deallocate(ttnn_slice_687, False)
    ttnn_slice_688 = ttnn.slice(
        v_30,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_688, 13)
    ttnn.deallocate(ttnn_slice_688, False)
    ttnn_slice_689 = ttnn.slice(
        v_30,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_689, 14)
    ttnn.deallocate(ttnn_slice_689, False)
    ttnn_slice_690 = ttnn.slice(
        v_30,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_690, 15)
    ttnn.deallocate(ttnn_slice_690, False)
    ttnn_slice_691 = ttnn.slice(
        v_30,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_691, 16)
    ttnn.deallocate(ttnn_slice_691, False)
    ttnn_slice_692 = ttnn.slice(
        v_30,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_692, 17)
    ttnn.deallocate(ttnn_slice_692, False)
    ttnn_slice_693 = ttnn.slice(
        v_30,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_693, 18)
    ttnn.deallocate(ttnn_slice_693, False)
    ttnn_slice_694 = ttnn.slice(
        v_30,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_694, 19)
    ttnn.deallocate(ttnn_slice_694, False)
    ttnn_slice_695 = ttnn.slice(
        v_30,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_695, 20)
    ttnn.deallocate(ttnn_slice_695, False)
    ttnn_slice_696 = ttnn.slice(
        v_30,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_696, 21)
    ttnn.deallocate(ttnn_slice_696, False)
    ttnn_slice_697 = ttnn.slice(
        v_30,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_697, 22)
    ttnn.deallocate(ttnn_slice_697, False)
    ttnn_slice_698 = ttnn.slice(
        v_30,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_698, 23)
    ttnn.deallocate(ttnn_slice_698, False)
    ttnn_slice_699 = ttnn.slice(
        v_30,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_699, 24)
    ttnn.deallocate(ttnn_slice_699, False)
    ttnn_slice_700 = ttnn.slice(
        v_30,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_700, 25)
    ttnn.deallocate(ttnn_slice_700, False)
    ttnn_slice_701 = ttnn.slice(
        v_30,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_701, 26)
    ttnn.deallocate(ttnn_slice_701, False)
    ttnn_slice_702 = ttnn.slice(
        v_30,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_702, 27)
    ttnn.deallocate(ttnn_slice_702, False)
    ttnn_slice_703 = ttnn.slice(
        v_30,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_703, 28)
    ttnn.deallocate(ttnn_slice_703, False)
    ttnn_slice_704 = ttnn.slice(
        v_30,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_704, 29)
    ttnn.deallocate(ttnn_slice_704, False)
    ttnn_slice_705 = ttnn.slice(
        v_30,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_30, ttnn_slice_705, 30)
    ttnn.deallocate(ttnn_slice_705, False)
    ttnn_slice_706 = ttnn.slice(
        v_30,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_30, False)
    ttnn.fill_cache(args_30, ttnn_slice_706, 31)
    ttnn.deallocate(ttnn_slice_706, False)
    ttnn_to_memory_config_75 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_28 = ttnn.add(
        args_28,
        ttnn_to_memory_config_75,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_75, False)
    ttnn.deallocate(args_28, False)
    ttnn_to_memory_config_76 = ttnn.to_memory_config(
        ttnn_add_28,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_28, False)
    ttnn_to_memory_config_77 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_78 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_19 = ttnn.experimental.rotary_embedding(
        v_28,
        ttnn_to_memory_config_78,
        ttnn_to_memory_config_77,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_78, False)
    ttnn.deallocate(ttnn_to_memory_config_77, False)
    ttnn.deallocate(v_28, False)
    ttnn_slice_707 = ttnn.slice(
        ttnn_experimental_rotary_embedding_19,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_19, False)
    ttnn_to_memory_config_79 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_9 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_707,
            args_29,
            args_30,
            attn_mask=ttnn_to_memory_config_79,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_79, False)
    ttnn.deallocate(ttnn_slice_707, False)
    ttnn_transformer_concatenate_heads_9 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_9,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_9, False)
    ttnn_reshape_23 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_9,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_9, False)
    ttnn_matmul_47 = ttnn.matmul(
        ttnn_reshape_23,
        ce_cache__main["main_const_eval_63"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_23, False)
    ttnn_add_29 = ttnn.add(
        ttnn_matmul_47,
        ttnn_add_27,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_47, False)
    ttnn.deallocate(ttnn_add_27, False)
    ttnn_rms_norm_19 = ttnn.rms_norm(
        ttnn_add_29,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.9.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_48 = ttnn.matmul(
        ttnn_rms_norm_19,
        ce_cache__main["main_const_eval_10"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_49 = ttnn.matmul(
        ttnn_rms_norm_19,
        ce_cache__main["main_const_eval_60"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_19, False)
    ttnn_multiply_9 = ttnn.multiply(
        ttnn_matmul_48,
        ttnn_matmul_49,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_49, False)
    ttnn.deallocate(ttnn_matmul_48, False)
    ttnn_matmul_50 = ttnn.matmul(
        ttnn_multiply_9,
        ce_cache__main["main_const_eval_45"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_9, False)
    ttnn_add_30 = ttnn.add(
        ttnn_matmul_50,
        ttnn_add_29,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_50, False)
    ttnn.deallocate(ttnn_add_29, False)
    ttnn_rms_norm_20 = ttnn.rms_norm(
        ttnn_add_30,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.10.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_51 = ttnn.matmul(
        ttnn_rms_norm_20,
        ce_cache__main["main_const_eval_27"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_20, False)
    ttnn_reshape_24 = ttnn.reshape(
        ttnn_matmul_51,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_51, False)
    v_31, v_32, v_33 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_24,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_24, False)
    ttnn_to_memory_config_80 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_81 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_20 = ttnn.experimental.rotary_embedding(
        v_32,
        ttnn_to_memory_config_81,
        ttnn_to_memory_config_80,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_81, False)
    ttnn.deallocate(ttnn_to_memory_config_80, False)
    ttnn.deallocate(v_32, False)
    ttnn_slice_708 = ttnn.slice(
        ttnn_experimental_rotary_embedding_20,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_20, False)
    ttnn_slice_709 = ttnn.slice(
        ttnn_slice_708,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_709, 0)
    ttnn.deallocate(ttnn_slice_709, False)
    ttnn_slice_710 = ttnn.slice(
        ttnn_slice_708,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_710, 1)
    ttnn.deallocate(ttnn_slice_710, False)
    ttnn_slice_711 = ttnn.slice(
        ttnn_slice_708,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_711, 2)
    ttnn.deallocate(ttnn_slice_711, False)
    ttnn_slice_712 = ttnn.slice(
        ttnn_slice_708,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_712, 3)
    ttnn.deallocate(ttnn_slice_712, False)
    ttnn_slice_713 = ttnn.slice(
        ttnn_slice_708,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_713, 4)
    ttnn.deallocate(ttnn_slice_713, False)
    ttnn_slice_714 = ttnn.slice(
        ttnn_slice_708,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_714, 5)
    ttnn.deallocate(ttnn_slice_714, False)
    ttnn_slice_715 = ttnn.slice(
        ttnn_slice_708,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_715, 6)
    ttnn.deallocate(ttnn_slice_715, False)
    ttnn_slice_716 = ttnn.slice(
        ttnn_slice_708,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_716, 7)
    ttnn.deallocate(ttnn_slice_716, False)
    ttnn_slice_717 = ttnn.slice(
        ttnn_slice_708,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_717, 8)
    ttnn.deallocate(ttnn_slice_717, False)
    ttnn_slice_718 = ttnn.slice(
        ttnn_slice_708,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_718, 9)
    ttnn.deallocate(ttnn_slice_718, False)
    ttnn_slice_719 = ttnn.slice(
        ttnn_slice_708,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_719, 10)
    ttnn.deallocate(ttnn_slice_719, False)
    ttnn_slice_720 = ttnn.slice(
        ttnn_slice_708,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_720, 11)
    ttnn.deallocate(ttnn_slice_720, False)
    ttnn_slice_721 = ttnn.slice(
        ttnn_slice_708,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_721, 12)
    ttnn.deallocate(ttnn_slice_721, False)
    ttnn_slice_722 = ttnn.slice(
        ttnn_slice_708,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_722, 13)
    ttnn.deallocate(ttnn_slice_722, False)
    ttnn_slice_723 = ttnn.slice(
        ttnn_slice_708,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_723, 14)
    ttnn.deallocate(ttnn_slice_723, False)
    ttnn_slice_724 = ttnn.slice(
        ttnn_slice_708,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_724, 15)
    ttnn.deallocate(ttnn_slice_724, False)
    ttnn_slice_725 = ttnn.slice(
        ttnn_slice_708,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_725, 16)
    ttnn.deallocate(ttnn_slice_725, False)
    ttnn_slice_726 = ttnn.slice(
        ttnn_slice_708,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_726, 17)
    ttnn.deallocate(ttnn_slice_726, False)
    ttnn_slice_727 = ttnn.slice(
        ttnn_slice_708,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_727, 18)
    ttnn.deallocate(ttnn_slice_727, False)
    ttnn_slice_728 = ttnn.slice(
        ttnn_slice_708,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_728, 19)
    ttnn.deallocate(ttnn_slice_728, False)
    ttnn_slice_729 = ttnn.slice(
        ttnn_slice_708,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_729, 20)
    ttnn.deallocate(ttnn_slice_729, False)
    ttnn_slice_730 = ttnn.slice(
        ttnn_slice_708,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_730, 21)
    ttnn.deallocate(ttnn_slice_730, False)
    ttnn_slice_731 = ttnn.slice(
        ttnn_slice_708,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_731, 22)
    ttnn.deallocate(ttnn_slice_731, False)
    ttnn_slice_732 = ttnn.slice(
        ttnn_slice_708,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_732, 23)
    ttnn.deallocate(ttnn_slice_732, False)
    ttnn_slice_733 = ttnn.slice(
        ttnn_slice_708,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_733, 24)
    ttnn.deallocate(ttnn_slice_733, False)
    ttnn_slice_734 = ttnn.slice(
        ttnn_slice_708,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_734, 25)
    ttnn.deallocate(ttnn_slice_734, False)
    ttnn_slice_735 = ttnn.slice(
        ttnn_slice_708,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_735, 26)
    ttnn.deallocate(ttnn_slice_735, False)
    ttnn_slice_736 = ttnn.slice(
        ttnn_slice_708,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_736, 27)
    ttnn.deallocate(ttnn_slice_736, False)
    ttnn_slice_737 = ttnn.slice(
        ttnn_slice_708,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_737, 28)
    ttnn.deallocate(ttnn_slice_737, False)
    ttnn_slice_738 = ttnn.slice(
        ttnn_slice_708,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_738, 29)
    ttnn.deallocate(ttnn_slice_738, False)
    ttnn_slice_739 = ttnn.slice(
        ttnn_slice_708,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_32, ttnn_slice_739, 30)
    ttnn.deallocate(ttnn_slice_739, False)
    ttnn_slice_740 = ttnn.slice(
        ttnn_slice_708,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_708, False)
    ttnn.fill_cache(args_32, ttnn_slice_740, 31)
    ttnn.deallocate(ttnn_slice_740, False)
    ttnn_slice_741 = ttnn.slice(
        v_33,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_741, 0)
    ttnn.deallocate(ttnn_slice_741, False)
    ttnn_slice_742 = ttnn.slice(
        v_33,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_742, 1)
    ttnn.deallocate(ttnn_slice_742, False)
    ttnn_slice_743 = ttnn.slice(
        v_33,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_743, 2)
    ttnn.deallocate(ttnn_slice_743, False)
    ttnn_slice_744 = ttnn.slice(
        v_33,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_744, 3)
    ttnn.deallocate(ttnn_slice_744, False)
    ttnn_slice_745 = ttnn.slice(
        v_33,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_745, 4)
    ttnn.deallocate(ttnn_slice_745, False)
    ttnn_slice_746 = ttnn.slice(
        v_33,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_746, 5)
    ttnn.deallocate(ttnn_slice_746, False)
    ttnn_slice_747 = ttnn.slice(
        v_33,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_747, 6)
    ttnn.deallocate(ttnn_slice_747, False)
    ttnn_slice_748 = ttnn.slice(
        v_33,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_748, 7)
    ttnn.deallocate(ttnn_slice_748, False)
    ttnn_slice_749 = ttnn.slice(
        v_33,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_749, 8)
    ttnn.deallocate(ttnn_slice_749, False)
    ttnn_slice_750 = ttnn.slice(
        v_33,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_750, 9)
    ttnn.deallocate(ttnn_slice_750, False)
    ttnn_slice_751 = ttnn.slice(
        v_33,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_751, 10)
    ttnn.deallocate(ttnn_slice_751, False)
    ttnn_slice_752 = ttnn.slice(
        v_33,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_752, 11)
    ttnn.deallocate(ttnn_slice_752, False)
    ttnn_slice_753 = ttnn.slice(
        v_33,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_753, 12)
    ttnn.deallocate(ttnn_slice_753, False)
    ttnn_slice_754 = ttnn.slice(
        v_33,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_754, 13)
    ttnn.deallocate(ttnn_slice_754, False)
    ttnn_slice_755 = ttnn.slice(
        v_33,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_755, 14)
    ttnn.deallocate(ttnn_slice_755, False)
    ttnn_slice_756 = ttnn.slice(
        v_33,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_756, 15)
    ttnn.deallocate(ttnn_slice_756, False)
    ttnn_slice_757 = ttnn.slice(
        v_33,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_757, 16)
    ttnn.deallocate(ttnn_slice_757, False)
    ttnn_slice_758 = ttnn.slice(
        v_33,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_758, 17)
    ttnn.deallocate(ttnn_slice_758, False)
    ttnn_slice_759 = ttnn.slice(
        v_33,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_759, 18)
    ttnn.deallocate(ttnn_slice_759, False)
    ttnn_slice_760 = ttnn.slice(
        v_33,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_760, 19)
    ttnn.deallocate(ttnn_slice_760, False)
    ttnn_slice_761 = ttnn.slice(
        v_33,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_761, 20)
    ttnn.deallocate(ttnn_slice_761, False)
    ttnn_slice_762 = ttnn.slice(
        v_33,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_762, 21)
    ttnn.deallocate(ttnn_slice_762, False)
    ttnn_slice_763 = ttnn.slice(
        v_33,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_763, 22)
    ttnn.deallocate(ttnn_slice_763, False)
    ttnn_slice_764 = ttnn.slice(
        v_33,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_764, 23)
    ttnn.deallocate(ttnn_slice_764, False)
    ttnn_slice_765 = ttnn.slice(
        v_33,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_765, 24)
    ttnn.deallocate(ttnn_slice_765, False)
    ttnn_slice_766 = ttnn.slice(
        v_33,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_766, 25)
    ttnn.deallocate(ttnn_slice_766, False)
    ttnn_slice_767 = ttnn.slice(
        v_33,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_767, 26)
    ttnn.deallocate(ttnn_slice_767, False)
    ttnn_slice_768 = ttnn.slice(
        v_33,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_768, 27)
    ttnn.deallocate(ttnn_slice_768, False)
    ttnn_slice_769 = ttnn.slice(
        v_33,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_769, 28)
    ttnn.deallocate(ttnn_slice_769, False)
    ttnn_slice_770 = ttnn.slice(
        v_33,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_770, 29)
    ttnn.deallocate(ttnn_slice_770, False)
    ttnn_slice_771 = ttnn.slice(
        v_33,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_33, ttnn_slice_771, 30)
    ttnn.deallocate(ttnn_slice_771, False)
    ttnn_slice_772 = ttnn.slice(
        v_33,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_33, False)
    ttnn.fill_cache(args_33, ttnn_slice_772, 31)
    ttnn.deallocate(ttnn_slice_772, False)
    ttnn_to_memory_config_82 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_31 = ttnn.add(
        args_31,
        ttnn_to_memory_config_82,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_82, False)
    ttnn.deallocate(args_31, False)
    ttnn_to_memory_config_83 = ttnn.to_memory_config(
        ttnn_add_31,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_31, False)
    ttnn_to_memory_config_84 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_85 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_21 = ttnn.experimental.rotary_embedding(
        v_31,
        ttnn_to_memory_config_85,
        ttnn_to_memory_config_84,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_85, False)
    ttnn.deallocate(ttnn_to_memory_config_84, False)
    ttnn.deallocate(v_31, False)
    ttnn_slice_773 = ttnn.slice(
        ttnn_experimental_rotary_embedding_21,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_21, False)
    ttnn_to_memory_config_86 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_10 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_773,
            args_32,
            args_33,
            attn_mask=ttnn_to_memory_config_86,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_86, False)
    ttnn.deallocate(ttnn_slice_773, False)
    ttnn_transformer_concatenate_heads_10 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_10,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_10, False)
    ttnn_reshape_25 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_10,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_10, False)
    ttnn_matmul_52 = ttnn.matmul(
        ttnn_reshape_25,
        ce_cache__main["main_const_eval_19"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_25, False)
    ttnn_add_32 = ttnn.add(
        ttnn_matmul_52,
        ttnn_add_30,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_52, False)
    ttnn.deallocate(ttnn_add_30, False)
    ttnn_rms_norm_21 = ttnn.rms_norm(
        ttnn_add_32,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.10.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_53 = ttnn.matmul(
        ttnn_rms_norm_21,
        ce_cache__main["main_const_eval_72"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_54 = ttnn.matmul(
        ttnn_rms_norm_21,
        ce_cache__main["main_const_eval_85"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_21, False)
    ttnn_multiply_10 = ttnn.multiply(
        ttnn_matmul_53,
        ttnn_matmul_54,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_54, False)
    ttnn.deallocate(ttnn_matmul_53, False)
    ttnn_matmul_55 = ttnn.matmul(
        ttnn_multiply_10,
        ce_cache__main["main_const_eval_71"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_10, False)
    ttnn_add_33 = ttnn.add(
        ttnn_matmul_55,
        ttnn_add_32,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_55, False)
    ttnn.deallocate(ttnn_add_32, False)
    ttnn_rms_norm_22 = ttnn.rms_norm(
        ttnn_add_33,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.11.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_56 = ttnn.matmul(
        ttnn_rms_norm_22,
        ce_cache__main["main_const_eval_17"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_22, False)
    ttnn_reshape_26 = ttnn.reshape(
        ttnn_matmul_56,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_56, False)
    v_34, v_35, v_36 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_26,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_26, False)
    ttnn_to_memory_config_87 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_88 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_22 = ttnn.experimental.rotary_embedding(
        v_35,
        ttnn_to_memory_config_88,
        ttnn_to_memory_config_87,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_88, False)
    ttnn.deallocate(ttnn_to_memory_config_87, False)
    ttnn.deallocate(v_35, False)
    ttnn_slice_774 = ttnn.slice(
        ttnn_experimental_rotary_embedding_22,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_22, False)
    ttnn_slice_775 = ttnn.slice(
        ttnn_slice_774,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_775, 0)
    ttnn.deallocate(ttnn_slice_775, False)
    ttnn_slice_776 = ttnn.slice(
        ttnn_slice_774,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_776, 1)
    ttnn.deallocate(ttnn_slice_776, False)
    ttnn_slice_777 = ttnn.slice(
        ttnn_slice_774,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_777, 2)
    ttnn.deallocate(ttnn_slice_777, False)
    ttnn_slice_778 = ttnn.slice(
        ttnn_slice_774,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_778, 3)
    ttnn.deallocate(ttnn_slice_778, False)
    ttnn_slice_779 = ttnn.slice(
        ttnn_slice_774,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_779, 4)
    ttnn.deallocate(ttnn_slice_779, False)
    ttnn_slice_780 = ttnn.slice(
        ttnn_slice_774,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_780, 5)
    ttnn.deallocate(ttnn_slice_780, False)
    ttnn_slice_781 = ttnn.slice(
        ttnn_slice_774,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_781, 6)
    ttnn.deallocate(ttnn_slice_781, False)
    ttnn_slice_782 = ttnn.slice(
        ttnn_slice_774,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_782, 7)
    ttnn.deallocate(ttnn_slice_782, False)
    ttnn_slice_783 = ttnn.slice(
        ttnn_slice_774,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_783, 8)
    ttnn.deallocate(ttnn_slice_783, False)
    ttnn_slice_784 = ttnn.slice(
        ttnn_slice_774,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_784, 9)
    ttnn.deallocate(ttnn_slice_784, False)
    ttnn_slice_785 = ttnn.slice(
        ttnn_slice_774,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_785, 10)
    ttnn.deallocate(ttnn_slice_785, False)
    ttnn_slice_786 = ttnn.slice(
        ttnn_slice_774,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_786, 11)
    ttnn.deallocate(ttnn_slice_786, False)
    ttnn_slice_787 = ttnn.slice(
        ttnn_slice_774,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_787, 12)
    ttnn.deallocate(ttnn_slice_787, False)
    ttnn_slice_788 = ttnn.slice(
        ttnn_slice_774,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_788, 13)
    ttnn.deallocate(ttnn_slice_788, False)
    ttnn_slice_789 = ttnn.slice(
        ttnn_slice_774,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_789, 14)
    ttnn.deallocate(ttnn_slice_789, False)
    ttnn_slice_790 = ttnn.slice(
        ttnn_slice_774,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_790, 15)
    ttnn.deallocate(ttnn_slice_790, False)
    ttnn_slice_791 = ttnn.slice(
        ttnn_slice_774,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_791, 16)
    ttnn.deallocate(ttnn_slice_791, False)
    ttnn_slice_792 = ttnn.slice(
        ttnn_slice_774,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_792, 17)
    ttnn.deallocate(ttnn_slice_792, False)
    ttnn_slice_793 = ttnn.slice(
        ttnn_slice_774,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_793, 18)
    ttnn.deallocate(ttnn_slice_793, False)
    ttnn_slice_794 = ttnn.slice(
        ttnn_slice_774,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_794, 19)
    ttnn.deallocate(ttnn_slice_794, False)
    ttnn_slice_795 = ttnn.slice(
        ttnn_slice_774,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_795, 20)
    ttnn.deallocate(ttnn_slice_795, False)
    ttnn_slice_796 = ttnn.slice(
        ttnn_slice_774,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_796, 21)
    ttnn.deallocate(ttnn_slice_796, False)
    ttnn_slice_797 = ttnn.slice(
        ttnn_slice_774,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_797, 22)
    ttnn.deallocate(ttnn_slice_797, False)
    ttnn_slice_798 = ttnn.slice(
        ttnn_slice_774,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_798, 23)
    ttnn.deallocate(ttnn_slice_798, False)
    ttnn_slice_799 = ttnn.slice(
        ttnn_slice_774,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_799, 24)
    ttnn.deallocate(ttnn_slice_799, False)
    ttnn_slice_800 = ttnn.slice(
        ttnn_slice_774,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_800, 25)
    ttnn.deallocate(ttnn_slice_800, False)
    ttnn_slice_801 = ttnn.slice(
        ttnn_slice_774,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_801, 26)
    ttnn.deallocate(ttnn_slice_801, False)
    ttnn_slice_802 = ttnn.slice(
        ttnn_slice_774,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_802, 27)
    ttnn.deallocate(ttnn_slice_802, False)
    ttnn_slice_803 = ttnn.slice(
        ttnn_slice_774,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_803, 28)
    ttnn.deallocate(ttnn_slice_803, False)
    ttnn_slice_804 = ttnn.slice(
        ttnn_slice_774,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_804, 29)
    ttnn.deallocate(ttnn_slice_804, False)
    ttnn_slice_805 = ttnn.slice(
        ttnn_slice_774,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_35, ttnn_slice_805, 30)
    ttnn.deallocate(ttnn_slice_805, False)
    ttnn_slice_806 = ttnn.slice(
        ttnn_slice_774,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_774, False)
    ttnn.fill_cache(args_35, ttnn_slice_806, 31)
    ttnn.deallocate(ttnn_slice_806, False)
    ttnn_slice_807 = ttnn.slice(
        v_36,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_807, 0)
    ttnn.deallocate(ttnn_slice_807, False)
    ttnn_slice_808 = ttnn.slice(
        v_36,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_808, 1)
    ttnn.deallocate(ttnn_slice_808, False)
    ttnn_slice_809 = ttnn.slice(
        v_36,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_809, 2)
    ttnn.deallocate(ttnn_slice_809, False)
    ttnn_slice_810 = ttnn.slice(
        v_36,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_810, 3)
    ttnn.deallocate(ttnn_slice_810, False)
    ttnn_slice_811 = ttnn.slice(
        v_36,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_811, 4)
    ttnn.deallocate(ttnn_slice_811, False)
    ttnn_slice_812 = ttnn.slice(
        v_36,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_812, 5)
    ttnn.deallocate(ttnn_slice_812, False)
    ttnn_slice_813 = ttnn.slice(
        v_36,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_813, 6)
    ttnn.deallocate(ttnn_slice_813, False)
    ttnn_slice_814 = ttnn.slice(
        v_36,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_814, 7)
    ttnn.deallocate(ttnn_slice_814, False)
    ttnn_slice_815 = ttnn.slice(
        v_36,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_815, 8)
    ttnn.deallocate(ttnn_slice_815, False)
    ttnn_slice_816 = ttnn.slice(
        v_36,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_816, 9)
    ttnn.deallocate(ttnn_slice_816, False)
    ttnn_slice_817 = ttnn.slice(
        v_36,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_817, 10)
    ttnn.deallocate(ttnn_slice_817, False)
    ttnn_slice_818 = ttnn.slice(
        v_36,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_818, 11)
    ttnn.deallocate(ttnn_slice_818, False)
    ttnn_slice_819 = ttnn.slice(
        v_36,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_819, 12)
    ttnn.deallocate(ttnn_slice_819, False)
    ttnn_slice_820 = ttnn.slice(
        v_36,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_820, 13)
    ttnn.deallocate(ttnn_slice_820, False)
    ttnn_slice_821 = ttnn.slice(
        v_36,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_821, 14)
    ttnn.deallocate(ttnn_slice_821, False)
    ttnn_slice_822 = ttnn.slice(
        v_36,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_822, 15)
    ttnn.deallocate(ttnn_slice_822, False)
    ttnn_slice_823 = ttnn.slice(
        v_36,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_823, 16)
    ttnn.deallocate(ttnn_slice_823, False)
    ttnn_slice_824 = ttnn.slice(
        v_36,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_824, 17)
    ttnn.deallocate(ttnn_slice_824, False)
    ttnn_slice_825 = ttnn.slice(
        v_36,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_825, 18)
    ttnn.deallocate(ttnn_slice_825, False)
    ttnn_slice_826 = ttnn.slice(
        v_36,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_826, 19)
    ttnn.deallocate(ttnn_slice_826, False)
    ttnn_slice_827 = ttnn.slice(
        v_36,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_827, 20)
    ttnn.deallocate(ttnn_slice_827, False)
    ttnn_slice_828 = ttnn.slice(
        v_36,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_828, 21)
    ttnn.deallocate(ttnn_slice_828, False)
    ttnn_slice_829 = ttnn.slice(
        v_36,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_829, 22)
    ttnn.deallocate(ttnn_slice_829, False)
    ttnn_slice_830 = ttnn.slice(
        v_36,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_830, 23)
    ttnn.deallocate(ttnn_slice_830, False)
    ttnn_slice_831 = ttnn.slice(
        v_36,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_831, 24)
    ttnn.deallocate(ttnn_slice_831, False)
    ttnn_slice_832 = ttnn.slice(
        v_36,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_832, 25)
    ttnn.deallocate(ttnn_slice_832, False)
    ttnn_slice_833 = ttnn.slice(
        v_36,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_833, 26)
    ttnn.deallocate(ttnn_slice_833, False)
    ttnn_slice_834 = ttnn.slice(
        v_36,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_834, 27)
    ttnn.deallocate(ttnn_slice_834, False)
    ttnn_slice_835 = ttnn.slice(
        v_36,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_835, 28)
    ttnn.deallocate(ttnn_slice_835, False)
    ttnn_slice_836 = ttnn.slice(
        v_36,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_836, 29)
    ttnn.deallocate(ttnn_slice_836, False)
    ttnn_slice_837 = ttnn.slice(
        v_36,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_36, ttnn_slice_837, 30)
    ttnn.deallocate(ttnn_slice_837, False)
    ttnn_slice_838 = ttnn.slice(
        v_36,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_36, False)
    ttnn.fill_cache(args_36, ttnn_slice_838, 31)
    ttnn.deallocate(ttnn_slice_838, False)
    ttnn_to_memory_config_89 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_34 = ttnn.add(
        args_34,
        ttnn_to_memory_config_89,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_89, False)
    ttnn.deallocate(args_34, False)
    ttnn_to_memory_config_90 = ttnn.to_memory_config(
        ttnn_add_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_34, False)
    ttnn_to_memory_config_91 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_92 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_23 = ttnn.experimental.rotary_embedding(
        v_34,
        ttnn_to_memory_config_92,
        ttnn_to_memory_config_91,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_92, False)
    ttnn.deallocate(ttnn_to_memory_config_91, False)
    ttnn.deallocate(v_34, False)
    ttnn_slice_839 = ttnn.slice(
        ttnn_experimental_rotary_embedding_23,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_23, False)
    ttnn_to_memory_config_93 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_11 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_839,
            args_35,
            args_36,
            attn_mask=ttnn_to_memory_config_93,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_93, False)
    ttnn.deallocate(ttnn_slice_839, False)
    ttnn_transformer_concatenate_heads_11 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_11,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_11, False)
    ttnn_reshape_27 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_11,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_11, False)
    ttnn_matmul_57 = ttnn.matmul(
        ttnn_reshape_27,
        ce_cache__main["main_const_eval_36"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_27, False)
    ttnn_add_35 = ttnn.add(
        ttnn_matmul_57,
        ttnn_add_33,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_57, False)
    ttnn.deallocate(ttnn_add_33, False)
    ttnn_rms_norm_23 = ttnn.rms_norm(
        ttnn_add_35,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.11.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_58 = ttnn.matmul(
        ttnn_rms_norm_23,
        ce_cache__main["main_const_eval_54"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_59 = ttnn.matmul(
        ttnn_rms_norm_23,
        ce_cache__main["main_const_eval_14"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_23, False)
    ttnn_multiply_11 = ttnn.multiply(
        ttnn_matmul_58,
        ttnn_matmul_59,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_59, False)
    ttnn.deallocate(ttnn_matmul_58, False)
    ttnn_matmul_60 = ttnn.matmul(
        ttnn_multiply_11,
        ce_cache__main["main_const_eval_76"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_11, False)
    ttnn_add_36 = ttnn.add(
        ttnn_matmul_60,
        ttnn_add_35,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_60, False)
    ttnn.deallocate(ttnn_add_35, False)
    ttnn_rms_norm_24 = ttnn.rms_norm(
        ttnn_add_36,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.12.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_61 = ttnn.matmul(
        ttnn_rms_norm_24,
        ce_cache__main["main_const_eval_81"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_24, False)
    ttnn_reshape_28 = ttnn.reshape(
        ttnn_matmul_61,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_61, False)
    v_37, v_38, v_39 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_28,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_28, False)
    ttnn_to_memory_config_94 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_95 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_24 = ttnn.experimental.rotary_embedding(
        v_38,
        ttnn_to_memory_config_95,
        ttnn_to_memory_config_94,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_95, False)
    ttnn.deallocate(ttnn_to_memory_config_94, False)
    ttnn.deallocate(v_38, False)
    ttnn_slice_840 = ttnn.slice(
        ttnn_experimental_rotary_embedding_24,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_24, False)
    ttnn_slice_841 = ttnn.slice(
        ttnn_slice_840,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_841, 0)
    ttnn.deallocate(ttnn_slice_841, False)
    ttnn_slice_842 = ttnn.slice(
        ttnn_slice_840,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_842, 1)
    ttnn.deallocate(ttnn_slice_842, False)
    ttnn_slice_843 = ttnn.slice(
        ttnn_slice_840,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_843, 2)
    ttnn.deallocate(ttnn_slice_843, False)
    ttnn_slice_844 = ttnn.slice(
        ttnn_slice_840,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_844, 3)
    ttnn.deallocate(ttnn_slice_844, False)
    ttnn_slice_845 = ttnn.slice(
        ttnn_slice_840,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_845, 4)
    ttnn.deallocate(ttnn_slice_845, False)
    ttnn_slice_846 = ttnn.slice(
        ttnn_slice_840,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_846, 5)
    ttnn.deallocate(ttnn_slice_846, False)
    ttnn_slice_847 = ttnn.slice(
        ttnn_slice_840,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_847, 6)
    ttnn.deallocate(ttnn_slice_847, False)
    ttnn_slice_848 = ttnn.slice(
        ttnn_slice_840,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_848, 7)
    ttnn.deallocate(ttnn_slice_848, False)
    ttnn_slice_849 = ttnn.slice(
        ttnn_slice_840,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_849, 8)
    ttnn.deallocate(ttnn_slice_849, False)
    ttnn_slice_850 = ttnn.slice(
        ttnn_slice_840,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_850, 9)
    ttnn.deallocate(ttnn_slice_850, False)
    ttnn_slice_851 = ttnn.slice(
        ttnn_slice_840,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_851, 10)
    ttnn.deallocate(ttnn_slice_851, False)
    ttnn_slice_852 = ttnn.slice(
        ttnn_slice_840,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_852, 11)
    ttnn.deallocate(ttnn_slice_852, False)
    ttnn_slice_853 = ttnn.slice(
        ttnn_slice_840,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_853, 12)
    ttnn.deallocate(ttnn_slice_853, False)
    ttnn_slice_854 = ttnn.slice(
        ttnn_slice_840,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_854, 13)
    ttnn.deallocate(ttnn_slice_854, False)
    ttnn_slice_855 = ttnn.slice(
        ttnn_slice_840,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_855, 14)
    ttnn.deallocate(ttnn_slice_855, False)
    ttnn_slice_856 = ttnn.slice(
        ttnn_slice_840,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_856, 15)
    ttnn.deallocate(ttnn_slice_856, False)
    ttnn_slice_857 = ttnn.slice(
        ttnn_slice_840,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_857, 16)
    ttnn.deallocate(ttnn_slice_857, False)
    ttnn_slice_858 = ttnn.slice(
        ttnn_slice_840,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_858, 17)
    ttnn.deallocate(ttnn_slice_858, False)
    ttnn_slice_859 = ttnn.slice(
        ttnn_slice_840,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_859, 18)
    ttnn.deallocate(ttnn_slice_859, False)
    ttnn_slice_860 = ttnn.slice(
        ttnn_slice_840,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_860, 19)
    ttnn.deallocate(ttnn_slice_860, False)
    ttnn_slice_861 = ttnn.slice(
        ttnn_slice_840,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_861, 20)
    ttnn.deallocate(ttnn_slice_861, False)
    ttnn_slice_862 = ttnn.slice(
        ttnn_slice_840,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_862, 21)
    ttnn.deallocate(ttnn_slice_862, False)
    ttnn_slice_863 = ttnn.slice(
        ttnn_slice_840,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_863, 22)
    ttnn.deallocate(ttnn_slice_863, False)
    ttnn_slice_864 = ttnn.slice(
        ttnn_slice_840,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_864, 23)
    ttnn.deallocate(ttnn_slice_864, False)
    ttnn_slice_865 = ttnn.slice(
        ttnn_slice_840,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_865, 24)
    ttnn.deallocate(ttnn_slice_865, False)
    ttnn_slice_866 = ttnn.slice(
        ttnn_slice_840,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_866, 25)
    ttnn.deallocate(ttnn_slice_866, False)
    ttnn_slice_867 = ttnn.slice(
        ttnn_slice_840,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_867, 26)
    ttnn.deallocate(ttnn_slice_867, False)
    ttnn_slice_868 = ttnn.slice(
        ttnn_slice_840,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_868, 27)
    ttnn.deallocate(ttnn_slice_868, False)
    ttnn_slice_869 = ttnn.slice(
        ttnn_slice_840,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_869, 28)
    ttnn.deallocate(ttnn_slice_869, False)
    ttnn_slice_870 = ttnn.slice(
        ttnn_slice_840,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_870, 29)
    ttnn.deallocate(ttnn_slice_870, False)
    ttnn_slice_871 = ttnn.slice(
        ttnn_slice_840,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_38, ttnn_slice_871, 30)
    ttnn.deallocate(ttnn_slice_871, False)
    ttnn_slice_872 = ttnn.slice(
        ttnn_slice_840,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_840, False)
    ttnn.fill_cache(args_38, ttnn_slice_872, 31)
    ttnn.deallocate(ttnn_slice_872, False)
    ttnn_slice_873 = ttnn.slice(
        v_39,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_873, 0)
    ttnn.deallocate(ttnn_slice_873, False)
    ttnn_slice_874 = ttnn.slice(
        v_39,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_874, 1)
    ttnn.deallocate(ttnn_slice_874, False)
    ttnn_slice_875 = ttnn.slice(
        v_39,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_875, 2)
    ttnn.deallocate(ttnn_slice_875, False)
    ttnn_slice_876 = ttnn.slice(
        v_39,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_876, 3)
    ttnn.deallocate(ttnn_slice_876, False)
    ttnn_slice_877 = ttnn.slice(
        v_39,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_877, 4)
    ttnn.deallocate(ttnn_slice_877, False)
    ttnn_slice_878 = ttnn.slice(
        v_39,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_878, 5)
    ttnn.deallocate(ttnn_slice_878, False)
    ttnn_slice_879 = ttnn.slice(
        v_39,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_879, 6)
    ttnn.deallocate(ttnn_slice_879, False)
    ttnn_slice_880 = ttnn.slice(
        v_39,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_880, 7)
    ttnn.deallocate(ttnn_slice_880, False)
    ttnn_slice_881 = ttnn.slice(
        v_39,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_881, 8)
    ttnn.deallocate(ttnn_slice_881, False)
    ttnn_slice_882 = ttnn.slice(
        v_39,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_882, 9)
    ttnn.deallocate(ttnn_slice_882, False)
    ttnn_slice_883 = ttnn.slice(
        v_39,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_883, 10)
    ttnn.deallocate(ttnn_slice_883, False)
    ttnn_slice_884 = ttnn.slice(
        v_39,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_884, 11)
    ttnn.deallocate(ttnn_slice_884, False)
    ttnn_slice_885 = ttnn.slice(
        v_39,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_885, 12)
    ttnn.deallocate(ttnn_slice_885, False)
    ttnn_slice_886 = ttnn.slice(
        v_39,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_886, 13)
    ttnn.deallocate(ttnn_slice_886, False)
    ttnn_slice_887 = ttnn.slice(
        v_39,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_887, 14)
    ttnn.deallocate(ttnn_slice_887, False)
    ttnn_slice_888 = ttnn.slice(
        v_39,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_888, 15)
    ttnn.deallocate(ttnn_slice_888, False)
    ttnn_slice_889 = ttnn.slice(
        v_39,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_889, 16)
    ttnn.deallocate(ttnn_slice_889, False)
    ttnn_slice_890 = ttnn.slice(
        v_39,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_890, 17)
    ttnn.deallocate(ttnn_slice_890, False)
    ttnn_slice_891 = ttnn.slice(
        v_39,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_891, 18)
    ttnn.deallocate(ttnn_slice_891, False)
    ttnn_slice_892 = ttnn.slice(
        v_39,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_892, 19)
    ttnn.deallocate(ttnn_slice_892, False)
    ttnn_slice_893 = ttnn.slice(
        v_39,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_893, 20)
    ttnn.deallocate(ttnn_slice_893, False)
    ttnn_slice_894 = ttnn.slice(
        v_39,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_894, 21)
    ttnn.deallocate(ttnn_slice_894, False)
    ttnn_slice_895 = ttnn.slice(
        v_39,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_895, 22)
    ttnn.deallocate(ttnn_slice_895, False)
    ttnn_slice_896 = ttnn.slice(
        v_39,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_896, 23)
    ttnn.deallocate(ttnn_slice_896, False)
    ttnn_slice_897 = ttnn.slice(
        v_39,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_897, 24)
    ttnn.deallocate(ttnn_slice_897, False)
    ttnn_slice_898 = ttnn.slice(
        v_39,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_898, 25)
    ttnn.deallocate(ttnn_slice_898, False)
    ttnn_slice_899 = ttnn.slice(
        v_39,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_899, 26)
    ttnn.deallocate(ttnn_slice_899, False)
    ttnn_slice_900 = ttnn.slice(
        v_39,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_900, 27)
    ttnn.deallocate(ttnn_slice_900, False)
    ttnn_slice_901 = ttnn.slice(
        v_39,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_901, 28)
    ttnn.deallocate(ttnn_slice_901, False)
    ttnn_slice_902 = ttnn.slice(
        v_39,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_902, 29)
    ttnn.deallocate(ttnn_slice_902, False)
    ttnn_slice_903 = ttnn.slice(
        v_39,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_39, ttnn_slice_903, 30)
    ttnn.deallocate(ttnn_slice_903, False)
    ttnn_slice_904 = ttnn.slice(
        v_39,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_39, False)
    ttnn.fill_cache(args_39, ttnn_slice_904, 31)
    ttnn.deallocate(ttnn_slice_904, False)
    ttnn_to_memory_config_96 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_37 = ttnn.add(
        args_37,
        ttnn_to_memory_config_96,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_96, False)
    ttnn.deallocate(args_37, False)
    ttnn_to_memory_config_97 = ttnn.to_memory_config(
        ttnn_add_37,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_37, False)
    ttnn_to_memory_config_98 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_99 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_25 = ttnn.experimental.rotary_embedding(
        v_37,
        ttnn_to_memory_config_99,
        ttnn_to_memory_config_98,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_99, False)
    ttnn.deallocate(ttnn_to_memory_config_98, False)
    ttnn.deallocate(v_37, False)
    ttnn_slice_905 = ttnn.slice(
        ttnn_experimental_rotary_embedding_25,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_25, False)
    ttnn_to_memory_config_100 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_12 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_905,
            args_38,
            args_39,
            attn_mask=ttnn_to_memory_config_100,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_100, False)
    ttnn.deallocate(ttnn_slice_905, False)
    ttnn_transformer_concatenate_heads_12 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_12, False)
    ttnn_reshape_29 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_12,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_12, False)
    ttnn_matmul_62 = ttnn.matmul(
        ttnn_reshape_29,
        ce_cache__main["main_const_eval_56"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_29, False)
    ttnn_add_38 = ttnn.add(
        ttnn_matmul_62,
        ttnn_add_36,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_62, False)
    ttnn.deallocate(ttnn_add_36, False)
    ttnn_rms_norm_25 = ttnn.rms_norm(
        ttnn_add_38,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.12.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_63 = ttnn.matmul(
        ttnn_rms_norm_25,
        ce_cache__main["main_const_eval_9"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_64 = ttnn.matmul(
        ttnn_rms_norm_25,
        ce_cache__main["main_const_eval_43"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_25, False)
    ttnn_multiply_12 = ttnn.multiply(
        ttnn_matmul_63,
        ttnn_matmul_64,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_64, False)
    ttnn.deallocate(ttnn_matmul_63, False)
    ttnn_matmul_65 = ttnn.matmul(
        ttnn_multiply_12,
        ce_cache__main["main_const_eval_49"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_12, False)
    ttnn_add_39 = ttnn.add(
        ttnn_matmul_65,
        ttnn_add_38,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_65, False)
    ttnn.deallocate(ttnn_add_38, False)
    ttnn_rms_norm_26 = ttnn.rms_norm(
        ttnn_add_39,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.13.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_66 = ttnn.matmul(
        ttnn_rms_norm_26,
        ce_cache__main["main_const_eval_74"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_26, False)
    ttnn_reshape_30 = ttnn.reshape(
        ttnn_matmul_66,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_66, False)
    v_40, v_41, v_42 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_30,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_30, False)
    ttnn_to_memory_config_101 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_102 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_26 = ttnn.experimental.rotary_embedding(
        v_41,
        ttnn_to_memory_config_102,
        ttnn_to_memory_config_101,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_102, False)
    ttnn.deallocate(ttnn_to_memory_config_101, False)
    ttnn.deallocate(v_41, False)
    ttnn_slice_906 = ttnn.slice(
        ttnn_experimental_rotary_embedding_26,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_26, False)
    ttnn_slice_907 = ttnn.slice(
        ttnn_slice_906,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_907, 0)
    ttnn.deallocate(ttnn_slice_907, False)
    ttnn_slice_908 = ttnn.slice(
        ttnn_slice_906,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_908, 1)
    ttnn.deallocate(ttnn_slice_908, False)
    ttnn_slice_909 = ttnn.slice(
        ttnn_slice_906,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_909, 2)
    ttnn.deallocate(ttnn_slice_909, False)
    ttnn_slice_910 = ttnn.slice(
        ttnn_slice_906,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_910, 3)
    ttnn.deallocate(ttnn_slice_910, False)
    ttnn_slice_911 = ttnn.slice(
        ttnn_slice_906,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_911, 4)
    ttnn.deallocate(ttnn_slice_911, False)
    ttnn_slice_912 = ttnn.slice(
        ttnn_slice_906,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_912, 5)
    ttnn.deallocate(ttnn_slice_912, False)
    ttnn_slice_913 = ttnn.slice(
        ttnn_slice_906,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_913, 6)
    ttnn.deallocate(ttnn_slice_913, False)
    ttnn_slice_914 = ttnn.slice(
        ttnn_slice_906,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_914, 7)
    ttnn.deallocate(ttnn_slice_914, False)
    ttnn_slice_915 = ttnn.slice(
        ttnn_slice_906,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_915, 8)
    ttnn.deallocate(ttnn_slice_915, False)
    ttnn_slice_916 = ttnn.slice(
        ttnn_slice_906,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_916, 9)
    ttnn.deallocate(ttnn_slice_916, False)
    ttnn_slice_917 = ttnn.slice(
        ttnn_slice_906,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_917, 10)
    ttnn.deallocate(ttnn_slice_917, False)
    ttnn_slice_918 = ttnn.slice(
        ttnn_slice_906,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_918, 11)
    ttnn.deallocate(ttnn_slice_918, False)
    ttnn_slice_919 = ttnn.slice(
        ttnn_slice_906,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_919, 12)
    ttnn.deallocate(ttnn_slice_919, False)
    ttnn_slice_920 = ttnn.slice(
        ttnn_slice_906,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_920, 13)
    ttnn.deallocate(ttnn_slice_920, False)
    ttnn_slice_921 = ttnn.slice(
        ttnn_slice_906,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_921, 14)
    ttnn.deallocate(ttnn_slice_921, False)
    ttnn_slice_922 = ttnn.slice(
        ttnn_slice_906,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_922, 15)
    ttnn.deallocate(ttnn_slice_922, False)
    ttnn_slice_923 = ttnn.slice(
        ttnn_slice_906,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_923, 16)
    ttnn.deallocate(ttnn_slice_923, False)
    ttnn_slice_924 = ttnn.slice(
        ttnn_slice_906,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_924, 17)
    ttnn.deallocate(ttnn_slice_924, False)
    ttnn_slice_925 = ttnn.slice(
        ttnn_slice_906,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_925, 18)
    ttnn.deallocate(ttnn_slice_925, False)
    ttnn_slice_926 = ttnn.slice(
        ttnn_slice_906,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_926, 19)
    ttnn.deallocate(ttnn_slice_926, False)
    ttnn_slice_927 = ttnn.slice(
        ttnn_slice_906,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_927, 20)
    ttnn.deallocate(ttnn_slice_927, False)
    ttnn_slice_928 = ttnn.slice(
        ttnn_slice_906,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_928, 21)
    ttnn.deallocate(ttnn_slice_928, False)
    ttnn_slice_929 = ttnn.slice(
        ttnn_slice_906,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_929, 22)
    ttnn.deallocate(ttnn_slice_929, False)
    ttnn_slice_930 = ttnn.slice(
        ttnn_slice_906,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_930, 23)
    ttnn.deallocate(ttnn_slice_930, False)
    ttnn_slice_931 = ttnn.slice(
        ttnn_slice_906,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_931, 24)
    ttnn.deallocate(ttnn_slice_931, False)
    ttnn_slice_932 = ttnn.slice(
        ttnn_slice_906,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_932, 25)
    ttnn.deallocate(ttnn_slice_932, False)
    ttnn_slice_933 = ttnn.slice(
        ttnn_slice_906,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_933, 26)
    ttnn.deallocate(ttnn_slice_933, False)
    ttnn_slice_934 = ttnn.slice(
        ttnn_slice_906,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_934, 27)
    ttnn.deallocate(ttnn_slice_934, False)
    ttnn_slice_935 = ttnn.slice(
        ttnn_slice_906,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_935, 28)
    ttnn.deallocate(ttnn_slice_935, False)
    ttnn_slice_936 = ttnn.slice(
        ttnn_slice_906,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_936, 29)
    ttnn.deallocate(ttnn_slice_936, False)
    ttnn_slice_937 = ttnn.slice(
        ttnn_slice_906,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_41, ttnn_slice_937, 30)
    ttnn.deallocate(ttnn_slice_937, False)
    ttnn_slice_938 = ttnn.slice(
        ttnn_slice_906,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_906, False)
    ttnn.fill_cache(args_41, ttnn_slice_938, 31)
    ttnn.deallocate(ttnn_slice_938, False)
    ttnn_slice_939 = ttnn.slice(
        v_42,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_939, 0)
    ttnn.deallocate(ttnn_slice_939, False)
    ttnn_slice_940 = ttnn.slice(
        v_42,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_940, 1)
    ttnn.deallocate(ttnn_slice_940, False)
    ttnn_slice_941 = ttnn.slice(
        v_42,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_941, 2)
    ttnn.deallocate(ttnn_slice_941, False)
    ttnn_slice_942 = ttnn.slice(
        v_42,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_942, 3)
    ttnn.deallocate(ttnn_slice_942, False)
    ttnn_slice_943 = ttnn.slice(
        v_42,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_943, 4)
    ttnn.deallocate(ttnn_slice_943, False)
    ttnn_slice_944 = ttnn.slice(
        v_42,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_944, 5)
    ttnn.deallocate(ttnn_slice_944, False)
    ttnn_slice_945 = ttnn.slice(
        v_42,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_945, 6)
    ttnn.deallocate(ttnn_slice_945, False)
    ttnn_slice_946 = ttnn.slice(
        v_42,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_946, 7)
    ttnn.deallocate(ttnn_slice_946, False)
    ttnn_slice_947 = ttnn.slice(
        v_42,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_947, 8)
    ttnn.deallocate(ttnn_slice_947, False)
    ttnn_slice_948 = ttnn.slice(
        v_42,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_948, 9)
    ttnn.deallocate(ttnn_slice_948, False)
    ttnn_slice_949 = ttnn.slice(
        v_42,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_949, 10)
    ttnn.deallocate(ttnn_slice_949, False)
    ttnn_slice_950 = ttnn.slice(
        v_42,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_950, 11)
    ttnn.deallocate(ttnn_slice_950, False)
    ttnn_slice_951 = ttnn.slice(
        v_42,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_951, 12)
    ttnn.deallocate(ttnn_slice_951, False)
    ttnn_slice_952 = ttnn.slice(
        v_42,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_952, 13)
    ttnn.deallocate(ttnn_slice_952, False)
    ttnn_slice_953 = ttnn.slice(
        v_42,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_953, 14)
    ttnn.deallocate(ttnn_slice_953, False)
    ttnn_slice_954 = ttnn.slice(
        v_42,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_954, 15)
    ttnn.deallocate(ttnn_slice_954, False)
    ttnn_slice_955 = ttnn.slice(
        v_42,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_955, 16)
    ttnn.deallocate(ttnn_slice_955, False)
    ttnn_slice_956 = ttnn.slice(
        v_42,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_956, 17)
    ttnn.deallocate(ttnn_slice_956, False)
    ttnn_slice_957 = ttnn.slice(
        v_42,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_957, 18)
    ttnn.deallocate(ttnn_slice_957, False)
    ttnn_slice_958 = ttnn.slice(
        v_42,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_958, 19)
    ttnn.deallocate(ttnn_slice_958, False)
    ttnn_slice_959 = ttnn.slice(
        v_42,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_959, 20)
    ttnn.deallocate(ttnn_slice_959, False)
    ttnn_slice_960 = ttnn.slice(
        v_42,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_960, 21)
    ttnn.deallocate(ttnn_slice_960, False)
    ttnn_slice_961 = ttnn.slice(
        v_42,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_961, 22)
    ttnn.deallocate(ttnn_slice_961, False)
    ttnn_slice_962 = ttnn.slice(
        v_42,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_962, 23)
    ttnn.deallocate(ttnn_slice_962, False)
    ttnn_slice_963 = ttnn.slice(
        v_42,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_963, 24)
    ttnn.deallocate(ttnn_slice_963, False)
    ttnn_slice_964 = ttnn.slice(
        v_42,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_964, 25)
    ttnn.deallocate(ttnn_slice_964, False)
    ttnn_slice_965 = ttnn.slice(
        v_42,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_965, 26)
    ttnn.deallocate(ttnn_slice_965, False)
    ttnn_slice_966 = ttnn.slice(
        v_42,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_966, 27)
    ttnn.deallocate(ttnn_slice_966, False)
    ttnn_slice_967 = ttnn.slice(
        v_42,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_967, 28)
    ttnn.deallocate(ttnn_slice_967, False)
    ttnn_slice_968 = ttnn.slice(
        v_42,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_968, 29)
    ttnn.deallocate(ttnn_slice_968, False)
    ttnn_slice_969 = ttnn.slice(
        v_42,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_42, ttnn_slice_969, 30)
    ttnn.deallocate(ttnn_slice_969, False)
    ttnn_slice_970 = ttnn.slice(
        v_42,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_42, False)
    ttnn.fill_cache(args_42, ttnn_slice_970, 31)
    ttnn.deallocate(ttnn_slice_970, False)
    ttnn_to_memory_config_103 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_40 = ttnn.add(
        args_40,
        ttnn_to_memory_config_103,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_103, False)
    ttnn.deallocate(args_40, False)
    ttnn_to_memory_config_104 = ttnn.to_memory_config(
        ttnn_add_40,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_40, False)
    ttnn_to_memory_config_105 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_106 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_27 = ttnn.experimental.rotary_embedding(
        v_40,
        ttnn_to_memory_config_106,
        ttnn_to_memory_config_105,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_106, False)
    ttnn.deallocate(ttnn_to_memory_config_105, False)
    ttnn.deallocate(v_40, False)
    ttnn_slice_971 = ttnn.slice(
        ttnn_experimental_rotary_embedding_27,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_27, False)
    ttnn_to_memory_config_107 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_13 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_971,
            args_41,
            args_42,
            attn_mask=ttnn_to_memory_config_107,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_107, False)
    ttnn.deallocate(ttnn_slice_971, False)
    ttnn_transformer_concatenate_heads_13 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_13,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_13, False)
    ttnn_reshape_31 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_13,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_13, False)
    ttnn_matmul_67 = ttnn.matmul(
        ttnn_reshape_31,
        ce_cache__main["main_const_eval_87"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_31, False)
    ttnn_add_41 = ttnn.add(
        ttnn_matmul_67,
        ttnn_add_39,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_67, False)
    ttnn.deallocate(ttnn_add_39, False)
    ttnn_rms_norm_27 = ttnn.rms_norm(
        ttnn_add_41,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.13.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_68 = ttnn.matmul(
        ttnn_rms_norm_27,
        ce_cache__main["main_const_eval_75"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_69 = ttnn.matmul(
        ttnn_rms_norm_27,
        ce_cache__main["main_const_eval_73"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_27, False)
    ttnn_multiply_13 = ttnn.multiply(
        ttnn_matmul_68,
        ttnn_matmul_69,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_69, False)
    ttnn.deallocate(ttnn_matmul_68, False)
    ttnn_matmul_70 = ttnn.matmul(
        ttnn_multiply_13,
        ce_cache__main["main_const_eval_28"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_13, False)
    ttnn_add_42 = ttnn.add(
        ttnn_matmul_70,
        ttnn_add_41,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_70, False)
    ttnn.deallocate(ttnn_add_41, False)
    ttnn_rms_norm_28 = ttnn.rms_norm(
        ttnn_add_42,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.14.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_71 = ttnn.matmul(
        ttnn_rms_norm_28,
        ce_cache__main["main_const_eval_3"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_28, False)
    ttnn_reshape_32 = ttnn.reshape(
        ttnn_matmul_71,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_71, False)
    v_43, v_44, v_45 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_32,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_32, False)
    ttnn_to_memory_config_108 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_109 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_28 = ttnn.experimental.rotary_embedding(
        v_44,
        ttnn_to_memory_config_109,
        ttnn_to_memory_config_108,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_109, False)
    ttnn.deallocate(ttnn_to_memory_config_108, False)
    ttnn.deallocate(v_44, False)
    ttnn_slice_972 = ttnn.slice(
        ttnn_experimental_rotary_embedding_28,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_28, False)
    ttnn_slice_973 = ttnn.slice(
        ttnn_slice_972,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_973, 0)
    ttnn.deallocate(ttnn_slice_973, False)
    ttnn_slice_974 = ttnn.slice(
        ttnn_slice_972,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_974, 1)
    ttnn.deallocate(ttnn_slice_974, False)
    ttnn_slice_975 = ttnn.slice(
        ttnn_slice_972,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_975, 2)
    ttnn.deallocate(ttnn_slice_975, False)
    ttnn_slice_976 = ttnn.slice(
        ttnn_slice_972,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_976, 3)
    ttnn.deallocate(ttnn_slice_976, False)
    ttnn_slice_977 = ttnn.slice(
        ttnn_slice_972,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_977, 4)
    ttnn.deallocate(ttnn_slice_977, False)
    ttnn_slice_978 = ttnn.slice(
        ttnn_slice_972,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_978, 5)
    ttnn.deallocate(ttnn_slice_978, False)
    ttnn_slice_979 = ttnn.slice(
        ttnn_slice_972,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_979, 6)
    ttnn.deallocate(ttnn_slice_979, False)
    ttnn_slice_980 = ttnn.slice(
        ttnn_slice_972,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_980, 7)
    ttnn.deallocate(ttnn_slice_980, False)
    ttnn_slice_981 = ttnn.slice(
        ttnn_slice_972,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_981, 8)
    ttnn.deallocate(ttnn_slice_981, False)
    ttnn_slice_982 = ttnn.slice(
        ttnn_slice_972,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_982, 9)
    ttnn.deallocate(ttnn_slice_982, False)
    ttnn_slice_983 = ttnn.slice(
        ttnn_slice_972,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_983, 10)
    ttnn.deallocate(ttnn_slice_983, False)
    ttnn_slice_984 = ttnn.slice(
        ttnn_slice_972,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_984, 11)
    ttnn.deallocate(ttnn_slice_984, False)
    ttnn_slice_985 = ttnn.slice(
        ttnn_slice_972,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_985, 12)
    ttnn.deallocate(ttnn_slice_985, False)
    ttnn_slice_986 = ttnn.slice(
        ttnn_slice_972,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_986, 13)
    ttnn.deallocate(ttnn_slice_986, False)
    ttnn_slice_987 = ttnn.slice(
        ttnn_slice_972,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_987, 14)
    ttnn.deallocate(ttnn_slice_987, False)
    ttnn_slice_988 = ttnn.slice(
        ttnn_slice_972,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_988, 15)
    ttnn.deallocate(ttnn_slice_988, False)
    ttnn_slice_989 = ttnn.slice(
        ttnn_slice_972,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_989, 16)
    ttnn.deallocate(ttnn_slice_989, False)
    ttnn_slice_990 = ttnn.slice(
        ttnn_slice_972,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_990, 17)
    ttnn.deallocate(ttnn_slice_990, False)
    ttnn_slice_991 = ttnn.slice(
        ttnn_slice_972,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_991, 18)
    ttnn.deallocate(ttnn_slice_991, False)
    ttnn_slice_992 = ttnn.slice(
        ttnn_slice_972,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_992, 19)
    ttnn.deallocate(ttnn_slice_992, False)
    ttnn_slice_993 = ttnn.slice(
        ttnn_slice_972,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_993, 20)
    ttnn.deallocate(ttnn_slice_993, False)
    ttnn_slice_994 = ttnn.slice(
        ttnn_slice_972,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_994, 21)
    ttnn.deallocate(ttnn_slice_994, False)
    ttnn_slice_995 = ttnn.slice(
        ttnn_slice_972,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_995, 22)
    ttnn.deallocate(ttnn_slice_995, False)
    ttnn_slice_996 = ttnn.slice(
        ttnn_slice_972,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_996, 23)
    ttnn.deallocate(ttnn_slice_996, False)
    ttnn_slice_997 = ttnn.slice(
        ttnn_slice_972,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_997, 24)
    ttnn.deallocate(ttnn_slice_997, False)
    ttnn_slice_998 = ttnn.slice(
        ttnn_slice_972,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_998, 25)
    ttnn.deallocate(ttnn_slice_998, False)
    ttnn_slice_999 = ttnn.slice(
        ttnn_slice_972,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_999, 26)
    ttnn.deallocate(ttnn_slice_999, False)
    ttnn_slice_1000 = ttnn.slice(
        ttnn_slice_972,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_1000, 27)
    ttnn.deallocate(ttnn_slice_1000, False)
    ttnn_slice_1001 = ttnn.slice(
        ttnn_slice_972,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_1001, 28)
    ttnn.deallocate(ttnn_slice_1001, False)
    ttnn_slice_1002 = ttnn.slice(
        ttnn_slice_972,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_1002, 29)
    ttnn.deallocate(ttnn_slice_1002, False)
    ttnn_slice_1003 = ttnn.slice(
        ttnn_slice_972,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_44, ttnn_slice_1003, 30)
    ttnn.deallocate(ttnn_slice_1003, False)
    ttnn_slice_1004 = ttnn.slice(
        ttnn_slice_972,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_972, False)
    ttnn.fill_cache(args_44, ttnn_slice_1004, 31)
    ttnn.deallocate(ttnn_slice_1004, False)
    ttnn_slice_1005 = ttnn.slice(
        v_45,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1005, 0)
    ttnn.deallocate(ttnn_slice_1005, False)
    ttnn_slice_1006 = ttnn.slice(
        v_45,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1006, 1)
    ttnn.deallocate(ttnn_slice_1006, False)
    ttnn_slice_1007 = ttnn.slice(
        v_45,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1007, 2)
    ttnn.deallocate(ttnn_slice_1007, False)
    ttnn_slice_1008 = ttnn.slice(
        v_45,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1008, 3)
    ttnn.deallocate(ttnn_slice_1008, False)
    ttnn_slice_1009 = ttnn.slice(
        v_45,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1009, 4)
    ttnn.deallocate(ttnn_slice_1009, False)
    ttnn_slice_1010 = ttnn.slice(
        v_45,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1010, 5)
    ttnn.deallocate(ttnn_slice_1010, False)
    ttnn_slice_1011 = ttnn.slice(
        v_45,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1011, 6)
    ttnn.deallocate(ttnn_slice_1011, False)
    ttnn_slice_1012 = ttnn.slice(
        v_45,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1012, 7)
    ttnn.deallocate(ttnn_slice_1012, False)
    ttnn_slice_1013 = ttnn.slice(
        v_45,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1013, 8)
    ttnn.deallocate(ttnn_slice_1013, False)
    ttnn_slice_1014 = ttnn.slice(
        v_45,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1014, 9)
    ttnn.deallocate(ttnn_slice_1014, False)
    ttnn_slice_1015 = ttnn.slice(
        v_45,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1015, 10)
    ttnn.deallocate(ttnn_slice_1015, False)
    ttnn_slice_1016 = ttnn.slice(
        v_45,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1016, 11)
    ttnn.deallocate(ttnn_slice_1016, False)
    ttnn_slice_1017 = ttnn.slice(
        v_45,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1017, 12)
    ttnn.deallocate(ttnn_slice_1017, False)
    ttnn_slice_1018 = ttnn.slice(
        v_45,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1018, 13)
    ttnn.deallocate(ttnn_slice_1018, False)
    ttnn_slice_1019 = ttnn.slice(
        v_45,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1019, 14)
    ttnn.deallocate(ttnn_slice_1019, False)
    ttnn_slice_1020 = ttnn.slice(
        v_45,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1020, 15)
    ttnn.deallocate(ttnn_slice_1020, False)
    ttnn_slice_1021 = ttnn.slice(
        v_45,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1021, 16)
    ttnn.deallocate(ttnn_slice_1021, False)
    ttnn_slice_1022 = ttnn.slice(
        v_45,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1022, 17)
    ttnn.deallocate(ttnn_slice_1022, False)
    ttnn_slice_1023 = ttnn.slice(
        v_45,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1023, 18)
    ttnn.deallocate(ttnn_slice_1023, False)
    ttnn_slice_1024 = ttnn.slice(
        v_45,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1024, 19)
    ttnn.deallocate(ttnn_slice_1024, False)
    ttnn_slice_1025 = ttnn.slice(
        v_45,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1025, 20)
    ttnn.deallocate(ttnn_slice_1025, False)
    ttnn_slice_1026 = ttnn.slice(
        v_45,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1026, 21)
    ttnn.deallocate(ttnn_slice_1026, False)
    ttnn_slice_1027 = ttnn.slice(
        v_45,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1027, 22)
    ttnn.deallocate(ttnn_slice_1027, False)
    ttnn_slice_1028 = ttnn.slice(
        v_45,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1028, 23)
    ttnn.deallocate(ttnn_slice_1028, False)
    ttnn_slice_1029 = ttnn.slice(
        v_45,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1029, 24)
    ttnn.deallocate(ttnn_slice_1029, False)
    ttnn_slice_1030 = ttnn.slice(
        v_45,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1030, 25)
    ttnn.deallocate(ttnn_slice_1030, False)
    ttnn_slice_1031 = ttnn.slice(
        v_45,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1031, 26)
    ttnn.deallocate(ttnn_slice_1031, False)
    ttnn_slice_1032 = ttnn.slice(
        v_45,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1032, 27)
    ttnn.deallocate(ttnn_slice_1032, False)
    ttnn_slice_1033 = ttnn.slice(
        v_45,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1033, 28)
    ttnn.deallocate(ttnn_slice_1033, False)
    ttnn_slice_1034 = ttnn.slice(
        v_45,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1034, 29)
    ttnn.deallocate(ttnn_slice_1034, False)
    ttnn_slice_1035 = ttnn.slice(
        v_45,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_45, ttnn_slice_1035, 30)
    ttnn.deallocate(ttnn_slice_1035, False)
    ttnn_slice_1036 = ttnn.slice(
        v_45,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_45, False)
    ttnn.fill_cache(args_45, ttnn_slice_1036, 31)
    ttnn.deallocate(ttnn_slice_1036, False)
    ttnn_to_memory_config_110 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_43 = ttnn.add(
        args_43,
        ttnn_to_memory_config_110,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_110, False)
    ttnn.deallocate(args_43, False)
    ttnn_to_memory_config_111 = ttnn.to_memory_config(
        ttnn_add_43,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_43, False)
    ttnn_to_memory_config_112 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_113 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_29 = ttnn.experimental.rotary_embedding(
        v_43,
        ttnn_to_memory_config_113,
        ttnn_to_memory_config_112,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_113, False)
    ttnn.deallocate(ttnn_to_memory_config_112, False)
    ttnn.deallocate(v_43, False)
    ttnn_slice_1037 = ttnn.slice(
        ttnn_experimental_rotary_embedding_29,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_29, False)
    ttnn_to_memory_config_114 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_transformer_scaled_dot_product_attention_14 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_1037,
            args_44,
            args_45,
            attn_mask=ttnn_to_memory_config_114,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_114, False)
    ttnn.deallocate(ttnn_slice_1037, False)
    ttnn_transformer_concatenate_heads_14 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_14,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_14, False)
    ttnn_reshape_33 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_14,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_14, False)
    ttnn_matmul_72 = ttnn.matmul(
        ttnn_reshape_33,
        ce_cache__main["main_const_eval_23"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_33, False)
    ttnn_add_44 = ttnn.add(
        ttnn_matmul_72,
        ttnn_add_42,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_72, False)
    ttnn.deallocate(ttnn_add_42, False)
    ttnn_rms_norm_29 = ttnn.rms_norm(
        ttnn_add_44,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.14.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_73 = ttnn.matmul(
        ttnn_rms_norm_29,
        ce_cache__main["main_const_eval_44"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_74 = ttnn.matmul(
        ttnn_rms_norm_29,
        ce_cache__main["main_const_eval_67"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_29, False)
    ttnn_multiply_14 = ttnn.multiply(
        ttnn_matmul_73,
        ttnn_matmul_74,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_74, False)
    ttnn.deallocate(ttnn_matmul_73, False)
    ttnn_matmul_75 = ttnn.matmul(
        ttnn_multiply_14,
        ce_cache__main["main_const_eval_2"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_14, False)
    ttnn_add_45 = ttnn.add(
        ttnn_matmul_75,
        ttnn_add_44,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_75, False)
    ttnn.deallocate(ttnn_add_44, False)
    ttnn_rms_norm_30 = ttnn.rms_norm(
        ttnn_add_45,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.15.input_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_76 = ttnn.matmul(
        ttnn_rms_norm_30,
        ce_cache__main["main_const_eval_24"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 288],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=2,
            out_block_w=9,
            per_core_M=2,
            per_core_N=9,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_30, False)
    ttnn_reshape_34 = ttnn.reshape(
        ttnn_matmul_76,
        [32, 18, 3072],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_76, False)
    v_46, v_47, v_48 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_34,
        None,
        num_heads=32,
        num_kv_heads=8,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_34, False)
    ttnn_to_memory_config_115 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_to_memory_config_116 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn_experimental_rotary_embedding_30 = ttnn.experimental.rotary_embedding(
        v_47,
        ttnn_to_memory_config_116,
        ttnn_to_memory_config_115,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_116, False)
    ttnn.deallocate(ttnn_to_memory_config_115, False)
    ttnn.deallocate(v_47, False)
    ttnn_slice_1038 = ttnn.slice(
        ttnn_experimental_rotary_embedding_30,
        [0, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_30, False)
    ttnn_slice_1039 = ttnn.slice(
        ttnn_slice_1038,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1039, 0)
    ttnn.deallocate(ttnn_slice_1039, False)
    ttnn_slice_1040 = ttnn.slice(
        ttnn_slice_1038,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1040, 1)
    ttnn.deallocate(ttnn_slice_1040, False)
    ttnn_slice_1041 = ttnn.slice(
        ttnn_slice_1038,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1041, 2)
    ttnn.deallocate(ttnn_slice_1041, False)
    ttnn_slice_1042 = ttnn.slice(
        ttnn_slice_1038,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1042, 3)
    ttnn.deallocate(ttnn_slice_1042, False)
    ttnn_slice_1043 = ttnn.slice(
        ttnn_slice_1038,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1043, 4)
    ttnn.deallocate(ttnn_slice_1043, False)
    ttnn_slice_1044 = ttnn.slice(
        ttnn_slice_1038,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1044, 5)
    ttnn.deallocate(ttnn_slice_1044, False)
    ttnn_slice_1045 = ttnn.slice(
        ttnn_slice_1038,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1045, 6)
    ttnn.deallocate(ttnn_slice_1045, False)
    ttnn_slice_1046 = ttnn.slice(
        ttnn_slice_1038,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1046, 7)
    ttnn.deallocate(ttnn_slice_1046, False)
    ttnn_slice_1047 = ttnn.slice(
        ttnn_slice_1038,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1047, 8)
    ttnn.deallocate(ttnn_slice_1047, False)
    ttnn_slice_1048 = ttnn.slice(
        ttnn_slice_1038,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1048, 9)
    ttnn.deallocate(ttnn_slice_1048, False)
    ttnn_slice_1049 = ttnn.slice(
        ttnn_slice_1038,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1049, 10)
    ttnn.deallocate(ttnn_slice_1049, False)
    ttnn_slice_1050 = ttnn.slice(
        ttnn_slice_1038,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1050, 11)
    ttnn.deallocate(ttnn_slice_1050, False)
    ttnn_slice_1051 = ttnn.slice(
        ttnn_slice_1038,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1051, 12)
    ttnn.deallocate(ttnn_slice_1051, False)
    ttnn_slice_1052 = ttnn.slice(
        ttnn_slice_1038,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1052, 13)
    ttnn.deallocate(ttnn_slice_1052, False)
    ttnn_slice_1053 = ttnn.slice(
        ttnn_slice_1038,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1053, 14)
    ttnn.deallocate(ttnn_slice_1053, False)
    ttnn_slice_1054 = ttnn.slice(
        ttnn_slice_1038,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1054, 15)
    ttnn.deallocate(ttnn_slice_1054, False)
    ttnn_slice_1055 = ttnn.slice(
        ttnn_slice_1038,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1055, 16)
    ttnn.deallocate(ttnn_slice_1055, False)
    ttnn_slice_1056 = ttnn.slice(
        ttnn_slice_1038,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1056, 17)
    ttnn.deallocate(ttnn_slice_1056, False)
    ttnn_slice_1057 = ttnn.slice(
        ttnn_slice_1038,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1057, 18)
    ttnn.deallocate(ttnn_slice_1057, False)
    ttnn_slice_1058 = ttnn.slice(
        ttnn_slice_1038,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1058, 19)
    ttnn.deallocate(ttnn_slice_1058, False)
    ttnn_slice_1059 = ttnn.slice(
        ttnn_slice_1038,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1059, 20)
    ttnn.deallocate(ttnn_slice_1059, False)
    ttnn_slice_1060 = ttnn.slice(
        ttnn_slice_1038,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1060, 21)
    ttnn.deallocate(ttnn_slice_1060, False)
    ttnn_slice_1061 = ttnn.slice(
        ttnn_slice_1038,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1061, 22)
    ttnn.deallocate(ttnn_slice_1061, False)
    ttnn_slice_1062 = ttnn.slice(
        ttnn_slice_1038,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1062, 23)
    ttnn.deallocate(ttnn_slice_1062, False)
    ttnn_slice_1063 = ttnn.slice(
        ttnn_slice_1038,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1063, 24)
    ttnn.deallocate(ttnn_slice_1063, False)
    ttnn_slice_1064 = ttnn.slice(
        ttnn_slice_1038,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1064, 25)
    ttnn.deallocate(ttnn_slice_1064, False)
    ttnn_slice_1065 = ttnn.slice(
        ttnn_slice_1038,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1065, 26)
    ttnn.deallocate(ttnn_slice_1065, False)
    ttnn_slice_1066 = ttnn.slice(
        ttnn_slice_1038,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1066, 27)
    ttnn.deallocate(ttnn_slice_1066, False)
    ttnn_slice_1067 = ttnn.slice(
        ttnn_slice_1038,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1067, 28)
    ttnn.deallocate(ttnn_slice_1067, False)
    ttnn_slice_1068 = ttnn.slice(
        ttnn_slice_1038,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1068, 29)
    ttnn.deallocate(ttnn_slice_1068, False)
    ttnn_slice_1069 = ttnn.slice(
        ttnn_slice_1038,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_47, ttnn_slice_1069, 30)
    ttnn.deallocate(ttnn_slice_1069, False)
    ttnn_slice_1070 = ttnn.slice(
        ttnn_slice_1038,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_slice_1038, False)
    ttnn.fill_cache(args_47, ttnn_slice_1070, 31)
    ttnn.deallocate(ttnn_slice_1070, False)
    ttnn_slice_1071 = ttnn.slice(
        v_48,
        [0, 0, 0, 0],
        [1, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1071, 0)
    ttnn.deallocate(ttnn_slice_1071, False)
    ttnn_slice_1072 = ttnn.slice(
        v_48,
        [1, 0, 0, 0],
        [2, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1072, 1)
    ttnn.deallocate(ttnn_slice_1072, False)
    ttnn_slice_1073 = ttnn.slice(
        v_48,
        [2, 0, 0, 0],
        [3, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1073, 2)
    ttnn.deallocate(ttnn_slice_1073, False)
    ttnn_slice_1074 = ttnn.slice(
        v_48,
        [3, 0, 0, 0],
        [4, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1074, 3)
    ttnn.deallocate(ttnn_slice_1074, False)
    ttnn_slice_1075 = ttnn.slice(
        v_48,
        [4, 0, 0, 0],
        [5, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1075, 4)
    ttnn.deallocate(ttnn_slice_1075, False)
    ttnn_slice_1076 = ttnn.slice(
        v_48,
        [5, 0, 0, 0],
        [6, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1076, 5)
    ttnn.deallocate(ttnn_slice_1076, False)
    ttnn_slice_1077 = ttnn.slice(
        v_48,
        [6, 0, 0, 0],
        [7, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1077, 6)
    ttnn.deallocate(ttnn_slice_1077, False)
    ttnn_slice_1078 = ttnn.slice(
        v_48,
        [7, 0, 0, 0],
        [8, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1078, 7)
    ttnn.deallocate(ttnn_slice_1078, False)
    ttnn_slice_1079 = ttnn.slice(
        v_48,
        [8, 0, 0, 0],
        [9, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1079, 8)
    ttnn.deallocate(ttnn_slice_1079, False)
    ttnn_slice_1080 = ttnn.slice(
        v_48,
        [9, 0, 0, 0],
        [10, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1080, 9)
    ttnn.deallocate(ttnn_slice_1080, False)
    ttnn_slice_1081 = ttnn.slice(
        v_48,
        [10, 0, 0, 0],
        [11, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1081, 10)
    ttnn.deallocate(ttnn_slice_1081, False)
    ttnn_slice_1082 = ttnn.slice(
        v_48,
        [11, 0, 0, 0],
        [12, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1082, 11)
    ttnn.deallocate(ttnn_slice_1082, False)
    ttnn_slice_1083 = ttnn.slice(
        v_48,
        [12, 0, 0, 0],
        [13, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1083, 12)
    ttnn.deallocate(ttnn_slice_1083, False)
    ttnn_slice_1084 = ttnn.slice(
        v_48,
        [13, 0, 0, 0],
        [14, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1084, 13)
    ttnn.deallocate(ttnn_slice_1084, False)
    ttnn_slice_1085 = ttnn.slice(
        v_48,
        [14, 0, 0, 0],
        [15, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1085, 14)
    ttnn.deallocate(ttnn_slice_1085, False)
    ttnn_slice_1086 = ttnn.slice(
        v_48,
        [15, 0, 0, 0],
        [16, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1086, 15)
    ttnn.deallocate(ttnn_slice_1086, False)
    ttnn_slice_1087 = ttnn.slice(
        v_48,
        [16, 0, 0, 0],
        [17, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1087, 16)
    ttnn.deallocate(ttnn_slice_1087, False)
    ttnn_slice_1088 = ttnn.slice(
        v_48,
        [17, 0, 0, 0],
        [18, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1088, 17)
    ttnn.deallocate(ttnn_slice_1088, False)
    ttnn_slice_1089 = ttnn.slice(
        v_48,
        [18, 0, 0, 0],
        [19, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1089, 18)
    ttnn.deallocate(ttnn_slice_1089, False)
    ttnn_slice_1090 = ttnn.slice(
        v_48,
        [19, 0, 0, 0],
        [20, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1090, 19)
    ttnn.deallocate(ttnn_slice_1090, False)
    ttnn_slice_1091 = ttnn.slice(
        v_48,
        [20, 0, 0, 0],
        [21, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1091, 20)
    ttnn.deallocate(ttnn_slice_1091, False)
    ttnn_slice_1092 = ttnn.slice(
        v_48,
        [21, 0, 0, 0],
        [22, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1092, 21)
    ttnn.deallocate(ttnn_slice_1092, False)
    ttnn_slice_1093 = ttnn.slice(
        v_48,
        [22, 0, 0, 0],
        [23, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1093, 22)
    ttnn.deallocate(ttnn_slice_1093, False)
    ttnn_slice_1094 = ttnn.slice(
        v_48,
        [23, 0, 0, 0],
        [24, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1094, 23)
    ttnn.deallocate(ttnn_slice_1094, False)
    ttnn_slice_1095 = ttnn.slice(
        v_48,
        [24, 0, 0, 0],
        [25, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1095, 24)
    ttnn.deallocate(ttnn_slice_1095, False)
    ttnn_slice_1096 = ttnn.slice(
        v_48,
        [25, 0, 0, 0],
        [26, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1096, 25)
    ttnn.deallocate(ttnn_slice_1096, False)
    ttnn_slice_1097 = ttnn.slice(
        v_48,
        [26, 0, 0, 0],
        [27, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1097, 26)
    ttnn.deallocate(ttnn_slice_1097, False)
    ttnn_slice_1098 = ttnn.slice(
        v_48,
        [27, 0, 0, 0],
        [28, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1098, 27)
    ttnn.deallocate(ttnn_slice_1098, False)
    ttnn_slice_1099 = ttnn.slice(
        v_48,
        [28, 0, 0, 0],
        [29, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1099, 28)
    ttnn.deallocate(ttnn_slice_1099, False)
    ttnn_slice_1100 = ttnn.slice(
        v_48,
        [29, 0, 0, 0],
        [30, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1100, 29)
    ttnn.deallocate(ttnn_slice_1100, False)
    ttnn_slice_1101 = ttnn.slice(
        v_48,
        [30, 0, 0, 0],
        [31, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.fill_cache(args_48, ttnn_slice_1101, 30)
    ttnn.deallocate(ttnn_slice_1101, False)
    ttnn_slice_1102 = ttnn.slice(
        v_48,
        [31, 0, 0, 0],
        [32, 8, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(v_48, False)
    ttnn.fill_cache(args_48, ttnn_slice_1102, 31)
    ttnn.deallocate(ttnn_slice_1102, False)
    ttnn_to_memory_config_117 = ttnn.to_memory_config(
        var_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_46 = ttnn.add(
        args_46,
        ttnn_to_memory_config_117,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_117, False)
    ttnn.deallocate(args_46, False)
    ttnn_to_memory_config_118 = ttnn.to_memory_config(
        ttnn_add_46,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_46, False)
    ttnn_to_memory_config_119 = ttnn.to_memory_config(
        ttnn_typecast_149,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_149, False)
    ttnn_to_memory_config_120 = ttnn.to_memory_config(
        ttnn_typecast_148,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_148, False)
    ttnn_experimental_rotary_embedding_31 = ttnn.experimental.rotary_embedding(
        v_46,
        ttnn_to_memory_config_120,
        ttnn_to_memory_config_119,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_120, False)
    ttnn.deallocate(ttnn_to_memory_config_119, False)
    ttnn.deallocate(v_46, False)
    ttnn_slice_1103 = ttnn.slice(
        ttnn_experimental_rotary_embedding_31,
        [0, 0, 0, 0],
        [32, 32, 18, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_31, False)
    ttnn_to_memory_config_121 = ttnn.to_memory_config(
        ttnn_where_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_where_0, False)
    ttnn_transformer_scaled_dot_product_attention_15 = (
        ttnn.transformer.scaled_dot_product_attention(
            ttnn_slice_1103,
            args_47,
            args_48,
            attn_mask=ttnn_to_memory_config_121,
            is_causal=False,
            scale=0.1249999925494194,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_to_memory_config_121, False)
    ttnn.deallocate(ttnn_slice_1103, False)
    ttnn_transformer_concatenate_heads_15 = ttnn.transformer.concatenate_heads(
        ttnn_transformer_scaled_dot_product_attention_15,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_15, False)
    ttnn_reshape_35 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_15,
        [576, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_15, False)
    ttnn_matmul_77 = ttnn.matmul(
        ttnn_reshape_35,
        ce_cache__main["main_const_eval_39"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_35, False)
    ttnn_add_47 = ttnn.add(
        ttnn_matmul_77,
        ttnn_add_45,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_77, False)
    ttnn.deallocate(ttnn_add_45, False)
    ttnn_rms_norm_31 = ttnn.rms_norm(
        ttnn_add_47,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.layers.15.post_attention_layernorm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_matmul_78 = ttnn.matmul(
        ttnn_rms_norm_31,
        ce_cache__main["main_const_eval_32"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_matmul_79 = ttnn.matmul(
        ttnn_rms_norm_31,
        ce_cache__main["main_const_eval_46"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=2,
            out_block_w=24,
            per_core_M=2,
            per_core_N=24,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_31, False)
    ttnn_multiply_15 = ttnn.multiply(
        ttnn_matmul_78,
        ttnn_matmul_79,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 768],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_79, False)
    ttnn.deallocate(ttnn_matmul_78, False)
    ttnn_matmul_80 = ttnn.matmul(
        ttnn_multiply_15,
        ce_cache__main["main_const_eval_79"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=6,
            out_block_h=2,
            out_block_w=6,
            per_core_M=2,
            per_core_N=6,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_15, False)
    ttnn_add_48 = ttnn.add(
        ttnn_matmul_80,
        ttnn_add_47,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_matmul_80, False)
    ttnn.deallocate(ttnn_add_47, False)
    ttnn_rms_norm_32 = ttnn.rms_norm(
        ttnn_add_48,
        epsilon=9.9999997473787516e-06,
        weight=weights["model.norm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn.deallocate(ttnn_add_48, False)
    ttnn_matmul_81 = ttnn.matmul(
        ttnn_rms_norm_32,
        ce_cache__main["main_const_eval_82"],
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_rms_norm_32, False)
    ttnn_reshape_36 = ttnn.reshape(
        ttnn_matmul_81,
        [32, 18, 128256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [
        args_2,
        args_3,
        ttnn_to_memory_config_9,
        args_5,
        args_6,
        ttnn_to_memory_config_20,
        args_8,
        args_9,
        ttnn_to_memory_config_27,
        args_11,
        args_12,
        ttnn_to_memory_config_34,
        args_14,
        args_15,
        ttnn_to_memory_config_41,
        args_17,
        args_18,
        ttnn_to_memory_config_48,
        args_20,
        args_21,
        ttnn_to_memory_config_55,
        args_23,
        args_24,
        ttnn_to_memory_config_62,
        args_26,
        args_27,
        ttnn_to_memory_config_69,
        args_29,
        args_30,
        ttnn_to_memory_config_76,
        args_32,
        args_33,
        ttnn_to_memory_config_83,
        args_35,
        args_36,
        ttnn_to_memory_config_90,
        args_38,
        args_39,
        ttnn_to_memory_config_97,
        args_41,
        args_42,
        ttnn_to_memory_config_104,
        args_44,
        args_45,
        ttnn_to_memory_config_111,
        args_47,
        args_48,
        ttnn_to_memory_config_118,
        ttnn_matmul_81,
        ttnn_reshape_36,
    ]


def consteval__main(ce_cache, weights, device):
    if not ce_cache:
        main_const_eval_0_0 = main_const_eval_0(device)
        ce_cache["main_const_eval_0"] = main_const_eval_0_0[0]
        main_const_eval_1_0 = main_const_eval_1(
            [weights["model.layers.3.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_1"] = main_const_eval_1_0[0]
        main_const_eval_2_0 = main_const_eval_2(
            [weights["model.layers.14.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_2"] = main_const_eval_2_0[0]
        main_const_eval_3_0 = main_const_eval_3(
            [
                weights["model.layers.14.self_attn.k_proj.weight"],
                weights["model.layers.14.self_attn.v_proj.weight"],
                weights["model.layers.14.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_3"] = main_const_eval_3_0[0]
        main_const_eval_4_0 = main_const_eval_4(
            [
                weights["model.layers.4.self_attn.k_proj.weight"],
                weights["model.layers.4.self_attn.v_proj.weight"],
                weights["model.layers.4.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_4"] = main_const_eval_4_0[0]
        main_const_eval_5_0 = main_const_eval_5(
            [weights["model.layers.7.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_5"] = main_const_eval_5_0[0]
        main_const_eval_6_0 = main_const_eval_6(
            [weights["model.layers.2.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_6"] = main_const_eval_6_0[0]
        main_const_eval_7_0 = main_const_eval_7(
            [weights["model.layers.4.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_7"] = main_const_eval_7_0[0]
        main_const_eval_8_0 = main_const_eval_8(device)
        ce_cache["main_const_eval_8"] = main_const_eval_8_0[0]
        main_const_eval_9_0 = main_const_eval_9(
            [weights["model.layers.12.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_9"] = main_const_eval_9_0[0]
        main_const_eval_10_0 = main_const_eval_10(
            [weights["model.layers.9.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_10"] = main_const_eval_10_0[0]
        main_const_eval_11_0 = main_const_eval_11(
            [weights["model.layers.6.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_11"] = main_const_eval_11_0[0]
        main_const_eval_12_0 = main_const_eval_12(
            [weights["model.layers.6.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_12"] = main_const_eval_12_0[0]
        main_const_eval_13_0 = main_const_eval_13(
            [weights["model.layers.0.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_13"] = main_const_eval_13_0[0]
        main_const_eval_14_0 = main_const_eval_14(
            [weights["model.layers.11.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_14"] = main_const_eval_14_0[0]
        main_const_eval_15_0 = main_const_eval_15(
            [
                weights["model.layers.1.self_attn.k_proj.weight"],
                weights["model.layers.1.self_attn.v_proj.weight"],
                weights["model.layers.1.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_15"] = main_const_eval_15_0[0]
        main_const_eval_16_0 = main_const_eval_16(
            [
                weights["model.layers.7.self_attn.k_proj.weight"],
                weights["model.layers.7.self_attn.v_proj.weight"],
                weights["model.layers.7.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_16"] = main_const_eval_16_0[0]
        main_const_eval_17_0 = main_const_eval_17(
            [
                weights["model.layers.11.self_attn.k_proj.weight"],
                weights["model.layers.11.self_attn.v_proj.weight"],
                weights["model.layers.11.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_17"] = main_const_eval_17_0[0]
        main_const_eval_18_0 = main_const_eval_18(
            [weights["model.layers.0.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_18"] = main_const_eval_18_0[0]
        main_const_eval_19_0 = main_const_eval_19(
            [weights["model.layers.10.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_19"] = main_const_eval_19_0[0]
        main_const_eval_20_0 = main_const_eval_20(
            [weights["model.layers.1.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_20"] = main_const_eval_20_0[0]
        main_const_eval_21_0 = main_const_eval_21(
            [weights["model.layers.5.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_21"] = main_const_eval_21_0[0]
        main_const_eval_22_0 = main_const_eval_22(
            [weights["model.layers.8.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_22"] = main_const_eval_22_0[0]
        main_const_eval_23_0 = main_const_eval_23(
            [weights["model.layers.14.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_23"] = main_const_eval_23_0[0]
        main_const_eval_24_0 = main_const_eval_24(
            [
                weights["model.layers.15.self_attn.k_proj.weight"],
                weights["model.layers.15.self_attn.v_proj.weight"],
                weights["model.layers.15.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_24"] = main_const_eval_24_0[0]
        main_const_eval_25_0 = main_const_eval_25(
            [weights["model.rotary_emb.inv_freq"]], device
        )
        ce_cache["main_const_eval_25"] = main_const_eval_25_0[0]
        main_const_eval_26_0 = main_const_eval_26(
            [
                weights["model.layers.3.self_attn.k_proj.weight"],
                weights["model.layers.3.self_attn.v_proj.weight"],
                weights["model.layers.3.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_26"] = main_const_eval_26_0[0]
        main_const_eval_27_0 = main_const_eval_27(
            [
                weights["model.layers.10.self_attn.k_proj.weight"],
                weights["model.layers.10.self_attn.v_proj.weight"],
                weights["model.layers.10.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_27"] = main_const_eval_27_0[0]
        main_const_eval_28_0 = main_const_eval_28(
            [weights["model.layers.13.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_28"] = main_const_eval_28_0[0]
        main_const_eval_29_0 = main_const_eval_29(
            [weights["model.layers.8.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_29"] = main_const_eval_29_0[0]
        main_const_eval_30_0 = main_const_eval_30(
            [weights["model.layers.0.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_30"] = main_const_eval_30_0[0]
        main_const_eval_31_0 = main_const_eval_31(
            [weights["model.layers.3.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_31"] = main_const_eval_31_0[0]
        main_const_eval_32_0 = main_const_eval_32(
            [weights["model.layers.15.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_32"] = main_const_eval_32_0[0]
        main_const_eval_33_0 = main_const_eval_33(
            [weights["model.layers.1.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_33"] = main_const_eval_33_0[0]
        main_const_eval_34_0 = main_const_eval_34(
            [
                weights["model.layers.2.self_attn.k_proj.weight"],
                weights["model.layers.2.self_attn.v_proj.weight"],
                weights["model.layers.2.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_34"] = main_const_eval_34_0[0]
        main_const_eval_35_0 = main_const_eval_35(
            [weights["model.layers.5.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_35"] = main_const_eval_35_0[0]
        main_const_eval_36_0 = main_const_eval_36(
            [weights["model.layers.11.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_36"] = main_const_eval_36_0[0]
        main_const_eval_37_0 = main_const_eval_37(
            [
                weights["model.layers.6.self_attn.k_proj.weight"],
                weights["model.layers.6.self_attn.v_proj.weight"],
                weights["model.layers.6.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_37"] = main_const_eval_37_0[0]
        main_const_eval_38_0 = main_const_eval_38(
            [weights["model.layers.7.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_38"] = main_const_eval_38_0[0]
        main_const_eval_39_0 = main_const_eval_39(
            [weights["model.layers.15.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_39"] = main_const_eval_39_0[0]
        main_const_eval_40_0 = main_const_eval_40(
            [weights["model.layers.0.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_40"] = main_const_eval_40_0[0]
        main_const_eval_41_0 = main_const_eval_41(
            [weights["model.layers.4.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_41"] = main_const_eval_41_0[0]
        main_const_eval_42_0 = main_const_eval_42(
            [weights["model.layers.8.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_42"] = main_const_eval_42_0[0]
        main_const_eval_43_0 = main_const_eval_43(
            [weights["model.layers.12.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_43"] = main_const_eval_43_0[0]
        main_const_eval_44_0 = main_const_eval_44(
            [weights["model.layers.14.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_44"] = main_const_eval_44_0[0]
        main_const_eval_45_0 = main_const_eval_45(
            [weights["model.layers.9.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_45"] = main_const_eval_45_0[0]
        main_const_eval_46_0 = main_const_eval_46(
            [weights["model.layers.15.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_46"] = main_const_eval_46_0[0]
        main_const_eval_47_0 = main_const_eval_47(
            [
                weights["model.layers.0.self_attn.k_proj.weight"],
                weights["model.layers.0.self_attn.v_proj.weight"],
                weights["model.layers.0.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_47"] = main_const_eval_47_0[0]
        main_const_eval_48_0 = main_const_eval_48(
            [
                weights["model.layers.9.self_attn.k_proj.weight"],
                weights["model.layers.9.self_attn.v_proj.weight"],
                weights["model.layers.9.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_48"] = main_const_eval_48_0[0]
        main_const_eval_49_0 = main_const_eval_49(
            [weights["model.layers.12.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_49"] = main_const_eval_49_0[0]
        main_const_eval_50_0 = main_const_eval_50(device)
        ce_cache["main_const_eval_50"] = main_const_eval_50_0[0]
        main_const_eval_51_0 = main_const_eval_51(
            [weights["model.layers.2.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_51"] = main_const_eval_51_0[0]
        main_const_eval_52_0 = main_const_eval_52(
            [weights["model.layers.4.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_52"] = main_const_eval_52_0[0]
        main_const_eval_53_0 = main_const_eval_53(
            [weights["model.layers.7.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_53"] = main_const_eval_53_0[0]
        main_const_eval_54_0 = main_const_eval_54(
            [weights["model.layers.11.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_54"] = main_const_eval_54_0[0]
        main_const_eval_55_0 = main_const_eval_55(
            [weights["model.layers.6.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_55"] = main_const_eval_55_0[0]
        main_const_eval_56_0 = main_const_eval_56(
            [weights["model.layers.12.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_56"] = main_const_eval_56_0[0]
        main_const_eval_57_0 = main_const_eval_57(
            [weights["model.layers.7.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_57"] = main_const_eval_57_0[0]
        main_const_eval_58_0 = main_const_eval_58(
            [weights["model.layers.2.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_58"] = main_const_eval_58_0[0]
        main_const_eval_59_0 = main_const_eval_59(
            [weights["model.layers.8.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_59"] = main_const_eval_59_0[0]
        main_const_eval_60_0 = main_const_eval_60(
            [weights["model.layers.9.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_60"] = main_const_eval_60_0[0]
        main_const_eval_61_0 = main_const_eval_61(device)
        ce_cache["main_const_eval_61"] = main_const_eval_61_0[0]
        main_const_eval_62_0 = main_const_eval_62(
            [weights["model.layers.3.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_62"] = main_const_eval_62_0[0]
        main_const_eval_63_0 = main_const_eval_63(
            [weights["model.layers.9.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_63"] = main_const_eval_63_0[0]
        main_const_eval_64_0 = main_const_eval_64(
            [weights["model.layers.4.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_64"] = main_const_eval_64_0[0]
        main_const_eval_65_0 = main_const_eval_65(
            [weights["model.embed_tokens.weight"]], device
        )
        ce_cache["main_const_eval_65"] = main_const_eval_65_0[0]
        main_const_eval_66_0 = main_const_eval_66(
            [weights["model.layers.3.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_66"] = main_const_eval_66_0[0]
        main_const_eval_67_0 = main_const_eval_67(
            [weights["model.layers.14.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_67"] = main_const_eval_67_0[0]
        main_const_eval_68_0 = main_const_eval_68(
            [
                weights["model.layers.5.self_attn.k_proj.weight"],
                weights["model.layers.5.self_attn.v_proj.weight"],
                weights["model.layers.5.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_68"] = main_const_eval_68_0[0]
        main_const_eval_69_0 = main_const_eval_69(
            [
                weights["model.layers.8.self_attn.k_proj.weight"],
                weights["model.layers.8.self_attn.v_proj.weight"],
                weights["model.layers.8.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_69"] = main_const_eval_69_0[0]
        main_const_eval_70_0 = main_const_eval_70(
            [weights["model.layers.5.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_70"] = main_const_eval_70_0[0]
        main_const_eval_71_0 = main_const_eval_71(
            [weights["model.layers.10.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_71"] = main_const_eval_71_0[0]
        main_const_eval_72_0 = main_const_eval_72(
            [weights["model.layers.10.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_72"] = main_const_eval_72_0[0]
        main_const_eval_73_0 = main_const_eval_73(
            [weights["model.layers.13.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_73"] = main_const_eval_73_0[0]
        main_const_eval_74_0 = main_const_eval_74(
            [
                weights["model.layers.13.self_attn.k_proj.weight"],
                weights["model.layers.13.self_attn.v_proj.weight"],
                weights["model.layers.13.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_74"] = main_const_eval_74_0[0]
        main_const_eval_75_0 = main_const_eval_75(
            [weights["model.layers.13.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_75"] = main_const_eval_75_0[0]
        main_const_eval_76_0 = main_const_eval_76(
            [weights["model.layers.11.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_76"] = main_const_eval_76_0[0]
        main_const_eval_77_0 = main_const_eval_77(
            [weights["model.layers.6.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_77"] = main_const_eval_77_0[0]
        main_const_eval_78_0 = main_const_eval_78(
            [weights["model.layers.1.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_78"] = main_const_eval_78_0[0]
        main_const_eval_79_0 = main_const_eval_79(
            [weights["model.layers.15.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_79"] = main_const_eval_79_0[0]
        main_const_eval_80_0 = main_const_eval_80(
            [weights["model.layers.1.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_80"] = main_const_eval_80_0[0]
        main_const_eval_81_0 = main_const_eval_81(
            [
                weights["model.layers.12.self_attn.k_proj.weight"],
                weights["model.layers.12.self_attn.v_proj.weight"],
                weights["model.layers.12.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_81"] = main_const_eval_81_0[0]
        main_const_eval_82_0 = main_const_eval_82([weights["lm_head.weight"]], device)
        ce_cache["main_const_eval_82"] = main_const_eval_82_0[0]
        main_const_eval_83_0 = main_const_eval_83(device)
        ce_cache["main_const_eval_83"] = main_const_eval_83_0[0]
        main_const_eval_84_0 = main_const_eval_84(
            [weights["model.layers.2.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_84"] = main_const_eval_84_0[0]
        main_const_eval_85_0 = main_const_eval_85(
            [weights["model.layers.10.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_85"] = main_const_eval_85_0[0]
        main_const_eval_86_0 = main_const_eval_86(
            [weights["model.layers.5.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_86"] = main_const_eval_86_0[0]
        main_const_eval_87_0 = main_const_eval_87(
            [weights["model.layers.13.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_87"] = main_const_eval_87_0[0]
    return ce_cache


def load_activations_for__main(device):
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg8.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg18.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg20.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg30.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg32.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg41.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg42.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg44.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg53.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_14 = utils.load_tensor(
        "./tensors/arg54.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_15 = utils.load_tensor(
        "./tensors/arg56.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_16 = utils.load_tensor(
        "./tensors/arg65.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg66.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg68.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_19 = utils.load_tensor(
        "./tensors/arg77.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_20 = utils.load_tensor(
        "./tensors/arg78.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_21 = utils.load_tensor(
        "./tensors/arg80.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_22 = utils.load_tensor(
        "./tensors/arg89.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_23 = utils.load_tensor(
        "./tensors/arg90.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg92.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_25 = utils.load_tensor(
        "./tensors/arg101.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_26 = utils.load_tensor(
        "./tensors/arg102.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_27 = utils.load_tensor(
        "./tensors/arg104.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_28 = utils.load_tensor(
        "./tensors/arg113.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg114.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg116.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_31 = utils.load_tensor(
        "./tensors/arg125.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_32 = utils.load_tensor(
        "./tensors/arg126.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_33 = utils.load_tensor(
        "./tensors/arg128.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_34 = utils.load_tensor(
        "./tensors/arg137.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_35 = utils.load_tensor(
        "./tensors/arg138.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg140.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_37 = utils.load_tensor(
        "./tensors/arg149.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_38 = utils.load_tensor(
        "./tensors/arg150.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_39 = utils.load_tensor(
        "./tensors/arg152.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_40 = utils.load_tensor(
        "./tensors/arg161.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_41 = utils.load_tensor(
        "./tensors/arg162.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_42 = utils.load_tensor(
        "./tensors/arg164.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_43 = utils.load_tensor(
        "./tensors/arg173.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_44 = utils.load_tensor(
        "./tensors/arg174.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_45 = utils.load_tensor(
        "./tensors/arg176.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_46 = utils.load_tensor(
        "./tensors/arg185.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_47 = utils.load_tensor(
        "./tensors/arg186.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_48 = utils.load_tensor(
        "./tensors/arg188.tensorbin",
        ttnn.Layout.TILE,
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
    ]


def main():
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 1)),
        l1_small_size=1 << 15,
    )
    load_activations_for__main_0 = load_activations_for__main(device)
    load_weights_for__main_0 = load_weights_for__main_from_state_dict(device)
    _main_0 = _main(load_activations_for__main_0, load_weights_for__main_0, device)
    return 0


def test_main():
    exact_pcc = 1.0078125

    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 1)),
        l1_small_size=1 << 15,
    )
    interleaved_dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )

    def to_ttnn_int32_row_major(tensor):
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.DataType.INT32,
            layout=ttnn.Layout.ROW_MAJOR,
            device=device,
            memory_config=interleaved_dram_memory_config,
        )

    def to_ttnn_bfloat16_tile(tensor):
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=device,
            memory_config=interleaved_dram_memory_config,
        )

    pytorch_input = model_pt.load_input()
    layers = pytorch_input["past_key_values"].layers

    # Mirror load_activations_for__main() ordering:
    #   [0]   layer 0 cumulative_length (INT32, ROW_MAJOR)
    #   [1]   input_ids                 (INT32, ROW_MAJOR)
    #   [2:4] layer 0 keys, values      (BFLOAT16, TILE)
    # then for each subsequent layer: cumulative_length, keys, values
    activations = [
        to_ttnn_int32_row_major(layers[0].cumulative_length),
        to_ttnn_int32_row_major(pytorch_input["input_ids"]),
        to_ttnn_bfloat16_tile(layers[0].keys),
        to_ttnn_bfloat16_tile(layers[0].values),
    ]
    for layer in layers[1:]:
        activations.append(to_ttnn_int32_row_major(layer.cumulative_length))
        activations.append(to_ttnn_bfloat16_tile(layer.keys))
        activations.append(to_ttnn_bfloat16_tile(layer.values))

    weights = load_weights_for__main_from_state_dict(device)
    outputs = _main(activations, weights, device)

    ttnn_output = ttnn.to_torch(ttnn.from_device(outputs[-1]))
    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output, golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    main()
