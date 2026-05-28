import ttnn
import utils
import ttir_cpu
import torch
import model_pt
from utils import calculate_pcc
from model_ttnn import ModelTTNN


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
    model = ModelTTNN(device)
    _main_0 = model(load_activations_for__main_0)
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

    model = ModelTTNN(device)
    outputs = model(activations)

    ttnn_output = ttnn.to_torch(ttnn.from_device(outputs[-1]))
    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output, golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    main()
