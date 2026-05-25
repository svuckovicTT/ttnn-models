import ttnn
import ttir_cpu
import torch


def main_const_eval_0(arg, device):
    ttnn_to_device_0 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_to_device_0]


def cpu_hoisted_const_eval_70989bca(arg_0, arg_1, arg_2):
    ttnn_to_torch_0 = ttnn.to_torch(arg_0)
    ttnn_to_torch_1 = ttnn.to_torch(arg_1)
    ttnn_to_torch_2 = ttnn.to_torch(arg_2)
    ttir_cpu_permute_0 = ttir_cpu.permute(ttnn_to_torch_0, [1, 0])
    ttir_cpu_permute_1 = ttir_cpu.permute(ttnn_to_torch_1, [1, 0])
    ttir_cpu_permute_2 = ttir_cpu.permute(ttnn_to_torch_2, [1, 0])
    util_create_list_0 = [ttir_cpu_permute_2, ttir_cpu_permute_0, ttir_cpu_permute_1]
    ttir_cpu_concat_0 = ttir_cpu.concat(util_create_list_0, dim=1)
    ttnn_from_torch_1 = ttnn.from_torch(ttir_cpu_concat_0)
    return ttnn_from_torch_1


def main_const_eval_1(arg, device):
    ttnn_typecast_1 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_2 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_3 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_0 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_1, ttnn_typecast_2, ttnn_typecast_3
    )
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn_typecast_4 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_0, False)
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_typecast_4, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_4, False)
    ttnn_to_device_1 = ttnn.to_device(
        ttnn_to_layout_0,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    return [ttnn_to_device_1]


def cpu_hoisted_const_eval_d18be0b6(arg):
    ttnn_to_torch_3 = ttnn.to_torch(arg)
    ttir_cpu_constant_2 = ttir_cpu.constant(
        shape=[1, 1, 16],
        dtype=torch.float32,
        data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )
    ttir_cpu_reshape_0 = ttir_cpu.reshape(ttnn_to_torch_3, [1, 64, 1])
    ttir_cpu_matmul_0 = ttir_cpu.matmul(ttir_cpu_reshape_0, ttir_cpu_constant_2)
    ttir_cpu_permute_3 = ttir_cpu.permute(ttir_cpu_matmul_0, [0, 2, 1])
    ttir_cpu_reshape_1 = ttir_cpu.reshape(ttir_cpu_permute_3, [1, 1, 16, 64])
    util_create_list_1 = [ttir_cpu_reshape_1, ttir_cpu_reshape_1]
    ttir_cpu_concat_1 = ttir_cpu.concat(util_create_list_1, dim=3)
    ttir_cpu_cos_0 = ttir_cpu.cos(ttir_cpu_concat_1)
    ttir_cpu_sin_0 = ttir_cpu.sin(ttir_cpu_concat_1)
    ttnn_from_torch_2 = ttnn.from_torch(ttir_cpu_cos_0)
    ttnn_from_torch_3 = ttnn.from_torch(ttir_cpu_sin_0)
    return ttnn_from_torch_2, ttnn_from_torch_3


def main_const_eval_2(arg, device):
    v_0, v_1 = cpu_hoisted_const_eval_d18be0b6(arg[0])
    ttnn_typecast_5 = ttnn.typecast(v_0, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_typecast_5, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_5, False)
    ttnn_to_device_2 = ttnn.to_device(
        ttnn_to_layout_1,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_typecast_6 = ttnn.typecast(v_1, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_1, False)
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_typecast_6, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_6, False)
    ttnn_to_device_3 = ttnn.to_device(
        ttnn_to_layout_2,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    return [ttnn_to_device_2, ttnn_to_device_3]


def main_const_eval_3(arg, device):
    ttnn_typecast_7 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_8 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_9 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_1 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_7, ttnn_typecast_8, ttnn_typecast_9
    )
    ttnn.deallocate(ttnn_typecast_9, False)
    ttnn.deallocate(ttnn_typecast_8, False)
    ttnn.deallocate(ttnn_typecast_7, False)
    ttnn_typecast_10 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_1, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_1, False)
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_typecast_10, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_10, False)
    ttnn_to_device_4 = ttnn.to_device(
        ttnn_to_layout_3,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    return [ttnn_to_device_4]


def main_const_eval_4(arg, device):
    ttnn_typecast_11 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_12 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_13 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_2 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_11, ttnn_typecast_12, ttnn_typecast_13
    )
    ttnn.deallocate(ttnn_typecast_13, False)
    ttnn.deallocate(ttnn_typecast_12, False)
    ttnn.deallocate(ttnn_typecast_11, False)
    ttnn_typecast_14 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_2, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_2, False)
    ttnn_to_layout_4 = ttnn.to_layout(
        ttnn_typecast_14, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_14, False)
    ttnn_to_device_5 = ttnn.to_device(
        ttnn_to_layout_4,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    return [ttnn_to_device_5]


def main_const_eval_5(arg, device):
    ttnn_typecast_15 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_16 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_17 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_3 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_15, ttnn_typecast_16, ttnn_typecast_17
    )
    ttnn.deallocate(ttnn_typecast_17, False)
    ttnn.deallocate(ttnn_typecast_16, False)
    ttnn.deallocate(ttnn_typecast_15, False)
    ttnn_typecast_18 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_3, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_3, False)
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_typecast_18, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_18, False)
    ttnn_to_device_6 = ttnn.to_device(
        ttnn_to_layout_5,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_5, False)
    return [ttnn_to_device_6]


def main_const_eval_6(arg, device):
    ttnn_typecast_19 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_20 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_21 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_4 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_19, ttnn_typecast_20, ttnn_typecast_21
    )
    ttnn.deallocate(ttnn_typecast_21, False)
    ttnn.deallocate(ttnn_typecast_20, False)
    ttnn.deallocate(ttnn_typecast_19, False)
    ttnn_typecast_22 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_4, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_4, False)
    ttnn_to_layout_6 = ttnn.to_layout(
        ttnn_typecast_22, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_22, False)
    ttnn_to_device_7 = ttnn.to_device(
        ttnn_to_layout_6,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_6, False)
    return [ttnn_to_device_7]


def main_const_eval_7(arg, device):
    ttnn_typecast_23 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_24 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_25 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_5 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_23, ttnn_typecast_24, ttnn_typecast_25
    )
    ttnn.deallocate(ttnn_typecast_25, False)
    ttnn.deallocate(ttnn_typecast_24, False)
    ttnn.deallocate(ttnn_typecast_23, False)
    ttnn_typecast_26 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_5, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_5, False)
    ttnn_to_layout_7 = ttnn.to_layout(
        ttnn_typecast_26, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_26, False)
    ttnn_to_device_8 = ttnn.to_device(
        ttnn_to_layout_7,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_7, False)
    return [ttnn_to_device_8]


def main_const_eval_8(arg, device):
    ttnn_typecast_27 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_28 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_29 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_6 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_27, ttnn_typecast_28, ttnn_typecast_29
    )
    ttnn.deallocate(ttnn_typecast_29, False)
    ttnn.deallocate(ttnn_typecast_28, False)
    ttnn.deallocate(ttnn_typecast_27, False)
    ttnn_typecast_30 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_6, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_6, False)
    ttnn_to_layout_8 = ttnn.to_layout(
        ttnn_typecast_30, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_30, False)
    ttnn_to_device_9 = ttnn.to_device(
        ttnn_to_layout_8,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_8, False)
    return [ttnn_to_device_9]


def main_const_eval_9(arg, device):
    ttnn_typecast_31 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_32 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_33 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_7 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_31, ttnn_typecast_32, ttnn_typecast_33
    )
    ttnn.deallocate(ttnn_typecast_33, False)
    ttnn.deallocate(ttnn_typecast_32, False)
    ttnn.deallocate(ttnn_typecast_31, False)
    ttnn_typecast_34 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_7, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_7, False)
    ttnn_to_layout_9 = ttnn.to_layout(
        ttnn_typecast_34, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_34, False)
    ttnn_to_device_10 = ttnn.to_device(
        ttnn_to_layout_9,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_9, False)
    return [ttnn_to_device_10]


def main_const_eval_10(arg, device):
    ttnn_typecast_35 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_36 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_37 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_8 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_35, ttnn_typecast_36, ttnn_typecast_37
    )
    ttnn.deallocate(ttnn_typecast_37, False)
    ttnn.deallocate(ttnn_typecast_36, False)
    ttnn.deallocate(ttnn_typecast_35, False)
    ttnn_typecast_38 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_8, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_8, False)
    ttnn_to_layout_10 = ttnn.to_layout(
        ttnn_typecast_38, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_38, False)
    ttnn_to_device_11 = ttnn.to_device(
        ttnn_to_layout_10,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_10, False)
    return [ttnn_to_device_11]


def main_const_eval_11(arg, device):
    ttnn_typecast_39 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_40 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_41 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_9 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_39, ttnn_typecast_40, ttnn_typecast_41
    )
    ttnn.deallocate(ttnn_typecast_41, False)
    ttnn.deallocate(ttnn_typecast_40, False)
    ttnn.deallocate(ttnn_typecast_39, False)
    ttnn_typecast_42 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_9, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_9, False)
    ttnn_to_layout_11 = ttnn.to_layout(
        ttnn_typecast_42, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_42, False)
    ttnn_to_device_12 = ttnn.to_device(
        ttnn_to_layout_11,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_11, False)
    return [ttnn_to_device_12]


def main_const_eval_12(arg, device):
    ttnn_typecast_43 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_44 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_45 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_10 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_43, ttnn_typecast_44, ttnn_typecast_45
    )
    ttnn.deallocate(ttnn_typecast_45, False)
    ttnn.deallocate(ttnn_typecast_44, False)
    ttnn.deallocate(ttnn_typecast_43, False)
    ttnn_typecast_46 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_10, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_10, False)
    ttnn_to_layout_12 = ttnn.to_layout(
        ttnn_typecast_46, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_46, False)
    ttnn_to_device_13 = ttnn.to_device(
        ttnn_to_layout_12,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_12, False)
    return [ttnn_to_device_13]


def main_const_eval_13(arg, device):
    ttnn_typecast_47 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_48 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_49 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_11 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_47, ttnn_typecast_48, ttnn_typecast_49
    )
    ttnn.deallocate(ttnn_typecast_49, False)
    ttnn.deallocate(ttnn_typecast_48, False)
    ttnn.deallocate(ttnn_typecast_47, False)
    ttnn_typecast_50 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_11, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_11, False)
    ttnn_to_layout_13 = ttnn.to_layout(
        ttnn_typecast_50, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_50, False)
    ttnn_to_device_14 = ttnn.to_device(
        ttnn_to_layout_13,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_13, False)
    return [ttnn_to_device_14]


def main_const_eval_14(arg, device):
    ttnn_typecast_51 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_52 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_53 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_12 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_51, ttnn_typecast_52, ttnn_typecast_53
    )
    ttnn.deallocate(ttnn_typecast_53, False)
    ttnn.deallocate(ttnn_typecast_52, False)
    ttnn.deallocate(ttnn_typecast_51, False)
    ttnn_typecast_54 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_12, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_12, False)
    ttnn_to_layout_14 = ttnn.to_layout(
        ttnn_typecast_54, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_54, False)
    ttnn_to_device_15 = ttnn.to_device(
        ttnn_to_layout_14,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_14, False)
    return [ttnn_to_device_15]


def main_const_eval_15(arg, device):
    ttnn_typecast_55 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_56 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_57 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_13 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_55, ttnn_typecast_56, ttnn_typecast_57
    )
    ttnn.deallocate(ttnn_typecast_57, False)
    ttnn.deallocate(ttnn_typecast_56, False)
    ttnn.deallocate(ttnn_typecast_55, False)
    ttnn_typecast_58 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_13, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_13, False)
    ttnn_to_layout_15 = ttnn.to_layout(
        ttnn_typecast_58, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_58, False)
    ttnn_to_device_16 = ttnn.to_device(
        ttnn_to_layout_15,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_15, False)
    return [ttnn_to_device_16]


def main_const_eval_16(arg, device):
    ttnn_typecast_59 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_60 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_61 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_14 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_59, ttnn_typecast_60, ttnn_typecast_61
    )
    ttnn.deallocate(ttnn_typecast_61, False)
    ttnn.deallocate(ttnn_typecast_60, False)
    ttnn.deallocate(ttnn_typecast_59, False)
    ttnn_typecast_62 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_14, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_14, False)
    ttnn_to_layout_16 = ttnn.to_layout(
        ttnn_typecast_62, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_62, False)
    ttnn_to_device_17 = ttnn.to_device(
        ttnn_to_layout_16,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_16, False)
    return [ttnn_to_device_17]


def main_const_eval_17(arg, device):
    ttnn_typecast_63 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_64 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_65 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_15 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_63, ttnn_typecast_64, ttnn_typecast_65
    )
    ttnn.deallocate(ttnn_typecast_65, False)
    ttnn.deallocate(ttnn_typecast_64, False)
    ttnn.deallocate(ttnn_typecast_63, False)
    ttnn_typecast_66 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_15, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_15, False)
    ttnn_to_layout_17 = ttnn.to_layout(
        ttnn_typecast_66, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_66, False)
    ttnn_to_device_18 = ttnn.to_device(
        ttnn_to_layout_17,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_17, False)
    return [ttnn_to_device_18]


def main_const_eval_18(arg, device):
    ttnn_typecast_67 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_68 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_69 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_16 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_67, ttnn_typecast_68, ttnn_typecast_69
    )
    ttnn.deallocate(ttnn_typecast_69, False)
    ttnn.deallocate(ttnn_typecast_68, False)
    ttnn.deallocate(ttnn_typecast_67, False)
    ttnn_typecast_70 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_16, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_16, False)
    ttnn_to_layout_18 = ttnn.to_layout(
        ttnn_typecast_70, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_70, False)
    ttnn_to_device_19 = ttnn.to_device(
        ttnn_to_layout_18,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_18, False)
    return [ttnn_to_device_19]


def main_const_eval_19(arg, device):
    ttnn_typecast_71 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_72 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_73 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_17 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_71, ttnn_typecast_72, ttnn_typecast_73
    )
    ttnn.deallocate(ttnn_typecast_73, False)
    ttnn.deallocate(ttnn_typecast_72, False)
    ttnn.deallocate(ttnn_typecast_71, False)
    ttnn_typecast_74 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_17, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_17, False)
    ttnn_to_layout_19 = ttnn.to_layout(
        ttnn_typecast_74, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_74, False)
    ttnn_to_device_20 = ttnn.to_device(
        ttnn_to_layout_19,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_19, False)
    return [ttnn_to_device_20]


def main_const_eval_20(arg, device):
    ttnn_typecast_75 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_76 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_77 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_18 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_75, ttnn_typecast_76, ttnn_typecast_77
    )
    ttnn.deallocate(ttnn_typecast_77, False)
    ttnn.deallocate(ttnn_typecast_76, False)
    ttnn.deallocate(ttnn_typecast_75, False)
    ttnn_typecast_78 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_18, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_18, False)
    ttnn_to_layout_20 = ttnn.to_layout(
        ttnn_typecast_78, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_78, False)
    ttnn_to_device_21 = ttnn.to_device(
        ttnn_to_layout_20,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_20, False)
    return [ttnn_to_device_21]


def cpu_hoisted_const_eval_17957e90():
    ttir_cpu_constant_0 = ttir_cpu.constant(
        shape=[1, 1, 16, 1],
        dtype=torch.int32,
        data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )
    ttir_cpu_constant_1 = ttir_cpu.constant(
        shape=[1, 1, 1, 16],
        dtype=torch.int32,
        data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )
    ttir_cpu_full_0 = ttir_cpu.full(
        shape=[1, 1, 1, 1], fill_value=float('-inf'), dtype=torch.float32
    )
    ttir_cpu_full_1 = ttir_cpu.full(
        shape=[1, 1, 1, 1], fill_value=0, dtype=torch.float32
    )
    ttir_cpu_ge_0 = ttir_cpu.ge(ttir_cpu_constant_0, ttir_cpu_constant_1)
    ttir_cpu_where_0 = ttir_cpu.where(ttir_cpu_ge_0, ttir_cpu_full_1, ttir_cpu_full_0)
    ttnn_from_torch_0 = ttnn.from_torch(ttir_cpu_where_0)
    return ttnn_from_torch_0


def main_const_eval_21(device):
    cpu_hoisted_const_eval_17957e90_0 = cpu_hoisted_const_eval_17957e90()
    ttnn_typecast_79 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_21 = ttnn.to_layout(
        ttnn_typecast_79, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_79, False)
    ttnn_to_device_22 = ttnn.to_device(
        ttnn_to_layout_21,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_21, False)
    ttnn_typecast_80 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_22 = ttnn.to_layout(
        ttnn_typecast_80, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_80, False)
    ttnn_to_device_23 = ttnn.to_device(
        ttnn_to_layout_22,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_22, False)
    ttnn_typecast_81 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_23 = ttnn.to_layout(
        ttnn_typecast_81, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_81, False)
    ttnn_to_device_24 = ttnn.to_device(
        ttnn_to_layout_23,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_23, False)
    ttnn_typecast_82 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_24 = ttnn.to_layout(
        ttnn_typecast_82, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_82, False)
    ttnn_to_device_25 = ttnn.to_device(
        ttnn_to_layout_24,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_24, False)
    ttnn_typecast_83 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_25 = ttnn.to_layout(
        ttnn_typecast_83, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_83, False)
    ttnn_to_device_26 = ttnn.to_device(
        ttnn_to_layout_25,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_25, False)
    ttnn_typecast_84 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_26 = ttnn.to_layout(
        ttnn_typecast_84, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_84, False)
    ttnn_to_device_27 = ttnn.to_device(
        ttnn_to_layout_26,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_26, False)
    ttnn_typecast_85 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_27 = ttnn.to_layout(
        ttnn_typecast_85, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_85, False)
    ttnn_to_device_28 = ttnn.to_device(
        ttnn_to_layout_27,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_27, False)
    ttnn_typecast_86 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_28 = ttnn.to_layout(
        ttnn_typecast_86, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_86, False)
    ttnn_to_device_29 = ttnn.to_device(
        ttnn_to_layout_28,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_28, False)
    ttnn_typecast_87 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_29 = ttnn.to_layout(
        ttnn_typecast_87, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_87, False)
    ttnn_to_device_30 = ttnn.to_device(
        ttnn_to_layout_29,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_29, False)
    ttnn_typecast_88 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_30 = ttnn.to_layout(
        ttnn_typecast_88, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_88, False)
    ttnn_to_device_31 = ttnn.to_device(
        ttnn_to_layout_30,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_30, False)
    ttnn_typecast_89 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_31 = ttnn.to_layout(
        ttnn_typecast_89, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_89, False)
    ttnn_to_device_32 = ttnn.to_device(
        ttnn_to_layout_31,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_31, False)
    ttnn_typecast_90 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_32 = ttnn.to_layout(
        ttnn_typecast_90, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_90, False)
    ttnn_to_device_33 = ttnn.to_device(
        ttnn_to_layout_32,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_32, False)
    ttnn_typecast_91 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_33 = ttnn.to_layout(
        ttnn_typecast_91, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_91, False)
    ttnn_to_device_34 = ttnn.to_device(
        ttnn_to_layout_33,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_33, False)
    ttnn_typecast_92 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_34 = ttnn.to_layout(
        ttnn_typecast_92, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_92, False)
    ttnn_to_device_35 = ttnn.to_device(
        ttnn_to_layout_34,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_34, False)
    ttnn_typecast_93 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_35 = ttnn.to_layout(
        ttnn_typecast_93, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_93, False)
    ttnn_to_device_36 = ttnn.to_device(
        ttnn_to_layout_35,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_35, False)
    ttnn_typecast_94 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_36 = ttnn.to_layout(
        ttnn_typecast_94, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_94, False)
    ttnn_to_device_37 = ttnn.to_device(
        ttnn_to_layout_36,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_36, False)
    ttnn_typecast_95 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_37 = ttnn.to_layout(
        ttnn_typecast_95, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_95, False)
    ttnn_to_device_38 = ttnn.to_device(
        ttnn_to_layout_37,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_37, False)
    ttnn_typecast_96 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_38 = ttnn.to_layout(
        ttnn_typecast_96, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_96, False)
    ttnn_to_device_39 = ttnn.to_device(
        ttnn_to_layout_38,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_38, False)
    ttnn_typecast_97 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_39 = ttnn.to_layout(
        ttnn_typecast_97, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_97, False)
    ttnn_to_device_40 = ttnn.to_device(
        ttnn_to_layout_39,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_39, False)
    ttnn_typecast_98 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_40 = ttnn.to_layout(
        ttnn_typecast_98, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_98, False)
    ttnn_to_device_41 = ttnn.to_device(
        ttnn_to_layout_40,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_40, False)
    ttnn_typecast_99 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_41 = ttnn.to_layout(
        ttnn_typecast_99, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_99, False)
    ttnn_to_device_42 = ttnn.to_device(
        ttnn_to_layout_41,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_41, False)
    ttnn_typecast_100 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_42 = ttnn.to_layout(
        ttnn_typecast_100, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_100, False)
    ttnn_to_device_43 = ttnn.to_device(
        ttnn_to_layout_42,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_42, False)
    ttnn_typecast_101 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_43 = ttnn.to_layout(
        ttnn_typecast_101, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_101, False)
    ttnn_to_device_44 = ttnn.to_device(
        ttnn_to_layout_43,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_43, False)
    ttnn_typecast_102 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_44 = ttnn.to_layout(
        ttnn_typecast_102, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_102, False)
    ttnn_to_device_45 = ttnn.to_device(
        ttnn_to_layout_44,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_44, False)
    ttnn_typecast_103 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_45 = ttnn.to_layout(
        ttnn_typecast_103, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_103, False)
    ttnn_to_device_46 = ttnn.to_device(
        ttnn_to_layout_45,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_45, False)
    ttnn_typecast_104 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_46 = ttnn.to_layout(
        ttnn_typecast_104, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_104, False)
    ttnn_to_device_47 = ttnn.to_device(
        ttnn_to_layout_46,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_46, False)
    ttnn_typecast_105 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_47 = ttnn.to_layout(
        ttnn_typecast_105, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_105, False)
    ttnn_to_device_48 = ttnn.to_device(
        ttnn_to_layout_47,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_47, False)
    ttnn_typecast_106 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_48 = ttnn.to_layout(
        ttnn_typecast_106, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_106, False)
    ttnn_to_device_49 = ttnn.to_device(
        ttnn_to_layout_48,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_48, False)
    ttnn_typecast_107 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_49 = ttnn.to_layout(
        ttnn_typecast_107, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_107, False)
    ttnn_to_device_50 = ttnn.to_device(
        ttnn_to_layout_49,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_49, False)
    ttnn_typecast_108 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_50 = ttnn.to_layout(
        ttnn_typecast_108, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_108, False)
    ttnn_to_device_51 = ttnn.to_device(
        ttnn_to_layout_50,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_50, False)
    ttnn_typecast_109 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_51 = ttnn.to_layout(
        ttnn_typecast_109, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_109, False)
    ttnn_to_device_52 = ttnn.to_device(
        ttnn_to_layout_51,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_51, False)
    ttnn_typecast_110 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_52 = ttnn.to_layout(
        ttnn_typecast_110, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_110, False)
    ttnn_to_device_53 = ttnn.to_device(
        ttnn_to_layout_52,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_52, False)
    ttnn_typecast_111 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_53 = ttnn.to_layout(
        ttnn_typecast_111, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_111, False)
    ttnn_to_device_54 = ttnn.to_device(
        ttnn_to_layout_53,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_53, False)
    ttnn_typecast_112 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_54 = ttnn.to_layout(
        ttnn_typecast_112, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_112, False)
    ttnn_to_device_55 = ttnn.to_device(
        ttnn_to_layout_54,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_54, False)
    ttnn_typecast_113 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn_to_layout_55 = ttnn.to_layout(
        ttnn_typecast_113, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_113, False)
    ttnn_to_device_56 = ttnn.to_device(
        ttnn_to_layout_55,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_55, False)
    ttnn_typecast_114 = ttnn.typecast(
        cpu_hoisted_const_eval_17957e90_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_17957e90_0, False)
    ttnn_to_layout_56 = ttnn.to_layout(
        ttnn_typecast_114, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_114, False)
    ttnn_to_device_57 = ttnn.to_device(
        ttnn_to_layout_56,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_56, False)
    return [
        ttnn_to_device_22,
        ttnn_to_device_23,
        ttnn_to_device_24,
        ttnn_to_device_25,
        ttnn_to_device_26,
        ttnn_to_device_27,
        ttnn_to_device_28,
        ttnn_to_device_29,
        ttnn_to_device_30,
        ttnn_to_device_31,
        ttnn_to_device_32,
        ttnn_to_device_33,
        ttnn_to_device_34,
        ttnn_to_device_35,
        ttnn_to_device_36,
        ttnn_to_device_37,
        ttnn_to_device_38,
        ttnn_to_device_39,
        ttnn_to_device_40,
        ttnn_to_device_41,
        ttnn_to_device_42,
        ttnn_to_device_43,
        ttnn_to_device_44,
        ttnn_to_device_45,
        ttnn_to_device_46,
        ttnn_to_device_47,
        ttnn_to_device_48,
        ttnn_to_device_49,
        ttnn_to_device_50,
        ttnn_to_device_51,
        ttnn_to_device_52,
        ttnn_to_device_53,
        ttnn_to_device_54,
        ttnn_to_device_55,
        ttnn_to_device_56,
        ttnn_to_device_57,
    ]


def main_const_eval_22(arg, device):
    ttnn_typecast_115 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_116 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_117 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_19 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_115, ttnn_typecast_116, ttnn_typecast_117
    )
    ttnn.deallocate(ttnn_typecast_117, False)
    ttnn.deallocate(ttnn_typecast_116, False)
    ttnn.deallocate(ttnn_typecast_115, False)
    ttnn_typecast_118 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_19, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_19, False)
    ttnn_to_layout_57 = ttnn.to_layout(
        ttnn_typecast_118, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_118, False)
    ttnn_to_device_58 = ttnn.to_device(
        ttnn_to_layout_57,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_57, False)
    return [ttnn_to_device_58]


def main_const_eval_23(arg, device):
    ttnn_typecast_119 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_120 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_121 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_20 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_119, ttnn_typecast_120, ttnn_typecast_121
    )
    ttnn.deallocate(ttnn_typecast_121, False)
    ttnn.deallocate(ttnn_typecast_120, False)
    ttnn.deallocate(ttnn_typecast_119, False)
    ttnn_typecast_122 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_20, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_20, False)
    ttnn_to_layout_58 = ttnn.to_layout(
        ttnn_typecast_122, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_122, False)
    ttnn_to_device_59 = ttnn.to_device(
        ttnn_to_layout_58,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_58, False)
    return [ttnn_to_device_59]


def main_const_eval_24(arg, device):
    ttnn_typecast_123 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_124 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_125 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_21 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_123, ttnn_typecast_124, ttnn_typecast_125
    )
    ttnn.deallocate(ttnn_typecast_125, False)
    ttnn.deallocate(ttnn_typecast_124, False)
    ttnn.deallocate(ttnn_typecast_123, False)
    ttnn_typecast_126 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_21, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_21, False)
    ttnn_to_layout_59 = ttnn.to_layout(
        ttnn_typecast_126, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_126, False)
    ttnn_to_device_60 = ttnn.to_device(
        ttnn_to_layout_59,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_59, False)
    return [ttnn_to_device_60]


def main_const_eval_25(arg, device):
    ttnn_typecast_127 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_128 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_129 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_22 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_127, ttnn_typecast_128, ttnn_typecast_129
    )
    ttnn.deallocate(ttnn_typecast_129, False)
    ttnn.deallocate(ttnn_typecast_128, False)
    ttnn.deallocate(ttnn_typecast_127, False)
    ttnn_typecast_130 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_22, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_22, False)
    ttnn_to_layout_60 = ttnn.to_layout(
        ttnn_typecast_130, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_130, False)
    ttnn_to_device_61 = ttnn.to_device(
        ttnn_to_layout_60,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_60, False)
    return [ttnn_to_device_61]


def main_const_eval_26(arg, device):
    ttnn_typecast_131 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_132 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_133 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_23 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_131, ttnn_typecast_132, ttnn_typecast_133
    )
    ttnn.deallocate(ttnn_typecast_133, False)
    ttnn.deallocate(ttnn_typecast_132, False)
    ttnn.deallocate(ttnn_typecast_131, False)
    ttnn_typecast_134 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_23, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_23, False)
    ttnn_to_layout_61 = ttnn.to_layout(
        ttnn_typecast_134, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_134, False)
    ttnn_to_device_62 = ttnn.to_device(
        ttnn_to_layout_61,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_61, False)
    return [ttnn_to_device_62]


def main_const_eval_27(arg, device):
    ttnn_typecast_135 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_136 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_137 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_24 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_135, ttnn_typecast_136, ttnn_typecast_137
    )
    ttnn.deallocate(ttnn_typecast_137, False)
    ttnn.deallocate(ttnn_typecast_136, False)
    ttnn.deallocate(ttnn_typecast_135, False)
    ttnn_typecast_138 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_24, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_24, False)
    ttnn_to_layout_62 = ttnn.to_layout(
        ttnn_typecast_138, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_138, False)
    ttnn_to_device_63 = ttnn.to_device(
        ttnn_to_layout_62,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_62, False)
    return [ttnn_to_device_63]


def main_const_eval_28(arg, device):
    ttnn_typecast_139 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_140 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_141 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_25 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_139, ttnn_typecast_140, ttnn_typecast_141
    )
    ttnn.deallocate(ttnn_typecast_141, False)
    ttnn.deallocate(ttnn_typecast_140, False)
    ttnn.deallocate(ttnn_typecast_139, False)
    ttnn_typecast_142 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_25, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_25, False)
    ttnn_to_layout_63 = ttnn.to_layout(
        ttnn_typecast_142, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_142, False)
    ttnn_to_device_64 = ttnn.to_device(
        ttnn_to_layout_63,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_63, False)
    return [ttnn_to_device_64]


def main_const_eval_29(arg, device):
    ttnn_typecast_143 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_144 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_145 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_26 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_143, ttnn_typecast_144, ttnn_typecast_145
    )
    ttnn.deallocate(ttnn_typecast_145, False)
    ttnn.deallocate(ttnn_typecast_144, False)
    ttnn.deallocate(ttnn_typecast_143, False)
    ttnn_typecast_146 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_26, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_26, False)
    ttnn_to_layout_64 = ttnn.to_layout(
        ttnn_typecast_146, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_146, False)
    ttnn_to_device_65 = ttnn.to_device(
        ttnn_to_layout_64,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_64, False)
    return [ttnn_to_device_65]


def main_const_eval_30(arg, device):
    ttnn_typecast_147 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_148 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_149 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_27 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_147, ttnn_typecast_148, ttnn_typecast_149
    )
    ttnn.deallocate(ttnn_typecast_149, False)
    ttnn.deallocate(ttnn_typecast_148, False)
    ttnn.deallocate(ttnn_typecast_147, False)
    ttnn_typecast_150 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_27, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_27, False)
    ttnn_to_layout_65 = ttnn.to_layout(
        ttnn_typecast_150, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_150, False)
    ttnn_to_device_66 = ttnn.to_device(
        ttnn_to_layout_65,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_65, False)
    return [ttnn_to_device_66]


def main_const_eval_31(arg, device):
    ttnn_typecast_151 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_152 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_153 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_28 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_151, ttnn_typecast_152, ttnn_typecast_153
    )
    ttnn.deallocate(ttnn_typecast_153, False)
    ttnn.deallocate(ttnn_typecast_152, False)
    ttnn.deallocate(ttnn_typecast_151, False)
    ttnn_typecast_154 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_28, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_28, False)
    ttnn_to_layout_66 = ttnn.to_layout(
        ttnn_typecast_154, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_154, False)
    ttnn_to_device_67 = ttnn.to_device(
        ttnn_to_layout_66,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_66, False)
    return [ttnn_to_device_67]


def main_const_eval_32(arg, device):
    ttnn_typecast_155 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_156 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_157 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_29 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_155, ttnn_typecast_156, ttnn_typecast_157
    )
    ttnn.deallocate(ttnn_typecast_157, False)
    ttnn.deallocate(ttnn_typecast_156, False)
    ttnn.deallocate(ttnn_typecast_155, False)
    ttnn_typecast_158 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_29, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_29, False)
    ttnn_to_layout_67 = ttnn.to_layout(
        ttnn_typecast_158, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_158, False)
    ttnn_to_device_68 = ttnn.to_device(
        ttnn_to_layout_67,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_67, False)
    return [ttnn_to_device_68]


def main_const_eval_33(arg, device):
    ttnn_typecast_159 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_160 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_161 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_30 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_159, ttnn_typecast_160, ttnn_typecast_161
    )
    ttnn.deallocate(ttnn_typecast_161, False)
    ttnn.deallocate(ttnn_typecast_160, False)
    ttnn.deallocate(ttnn_typecast_159, False)
    ttnn_typecast_162 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_30, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_30, False)
    ttnn_to_layout_68 = ttnn.to_layout(
        ttnn_typecast_162, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_162, False)
    ttnn_to_device_69 = ttnn.to_device(
        ttnn_to_layout_68,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_68, False)
    return [ttnn_to_device_69]


def main_const_eval_34(arg, device):
    ttnn_typecast_163 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_164 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_165 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_31 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_163, ttnn_typecast_164, ttnn_typecast_165
    )
    ttnn.deallocate(ttnn_typecast_165, False)
    ttnn.deallocate(ttnn_typecast_164, False)
    ttnn.deallocate(ttnn_typecast_163, False)
    ttnn_typecast_166 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_31, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_31, False)
    ttnn_to_layout_69 = ttnn.to_layout(
        ttnn_typecast_166, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_166, False)
    ttnn_to_device_70 = ttnn.to_device(
        ttnn_to_layout_69,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_69, False)
    return [ttnn_to_device_70]


def main_const_eval_35(arg, device):
    ttnn_typecast_167 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_168 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_169 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_32 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_167, ttnn_typecast_168, ttnn_typecast_169
    )
    ttnn.deallocate(ttnn_typecast_169, False)
    ttnn.deallocate(ttnn_typecast_168, False)
    ttnn.deallocate(ttnn_typecast_167, False)
    ttnn_typecast_170 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_32, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_32, False)
    ttnn_to_layout_70 = ttnn.to_layout(
        ttnn_typecast_170, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_170, False)
    ttnn_to_device_71 = ttnn.to_device(
        ttnn_to_layout_70,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_70, False)
    return [ttnn_to_device_71]


def main_const_eval_36(arg, device):
    ttnn_typecast_171 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_172 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_173 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_33 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_171, ttnn_typecast_172, ttnn_typecast_173
    )
    ttnn.deallocate(ttnn_typecast_173, False)
    ttnn.deallocate(ttnn_typecast_172, False)
    ttnn.deallocate(ttnn_typecast_171, False)
    ttnn_typecast_174 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_33, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_33, False)
    ttnn_to_layout_71 = ttnn.to_layout(
        ttnn_typecast_174, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_174, False)
    ttnn_to_device_72 = ttnn.to_device(
        ttnn_to_layout_71,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_71, False)
    return [ttnn_to_device_72]


def main_const_eval_37(arg, device):
    ttnn_typecast_175 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_176 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_177 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_34 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_175, ttnn_typecast_176, ttnn_typecast_177
    )
    ttnn.deallocate(ttnn_typecast_177, False)
    ttnn.deallocate(ttnn_typecast_176, False)
    ttnn.deallocate(ttnn_typecast_175, False)
    ttnn_typecast_178 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_34, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_34, False)
    ttnn_to_layout_72 = ttnn.to_layout(
        ttnn_typecast_178, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_178, False)
    ttnn_to_device_73 = ttnn.to_device(
        ttnn_to_layout_72,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_72, False)
    return [ttnn_to_device_73]


def main_const_eval_38(arg, device):
    ttnn_typecast_179 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_180 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_181 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_70989bca_35 = cpu_hoisted_const_eval_70989bca(
        ttnn_typecast_179, ttnn_typecast_180, ttnn_typecast_181
    )
    ttnn.deallocate(ttnn_typecast_181, False)
    ttnn.deallocate(ttnn_typecast_180, False)
    ttnn.deallocate(ttnn_typecast_179, False)
    ttnn_typecast_182 = ttnn.typecast(
        cpu_hoisted_const_eval_70989bca_35, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_70989bca_35, False)
    ttnn_to_layout_73 = ttnn.to_layout(
        ttnn_typecast_182, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_182, False)
    ttnn_to_device_74 = ttnn.to_device(
        ttnn_to_layout_73,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_73, False)
    return [ttnn_to_device_74]


def consteval__main(ce_cache, weights, device):
    if not ce_cache:
        main_const_eval_0_0 = main_const_eval_0(
            [weights["L__self___model_embed_tokens.weight"]], device
        )
        ce_cache["main_const_eval_0"] = main_const_eval_0_0[0]
        main_const_eval_1_0 = main_const_eval_1(
            [
                weights["L__self___model_layers_20_self_attn_v_proj.weight"],
                weights["L__self___model_layers_20_self_attn_k_proj.weight"],
                weights["L__self___model_layers_20_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_1"] = main_const_eval_1_0[0]
        main_const_eval_2_0 = main_const_eval_2(
            [weights["L__self___model_rotary_emb_inv_freq"]], device
        )
        ce_cache["main_const_eval_2"] = [main_const_eval_2_0[0], main_const_eval_2_0[1]]
        main_const_eval_3_0 = main_const_eval_3(
            [
                weights["L__self___model_layers_24_self_attn_v_proj.weight"],
                weights["L__self___model_layers_24_self_attn_k_proj.weight"],
                weights["L__self___model_layers_24_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_3"] = main_const_eval_3_0[0]
        main_const_eval_4_0 = main_const_eval_4(
            [
                weights["L__self___model_layers_16_self_attn_v_proj.weight"],
                weights["L__self___model_layers_16_self_attn_k_proj.weight"],
                weights["L__self___model_layers_16_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_4"] = main_const_eval_4_0[0]
        main_const_eval_5_0 = main_const_eval_5(
            [
                weights["L__self___model_layers_3_self_attn_v_proj.weight"],
                weights["L__self___model_layers_3_self_attn_k_proj.weight"],
                weights["L__self___model_layers_3_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_5"] = main_const_eval_5_0[0]
        main_const_eval_6_0 = main_const_eval_6(
            [
                weights["L__self___model_layers_1_self_attn_v_proj.weight"],
                weights["L__self___model_layers_1_self_attn_k_proj.weight"],
                weights["L__self___model_layers_1_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_6"] = main_const_eval_6_0[0]
        main_const_eval_7_0 = main_const_eval_7(
            [
                weights["L__self___model_layers_26_self_attn_v_proj.weight"],
                weights["L__self___model_layers_26_self_attn_k_proj.weight"],
                weights["L__self___model_layers_26_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_7"] = main_const_eval_7_0[0]
        main_const_eval_8_0 = main_const_eval_8(
            [
                weights["L__self___model_layers_18_self_attn_v_proj.weight"],
                weights["L__self___model_layers_18_self_attn_k_proj.weight"],
                weights["L__self___model_layers_18_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_8"] = main_const_eval_8_0[0]
        main_const_eval_9_0 = main_const_eval_9(
            [
                weights["L__self___model_layers_28_self_attn_v_proj.weight"],
                weights["L__self___model_layers_28_self_attn_k_proj.weight"],
                weights["L__self___model_layers_28_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_9"] = main_const_eval_9_0[0]
        main_const_eval_10_0 = main_const_eval_10(
            [
                weights["L__self___model_layers_35_self_attn_v_proj.weight"],
                weights["L__self___model_layers_35_self_attn_k_proj.weight"],
                weights["L__self___model_layers_35_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_10"] = main_const_eval_10_0[0]
        main_const_eval_11_0 = main_const_eval_11(
            [
                weights["L__self___model_layers_8_self_attn_v_proj.weight"],
                weights["L__self___model_layers_8_self_attn_k_proj.weight"],
                weights["L__self___model_layers_8_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_11"] = main_const_eval_11_0[0]
        main_const_eval_12_0 = main_const_eval_12(
            [
                weights["L__self___model_layers_9_self_attn_v_proj.weight"],
                weights["L__self___model_layers_9_self_attn_k_proj.weight"],
                weights["L__self___model_layers_9_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_12"] = main_const_eval_12_0[0]
        main_const_eval_13_0 = main_const_eval_13(
            [
                weights["L__self___model_layers_5_self_attn_v_proj.weight"],
                weights["L__self___model_layers_5_self_attn_k_proj.weight"],
                weights["L__self___model_layers_5_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_13"] = main_const_eval_13_0[0]
        main_const_eval_14_0 = main_const_eval_14(
            [
                weights["L__self___model_layers_15_self_attn_v_proj.weight"],
                weights["L__self___model_layers_15_self_attn_k_proj.weight"],
                weights["L__self___model_layers_15_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_14"] = main_const_eval_14_0[0]
        main_const_eval_15_0 = main_const_eval_15(
            [
                weights["L__self___model_layers_2_self_attn_v_proj.weight"],
                weights["L__self___model_layers_2_self_attn_k_proj.weight"],
                weights["L__self___model_layers_2_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_15"] = main_const_eval_15_0[0]
        main_const_eval_16_0 = main_const_eval_16(
            [
                weights["L__self___model_layers_12_self_attn_v_proj.weight"],
                weights["L__self___model_layers_12_self_attn_k_proj.weight"],
                weights["L__self___model_layers_12_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_16"] = main_const_eval_16_0[0]
        main_const_eval_17_0 = main_const_eval_17(
            [
                weights["L__self___model_layers_22_self_attn_v_proj.weight"],
                weights["L__self___model_layers_22_self_attn_k_proj.weight"],
                weights["L__self___model_layers_22_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_17"] = main_const_eval_17_0[0]
        main_const_eval_18_0 = main_const_eval_18(
            [
                weights["L__self___model_layers_25_self_attn_v_proj.weight"],
                weights["L__self___model_layers_25_self_attn_k_proj.weight"],
                weights["L__self___model_layers_25_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_18"] = main_const_eval_18_0[0]
        main_const_eval_19_0 = main_const_eval_19(
            [
                weights["L__self___model_layers_0_self_attn_v_proj.weight"],
                weights["L__self___model_layers_0_self_attn_k_proj.weight"],
                weights["L__self___model_layers_0_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_19"] = main_const_eval_19_0[0]
        main_const_eval_20_0 = main_const_eval_20(
            [
                weights["L__self___model_layers_23_self_attn_v_proj.weight"],
                weights["L__self___model_layers_23_self_attn_k_proj.weight"],
                weights["L__self___model_layers_23_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_20"] = main_const_eval_20_0[0]
        main_const_eval_21_0 = main_const_eval_21(device)
        ce_cache["main_const_eval_21"] = [
            main_const_eval_21_0[0],
            main_const_eval_21_0[1],
            main_const_eval_21_0[2],
            main_const_eval_21_0[3],
            main_const_eval_21_0[4],
            main_const_eval_21_0[5],
            main_const_eval_21_0[6],
            main_const_eval_21_0[7],
            main_const_eval_21_0[8],
            main_const_eval_21_0[9],
            main_const_eval_21_0[10],
            main_const_eval_21_0[11],
            main_const_eval_21_0[12],
            main_const_eval_21_0[13],
            main_const_eval_21_0[14],
            main_const_eval_21_0[15],
            main_const_eval_21_0[16],
            main_const_eval_21_0[17],
            main_const_eval_21_0[18],
            main_const_eval_21_0[19],
            main_const_eval_21_0[20],
            main_const_eval_21_0[21],
            main_const_eval_21_0[22],
            main_const_eval_21_0[23],
            main_const_eval_21_0[24],
            main_const_eval_21_0[25],
            main_const_eval_21_0[26],
            main_const_eval_21_0[27],
            main_const_eval_21_0[28],
            main_const_eval_21_0[29],
            main_const_eval_21_0[30],
            main_const_eval_21_0[31],
            main_const_eval_21_0[32],
            main_const_eval_21_0[33],
            main_const_eval_21_0[34],
            main_const_eval_21_0[35],
        ]
        main_const_eval_22_0 = main_const_eval_22(
            [
                weights["L__self___model_layers_19_self_attn_v_proj.weight"],
                weights["L__self___model_layers_19_self_attn_k_proj.weight"],
                weights["L__self___model_layers_19_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_22"] = main_const_eval_22_0[0]
        main_const_eval_23_0 = main_const_eval_23(
            [
                weights["L__self___model_layers_29_self_attn_v_proj.weight"],
                weights["L__self___model_layers_29_self_attn_k_proj.weight"],
                weights["L__self___model_layers_29_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_23"] = main_const_eval_23_0[0]
        main_const_eval_24_0 = main_const_eval_24(
            [
                weights["L__self___model_layers_32_self_attn_v_proj.weight"],
                weights["L__self___model_layers_32_self_attn_k_proj.weight"],
                weights["L__self___model_layers_32_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_24"] = main_const_eval_24_0[0]
        main_const_eval_25_0 = main_const_eval_25(
            [
                weights["L__self___model_layers_31_self_attn_v_proj.weight"],
                weights["L__self___model_layers_31_self_attn_k_proj.weight"],
                weights["L__self___model_layers_31_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_25"] = main_const_eval_25_0[0]
        main_const_eval_26_0 = main_const_eval_26(
            [
                weights["L__self___model_layers_6_self_attn_v_proj.weight"],
                weights["L__self___model_layers_6_self_attn_k_proj.weight"],
                weights["L__self___model_layers_6_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_26"] = main_const_eval_26_0[0]
        main_const_eval_27_0 = main_const_eval_27(
            [
                weights["L__self___model_layers_21_self_attn_v_proj.weight"],
                weights["L__self___model_layers_21_self_attn_k_proj.weight"],
                weights["L__self___model_layers_21_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_27"] = main_const_eval_27_0[0]
        main_const_eval_28_0 = main_const_eval_28(
            [
                weights["L__self___model_layers_10_self_attn_v_proj.weight"],
                weights["L__self___model_layers_10_self_attn_k_proj.weight"],
                weights["L__self___model_layers_10_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_28"] = main_const_eval_28_0[0]
        main_const_eval_29_0 = main_const_eval_29(
            [
                weights["L__self___model_layers_11_self_attn_v_proj.weight"],
                weights["L__self___model_layers_11_self_attn_k_proj.weight"],
                weights["L__self___model_layers_11_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_29"] = main_const_eval_29_0[0]
        main_const_eval_30_0 = main_const_eval_30(
            [
                weights["L__self___model_layers_13_self_attn_v_proj.weight"],
                weights["L__self___model_layers_13_self_attn_k_proj.weight"],
                weights["L__self___model_layers_13_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_30"] = main_const_eval_30_0[0]
        main_const_eval_31_0 = main_const_eval_31(
            [
                weights["L__self___model_layers_14_self_attn_v_proj.weight"],
                weights["L__self___model_layers_14_self_attn_k_proj.weight"],
                weights["L__self___model_layers_14_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_31"] = main_const_eval_31_0[0]
        main_const_eval_32_0 = main_const_eval_32(
            [
                weights["L__self___model_layers_33_self_attn_v_proj.weight"],
                weights["L__self___model_layers_33_self_attn_k_proj.weight"],
                weights["L__self___model_layers_33_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_32"] = main_const_eval_32_0[0]
        main_const_eval_33_0 = main_const_eval_33(
            [
                weights["L__self___model_layers_34_self_attn_v_proj.weight"],
                weights["L__self___model_layers_34_self_attn_k_proj.weight"],
                weights["L__self___model_layers_34_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_33"] = main_const_eval_33_0[0]
        main_const_eval_34_0 = main_const_eval_34(
            [
                weights["L__self___model_layers_4_self_attn_v_proj.weight"],
                weights["L__self___model_layers_4_self_attn_k_proj.weight"],
                weights["L__self___model_layers_4_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_34"] = main_const_eval_34_0[0]
        main_const_eval_35_0 = main_const_eval_35(
            [
                weights["L__self___model_layers_17_self_attn_v_proj.weight"],
                weights["L__self___model_layers_17_self_attn_k_proj.weight"],
                weights["L__self___model_layers_17_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_35"] = main_const_eval_35_0[0]
        main_const_eval_36_0 = main_const_eval_36(
            [
                weights["L__self___model_layers_30_self_attn_v_proj.weight"],
                weights["L__self___model_layers_30_self_attn_k_proj.weight"],
                weights["L__self___model_layers_30_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_36"] = main_const_eval_36_0[0]
        main_const_eval_37_0 = main_const_eval_37(
            [
                weights["L__self___model_layers_7_self_attn_v_proj.weight"],
                weights["L__self___model_layers_7_self_attn_k_proj.weight"],
                weights["L__self___model_layers_7_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_37"] = main_const_eval_37_0[0]
        main_const_eval_38_0 = main_const_eval_38(
            [
                weights["L__self___model_layers_27_self_attn_v_proj.weight"],
                weights["L__self___model_layers_27_self_attn_k_proj.weight"],
                weights["L__self___model_layers_27_self_attn_q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_38"] = main_const_eval_38_0[0]
    return ce_cache
