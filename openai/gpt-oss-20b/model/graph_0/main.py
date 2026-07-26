import ttnn
import utils
from utils import calculate_pcc
import ttir_cpu
import torch


def main_const_eval_0(arg):
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_to_device_0 = ttnn.to_device(
        arg[0],
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_to_device_0]


def cpu_hoisted_const_eval_4dba464f(arg):
    def cpu_hoisted_const_eval_4dba464f_impl(arg_0):
        ttir_cpu_permute_6 = ttir_cpu.permute(arg_0, [1, 0])
        return ttir_cpu_permute_6

    util_create_list_16 = [arg]
    utils_execute_cpu_hoisted_function_12 = utils.execute_cpu_hoisted_function(
        util_create_list_16, cpu_hoisted_const_eval_4dba464f_impl
    )
    return utils_execute_cpu_hoisted_function_12


def main_const_eval_1(arg):
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_0 = ttnn.from_device(arg[0])
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_from_device_0, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_0, False)
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_typecast_0, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    cpu_hoisted_const_eval_4dba464f_0 = cpu_hoisted_const_eval_4dba464f(
        ttnn_to_layout_0
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_typecast_1 = ttnn.typecast(
        cpu_hoisted_const_eval_4dba464f_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_4dba464f_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_typecast_1, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn_to_device_1 = ttnn.to_device(
        ttnn_to_layout_1,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_from_device_1 = ttnn.from_device(ttnn_to_device_1)
    ttnn.deallocate(ttnn_to_device_1, False)
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_from_device_1, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_1, False)
    ttnn_to_device_2 = ttnn.to_device(
        ttnn_typecast_2,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_2, False)
    return [ttnn_to_device_2]


def cpu_hoisted_const_eval_319e9113(arg_0, arg_1, arg_2):
    def cpu_hoisted_const_eval_319e9113_impl(arg, arg_0_1, arg_2_3):
        ttir_cpu_permute_0 = ttir_cpu.permute(arg, [1, 0])
        ttir_cpu_permute_1 = ttir_cpu.permute(arg_0_1, [1, 0])
        ttir_cpu_permute_2 = ttir_cpu.permute(arg_2_3, [1, 0])
        util_create_list_3 = [
            ttir_cpu_permute_0,
            ttir_cpu_permute_1,
            ttir_cpu_permute_2,
        ]
        ttir_cpu_concat_0 = ttir_cpu.concat(util_create_list_3, dim=1)
        return ttir_cpu_concat_0

    util_create_list_4 = [arg_0, arg_1, arg_2]
    utils_execute_cpu_hoisted_function_3 = utils.execute_cpu_hoisted_function(
        util_create_list_4, cpu_hoisted_const_eval_319e9113_impl
    )
    return utils_execute_cpu_hoisted_function_3


def main_const_eval_2(arg):
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_2 = ttnn.from_device(arg[0])
    ttnn_typecast_3 = ttnn.typecast(
        ttnn_from_device_2, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_2, False)
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_typecast_3, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn_from_device_3 = ttnn.from_device(arg[1])
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_from_device_3, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_3, False)
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_typecast_4, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_4, False)
    ttnn_from_device_4 = ttnn.from_device(arg[2])
    ttnn_typecast_5 = ttnn.typecast(
        ttnn_from_device_4, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_4, False)
    ttnn_to_layout_4 = ttnn.to_layout(
        ttnn_typecast_5, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_5, False)
    cpu_hoisted_const_eval_319e9113_0 = cpu_hoisted_const_eval_319e9113(
        ttnn_to_layout_2, ttnn_to_layout_3, ttnn_to_layout_4
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_typecast_6 = ttnn.typecast(
        cpu_hoisted_const_eval_319e9113_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_319e9113_0, False)
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_typecast_6, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_6, False)
    ttnn_to_device_3 = ttnn.to_device(
        ttnn_to_layout_5,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_5, False)
    ttnn_slice_0 = ttnn.slice(
        ttnn_to_device_3,
        [0, 0],
        [2880, 128],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_1 = ttnn.slice(
        ttnn_to_device_3,
        [0, 128],
        [2880, 256],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_2 = ttnn.slice(
        ttnn_to_device_3,
        [0, 256],
        [2880, 1280],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_3, False)
    ttnn_concat_0 = ttnn.concat(
        [ttnn_slice_2, ttnn_slice_0, ttnn_slice_1],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_2, False)
    ttnn.deallocate(ttnn_slice_1, False)
    ttnn.deallocate(ttnn_slice_0, False)
    ttnn_from_device_5 = ttnn.from_device(ttnn_concat_0)
    ttnn.deallocate(ttnn_concat_0, False)
    ttnn_typecast_7 = ttnn.typecast(
        ttnn_from_device_5, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_5, False)
    ttnn_to_device_4 = ttnn.to_device(
        ttnn_typecast_7,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_7, False)
    return [ttnn_to_device_4]


def main_const_eval_3(arg):
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_6 = ttnn.from_device(arg[0])
    ttnn_typecast_8 = ttnn.typecast(
        ttnn_from_device_6, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_6, False)
    ttnn_to_device_5 = ttnn.to_device(
        ttnn_typecast_8,
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_8, False)
    return [ttnn_to_device_5]


def main_const_eval_4():
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_zeros_0 = ttnn.zeros(
        shape=ttnn.Shape([544]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_zeros_0]


def cpu_hoisted_const_eval_32bbe2d5():
    def cpu_hoisted_const_eval_32bbe2d5_impl():
        ttir_cpu_zeros_0 = ttir_cpu.zeros(shape=[1, 1, 1, 1], dtype=torch.float32)
        ttir_cpu_constant_1 = ttir_cpu.constant(
            shape=[1, 1, 17, 1],
            dtype=torch.int32,
            data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        ttir_cpu_constant_2 = ttir_cpu.constant(
            shape=[1, 1, 1, 17],
            dtype=torch.int32,
            data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        ttir_cpu_full_2 = ttir_cpu.full(
            shape=[1, 1, 1, 1], fill_value=-3.38953139e38, dtype=torch.float32
        )
        ttir_cpu_ge_0 = ttir_cpu.ge(ttir_cpu_constant_1, ttir_cpu_constant_2)
        ttir_cpu_where_0 = ttir_cpu.where(
            ttir_cpu_ge_0, ttir_cpu_zeros_0, ttir_cpu_full_2
        )
        return ttir_cpu_where_0

    util_create_list_15 = []
    utils_execute_cpu_hoisted_function_11 = utils.execute_cpu_hoisted_function(
        util_create_list_15, cpu_hoisted_const_eval_32bbe2d5_impl
    )
    return utils_execute_cpu_hoisted_function_11


def main_const_eval_5():
    utils_DeviceGetter_get_device_5 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    cpu_hoisted_const_eval_32bbe2d5_0 = cpu_hoisted_const_eval_32bbe2d5()
    ttnn_typecast_9 = ttnn.typecast(
        cpu_hoisted_const_eval_32bbe2d5_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_32bbe2d5_0, False)
    ttnn_to_layout_6 = ttnn.to_layout(
        ttnn_typecast_9, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_9, False)
    ttnn_to_device_6 = ttnn.to_device(
        ttnn_to_layout_6,
        device=utils_DeviceGetter_get_device_5,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_6, False)
    return [ttnn_to_device_6]


def cpu_hoisted_const_eval_bd8d86d6(arg):
    def cpu_hoisted_const_eval_bd8d86d6_impl(arg_0):
        ttir_cpu_full_1 = ttir_cpu.full(
            shape=[1, 1, 1, 1], fill_value=1.34657359, dtype=torch.float32
        )
        ttir_cpu_constant_0 = ttir_cpu.constant(
            shape=[1, 1, 17],
            dtype=torch.float32,
            data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        ttir_cpu_reshape_5 = ttir_cpu.reshape(arg_0, [1, 32, 1])
        ttir_cpu_multiply_1 = ttir_cpu.multiply(ttir_cpu_reshape_5, ttir_cpu_constant_0)
        ttir_cpu_permute_4 = ttir_cpu.permute(ttir_cpu_multiply_1, [0, 2, 1])
        ttir_cpu_reshape_6 = ttir_cpu.reshape(ttir_cpu_permute_4, [1, 1, 17, 32])
        ttir_cpu_cos_0 = ttir_cpu.cos(ttir_cpu_reshape_6)
        ttir_cpu_multiply_2 = ttir_cpu.multiply(ttir_cpu_cos_0, ttir_cpu_full_1)
        ttir_cpu_sin_0 = ttir_cpu.sin(ttir_cpu_reshape_6)
        ttir_cpu_multiply_3 = ttir_cpu.multiply(ttir_cpu_sin_0, ttir_cpu_full_1)
        util_create_list_11 = [ttir_cpu_multiply_2, ttir_cpu_multiply_2]
        ttir_cpu_concat_2 = ttir_cpu.concat(util_create_list_11, dim=3)
        util_create_list_12 = [ttir_cpu_multiply_3, ttir_cpu_multiply_3]
        ttir_cpu_concat_3 = ttir_cpu.concat(util_create_list_12, dim=3)
        return ttir_cpu_multiply_1, ttir_cpu_concat_2, ttir_cpu_concat_3

    util_create_list_13 = [arg]
    v_0, v_1, v_2 = utils.execute_cpu_hoisted_function(
        util_create_list_13, cpu_hoisted_const_eval_bd8d86d6_impl
    )
    return v_0, v_1, v_2


def main_const_eval_6(arg):
    utils_DeviceGetter_get_device_6 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_typecast_10 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    v_0, v_1, v_2 = cpu_hoisted_const_eval_bd8d86d6(ttnn_typecast_10)
    ttnn.deallocate(ttnn_typecast_10, False)
    ttnn_typecast_11 = ttnn.typecast(v_1, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_1, False)
    ttnn_to_layout_7 = ttnn.to_layout(
        ttnn_typecast_11, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_11, False)
    ttnn_to_device_7 = ttnn.to_device(
        ttnn_to_layout_7,
        device=utils_DeviceGetter_get_device_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_7, False)
    ttnn_typecast_12 = ttnn.typecast(v_2, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_2, False)
    ttnn_to_layout_8 = ttnn.to_layout(
        ttnn_typecast_12, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_12, False)
    ttnn_to_device_8 = ttnn.to_device(
        ttnn_to_layout_8,
        device=utils_DeviceGetter_get_device_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_8, False)
    ttnn_to_layout_9 = ttnn.to_layout(v_0, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(v_0, False)
    ttnn_to_device_9 = ttnn.to_device(
        ttnn_to_layout_9,
        device=utils_DeviceGetter_get_device_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_9, False)
    return [ttnn_to_device_7, ttnn_to_device_8, ttnn_to_device_9]


def cpu_hoisted_const_eval_d1e42923(arg):
    def cpu_hoisted_const_eval_d1e42923_impl(arg_0):
        ttir_cpu_permute_5 = ttir_cpu.permute(arg_0, [1, 0])
        return ttir_cpu_permute_5

    util_create_list_14 = [arg]
    utils_execute_cpu_hoisted_function_10 = utils.execute_cpu_hoisted_function(
        util_create_list_14, cpu_hoisted_const_eval_d1e42923_impl
    )
    return utils_execute_cpu_hoisted_function_10


def main_const_eval_7(arg):
    utils_DeviceGetter_get_device_7 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_7 = ttnn.from_device(arg[0])
    ttnn_typecast_13 = ttnn.typecast(
        ttnn_from_device_7, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_7, False)
    ttnn_to_layout_10 = ttnn.to_layout(
        ttnn_typecast_13, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_13, False)
    cpu_hoisted_const_eval_d1e42923_0 = cpu_hoisted_const_eval_d1e42923(
        ttnn_to_layout_10
    )
    ttnn.deallocate(ttnn_to_layout_10, False)
    ttnn_to_layout_11 = ttnn.to_layout(
        cpu_hoisted_const_eval_d1e42923_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_d1e42923_0, False)
    ttnn_to_device_10 = ttnn.to_device(
        ttnn_to_layout_11,
        device=utils_DeviceGetter_get_device_7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_11, False)
    ttnn_from_device_8 = ttnn.from_device(ttnn_to_device_10)
    ttnn.deallocate(ttnn_to_device_10, False)
    ttnn_typecast_14 = ttnn.typecast(
        ttnn_from_device_8, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_8, False)
    ttnn_to_device_11 = ttnn.to_device(
        ttnn_typecast_14,
        device=utils_DeviceGetter_get_device_7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_14, False)
    return [ttnn_to_device_11]


def main_const_eval_8(arg):
    utils_DeviceGetter_get_device_8 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_9 = ttnn.from_device(arg[0])
    ttnn_typecast_15 = ttnn.typecast(
        ttnn_from_device_9, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_9, False)
    ttnn_to_layout_12 = ttnn.to_layout(
        ttnn_typecast_15, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_15, False)
    ttnn_from_device_10 = ttnn.from_device(arg[1])
    ttnn_typecast_16 = ttnn.typecast(
        ttnn_from_device_10, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_10, False)
    ttnn_to_layout_13 = ttnn.to_layout(
        ttnn_typecast_16, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_16, False)
    ttnn_from_device_11 = ttnn.from_device(arg[2])
    ttnn_typecast_17 = ttnn.typecast(
        ttnn_from_device_11, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_11, False)
    ttnn_to_layout_14 = ttnn.to_layout(
        ttnn_typecast_17, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_17, False)
    cpu_hoisted_const_eval_319e9113_1 = cpu_hoisted_const_eval_319e9113(
        ttnn_to_layout_12, ttnn_to_layout_13, ttnn_to_layout_14
    )
    ttnn.deallocate(ttnn_to_layout_14, False)
    ttnn.deallocate(ttnn_to_layout_13, False)
    ttnn.deallocate(ttnn_to_layout_12, False)
    ttnn_typecast_18 = ttnn.typecast(
        cpu_hoisted_const_eval_319e9113_1, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_319e9113_1, False)
    ttnn_to_layout_15 = ttnn.to_layout(
        ttnn_typecast_18, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_18, False)
    ttnn_to_device_12 = ttnn.to_device(
        ttnn_to_layout_15,
        device=utils_DeviceGetter_get_device_8,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_15, False)
    ttnn_slice_3 = ttnn.slice(
        ttnn_to_device_12,
        [0, 0],
        [2880, 128],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_4 = ttnn.slice(
        ttnn_to_device_12,
        [0, 128],
        [2880, 256],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_5 = ttnn.slice(
        ttnn_to_device_12,
        [0, 256],
        [2880, 1280],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_12, False)
    ttnn_concat_1 = ttnn.concat(
        [ttnn_slice_5, ttnn_slice_3, ttnn_slice_4],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_5, False)
    ttnn.deallocate(ttnn_slice_4, False)
    ttnn.deallocate(ttnn_slice_3, False)
    ttnn_from_device_12 = ttnn.from_device(ttnn_concat_1)
    ttnn.deallocate(ttnn_concat_1, False)
    ttnn_typecast_19 = ttnn.typecast(
        ttnn_from_device_12, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_12, False)
    ttnn_to_device_13 = ttnn.to_device(
        ttnn_typecast_19,
        device=utils_DeviceGetter_get_device_8,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_19, False)
    return [ttnn_to_device_13]


def cpu_hoisted_const_eval_332468e5(arg):
    def cpu_hoisted_const_eval_332468e5_impl(arg_0):
        ttir_cpu_permute_3 = ttir_cpu.permute(arg_0, [1, 0])
        return ttir_cpu_permute_3

    util_create_list_10 = [arg]
    utils_execute_cpu_hoisted_function_8 = utils.execute_cpu_hoisted_function(
        util_create_list_10, cpu_hoisted_const_eval_332468e5_impl
    )
    return utils_execute_cpu_hoisted_function_8


def main_const_eval_9(arg):
    utils_DeviceGetter_get_device_9 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_13 = ttnn.from_device(arg[0])
    ttnn_typecast_20 = ttnn.typecast(
        ttnn_from_device_13, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_13, False)
    ttnn_to_layout_16 = ttnn.to_layout(
        ttnn_typecast_20, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_20, False)
    cpu_hoisted_const_eval_332468e5_0 = cpu_hoisted_const_eval_332468e5(
        ttnn_to_layout_16
    )
    ttnn.deallocate(ttnn_to_layout_16, False)
    ttnn_typecast_21 = ttnn.typecast(
        cpu_hoisted_const_eval_332468e5_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_332468e5_0, False)
    ttnn_to_layout_17 = ttnn.to_layout(
        ttnn_typecast_21, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_21, False)
    ttnn_to_device_14 = ttnn.to_device(
        ttnn_to_layout_17,
        device=utils_DeviceGetter_get_device_9,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_17, False)
    ttnn_from_device_14 = ttnn.from_device(ttnn_to_device_14)
    ttnn.deallocate(ttnn_to_device_14, False)
    ttnn_typecast_22 = ttnn.typecast(
        ttnn_from_device_14, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_14, False)
    ttnn_to_device_15 = ttnn.to_device(
        ttnn_typecast_22,
        device=utils_DeviceGetter_get_device_9,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_22, False)
    return [ttnn_to_device_15]


def cpu_hoisted_const_eval_9cf1badc(arg):
    def cpu_hoisted_const_eval_9cf1badc_impl(arg_0):
        ttir_cpu_reshape_3 = ttir_cpu.reshape(arg_0, [32, 1, 5760])
        return ttir_cpu_reshape_3

    util_create_list_8 = [arg]
    utils_execute_cpu_hoisted_function_6 = utils.execute_cpu_hoisted_function(
        util_create_list_8, cpu_hoisted_const_eval_9cf1badc_impl
    )
    return utils_execute_cpu_hoisted_function_6


def main_const_eval_10(arg):
    utils_DeviceGetter_get_device_10 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_15 = ttnn.from_device(arg[0])
    ttnn_typecast_23 = ttnn.typecast(
        ttnn_from_device_15, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_15, False)
    ttnn_to_layout_18 = ttnn.to_layout(
        ttnn_typecast_23, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_23, False)
    cpu_hoisted_const_eval_9cf1badc_0 = cpu_hoisted_const_eval_9cf1badc(
        ttnn_to_layout_18
    )
    ttnn.deallocate(ttnn_to_layout_18, False)
    ttnn_typecast_24 = ttnn.typecast(
        cpu_hoisted_const_eval_9cf1badc_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_9cf1badc_0, False)
    ttnn_to_layout_19 = ttnn.to_layout(
        ttnn_typecast_24, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_24, False)
    ttnn_to_device_16 = ttnn.to_device(
        ttnn_to_layout_19,
        device=utils_DeviceGetter_get_device_10,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_19, False)
    return [ttnn_to_device_16]


def main_const_eval_11():
    utils_DeviceGetter_get_device_11 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_ones_0 = ttnn.ones(
        shape=ttnn.Shape([1, 1, 1]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_11,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_ones_0]


def main_const_eval_12(arg):
    utils_DeviceGetter_get_device_12 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_16 = ttnn.from_device(arg[0])
    ttnn_typecast_25 = ttnn.typecast(
        ttnn_from_device_16, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_16, False)
    ttnn_to_layout_20 = ttnn.to_layout(
        ttnn_typecast_25, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_25, False)
    cpu_hoisted_const_eval_4dba464f_1 = cpu_hoisted_const_eval_4dba464f(
        ttnn_to_layout_20
    )
    ttnn.deallocate(ttnn_to_layout_20, False)
    ttnn_typecast_26 = ttnn.typecast(
        cpu_hoisted_const_eval_4dba464f_1, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_4dba464f_1, False)
    ttnn_to_layout_21 = ttnn.to_layout(
        ttnn_typecast_26, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_26, False)
    ttnn_to_device_17 = ttnn.to_device(
        ttnn_to_layout_21,
        device=utils_DeviceGetter_get_device_12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_21, False)
    ttnn_from_device_17 = ttnn.from_device(ttnn_to_device_17)
    ttnn.deallocate(ttnn_to_device_17, False)
    ttnn_typecast_27 = ttnn.typecast(
        ttnn_from_device_17, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_17, False)
    ttnn_to_device_18 = ttnn.to_device(
        ttnn_typecast_27,
        device=utils_DeviceGetter_get_device_12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_27, False)
    return [ttnn_to_device_18]


def cpu_hoisted_const_eval_6fd7295c():
    def cpu_hoisted_const_eval_6fd7295c_impl():
        ttir_cpu_full_0 = ttir_cpu.full(
            shape=[1, 1, 1], fill_value=32, dtype=torch.int32
        )
        ttir_cpu_arange_0 = ttir_cpu.arange(
            0, 17, 1, arange_dimension=0, shape=[17], dtype=torch.int32
        )
        ttir_cpu_reshape_0 = ttir_cpu.reshape(ttir_cpu_arange_0, [17, 1, 1])
        ttir_cpu_multiply_0 = ttir_cpu.multiply(ttir_cpu_reshape_0, ttir_cpu_full_0)
        ttir_cpu_broadcast_0 = ttir_cpu.broadcast(ttir_cpu_multiply_0, [17, 4, 1])
        return ttir_cpu_broadcast_0

    util_create_list_0 = []
    utils_execute_cpu_hoisted_function_0 = utils.execute_cpu_hoisted_function(
        util_create_list_0, cpu_hoisted_const_eval_6fd7295c_impl
    )
    return utils_execute_cpu_hoisted_function_0


def main_const_eval_13():
    utils_DeviceGetter_get_device_13 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    cpu_hoisted_const_eval_6fd7295c_0 = cpu_hoisted_const_eval_6fd7295c()
    ttnn_to_layout_22 = ttnn.to_layout(
        cpu_hoisted_const_eval_6fd7295c_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_6fd7295c_0, False)
    ttnn_to_device_19 = ttnn.to_device(
        ttnn_to_layout_22,
        device=utils_DeviceGetter_get_device_13,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_22, False)
    return [ttnn_to_device_19]


def main_const_eval_14():
    utils_DeviceGetter_get_device_14 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=0.125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_14,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_full_0]


def cpu_hoisted_const_eval_6914797c(arg):
    def cpu_hoisted_const_eval_6914797c_impl(arg_0):
        ttir_cpu_reshape_4 = ttir_cpu.reshape(arg_0, [32, 1, 2880])
        return ttir_cpu_reshape_4

    util_create_list_9 = [arg]
    utils_execute_cpu_hoisted_function_7 = utils.execute_cpu_hoisted_function(
        util_create_list_9, cpu_hoisted_const_eval_6914797c_impl
    )
    return utils_execute_cpu_hoisted_function_7


def main_const_eval_15(arg):
    utils_DeviceGetter_get_device_15 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_18 = ttnn.from_device(arg[0])
    ttnn_typecast_28 = ttnn.typecast(
        ttnn_from_device_18, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_18, False)
    ttnn_to_layout_23 = ttnn.to_layout(
        ttnn_typecast_28, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_28, False)
    cpu_hoisted_const_eval_6914797c_0 = cpu_hoisted_const_eval_6914797c(
        ttnn_to_layout_23
    )
    ttnn.deallocate(ttnn_to_layout_23, False)
    ttnn_typecast_29 = ttnn.typecast(
        cpu_hoisted_const_eval_6914797c_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_6914797c_0, False)
    ttnn_to_layout_24 = ttnn.to_layout(
        ttnn_typecast_29, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_29, False)
    ttnn_to_device_20 = ttnn.to_device(
        ttnn_to_layout_24,
        device=utils_DeviceGetter_get_device_15,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_24, False)
    return [ttnn_to_device_20]


def main_const_eval_16(arg):
    utils_DeviceGetter_get_device_16 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_19 = ttnn.from_device(arg[0])
    ttnn_typecast_30 = ttnn.typecast(
        ttnn_from_device_19, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_19, False)
    ttnn_to_device_21 = ttnn.to_device(
        ttnn_typecast_30,
        device=utils_DeviceGetter_get_device_16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_30, False)
    return [ttnn_to_device_21]


def main_const_eval_17(arg):
    utils_DeviceGetter_get_device_17 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_20 = ttnn.from_device(arg[0])
    ttnn_typecast_31 = ttnn.typecast(
        ttnn_from_device_20, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_20, False)
    ttnn_to_layout_25 = ttnn.to_layout(
        ttnn_typecast_31, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_31, False)
    cpu_hoisted_const_eval_6914797c_1 = cpu_hoisted_const_eval_6914797c(
        ttnn_to_layout_25
    )
    ttnn.deallocate(ttnn_to_layout_25, False)
    ttnn_typecast_32 = ttnn.typecast(
        cpu_hoisted_const_eval_6914797c_1, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_6914797c_1, False)
    ttnn_to_layout_26 = ttnn.to_layout(
        ttnn_typecast_32, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_32, False)
    ttnn_to_device_22 = ttnn.to_device(
        ttnn_to_layout_26,
        device=utils_DeviceGetter_get_device_17,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_26, False)
    return [ttnn_to_device_22]


def main_const_eval_18():
    utils_DeviceGetter_get_device_18 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_18,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_full_1]


def main_const_eval_19(arg):
    utils_DeviceGetter_get_device_19 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_21 = ttnn.from_device(arg[0])
    ttnn_typecast_33 = ttnn.typecast(
        ttnn_from_device_21, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_21, False)
    ttnn_to_layout_27 = ttnn.to_layout(
        ttnn_typecast_33, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_33, False)
    cpu_hoisted_const_eval_9cf1badc_1 = cpu_hoisted_const_eval_9cf1badc(
        ttnn_to_layout_27
    )
    ttnn.deallocate(ttnn_to_layout_27, False)
    ttnn_typecast_34 = ttnn.typecast(
        cpu_hoisted_const_eval_9cf1badc_1, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_9cf1badc_1, False)
    ttnn_to_layout_28 = ttnn.to_layout(
        ttnn_typecast_34, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_34, False)
    ttnn_to_device_23 = ttnn.to_device(
        ttnn_to_layout_28,
        device=utils_DeviceGetter_get_device_19,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_28, False)
    return [ttnn_to_device_23]


def cpu_hoisted_const_eval_9fa64fc9(arg_0, arg_1, arg_2):
    def cpu_hoisted_const_eval_9fa64fc9_impl(arg, arg_0_1, arg_2_3):
        util_create_list_6 = [arg, arg_0_1, arg_2_3]
        ttir_cpu_concat_1 = ttir_cpu.concat(util_create_list_6, dim=0)
        return ttir_cpu_concat_1

    util_create_list_7 = [arg_0, arg_1, arg_2]
    utils_execute_cpu_hoisted_function_5 = utils.execute_cpu_hoisted_function(
        util_create_list_7, cpu_hoisted_const_eval_9fa64fc9_impl
    )
    return utils_execute_cpu_hoisted_function_5


def main_const_eval_20(arg):
    utils_DeviceGetter_get_device_20 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_to_device_24 = ttnn.to_device(
        arg[0],
        device=utils_DeviceGetter_get_device_20,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mesh_partition_0 = ttnn.mesh_partition(
        input_tensor=ttnn_to_device_24,
        dim=0,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_24, False)
    ttnn_to_device_25 = ttnn.to_device(
        arg[1],
        device=utils_DeviceGetter_get_device_20,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mesh_partition_1 = ttnn.mesh_partition(
        input_tensor=ttnn_to_device_25,
        dim=0,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_25, False)
    ttnn_to_device_26 = ttnn.to_device(
        arg[2],
        device=utils_DeviceGetter_get_device_20,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mesh_partition_2 = ttnn.mesh_partition(
        input_tensor=ttnn_to_device_26,
        dim=0,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_26, False)
    ttnn_from_device_22 = ttnn.from_device(ttnn_mesh_partition_0)
    ttnn.deallocate(ttnn_mesh_partition_0, False)
    ttnn_typecast_35 = ttnn.typecast(
        ttnn_from_device_22, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_22, False)
    ttnn_from_device_23 = ttnn.from_device(ttnn_mesh_partition_1)
    ttnn.deallocate(ttnn_mesh_partition_1, False)
    ttnn_typecast_36 = ttnn.typecast(
        ttnn_from_device_23, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_23, False)
    ttnn_from_device_24 = ttnn.from_device(ttnn_mesh_partition_2)
    ttnn.deallocate(ttnn_mesh_partition_2, False)
    ttnn_typecast_37 = ttnn.typecast(
        ttnn_from_device_24, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_24, False)
    cpu_hoisted_const_eval_9fa64fc9_0 = cpu_hoisted_const_eval_9fa64fc9(
        ttnn_typecast_35, ttnn_typecast_36, ttnn_typecast_37
    )
    ttnn.deallocate(ttnn_typecast_37, False)
    ttnn.deallocate(ttnn_typecast_36, False)
    ttnn.deallocate(ttnn_typecast_35, False)
    ttnn_typecast_38 = ttnn.typecast(
        cpu_hoisted_const_eval_9fa64fc9_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_9fa64fc9_0, False)
    ttnn_to_layout_29 = ttnn.to_layout(
        ttnn_typecast_38, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_38, False)
    ttnn_to_device_27 = ttnn.to_device(
        ttnn_to_layout_29,
        device=utils_DeviceGetter_get_device_20,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_29, False)
    ttnn_slice_6 = ttnn.slice(
        ttnn_to_device_27,
        [0],
        [128],
        [1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_7 = ttnn.slice(
        ttnn_to_device_27,
        [128],
        [256],
        [1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_8 = ttnn.slice(
        ttnn_to_device_27,
        [256],
        [1280],
        [1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_27, False)
    ttnn_concat_2 = ttnn.concat(
        [ttnn_slice_8, ttnn_slice_6, ttnn_slice_7],
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_8, False)
    ttnn.deallocate(ttnn_slice_7, False)
    ttnn.deallocate(ttnn_slice_6, False)
    return [ttnn_concat_2]


def cpu_hoisted_const_eval_393c291a(arg):
    def cpu_hoisted_const_eval_393c291a_impl(arg_0):
        ttir_cpu_reshape_2 = ttir_cpu.reshape(arg_0, [1, 16, 1, 1])
        ttir_cpu_broadcast_1 = ttir_cpu.broadcast(ttir_cpu_reshape_2, [1, 16, 17, 1])
        return ttir_cpu_broadcast_1

    util_create_list_5 = [arg]
    utils_execute_cpu_hoisted_function_4 = utils.execute_cpu_hoisted_function(
        util_create_list_5, cpu_hoisted_const_eval_393c291a_impl
    )
    return utils_execute_cpu_hoisted_function_4


def main_const_eval_21(arg):
    utils_DeviceGetter_get_device_21 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_to_device_28 = ttnn.to_device(
        arg[0],
        device=utils_DeviceGetter_get_device_21,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mesh_partition_3 = ttnn.mesh_partition(
        input_tensor=ttnn_to_device_28,
        dim=0,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_28, False)
    ttnn_from_device_25 = ttnn.from_device(ttnn_mesh_partition_3)
    ttnn.deallocate(ttnn_mesh_partition_3, False)
    ttnn_typecast_39 = ttnn.typecast(
        ttnn_from_device_25, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_25, False)
    cpu_hoisted_const_eval_393c291a_0 = cpu_hoisted_const_eval_393c291a(
        ttnn_typecast_39
    )
    ttnn.deallocate(ttnn_typecast_39, False)
    ttnn_typecast_40 = ttnn.typecast(
        cpu_hoisted_const_eval_393c291a_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_393c291a_0, False)
    ttnn_to_layout_30 = ttnn.to_layout(
        ttnn_typecast_40, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_40, False)
    ttnn_to_device_29 = ttnn.to_device(
        ttnn_to_layout_30,
        device=utils_DeviceGetter_get_device_21,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_30, False)
    return [ttnn_to_device_29]


def main_const_eval_22():
    utils_DeviceGetter_get_device_22 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_zeros_1 = ttnn.zeros(
        shape=ttnn.Shape([17, 32]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_22,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_zeros_1]


def cpu_hoisted_const_eval_82fd2f86(arg):
    def cpu_hoisted_const_eval_82fd2f86_impl(arg_0):
        ttir_cpu_reshape_1 = ttir_cpu.reshape(arg_0, [1, 2880])
        return ttir_cpu_reshape_1

    util_create_list_1 = [arg]
    utils_execute_cpu_hoisted_function_1 = utils.execute_cpu_hoisted_function(
        util_create_list_1, cpu_hoisted_const_eval_82fd2f86_impl
    )
    return utils_execute_cpu_hoisted_function_1


def main_const_eval_23(arg):
    utils_DeviceGetter_get_device_23 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_typecast_41 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_82fd2f86_0 = cpu_hoisted_const_eval_82fd2f86(
        ttnn_typecast_41
    )
    ttnn.deallocate(ttnn_typecast_41, False)
    ttnn_typecast_42 = ttnn.typecast(
        cpu_hoisted_const_eval_82fd2f86_0, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_82fd2f86_0, False)
    ttnn_to_layout_31 = ttnn.to_layout(
        ttnn_typecast_42, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_42, False)
    ttnn_to_device_30 = ttnn.to_device(
        ttnn_to_layout_31,
        device=utils_DeviceGetter_get_device_23,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_31, False)
    return [ttnn_to_device_30]


def main_const_eval_24(arg):
    utils_DeviceGetter_get_device_24 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_typecast_43 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_82fd2f86_1 = cpu_hoisted_const_eval_82fd2f86(
        ttnn_typecast_43
    )
    ttnn.deallocate(ttnn_typecast_43, False)
    ttnn_typecast_44 = ttnn.typecast(
        cpu_hoisted_const_eval_82fd2f86_1, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_82fd2f86_1, False)
    ttnn_to_layout_32 = ttnn.to_layout(
        ttnn_typecast_44, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_44, False)
    ttnn_to_device_31 = ttnn.to_device(
        ttnn_to_layout_32,
        device=utils_DeviceGetter_get_device_24,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_32, False)
    return [ttnn_to_device_31]


def main_const_eval_25():
    utils_DeviceGetter_get_device_25 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=1.703125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_25,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_full_2]


def main_const_eval_26(arg):
    utils_DeviceGetter_get_device_26 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_26 = ttnn.from_device(arg[0])
    ttnn_typecast_45 = ttnn.typecast(
        ttnn_from_device_26, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_26, False)
    ttnn_to_device_32 = ttnn.to_device(
        ttnn_typecast_45,
        device=utils_DeviceGetter_get_device_26,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_45, False)
    return [ttnn_to_device_32]


def main_const_eval_27(arg):
    utils_DeviceGetter_get_device_27 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_27 = ttnn.from_device(arg[0])
    ttnn_typecast_46 = ttnn.typecast(
        ttnn_from_device_27, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_27, False)
    ttnn_to_layout_33 = ttnn.to_layout(
        ttnn_typecast_46, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_46, False)
    cpu_hoisted_const_eval_d1e42923_1 = cpu_hoisted_const_eval_d1e42923(
        ttnn_to_layout_33
    )
    ttnn.deallocate(ttnn_to_layout_33, False)
    ttnn_to_layout_34 = ttnn.to_layout(
        cpu_hoisted_const_eval_d1e42923_1, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_d1e42923_1, False)
    ttnn_to_device_33 = ttnn.to_device(
        ttnn_to_layout_34,
        device=utils_DeviceGetter_get_device_27,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_34, False)
    ttnn_from_device_28 = ttnn.from_device(ttnn_to_device_33)
    ttnn.deallocate(ttnn_to_device_33, False)
    ttnn_typecast_47 = ttnn.typecast(
        ttnn_from_device_28, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_28, False)
    ttnn_to_device_34 = ttnn.to_device(
        ttnn_typecast_47,
        device=utils_DeviceGetter_get_device_27,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_47, False)
    return [ttnn_to_device_34]


def cpu_hoisted_const_eval_ba9090d7(arg):
    def cpu_hoisted_const_eval_ba9090d7_impl(arg_0):
        return arg_0

    util_create_list_2 = [arg]
    utils_execute_cpu_hoisted_function_2 = utils.execute_cpu_hoisted_function(
        util_create_list_2, cpu_hoisted_const_eval_ba9090d7_impl
    )
    return utils_execute_cpu_hoisted_function_2


def main_const_eval_28(arg):
    utils_DeviceGetter_get_device_28 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_typecast_48 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_ba9090d7_0 = cpu_hoisted_const_eval_ba9090d7(
        ttnn_typecast_48
    )
    ttnn.deallocate(ttnn_typecast_48, False)
    ttnn_to_layout_35 = ttnn.to_layout(
        cpu_hoisted_const_eval_ba9090d7_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_ba9090d7_0, False)
    ttnn_to_device_35 = ttnn.to_device(
        ttnn_to_layout_35,
        device=utils_DeviceGetter_get_device_28,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_35, False)
    return [ttnn_to_device_35]


def main_const_eval_29(arg):
    utils_DeviceGetter_get_device_29 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_to_device_36 = ttnn.to_device(
        arg[0],
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mesh_partition_4 = ttnn.mesh_partition(
        input_tensor=ttnn_to_device_36,
        dim=0,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_36, False)
    ttnn_from_device_29 = ttnn.from_device(ttnn_mesh_partition_4)
    ttnn.deallocate(ttnn_mesh_partition_4, False)
    ttnn_typecast_49 = ttnn.typecast(
        ttnn_from_device_29, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_29, False)
    cpu_hoisted_const_eval_393c291a_1 = cpu_hoisted_const_eval_393c291a(
        ttnn_typecast_49
    )
    ttnn.deallocate(ttnn_typecast_49, False)
    ttnn_typecast_50 = ttnn.typecast(
        cpu_hoisted_const_eval_393c291a_1, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_393c291a_1, False)
    ttnn_to_layout_36 = ttnn.to_layout(
        ttnn_typecast_50, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_50, False)
    ttnn_to_device_37 = ttnn.to_device(
        ttnn_to_layout_36,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_36, False)
    return [ttnn_to_device_37]


def main_const_eval_30(arg):
    utils_DeviceGetter_get_device_30 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_to_device_38 = ttnn.to_device(
        arg[0],
        device=utils_DeviceGetter_get_device_30,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mesh_partition_5 = ttnn.mesh_partition(
        input_tensor=ttnn_to_device_38,
        dim=0,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_38, False)
    ttnn_to_device_39 = ttnn.to_device(
        arg[1],
        device=utils_DeviceGetter_get_device_30,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mesh_partition_6 = ttnn.mesh_partition(
        input_tensor=ttnn_to_device_39,
        dim=0,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_39, False)
    ttnn_to_device_40 = ttnn.to_device(
        arg[2],
        device=utils_DeviceGetter_get_device_30,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mesh_partition_7 = ttnn.mesh_partition(
        input_tensor=ttnn_to_device_40,
        dim=0,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_40, False)
    ttnn_from_device_30 = ttnn.from_device(ttnn_mesh_partition_5)
    ttnn.deallocate(ttnn_mesh_partition_5, False)
    ttnn_typecast_51 = ttnn.typecast(
        ttnn_from_device_30, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_30, False)
    ttnn_from_device_31 = ttnn.from_device(ttnn_mesh_partition_6)
    ttnn.deallocate(ttnn_mesh_partition_6, False)
    ttnn_typecast_52 = ttnn.typecast(
        ttnn_from_device_31, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_31, False)
    ttnn_from_device_32 = ttnn.from_device(ttnn_mesh_partition_7)
    ttnn.deallocate(ttnn_mesh_partition_7, False)
    ttnn_typecast_53 = ttnn.typecast(
        ttnn_from_device_32, ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_32, False)
    cpu_hoisted_const_eval_9fa64fc9_1 = cpu_hoisted_const_eval_9fa64fc9(
        ttnn_typecast_51, ttnn_typecast_52, ttnn_typecast_53
    )
    ttnn.deallocate(ttnn_typecast_53, False)
    ttnn.deallocate(ttnn_typecast_52, False)
    ttnn.deallocate(ttnn_typecast_51, False)
    ttnn_typecast_54 = ttnn.typecast(
        cpu_hoisted_const_eval_9fa64fc9_1, ttnn.DataType.BFLOAT16, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_9fa64fc9_1, False)
    ttnn_to_layout_37 = ttnn.to_layout(
        ttnn_typecast_54, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(ttnn_typecast_54, False)
    ttnn_to_device_41 = ttnn.to_device(
        ttnn_to_layout_37,
        device=utils_DeviceGetter_get_device_30,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_37, False)
    ttnn_slice_9 = ttnn.slice(
        ttnn_to_device_41,
        [0],
        [128],
        [1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_10 = ttnn.slice(
        ttnn_to_device_41,
        [128],
        [256],
        [1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_11 = ttnn.slice(
        ttnn_to_device_41,
        [256],
        [1280],
        [1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_41, False)
    ttnn_concat_3 = ttnn.concat(
        [ttnn_slice_11, ttnn_slice_9, ttnn_slice_10],
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_11, False)
    ttnn.deallocate(ttnn_slice_10, False)
    ttnn.deallocate(ttnn_slice_9, False)
    return [ttnn_concat_3]


def main_const_eval_31(arg):
    utils_DeviceGetter_get_device_31 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_from_device_33 = ttnn.from_device(arg[0])
    ttnn_typecast_55 = ttnn.typecast(
        ttnn_from_device_33, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_33, False)
    ttnn_to_device_42 = ttnn.to_device(
        ttnn_typecast_55,
        device=utils_DeviceGetter_get_device_31,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_55, False)
    return [ttnn_to_device_42]


def main_const_eval_32(arg):
    utils_DeviceGetter_get_device_32 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_typecast_56 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_ba9090d7_1 = cpu_hoisted_const_eval_ba9090d7(
        ttnn_typecast_56
    )
    ttnn.deallocate(ttnn_typecast_56, False)
    ttnn_to_layout_38 = ttnn.to_layout(
        cpu_hoisted_const_eval_ba9090d7_1, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_ba9090d7_1, False)
    ttnn_to_device_43 = ttnn.to_device(
        ttnn_to_layout_38,
        device=utils_DeviceGetter_get_device_32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_38, False)
    return [ttnn_to_device_43]


ce_cache__main = {}


def _main(activations, weights):
    global ce_cache__main
    ce_cache__main = consteval__main(ce_cache__main, weights)
    var_0 = weights["model.layers.0.input_layernorm.parametrizations.weight.original"]
    primals_39 = activations[0]
    var_1 = weights["model.layers.1.input_layernorm.parametrizations.weight.original"]
    var_2 = weights[
        "model.layers.0.post_attention_layernorm.parametrizations.weight.original"
    ]
    var_3 = weights["model.norm.parametrizations.weight.original"]
    var_4 = weights[
        "model.layers.1.post_attention_layernorm.parametrizations.weight.original"
    ]
    var_5 = ce_cache__main["main_const_eval_4"]
    var_6 = ce_cache__main["main_const_eval_5"]
    var_7 = ce_cache__main["main_const_eval_6"]
    var_8 = var_7[0]
    var_9 = var_7[1]
    var_10 = ce_cache__main["main_const_eval_11"]
    var_11 = ce_cache__main["main_const_eval_13"]
    var_12 = ce_cache__main["main_const_eval_14"]
    var_13 = ce_cache__main["main_const_eval_18"]
    var_14 = ce_cache__main["main_const_eval_25"]
    ttnn_typecast_57 = ttnn.typecast(
        primals_39,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_typecast_57,
        [17],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_57, False)
    ttnn_embedding_0 = ttnn.embedding(
        ttnn_reshape_0,
        ce_cache__main["main_const_eval_0"],
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_embedding_0,
        [1, 17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_58 = ttnn.typecast(
        ttnn_reshape_1,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_pow_0 = ttnn.pow(
        ttnn_typecast_58,
        2.0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_58, False)
    ttnn_mean_0 = ttnn.mean(
        ttnn_pow_0,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_pow_0, False)
    ttnn_add_0 = ttnn.add(
        ttnn_mean_0,
        var_13,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_0, False)
    ttnn_rsqrt_0 = ttnn.rsqrt(
        ttnn_add_0,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_rms_norm_0 = ttnn.rms_norm(
        ttnn_reshape_1,
        epsilon=9.9999997473787516e-06,
        weight=var_0,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_rms_norm_0,
        [17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_linear_0 = ttnn.linear(
        ttnn_reshape_2,
        ce_cache__main["main_const_eval_2"],
        bias=ce_cache__main["main_const_eval_20"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_linear_0,
        [1, 17, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_0, False)
    v_15, v_16, v_17 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_3,
        None,
        num_heads=16,
        num_kv_heads=2,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_3, False)
    ttnn_experimental_rotary_embedding_0 = ttnn.experimental.rotary_embedding(
        v_17,
        var_8,
        var_9,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(v_17, False)
    ttnn_slice_12 = ttnn.slice(
        ttnn_experimental_rotary_embedding_0,
        [0, 0, 0, 0],
        [1, 2, 17, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_0, False)
    ttnn_experimental_rotary_embedding_1 = ttnn.experimental.rotary_embedding(
        v_15,
        var_8,
        var_9,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(v_15, False)
    ttnn_slice_13 = ttnn.slice(
        ttnn_experimental_rotary_embedding_1,
        [0, 0, 0, 0],
        [1, 16, 17, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_1, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_slice_12,
        [1, 2, 1, 17, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_0 = ttnn.repeat(
        ttnn_reshape_4,
        ttnn.Shape([1, 1, 8, 1, 1]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_4, False)
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_repeat_0,
        [1, 16, 17, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_0 = ttnn.permute(
        ttnn_reshape_5,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_5, False)
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_slice_13,
        ttnn_permute_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_matmul_0,
        var_12,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_add_1 = ttnn.add(
        ttnn_multiply_0,
        var_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_0, False)
    ttnn_concat_4 = ttnn.concat(
        [ttnn_add_1, ce_cache__main["main_const_eval_29"]],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_1, False)
    ttnn_softmax_0 = ttnn.softmax(
        ttnn_concat_4,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
        numeric_stable=True,
    )
    ttnn_slice_14 = ttnn.slice(
        ttnn_softmax_0,
        [0, 0, 0, 0],
        [1, 16, 17, 17],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_6 = ttnn.reshape(
        v_16,
        [1, 2, 1, 17, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_1 = ttnn.repeat(
        ttnn_reshape_6,
        ttnn.Shape([1, 1, 8, 1, 1]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_6, False)
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_repeat_1,
        [1, 16, 17, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_slice_14,
        ttnn_reshape_7,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_7, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_matmul_1,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
        ttnn_matmul_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_1, False)
    ttnn_reshape_8 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_0,
        [17, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_0, False)
    ttnn_matmul_2 = ttnn.matmul(
        ttnn_reshape_8,
        ce_cache__main["main_const_eval_12"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_8, False)
    ttnn_reshape_9 = ttnn.reshape(
        ttnn_matmul_2,
        [1, 1, 17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_2, False)
    ttnn_all_reduce_0 = ttnn.all_reduce(
        input_tensor=ttnn_reshape_9,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_9, False)
    ttnn_reshape_10 = ttnn.reshape(
        ttnn_all_reduce_0,
        [17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_reduce_0, False)
    ttnn_add_2 = ttnn.add(
        ttnn_reshape_10,
        ce_cache__main["main_const_eval_23"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_10, False)
    ttnn_add_3 = ttnn.add(
        ttnn_embedding_0,
        ttnn_add_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_2, False)
    ttnn.deallocate(ttnn_embedding_0, False)
    ttnn_reshape_11 = ttnn.reshape(
        ttnn_add_3,
        [1, 17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_3, False)
    ttnn_typecast_59 = ttnn.typecast(
        ttnn_reshape_11,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_pow_1 = ttnn.pow(
        ttnn_typecast_59,
        2.0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_59, False)
    ttnn_mean_1 = ttnn.mean(
        ttnn_pow_1,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_pow_1, False)
    ttnn_add_4 = ttnn.add(
        ttnn_mean_1,
        var_13,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_1, False)
    ttnn_rsqrt_1 = ttnn.rsqrt(
        ttnn_add_4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_4, False)
    ttnn_rms_norm_1 = ttnn.rms_norm(
        ttnn_reshape_11,
        epsilon=9.9999997473787516e-06,
        weight=var_2,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_reshape_12 = ttnn.reshape(
        ttnn_rms_norm_1,
        [17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_concat_5 = ttnn.concat(
        [
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
            ttnn_reshape_12,
        ],
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_13 = ttnn.reshape(
        ttnn_concat_5,
        [32, 17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_3 = ttnn.matmul(
        ttnn_reshape_13,
        ce_cache__main["main_const_eval_31"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_13, False)
    ttnn_add_5 = ttnn.add(
        ttnn_matmul_3,
        ce_cache__main["main_const_eval_10"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_15 = ttnn.slice(
        ttnn_add_5,
        [0, 0, 1],
        [32, 17, 5760],
        [1, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_clamp_0 = ttnn.clamp(
        ttnn_slice_15,
        -7.0,
        7.0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_15, False)
    ttnn_add_6 = ttnn.add(
        ttnn_clamp_0,
        var_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_clamp_0, False)
    ttnn_slice_16 = ttnn.slice(
        ttnn_add_5,
        [0, 0, 0],
        [32, 17, 5760],
        [1, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_5, False)
    ttnn_clamp_1 = ttnn.clamp(
        ttnn_slice_16,
        float("-inf"),
        7.0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_16, False)
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_clamp_1,
        var_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_0 = ttnn.sigmoid(
        ttnn_multiply_1,
        vector_mode=4,
        mode=ttnn.SigmoidMode.Accurate,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_1, False)
    ttnn_multiply_2 = ttnn.multiply(
        ttnn_clamp_1,
        ttnn_sigmoid_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_clamp_1, False)
    ttnn_multiply_3 = ttnn.multiply(
        ttnn_add_6,
        ttnn_multiply_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_2, False)
    ttnn.deallocate(ttnn_add_6, False)
    ttnn_matmul_4 = ttnn.matmul(
        ttnn_multiply_3,
        ce_cache__main["main_const_eval_26"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_add_7 = ttnn.add(
        ttnn_matmul_4,
        ce_cache__main["main_const_eval_15"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_60 = ttnn.typecast(
        ttnn_reshape_12,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_12, False)
    ttnn_linear_1 = ttnn.linear(
        ttnn_typecast_60,
        ce_cache__main["main_const_eval_27"],
        bias=ce_cache__main["main_const_eval_28"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.FLOAT32,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_typecast_60, False)
    ttnn_typecast_61 = ttnn.typecast(
        ttnn_linear_1,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_1, False)
    v_18, v_19 = ttnn.topk(
        ttnn_typecast_61,
        4,
        1,
        True,
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_18, False)
    ttnn_typecast_62 = ttnn.typecast(
        v_19,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_19, False)
    v_20, v_21 = ttnn.topk(
        ttnn_typecast_61,
        4,
        -1,
        True,
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_61, False)
    ttnn_typecast_63 = ttnn.typecast(
        v_21,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_21, False)
    ttnn_softmax_1 = ttnn.softmax(
        v_20,
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
        numeric_stable=True,
    )
    ttnn.deallocate(v_20, False)
    ttnn_reshape_14 = ttnn.reshape(
        ttnn_typecast_63,
        [17, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_8 = ttnn.add(
        var_11,
        ttnn_reshape_14,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_14, False)
    ttnn_reshape_15 = ttnn.reshape(
        ttnn_add_8,
        [68],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_8, False)
    ttnn_reshape_16 = ttnn.reshape(
        ttnn_softmax_1,
        [68],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_39 = ttnn.to_layout(
        ttnn_reshape_15,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_15, False)
    ttnn_to_layout_40 = ttnn.to_layout(
        ttnn_reshape_16,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_16, False)
    ttnn_scatter_0 = ttnn.scatter(
        input=var_5,
        dim=0,
        index=ttnn_to_layout_39,
        src=ttnn_to_layout_40,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        reduce=None,
    )
    ttnn.deallocate(ttnn_to_layout_40, False)
    ttnn.deallocate(ttnn_to_layout_39, False)
    ttnn_to_layout_41 = ttnn.to_layout(
        ttnn_scatter_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_scatter_0, False)
    ttnn_reshape_17 = ttnn.reshape(
        ttnn_to_layout_41,
        [17, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_41, False)
    ttnn_permute_2 = ttnn.permute(
        ttnn_reshape_17,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_18 = ttnn.reshape(
        ttnn_permute_2,
        [32, 17, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_2, False)
    ttnn_multiply_4 = ttnn.multiply(
        ttnn_add_7,
        ttnn_reshape_18,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_18, False)
    ttnn.deallocate(ttnn_add_7, False)
    ttnn_sum_0 = ttnn.sum(
        ttnn_multiply_4,
        [0],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_4, False)
    ttnn_add_9 = ttnn.add(
        ttnn_reshape_11,
        ttnn_sum_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_sum_0, False)
    ttnn_typecast_64 = ttnn.typecast(
        ttnn_add_9,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_pow_2 = ttnn.pow(
        ttnn_typecast_64,
        2.0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_64, False)
    ttnn_mean_2 = ttnn.mean(
        ttnn_pow_2,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_pow_2, False)
    ttnn_add_10 = ttnn.add(
        ttnn_mean_2,
        var_13,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_2, False)
    ttnn_rsqrt_2 = ttnn.rsqrt(
        ttnn_add_10,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_10, False)
    ttnn_rms_norm_2 = ttnn.rms_norm(
        ttnn_add_9,
        epsilon=9.9999997473787516e-06,
        weight=var_1,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_reshape_19 = ttnn.reshape(
        ttnn_rms_norm_2,
        [17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_linear_2 = ttnn.linear(
        ttnn_reshape_19,
        ce_cache__main["main_const_eval_8"],
        bias=ce_cache__main["main_const_eval_30"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_19, False)
    ttnn_reshape_20 = ttnn.reshape(
        ttnn_linear_2,
        [1, 17, 1280],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_2, False)
    v_22, v_23, v_24 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_20,
        None,
        num_heads=16,
        num_kv_heads=2,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_20, False)
    ttnn_experimental_rotary_embedding_2 = ttnn.experimental.rotary_embedding(
        v_24,
        var_8,
        var_9,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(v_24, False)
    ttnn_slice_17 = ttnn.slice(
        ttnn_experimental_rotary_embedding_2,
        [0, 0, 0, 0],
        [1, 2, 17, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_2, False)
    ttnn_experimental_rotary_embedding_3 = ttnn.experimental.rotary_embedding(
        v_22,
        var_8,
        var_9,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(v_22, False)
    ttnn_slice_18 = ttnn.slice(
        ttnn_experimental_rotary_embedding_3,
        [0, 0, 0, 0],
        [1, 16, 17, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_3, False)
    ttnn_reshape_21 = ttnn.reshape(
        ttnn_slice_17,
        [1, 2, 1, 17, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_2 = ttnn.repeat(
        ttnn_reshape_21,
        ttnn.Shape([1, 1, 8, 1, 1]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_21, False)
    ttnn_reshape_22 = ttnn.reshape(
        ttnn_repeat_2,
        [1, 16, 17, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_permute_3 = ttnn.permute(
        ttnn_reshape_22,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_22, False)
    ttnn_matmul_5 = ttnn.matmul(
        ttnn_slice_18,
        ttnn_permute_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_permute_3, False)
    ttnn_multiply_5 = ttnn.multiply(
        ttnn_matmul_5,
        var_12,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_5, False)
    ttnn_add_11 = ttnn.add(
        ttnn_multiply_5,
        var_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_5, False)
    ttnn_concat_6 = ttnn.concat(
        [ttnn_add_11, ce_cache__main["main_const_eval_21"]],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_11, False)
    ttnn_softmax_2 = ttnn.softmax(
        ttnn_concat_6,
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
        numeric_stable=True,
    )
    ttnn_slice_19 = ttnn.slice(
        ttnn_softmax_2,
        [0, 0, 0, 0],
        [1, 16, 17, 17],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_23 = ttnn.reshape(
        v_23,
        [1, 2, 1, 17, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_repeat_3 = ttnn.repeat(
        ttnn_reshape_23,
        ttnn.Shape([1, 1, 8, 1, 1]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_23, False)
    ttnn_reshape_24 = ttnn.reshape(
        ttnn_repeat_3,
        [1, 16, 17, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_6 = ttnn.matmul(
        ttnn_slice_19,
        ttnn_reshape_24,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_24, False)
    ttnn_permute_4 = ttnn.permute(
        ttnn_matmul_6,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
        ttnn_matmul_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_6, False)
    ttnn_reshape_25 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_1,
        [17, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_concatenate_heads_1, False)
    ttnn_matmul_7 = ttnn.matmul(
        ttnn_reshape_25,
        ce_cache__main["main_const_eval_1"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_25, False)
    ttnn_reshape_26 = ttnn.reshape(
        ttnn_matmul_7,
        [1, 1, 17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_7, False)
    ttnn_all_reduce_1 = ttnn.all_reduce(
        input_tensor=ttnn_reshape_26,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_26, False)
    ttnn_reshape_27 = ttnn.reshape(
        ttnn_all_reduce_1,
        [17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_reduce_1, False)
    ttnn_add_12 = ttnn.add(
        ttnn_reshape_27,
        ce_cache__main["main_const_eval_24"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_27, False)
    ttnn_reshape_28 = ttnn.reshape(
        ttnn_add_12,
        [1, 17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_12, False)
    ttnn_add_13 = ttnn.add(
        ttnn_add_9,
        ttnn_reshape_28,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_28, False)
    ttnn_typecast_65 = ttnn.typecast(
        ttnn_add_13,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_pow_3 = ttnn.pow(
        ttnn_typecast_65,
        2.0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_65, False)
    ttnn_mean_3 = ttnn.mean(
        ttnn_pow_3,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_pow_3, False)
    ttnn_add_14 = ttnn.add(
        ttnn_mean_3,
        var_13,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_3, False)
    ttnn_rsqrt_3 = ttnn.rsqrt(
        ttnn_add_14,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_14, False)
    ttnn_rms_norm_3 = ttnn.rms_norm(
        ttnn_add_13,
        epsilon=9.9999997473787516e-06,
        weight=var_4,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_reshape_29 = ttnn.reshape(
        ttnn_rms_norm_3,
        [17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_concat_7 = ttnn.concat(
        [
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
            ttnn_reshape_29,
        ],
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_30 = ttnn.reshape(
        ttnn_concat_7,
        [32, 17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_8 = ttnn.matmul(
        ttnn_reshape_30,
        ce_cache__main["main_const_eval_16"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_30, False)
    ttnn_add_15 = ttnn.add(
        ttnn_matmul_8,
        ce_cache__main["main_const_eval_19"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_20 = ttnn.slice(
        ttnn_add_15,
        [0, 0, 1],
        [32, 17, 5760],
        [1, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_clamp_2 = ttnn.clamp(
        ttnn_slice_20,
        -7.0,
        7.0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_20, False)
    ttnn_add_16 = ttnn.add(
        ttnn_clamp_2,
        var_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_clamp_2, False)
    ttnn_slice_21 = ttnn.slice(
        ttnn_add_15,
        [0, 0, 0],
        [32, 17, 5760],
        [1, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_15, False)
    ttnn_clamp_3 = ttnn.clamp(
        ttnn_slice_21,
        float("-inf"),
        7.0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_21, False)
    ttnn_multiply_6 = ttnn.multiply(
        ttnn_clamp_3,
        var_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sigmoid_1 = ttnn.sigmoid(
        ttnn_multiply_6,
        vector_mode=4,
        mode=ttnn.SigmoidMode.Accurate,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_6, False)
    ttnn_multiply_7 = ttnn.multiply(
        ttnn_clamp_3,
        ttnn_sigmoid_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_clamp_3, False)
    ttnn_multiply_8 = ttnn.multiply(
        ttnn_add_16,
        ttnn_multiply_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_multiply_7, False)
    ttnn.deallocate(ttnn_add_16, False)
    ttnn_matmul_9 = ttnn.matmul(
        ttnn_multiply_8,
        ce_cache__main["main_const_eval_3"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn_add_17 = ttnn.add(
        ttnn_matmul_9,
        ce_cache__main["main_const_eval_17"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_66 = ttnn.typecast(
        ttnn_reshape_29,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_29, False)
    ttnn_linear_3 = ttnn.linear(
        ttnn_typecast_66,
        ce_cache__main["main_const_eval_7"],
        bias=ce_cache__main["main_const_eval_32"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.FLOAT32,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_typecast_66, False)
    ttnn_typecast_67 = ttnn.typecast(
        ttnn_linear_3,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_3, False)
    v_25, v_26 = ttnn.topk(
        ttnn_typecast_67,
        4,
        1,
        True,
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_25, False)
    ttnn_typecast_68 = ttnn.typecast(
        v_26,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_26, False)
    v_27, v_28 = ttnn.topk(
        ttnn_typecast_67,
        4,
        -1,
        True,
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_67, False)
    ttnn_typecast_69 = ttnn.typecast(
        v_28,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_28, False)
    ttnn_softmax_3 = ttnn.softmax(
        v_27,
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
        numeric_stable=True,
    )
    ttnn.deallocate(v_27, False)
    ttnn_reshape_31 = ttnn.reshape(
        ttnn_typecast_69,
        [17, 4, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_add_18 = ttnn.add(
        var_11,
        ttnn_reshape_31,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_31, False)
    ttnn_reshape_32 = ttnn.reshape(
        ttnn_add_18,
        [68],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_18, False)
    ttnn_reshape_33 = ttnn.reshape(
        ttnn_softmax_3,
        [68],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_42 = ttnn.to_layout(
        ttnn_reshape_32,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_32, False)
    ttnn_to_layout_43 = ttnn.to_layout(
        ttnn_reshape_33,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_33, False)
    ttnn_scatter_1 = ttnn.scatter(
        input=var_5,
        dim=0,
        index=ttnn_to_layout_42,
        src=ttnn_to_layout_43,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        reduce=None,
    )
    ttnn.deallocate(ttnn_to_layout_43, False)
    ttnn.deallocate(ttnn_to_layout_42, False)
    ttnn_to_layout_44 = ttnn.to_layout(
        ttnn_scatter_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_scatter_1, False)
    ttnn_reshape_34 = ttnn.reshape(
        ttnn_to_layout_44,
        [17, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_44, False)
    ttnn_permute_5 = ttnn.permute(
        ttnn_reshape_34,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_35 = ttnn.reshape(
        ttnn_permute_5,
        [32, 17, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_5, False)
    ttnn_multiply_9 = ttnn.multiply(
        ttnn_add_17,
        ttnn_reshape_35,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_35, False)
    ttnn.deallocate(ttnn_add_17, False)
    ttnn_sum_1 = ttnn.sum(
        ttnn_multiply_9,
        [0],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_9, False)
    ttnn_add_19 = ttnn.add(
        ttnn_add_13,
        ttnn_sum_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_sum_1, False)
    ttnn_typecast_70 = ttnn.typecast(
        ttnn_add_19,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_pow_4 = ttnn.pow(
        ttnn_typecast_70,
        2.0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_70, False)
    ttnn_mean_4 = ttnn.mean(
        ttnn_pow_4,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_pow_4, False)
    ttnn_add_20 = ttnn.add(
        ttnn_mean_4,
        var_13,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mean_4, False)
    ttnn_rsqrt_4 = ttnn.rsqrt(
        ttnn_add_20,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_20, False)
    ttnn_rms_norm_4 = ttnn.rms_norm(
        ttnn_add_19,
        epsilon=9.9999997473787516e-06,
        weight=var_3,
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn_reshape_36 = ttnn.reshape(
        ttnn_rms_norm_4,
        [17, 2880],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_10 = ttnn.matmul(
        ttnn_reshape_36,
        ce_cache__main["main_const_eval_9"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_36, False)
    ttnn_reshape_37 = ttnn.reshape(
        ttnn_matmul_10,
        [1, 17, 201088],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_10, False)
    ttnn_to_layout_45 = ttnn.to_layout(
        ttnn_concat_4,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_4, False)
    ttnn_argmax_0 = ttnn.argmax(
        ttnn_to_layout_45,
        3,
        True,
        sub_core_grids=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_45, False)
    ttnn_to_layout_46 = ttnn.to_layout(
        ttnn_argmax_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_argmax_0, False)
    ttnn_typecast_71 = ttnn.typecast(
        ttnn_to_layout_46,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_46, False)
    ttnn_to_layout_47 = ttnn.to_layout(
        ttnn_concat_6,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_6, False)
    ttnn_argmax_1 = ttnn.argmax(
        ttnn_to_layout_47,
        3,
        True,
        sub_core_grids=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_47, False)
    ttnn_to_layout_48 = ttnn.to_layout(
        ttnn_argmax_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_argmax_1, False)
    ttnn_typecast_72 = ttnn.typecast(
        ttnn_to_layout_48,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_48, False)
    ttnn_to_layout_49 = ttnn.to_layout(
        primals_39,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(primals_39, False)
    return [
        v_16,
        ttnn_slice_12,
        v_23,
        ttnn_slice_17,
        ttnn_reshape_37,
        weights["model.layers.0.mlp.experts.gate_up_proj"],
        weights["model.layers.0.mlp.experts.gate_up_proj_bias"],
        weights["model.layers.0.mlp.experts.down_proj"],
        weights["model.layers.0.mlp.experts.down_proj_bias"],
        weights["model.layers.1.mlp.experts.gate_up_proj"],
        weights["model.layers.1.mlp.experts.gate_up_proj_bias"],
        weights["model.layers.1.mlp.experts.down_proj"],
        weights["model.layers.1.mlp.experts.down_proj_bias"],
        ttnn_to_layout_49,
        var_7[2],
        ttnn_reshape_1,
        var_0,
        ttnn_rsqrt_0,
        ttnn_rms_norm_0,
        weights["model.layers.0.self_attn.q_proj.parametrizations.weight.original"],
        weights["model.layers.0.self_attn.k_proj.parametrizations.weight.original"],
        weights["model.layers.0.self_attn.v_proj.parametrizations.weight.original"],
        ttnn_slice_13,
        ttnn_repeat_0,
        ttnn_repeat_1,
        ttnn_typecast_71,
        ttnn_slice_14,
        ttnn_permute_1,
        weights["model.layers.0.self_attn.o_proj.parametrizations.weight.original"],
        ttnn_reshape_11,
        var_2,
        ttnn_rsqrt_1,
        ttnn_rms_norm_1,
        weights["model.layers.0.mlp.router.parametrizations.weight.original"],
        ttnn_typecast_62,
        ttnn_typecast_63,
        ce_cache__main["main_const_eval_22"],
        ttnn_reshape_17,
        ttnn_concat_5,
        ttnn_matmul_3,
        ttnn_sigmoid_0,
        ttnn_multiply_3,
        ttnn_matmul_4,
        ttnn_add_9,
        var_1,
        ttnn_rsqrt_2,
        ttnn_rms_norm_2,
        weights["model.layers.1.self_attn.q_proj.parametrizations.weight.original"],
        weights["model.layers.1.self_attn.k_proj.parametrizations.weight.original"],
        weights["model.layers.1.self_attn.v_proj.parametrizations.weight.original"],
        ttnn_slice_18,
        ttnn_repeat_2,
        ttnn_repeat_3,
        ttnn_typecast_72,
        ttnn_slice_19,
        ttnn_permute_4,
        weights["model.layers.1.self_attn.o_proj.parametrizations.weight.original"],
        ttnn_add_13,
        var_4,
        ttnn_rsqrt_3,
        ttnn_rms_norm_3,
        weights["model.layers.1.mlp.router.parametrizations.weight.original"],
        ttnn_typecast_68,
        ttnn_typecast_69,
        ttnn_reshape_34,
        ttnn_concat_7,
        ttnn_matmul_8,
        ttnn_sigmoid_1,
        ttnn_multiply_8,
        ttnn_matmul_9,
        ttnn_add_19,
        var_3,
        ttnn_rsqrt_4,
        ttnn_rms_norm_4,
        weights["lm_head.parametrizations.weight.original"],
        ttnn_softmax_3,
        ttnn_softmax_2,
        ttnn_softmax_1,
        ttnn_softmax_0,
    ]


def consteval__main(ce_cache, weights):
    if not ce_cache:
        main_const_eval_0_0 = main_const_eval_0(
            [weights["model.embed_tokens.parametrizations.weight.original"]]
        )
        ce_cache["main_const_eval_0"] = main_const_eval_0_0[0]
        main_const_eval_1_0 = main_const_eval_1(
            [
                weights[
                    "model.layers.1.self_attn.o_proj.parametrizations.weight.original"
                ]
            ]
        )
        ce_cache["main_const_eval_1"] = main_const_eval_1_0[0]
        main_const_eval_2_0 = main_const_eval_2(
            [
                weights[
                    "model.layers.0.self_attn.v_proj.parametrizations.weight.original"
                ],
                weights[
                    "model.layers.0.self_attn.k_proj.parametrizations.weight.original"
                ],
                weights[
                    "model.layers.0.self_attn.q_proj.parametrizations.weight.original"
                ],
            ]
        )
        ce_cache["main_const_eval_2"] = main_const_eval_2_0[0]
        main_const_eval_3_0 = main_const_eval_3(
            [weights["model.layers.1.mlp.experts.down_proj"]]
        )
        ce_cache["main_const_eval_3"] = main_const_eval_3_0[0]
        main_const_eval_4_0 = main_const_eval_4()
        ce_cache["main_const_eval_4"] = main_const_eval_4_0[0]
        main_const_eval_5_0 = main_const_eval_5()
        ce_cache["main_const_eval_5"] = main_const_eval_5_0[0]
        main_const_eval_6_0 = main_const_eval_6([weights["model.rotary_emb.inv_freq"]])
        ce_cache["main_const_eval_6"] = [
            main_const_eval_6_0[0],
            main_const_eval_6_0[1],
            main_const_eval_6_0[2],
        ]
        main_const_eval_7_0 = main_const_eval_7(
            [weights["model.layers.1.mlp.router.parametrizations.weight.original"]]
        )
        ce_cache["main_const_eval_7"] = main_const_eval_7_0[0]
        main_const_eval_8_0 = main_const_eval_8(
            [
                weights[
                    "model.layers.1.self_attn.v_proj.parametrizations.weight.original"
                ],
                weights[
                    "model.layers.1.self_attn.k_proj.parametrizations.weight.original"
                ],
                weights[
                    "model.layers.1.self_attn.q_proj.parametrizations.weight.original"
                ],
            ]
        )
        ce_cache["main_const_eval_8"] = main_const_eval_8_0[0]
        main_const_eval_9_0 = main_const_eval_9(
            [weights["lm_head.parametrizations.weight.original"]]
        )
        ce_cache["main_const_eval_9"] = main_const_eval_9_0[0]
        main_const_eval_10_0 = main_const_eval_10(
            [weights["model.layers.0.mlp.experts.gate_up_proj_bias"]]
        )
        ce_cache["main_const_eval_10"] = main_const_eval_10_0[0]
        main_const_eval_11_0 = main_const_eval_11()
        ce_cache["main_const_eval_11"] = main_const_eval_11_0[0]
        main_const_eval_12_0 = main_const_eval_12(
            [
                weights[
                    "model.layers.0.self_attn.o_proj.parametrizations.weight.original"
                ]
            ]
        )
        ce_cache["main_const_eval_12"] = main_const_eval_12_0[0]
        main_const_eval_13_0 = main_const_eval_13()
        ce_cache["main_const_eval_13"] = main_const_eval_13_0[0]
        main_const_eval_14_0 = main_const_eval_14()
        ce_cache["main_const_eval_14"] = main_const_eval_14_0[0]
        main_const_eval_15_0 = main_const_eval_15(
            [weights["model.layers.0.mlp.experts.down_proj_bias"]]
        )
        ce_cache["main_const_eval_15"] = main_const_eval_15_0[0]
        main_const_eval_16_0 = main_const_eval_16(
            [weights["model.layers.1.mlp.experts.gate_up_proj"]]
        )
        ce_cache["main_const_eval_16"] = main_const_eval_16_0[0]
        main_const_eval_17_0 = main_const_eval_17(
            [weights["model.layers.1.mlp.experts.down_proj_bias"]]
        )
        ce_cache["main_const_eval_17"] = main_const_eval_17_0[0]
        main_const_eval_18_0 = main_const_eval_18()
        ce_cache["main_const_eval_18"] = main_const_eval_18_0[0]
        main_const_eval_19_0 = main_const_eval_19(
            [weights["model.layers.1.mlp.experts.gate_up_proj_bias"]]
        )
        ce_cache["main_const_eval_19"] = main_const_eval_19_0[0]
        main_const_eval_20_0 = main_const_eval_20(
            [
                weights["model.layers.0.self_attn.v_proj.bias"],
                weights["model.layers.0.self_attn.k_proj.bias"],
                weights["model.layers.0.self_attn.q_proj.bias"],
            ]
        )
        ce_cache["main_const_eval_20"] = main_const_eval_20_0[0]
        main_const_eval_21_0 = main_const_eval_21(
            [weights["model.layers.1.self_attn.sinks"]]
        )
        ce_cache["main_const_eval_21"] = main_const_eval_21_0[0]
        main_const_eval_22_0 = main_const_eval_22()
        ce_cache["main_const_eval_22"] = main_const_eval_22_0[0]
        main_const_eval_23_0 = main_const_eval_23(
            [weights["model.layers.0.self_attn.o_proj.bias"]]
        )
        ce_cache["main_const_eval_23"] = main_const_eval_23_0[0]
        main_const_eval_24_0 = main_const_eval_24(
            [weights["model.layers.1.self_attn.o_proj.bias"]]
        )
        ce_cache["main_const_eval_24"] = main_const_eval_24_0[0]
        main_const_eval_25_0 = main_const_eval_25()
        ce_cache["main_const_eval_25"] = main_const_eval_25_0[0]
        main_const_eval_26_0 = main_const_eval_26(
            [weights["model.layers.0.mlp.experts.down_proj"]]
        )
        ce_cache["main_const_eval_26"] = main_const_eval_26_0[0]
        main_const_eval_27_0 = main_const_eval_27(
            [weights["model.layers.0.mlp.router.parametrizations.weight.original"]]
        )
        ce_cache["main_const_eval_27"] = main_const_eval_27_0[0]
        main_const_eval_28_0 = main_const_eval_28(
            [weights["model.layers.0.mlp.router.bias"]]
        )
        ce_cache["main_const_eval_28"] = main_const_eval_28_0[0]
        main_const_eval_29_0 = main_const_eval_29(
            [weights["model.layers.0.self_attn.sinks"]]
        )
        ce_cache["main_const_eval_29"] = main_const_eval_29_0[0]
        main_const_eval_30_0 = main_const_eval_30(
            [
                weights["model.layers.1.self_attn.v_proj.bias"],
                weights["model.layers.1.self_attn.k_proj.bias"],
                weights["model.layers.1.self_attn.q_proj.bias"],
            ]
        )
        ce_cache["main_const_eval_30"] = main_const_eval_30_0[0]
        main_const_eval_31_0 = main_const_eval_31(
            [weights["model.layers.0.mlp.experts.gate_up_proj"]]
        )
        ce_cache["main_const_eval_31"] = main_const_eval_31_0[0]
        main_const_eval_32_0 = main_const_eval_32(
            [weights["model.layers.1.mlp.router.bias"]]
        )
        ce_cache["main_const_eval_32"] = main_const_eval_32_0[0]
    return ce_cache


def load_activations_for__main():
    utils_DeviceGetter_get_device_33 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_33,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [utils_load_tensor_0]


_main_weights = {}


def load_weights_for__main():
    utils_DeviceGetter_get_device_34 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    global _main_weights
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.self_attn.v_proj.bias"] = utils_load_tensor_1
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.0.self_attn.v_proj.parametrizations.weight.original"
    ] = utils_load_tensor_2
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.input_layernorm.parametrizations.weight.original"] = (
        utils_load_tensor_3
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.embed_tokens.parametrizations.weight.original"] = (
        utils_load_tensor_4
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.rotary_emb.inv_freq"] = utils_load_tensor_5
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.self_attn.k_proj.bias"] = utils_load_tensor_6
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg7.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.0.self_attn.k_proj.parametrizations.weight.original"
    ] = utils_load_tensor_7
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg8.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.self_attn.v_proj.bias"] = utils_load_tensor_8
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg9.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.1.self_attn.v_proj.parametrizations.weight.original"
    ] = utils_load_tensor_9
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg10.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.input_layernorm.parametrizations.weight.original"] = (
        utils_load_tensor_10
    )
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg11.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.mlp.router.bias"] = utils_load_tensor_11
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg12.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.mlp.router.parametrizations.weight.original"] = (
        utils_load_tensor_12
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg13.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.0.post_attention_layernorm.parametrizations.weight.original"
    ] = utils_load_tensor_13
    utils_load_tensor_14 = utils.load_tensor(
        "./tensors/arg14.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.self_attn.o_proj.bias"] = utils_load_tensor_14
    utils_load_tensor_15 = utils.load_tensor(
        "./tensors/arg15.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.0.self_attn.o_proj.parametrizations.weight.original"
    ] = utils_load_tensor_15
    utils_load_tensor_16 = utils.load_tensor(
        "./tensors/arg16.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.self_attn.sinks"] = utils_load_tensor_16
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.0.self_attn.q_proj.bias"] = utils_load_tensor_17
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg18.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.0.self_attn.q_proj.parametrizations.weight.original"
    ] = utils_load_tensor_18
    utils_load_tensor_19 = utils.load_tensor(
        "./tensors/arg19.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.mlp.experts.down_proj_bias"] = utils_load_tensor_19
    utils_load_tensor_20 = utils.load_tensor(
        "./tensors/arg20.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.mlp.experts.down_proj"] = utils_load_tensor_20
    utils_load_tensor_21 = utils.load_tensor(
        "./tensors/arg21.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.mlp.experts.gate_up_proj_bias"] = utils_load_tensor_21
    utils_load_tensor_22 = utils.load_tensor(
        "./tensors/arg22.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.0.mlp.experts.gate_up_proj"] = utils_load_tensor_22
    utils_load_tensor_23 = utils.load_tensor(
        "./tensors/arg23.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.self_attn.k_proj.bias"] = utils_load_tensor_23
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg24.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.1.self_attn.k_proj.parametrizations.weight.original"
    ] = utils_load_tensor_24
    utils_load_tensor_25 = utils.load_tensor(
        "./tensors/arg25.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["lm_head.parametrizations.weight.original"] = utils_load_tensor_25
    utils_load_tensor_26 = utils.load_tensor(
        "./tensors/arg26.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.norm.parametrizations.weight.original"] = utils_load_tensor_26
    utils_load_tensor_27 = utils.load_tensor(
        "./tensors/arg27.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.mlp.router.bias"] = utils_load_tensor_27
    utils_load_tensor_28 = utils.load_tensor(
        "./tensors/arg28.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.mlp.router.parametrizations.weight.original"] = (
        utils_load_tensor_28
    )
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.1.post_attention_layernorm.parametrizations.weight.original"
    ] = utils_load_tensor_29
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg30.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.self_attn.o_proj.bias"] = utils_load_tensor_30
    utils_load_tensor_31 = utils.load_tensor(
        "./tensors/arg31.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.1.self_attn.o_proj.parametrizations.weight.original"
    ] = utils_load_tensor_31
    utils_load_tensor_32 = utils.load_tensor(
        "./tensors/arg32.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.self_attn.sinks"] = utils_load_tensor_32
    utils_load_tensor_33 = utils.load_tensor(
        "./tensors/arg33.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.layers.1.self_attn.q_proj.bias"] = utils_load_tensor_33
    utils_load_tensor_34 = utils.load_tensor(
        "./tensors/arg34.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights[
        "model.layers.1.self_attn.q_proj.parametrizations.weight.original"
    ] = utils_load_tensor_34
    utils_load_tensor_35 = utils.load_tensor(
        "./tensors/arg35.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.mlp.experts.down_proj_bias"] = utils_load_tensor_35
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg36.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.mlp.experts.down_proj"] = utils_load_tensor_36
    utils_load_tensor_37 = utils.load_tensor(
        "./tensors/arg37.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.mlp.experts.gate_up_proj_bias"] = utils_load_tensor_37
    utils_load_tensor_38 = utils.load_tensor(
        "./tensors/arg38.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_34,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["model.layers.1.mlp.experts.gate_up_proj"] = utils_load_tensor_38
    return _main_weights


def main():
    load_activations_for__main_0 = load_activations_for__main()
    load_weights_for__main_0 = load_weights_for__main()
    _main_0 = _main(load_activations_for__main_0, load_weights_for__main_0)
    return 0


def test_main():
    import model_pt

    exact_pcc = 0.98

    def to_host_torch(tensor):
        tensor = ttnn.from_device(tensor)
        if utils.DeviceGetter._instance is not None:
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
        return ttnn.to_torch(tensor)

    pt_input = model_pt.load_input()
    device = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    ttnn_input = ttnn.from_torch(
        pt_input["input_ids"], dtype=ttnn.DataType.INT32, layout=ttnn.Layout.ROW_MAJOR
    )
    ttnn_input = ttnn.to_device(
        ttnn_input,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )

    weights = load_weights_for__main()
    outputs = _main([ttnn_input], weights)

    ttnn_output = to_host_torch(outputs[4])[:, -1]
    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output.to(torch.float32), golden_output.to(torch.float32))
    print(f"\nPCC: {pcc:.6f}")
    assert pcc > exact_pcc, f"PCC {pcc} is below expected {exact_pcc}"


if __name__ == "__main__":
    main()
