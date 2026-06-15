import ttnn
import utils
from utils import calculate_pcc

from model_ttnn import _main
from params import load_weights_for__main_from_state_dict


def main_const_eval_0(arg_0, device):
    ttnn_to_device_0 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_to_device_0]


def main_const_eval_1(arg_0, device):
    ttnn_to_device_1 = ttnn.to_device(
        arg_0[2],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_to_device_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_1, False)
    ttnn_to_device_2 = ttnn.to_device(
        arg_0[1],
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
    ttnn_to_device_3 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_to_device_3,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_3, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_0,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_to_layout_2,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_permute_2 = ttnn.permute(
        ttnn_to_layout_1,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_concat_0 = ttnn.concat(
        [ttnn_permute_0, ttnn_permute_1, ttnn_permute_2],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_2, False)
    ttnn.deallocate(ttnn_permute_1, False)
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_from_device_0 = ttnn.from_device(ttnn_concat_0)
    ttnn.deallocate(ttnn_concat_0, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_from_device_0, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_0, False)
    ttnn_to_device_4 = ttnn.to_device(
        ttnn_typecast_0,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    return [ttnn_to_device_4]


def main_const_eval_2(arg_0, device):
    ttnn_to_device_5 = ttnn.to_device(
        arg_0[2],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_to_device_5,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_5, False)
    ttnn_to_device_6 = ttnn.to_device(
        arg_0[1],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_4 = ttnn.to_layout(
        ttnn_to_device_6,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_6, False)
    ttnn_to_device_7 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_to_device_7,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_7, False)
    ttnn_permute_3 = ttnn.permute(
        ttnn_to_layout_3,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn_permute_4 = ttnn.permute(
        ttnn_to_layout_5,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_5, False)
    ttnn_permute_5 = ttnn.permute(
        ttnn_to_layout_4,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    ttnn_concat_1 = ttnn.concat(
        [ttnn_permute_3, ttnn_permute_4, ttnn_permute_5],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_5, False)
    ttnn.deallocate(ttnn_permute_4, False)
    ttnn.deallocate(ttnn_permute_3, False)
    ttnn_from_device_1 = ttnn.from_device(ttnn_concat_1)
    ttnn.deallocate(ttnn_concat_1, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_from_device_1, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_1, False)
    ttnn_to_device_8 = ttnn.to_device(
        ttnn_typecast_1,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    return [ttnn_to_device_8]


def main_const_eval_3(arg_0, device):
    ttnn_to_device_9 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_6 = ttnn.to_layout(
        ttnn_to_device_9,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_9, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_to_layout_6,
        [1, 160],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_6, False)
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_reshape_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_0, False)
    return [ttnn_typecast_2]


def main_const_eval_4(arg_0, device):
    ttnn_to_device_10 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_7 = ttnn.to_layout(
        ttnn_to_device_10,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_10, False)
    ttnn_permute_6 = ttnn.permute(
        ttnn_to_layout_7,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_7, False)
    ttnn_from_device_2 = ttnn.from_device(ttnn_permute_6)
    ttnn.deallocate(ttnn_permute_6, False)
    ttnn_typecast_3 = ttnn.typecast(
        ttnn_from_device_2, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_2, False)
    ttnn_to_device_11 = ttnn.to_device(
        ttnn_typecast_3,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_3, False)
    return [ttnn_to_device_11]


def main_const_eval_5(device):
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=-3.3895313892515355e38,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_full_0]


def main_const_eval_6(device):
    ttnn_Tensor_0 = ttnn.Tensor(
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
    return [ttnn_Tensor_0]


def main_const_eval_7(device):
    ttnn_zeros_0 = ttnn.zeros(
        shape=ttnn.Shape([16, 1]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_all_gather_0 = ttnn.all_gather(
        input_tensor=ttnn_zeros_0,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_zeros_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_all_gather_0,
        [64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_0, False)
    ttnn_to_layout_8 = ttnn.to_layout(
        ttnn_reshape_1,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    return [ttnn_to_layout_8]


def main_const_eval_8(arg_0, device):
    ttnn_to_device_12 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_9 = ttnn.to_layout(
        ttnn_to_device_12,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_12, False)
    ttnn_permute_7 = ttnn.permute(
        ttnn_to_layout_9,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_9, False)
    ttnn_from_device_3 = ttnn.from_device(ttnn_permute_7)
    ttnn.deallocate(ttnn_permute_7, False)
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_from_device_3, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_3, False)
    ttnn_to_device_13 = ttnn.to_device(
        ttnn_typecast_4,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_4, False)
    return [ttnn_to_device_13]


def main_const_eval_9(arg_0, device):
    ttnn_to_device_14 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_10 = ttnn.to_layout(
        ttnn_to_device_14,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_14, False)
    ttnn_permute_8 = ttnn.permute(
        ttnn_to_layout_10,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_10, False)
    ttnn_from_device_4 = ttnn.from_device(ttnn_permute_8)
    ttnn.deallocate(ttnn_permute_8, False)
    ttnn_typecast_5 = ttnn.typecast(
        ttnn_from_device_4, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_4, False)
    ttnn_to_device_15 = ttnn.to_device(
        ttnn_typecast_5,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_5, False)
    return [ttnn_to_device_15]


def main_const_eval_10(device):
    ttnn_zeros_1 = ttnn.zeros(
        shape=ttnn.Shape([1, 1]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_zeros_1]


def main_const_eval_11(device):
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=1,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_full_1]


def main_const_eval_12(arg_0, device):
    ttnn_to_device_16 = ttnn.to_device(
        arg_0[2],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_11 = ttnn.to_layout(
        ttnn_to_device_16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_16, False)
    ttnn_to_device_17 = ttnn.to_device(
        arg_0[1],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_12 = ttnn.to_layout(
        ttnn_to_device_17,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_17, False)
    ttnn_to_device_18 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_13 = ttnn.to_layout(
        ttnn_to_device_18,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_18, False)
    ttnn_permute_9 = ttnn.permute(
        ttnn_to_layout_11,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_11, False)
    ttnn_permute_10 = ttnn.permute(
        ttnn_to_layout_13,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_13, False)
    ttnn_permute_11 = ttnn.permute(
        ttnn_to_layout_12,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_12, False)
    ttnn_concat_2 = ttnn.concat(
        [ttnn_permute_9, ttnn_permute_10, ttnn_permute_11],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_11, False)
    ttnn.deallocate(ttnn_permute_10, False)
    ttnn.deallocate(ttnn_permute_9, False)
    ttnn_from_device_5 = ttnn.from_device(ttnn_concat_2)
    ttnn.deallocate(ttnn_concat_2, False)
    ttnn_typecast_6 = ttnn.typecast(
        ttnn_from_device_5, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_5, False)
    ttnn_to_device_19 = ttnn.to_device(
        ttnn_typecast_6,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_6, False)
    return [ttnn_to_device_19]


def main_const_eval_13(arg_0, device):
    ttnn_to_device_20 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_14 = ttnn.to_layout(
        ttnn_to_device_20,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_20, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_14,
        [1, 5, 5120, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_14, False)
    ttnn_from_device_6 = ttnn.from_device(ttnn_reshape_2)
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_typecast_7 = ttnn.typecast(
        ttnn_from_device_6, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_6, False)
    ttnn_to_device_21 = ttnn.to_device(
        ttnn_typecast_7,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_7, False)
    return [ttnn_to_device_21]


def main_const_eval_14(arg_0, device):
    ttnn_to_device_22 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_15 = ttnn.to_layout(
        ttnn_to_device_22,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_22, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_to_layout_15,
        [1, 5, 5120, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_15, False)
    ttnn_from_device_7 = ttnn.from_device(ttnn_reshape_3)
    ttnn.deallocate(ttnn_reshape_3, False)
    ttnn_typecast_8 = ttnn.typecast(
        ttnn_from_device_7, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_7, False)
    ttnn_to_device_23 = ttnn.to_device(
        ttnn_typecast_8,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_8, False)
    return [ttnn_to_device_23]


def main_const_eval_15(arg_0, device):
    ttnn_to_device_24 = ttnn.to_device(
        arg_0[2],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_16 = ttnn.to_layout(
        ttnn_to_device_24,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_24, False)
    ttnn_to_device_25 = ttnn.to_device(
        arg_0[1],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_17 = ttnn.to_layout(
        ttnn_to_device_25,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_25, False)
    ttnn_to_device_26 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_18 = ttnn.to_layout(
        ttnn_to_device_26,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_26, False)
    ttnn_concat_3 = ttnn.concat(
        [ttnn_to_layout_16, ttnn_to_layout_18, ttnn_to_layout_17],
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_18, False)
    ttnn.deallocate(ttnn_to_layout_17, False)
    ttnn.deallocate(ttnn_to_layout_16, False)
    return [ttnn_concat_3]


def main_const_eval_16(arg_0, device):
    ttnn_to_device_27 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_19 = ttnn.to_layout(
        ttnn_to_device_27,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_27, False)
    ttnn_permute_12 = ttnn.permute(
        ttnn_to_layout_19,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_19, False)
    ttnn_from_device_8 = ttnn.from_device(ttnn_permute_12)
    ttnn.deallocate(ttnn_permute_12, False)
    ttnn_typecast_9 = ttnn.typecast(
        ttnn_from_device_8, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_8, False)
    ttnn_to_device_28 = ttnn.to_device(
        ttnn_typecast_9,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_9, False)
    return [ttnn_to_device_28]


def main_const_eval_17(arg_0, device):
    ttnn_to_device_29 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_20 = ttnn.to_layout(
        ttnn_to_device_29,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_29, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_to_layout_20,
        [1, 5, 1536, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_20, False)
    ttnn_from_device_9 = ttnn.from_device(ttnn_reshape_4)
    ttnn.deallocate(ttnn_reshape_4, False)
    ttnn_typecast_10 = ttnn.typecast(
        ttnn_from_device_9, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_9, False)
    ttnn_to_device_30 = ttnn.to_device(
        ttnn_typecast_10,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_10, False)
    return [ttnn_to_device_30]


def main_const_eval_18(arg_0, device):
    ttnn_to_device_31 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_21 = ttnn.to_layout(
        ttnn_to_device_31,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_31, False)
    ttnn_permute_13 = ttnn.permute(
        ttnn_to_layout_21,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_21, False)
    ttnn_from_device_10 = ttnn.from_device(ttnn_permute_13)
    ttnn.deallocate(ttnn_permute_13, False)
    ttnn_typecast_11 = ttnn.typecast(
        ttnn_from_device_10, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_10, False)
    ttnn_to_device_32 = ttnn.to_device(
        ttnn_typecast_11,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_11, False)
    return [ttnn_to_device_32]


def main_const_eval_19(arg_0, device):
    ttnn_to_device_33 = ttnn.to_device(
        arg_0[2],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_22 = ttnn.to_layout(
        ttnn_to_device_33,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_33, False)
    ttnn_to_device_34 = ttnn.to_device(
        arg_0[1],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_23 = ttnn.to_layout(
        ttnn_to_device_34,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_34, False)
    ttnn_to_device_35 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_24 = ttnn.to_layout(
        ttnn_to_device_35,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_35, False)
    ttnn_permute_14 = ttnn.permute(
        ttnn_to_layout_22,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_22, False)
    ttnn_permute_15 = ttnn.permute(
        ttnn_to_layout_24,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_24, False)
    ttnn_permute_16 = ttnn.permute(
        ttnn_to_layout_23,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_23, False)
    ttnn_concat_4 = ttnn.concat(
        [ttnn_permute_14, ttnn_permute_15, ttnn_permute_16],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_16, False)
    ttnn.deallocate(ttnn_permute_15, False)
    ttnn.deallocate(ttnn_permute_14, False)
    ttnn_from_device_11 = ttnn.from_device(ttnn_concat_4)
    ttnn.deallocate(ttnn_concat_4, False)
    ttnn_typecast_12 = ttnn.typecast(
        ttnn_from_device_11, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_11, False)
    ttnn_to_device_36 = ttnn.to_device(
        ttnn_typecast_12,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_12, False)
    return [ttnn_to_device_36]


def main_const_eval_20(arg_0, device):
    ttnn_to_device_37 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_25 = ttnn.to_layout(
        ttnn_to_device_37,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_37, False)
    ttnn_permute_17 = ttnn.permute(
        ttnn_to_layout_25,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_25, False)
    ttnn_from_device_12 = ttnn.from_device(ttnn_permute_17)
    ttnn.deallocate(ttnn_permute_17, False)
    ttnn_typecast_13 = ttnn.typecast(
        ttnn_from_device_12, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_12, False)
    ttnn_to_device_38 = ttnn.to_device(
        ttnn_typecast_13,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_13, False)
    return [ttnn_to_device_38]


def main_const_eval_21(device):
    ttnn_zeros_2 = ttnn.zeros(
        shape=ttnn.Shape([1, 1, 1, 1]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_zeros_2]


def main_const_eval_22(arg_0, device):
    ttnn_to_device_39 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_26 = ttnn.to_layout(
        ttnn_to_device_39,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_39, False)
    ttnn_permute_18 = ttnn.permute(
        ttnn_to_layout_26,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_26, False)
    ttnn_from_device_13 = ttnn.from_device(ttnn_permute_18)
    ttnn.deallocate(ttnn_permute_18, False)
    ttnn_typecast_14 = ttnn.typecast(
        ttnn_from_device_13, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_13, False)
    ttnn_to_device_40 = ttnn.to_device(
        ttnn_typecast_14,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_14, False)
    return [ttnn_to_device_40]


def main_const_eval_23(arg_0, device):
    ttnn_to_device_41 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_27 = ttnn.to_layout(
        ttnn_to_device_41,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_41, False)
    ttnn_permute_19 = ttnn.permute(
        ttnn_to_layout_27,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_27, False)
    ttnn_from_device_14 = ttnn.from_device(ttnn_permute_19)
    ttnn.deallocate(ttnn_permute_19, False)
    ttnn_typecast_15 = ttnn.typecast(
        ttnn_from_device_14, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_14, False)
    ttnn_to_device_42 = ttnn.to_device(
        ttnn_typecast_15,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_15, False)
    return [ttnn_to_device_42]


def main_const_eval_24(device):
    ttnn_arange_0 = ttnn.arange(
        0,
        16,
        1,
        dtype=ttnn.DataType.INT32,
        device=device,
        layout=ttnn.Layout.TILE,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_arange_0,
        [16, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_arange_0, False)
    return [ttnn_reshape_5]


def main_const_eval_25(arg_0, device):
    ttnn_to_device_43 = ttnn.to_device(
        arg_0[2],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_28 = ttnn.to_layout(
        ttnn_to_device_43,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_43, False)
    ttnn_to_device_44 = ttnn.to_device(
        arg_0[1],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_29 = ttnn.to_layout(
        ttnn_to_device_44,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_44, False)
    ttnn_to_device_45 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_30 = ttnn.to_layout(
        ttnn_to_device_45,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_45, False)
    ttnn_concat_5 = ttnn.concat(
        [ttnn_to_layout_28, ttnn_to_layout_30, ttnn_to_layout_29],
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_30, False)
    ttnn.deallocate(ttnn_to_layout_29, False)
    ttnn.deallocate(ttnn_to_layout_28, False)
    return [ttnn_concat_5]


def main_const_eval_26(arg_0, device):
    ttnn_to_device_46 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_31 = ttnn.to_layout(
        ttnn_to_device_46,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_46, False)
    ttnn_permute_20 = ttnn.permute(
        ttnn_to_layout_31,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_31, False)
    ttnn_typecast_16 = ttnn.typecast(
        ttnn_permute_20,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_20, False)
    ttnn_from_device_15 = ttnn.from_device(ttnn_typecast_16)
    ttnn.deallocate(ttnn_typecast_16, False)
    ttnn_typecast_17 = ttnn.typecast(
        ttnn_from_device_15, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_15, False)
    ttnn_to_device_47 = ttnn.to_device(
        ttnn_typecast_17,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_17, False)
    return [ttnn_to_device_47]


def main_const_eval_27(arg_0, device):
    ttnn_to_device_48 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_32 = ttnn.to_layout(
        ttnn_to_device_48,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_48, False)
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_to_layout_32,
        [1, 32, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_32, False)
    ttnn_typecast_18 = ttnn.typecast(
        ttnn_reshape_6,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_6, False)
    return [ttnn_typecast_18]


def main_const_eval_28(arg_0, device):
    ttnn_to_device_49 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_33 = ttnn.to_layout(
        ttnn_to_device_49,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_49, False)
    ttnn_permute_21 = ttnn.permute(
        ttnn_to_layout_33,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_33, False)
    ttnn_from_device_16 = ttnn.from_device(ttnn_permute_21)
    ttnn.deallocate(ttnn_permute_21, False)
    ttnn_typecast_19 = ttnn.typecast(
        ttnn_from_device_16, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_16, False)
    ttnn_to_device_50 = ttnn.to_device(
        ttnn_typecast_19,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_19, False)
    return [ttnn_to_device_50]


def main_const_eval_29(device):
    ttnn_arange_1 = ttnn.arange(
        0,
        16,
        1,
        dtype=ttnn.DataType.UINT32,
        device=device,
        layout=ttnn.Layout.TILE,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_arange_1,
        [16, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_arange_1, False)
    ttnn_repeat_0 = ttnn.repeat(
        ttnn_reshape_7,
        ttnn.Shape([1, 8, 1]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_7, False)
    return [ttnn_repeat_0]


def main_const_eval_30(device):
    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=2.5,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_full_2]


def main_const_eval_31(device):
    ttnn_full_3 = ttnn.full(
        shape=ttnn.Shape([1, 1]),
        fill_value=9.9999996826552254e-21,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_full_3]


def main_const_eval_32(arg_0, device):
    ttnn_to_device_51 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_34 = ttnn.to_layout(
        ttnn_to_device_51,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_51, False)
    ttnn_permute_22 = ttnn.permute(
        ttnn_to_layout_34,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_34, False)
    ttnn_from_device_17 = ttnn.from_device(ttnn_permute_22)
    ttnn.deallocate(ttnn_permute_22, False)
    ttnn_typecast_20 = ttnn.typecast(
        ttnn_from_device_17, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_17, False)
    ttnn_to_device_52 = ttnn.to_device(
        ttnn_typecast_20,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_20, False)
    return [ttnn_to_device_52]


def main_const_eval_33(arg_0, device):
    ttnn_to_device_53 = ttnn.to_device(
        arg_0[2],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_35 = ttnn.to_layout(
        ttnn_to_device_53,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_53, False)
    ttnn_to_device_54 = ttnn.to_device(
        arg_0[1],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_36 = ttnn.to_layout(
        ttnn_to_device_54,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_54, False)
    ttnn_to_device_55 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_37 = ttnn.to_layout(
        ttnn_to_device_55,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_55, False)
    ttnn_concat_6 = ttnn.concat(
        [ttnn_to_layout_35, ttnn_to_layout_37, ttnn_to_layout_36],
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_37, False)
    ttnn.deallocate(ttnn_to_layout_36, False)
    ttnn.deallocate(ttnn_to_layout_35, False)
    return [ttnn_concat_6]


def main_const_eval_34(arg_0, device):
    ttnn_to_device_56 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_38 = ttnn.to_layout(
        ttnn_to_device_56,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_56, False)
    ttnn_permute_23 = ttnn.permute(
        ttnn_to_layout_38,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_38, False)
    ttnn_from_device_18 = ttnn.from_device(ttnn_permute_23)
    ttnn.deallocate(ttnn_permute_23, False)
    ttnn_typecast_21 = ttnn.typecast(
        ttnn_from_device_18, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_18, False)
    ttnn_to_device_57 = ttnn.to_device(
        ttnn_typecast_21,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_21, False)
    return [ttnn_to_device_57]


def main_const_eval_35(arg_0, device):
    ttnn_to_device_58 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_39 = ttnn.to_layout(
        ttnn_to_device_58,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_58, False)
    ttnn_permute_24 = ttnn.permute(
        ttnn_to_layout_39,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_39, False)
    ttnn_from_device_19 = ttnn.from_device(ttnn_permute_24)
    ttnn.deallocate(ttnn_permute_24, False)
    ttnn_typecast_22 = ttnn.typecast(
        ttnn_from_device_19, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_19, False)
    ttnn_to_device_59 = ttnn.to_device(
        ttnn_typecast_22,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_22, False)
    return [ttnn_to_device_59]


def main_const_eval_36(arg_0, device):
    ttnn_to_device_60 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_40 = ttnn.to_layout(
        ttnn_to_device_60,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_60, False)
    ttnn_permute_25 = ttnn.permute(
        ttnn_to_layout_40,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_40, False)
    ttnn_from_device_20 = ttnn.from_device(ttnn_permute_25)
    ttnn.deallocate(ttnn_permute_25, False)
    ttnn_typecast_23 = ttnn.typecast(
        ttnn_from_device_20, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_20, False)
    ttnn_to_device_61 = ttnn.to_device(
        ttnn_typecast_23,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_23, False)
    return [ttnn_to_device_61]


def main_const_eval_37(arg_0, device):
    ttnn_to_device_62 = ttnn.to_device(
        arg_0[2],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_41 = ttnn.to_layout(
        ttnn_to_device_62,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_62, False)
    ttnn_to_device_63 = ttnn.to_device(
        arg_0[1],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_42 = ttnn.to_layout(
        ttnn_to_device_63,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_63, False)
    ttnn_to_device_64 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_43 = ttnn.to_layout(
        ttnn_to_device_64,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_64, False)
    ttnn_concat_7 = ttnn.concat(
        [ttnn_to_layout_41, ttnn_to_layout_43, ttnn_to_layout_42],
        0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_43, False)
    ttnn.deallocate(ttnn_to_layout_42, False)
    ttnn.deallocate(ttnn_to_layout_41, False)
    return [ttnn_concat_7]


def main_const_eval_38(device):
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
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
        ],
        [1, 1, 160],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_Tensor_1]


def main_const_eval_39(arg_0, device):
    ttnn_typecast_24 = ttnn.typecast(arg_0[0], ttnn.DataType.UINT16, memory_config=None)
    ttnn_to_device_65 = ttnn.to_device(
        ttnn_typecast_24,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_24, False)
    return [ttnn_to_device_65]


def main_const_eval_40(arg_0, device):
    ttnn_to_device_66 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_44 = ttnn.to_layout(
        ttnn_to_device_66,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_66, False)
    ttnn_permute_26 = ttnn.permute(
        ttnn_to_layout_44,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_44, False)
    ttnn_from_device_21 = ttnn.from_device(ttnn_permute_26)
    ttnn.deallocate(ttnn_permute_26, False)
    ttnn_typecast_25 = ttnn.typecast(
        ttnn_from_device_21, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_21, False)
    ttnn_to_device_67 = ttnn.to_device(
        ttnn_typecast_25,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_25, False)
    return [ttnn_to_device_67]


def main_const_eval_41(arg_0, device):
    ttnn_to_device_68 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_45 = ttnn.to_layout(
        ttnn_to_device_68,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_68, False)
    ttnn_permute_27 = ttnn.permute(
        ttnn_to_layout_45,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_45, False)
    ttnn_from_device_22 = ttnn.from_device(ttnn_permute_27)
    ttnn.deallocate(ttnn_permute_27, False)
    ttnn_typecast_26 = ttnn.typecast(
        ttnn_from_device_22, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_22, False)
    ttnn_to_device_69 = ttnn.to_device(
        ttnn_typecast_26,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_26, False)
    return [ttnn_to_device_69]


def main_const_eval_42(arg_0, device):
    ttnn_to_device_70 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_46 = ttnn.to_layout(
        ttnn_to_device_70,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_70, False)
    ttnn_permute_28 = ttnn.permute(
        ttnn_to_layout_46,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_46, False)
    ttnn_from_device_23 = ttnn.from_device(ttnn_permute_28)
    ttnn.deallocate(ttnn_permute_28, False)
    ttnn_typecast_27 = ttnn.typecast(
        ttnn_from_device_23, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_23, False)
    ttnn_to_device_71 = ttnn.to_device(
        ttnn_typecast_27,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_27, False)
    return [ttnn_to_device_71]


def main_const_eval_43(device):
    ttnn_Tensor_2 = ttnn.Tensor(
        [160.0, 1.0],
        [2, 1],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_Tensor_2]


def main_const_eval_44(arg_0, device):
    ttnn_to_device_72 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_47 = ttnn.to_layout(
        ttnn_to_device_72,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_72, False)
    ttnn_permute_29 = ttnn.permute(
        ttnn_to_layout_47,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_47, False)
    ttnn_from_device_24 = ttnn.from_device(ttnn_permute_29)
    ttnn.deallocate(ttnn_permute_29, False)
    ttnn_typecast_28 = ttnn.typecast(
        ttnn_from_device_24, ttnn.DataType.BFLOAT8_B, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_24, False)
    ttnn_to_device_73 = ttnn.to_device(
        ttnn_typecast_28,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_28, False)
    return [ttnn_to_device_73]


def main_const_eval_45(device):
    ttnn_ones_0 = ttnn.ones(
        shape=ttnn.Shape([16, 1]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_all_gather_1 = ttnn.all_gather(
        input_tensor=ttnn_ones_0,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_ones_0, False)
    ttnn_reshape_8 = ttnn.reshape(
        ttnn_all_gather_1,
        [64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_1, False)
    ttnn_to_layout_48 = ttnn.to_layout(
        ttnn_reshape_8,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_8, False)
    return [ttnn_to_layout_48]



def consteval__main(ce_cache, weights, device):
    if not ce_cache:
        main_const_eval_0_0 = main_const_eval_0(
            [weights["model.model.embed_tokens.weight"]], device
        )
        ce_cache["main_const_eval_0"] = main_const_eval_0_0[0]
        main_const_eval_1_0 = main_const_eval_1(
            [
                weights["model.model.layers.3.self_attn.k_proj.weight"],
                weights["model.model.layers.3.self_attn.v_proj.weight"],
                weights["model.model.layers.3.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_1"] = main_const_eval_1_0[0]
        main_const_eval_2_0 = main_const_eval_2(
            [
                weights["model.model.layers.0.self_attn.k_proj.weight"],
                weights["model.model.layers.0.self_attn.v_proj.weight"],
                weights["model.model.layers.0.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_2"] = main_const_eval_2_0[0]
        main_const_eval_3_0 = main_const_eval_3(
            [
                weights[
                    "L__self___model_model_layers_3_mlp_mlp_router__route_fn___closure___0_cell_contents_e_score_correction_bias"
                ]
            ], device
        )
        ce_cache["main_const_eval_3"] = main_const_eval_3_0[0]
        main_const_eval_4_0 = main_const_eval_4(
            [weights["model.model.layers.2.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_4"] = main_const_eval_4_0[0]
        main_const_eval_5_0 = main_const_eval_5(device)
        ce_cache["main_const_eval_5"] = main_const_eval_5_0[0]
        main_const_eval_6_0 = main_const_eval_6(device)
        ce_cache["main_const_eval_6"] = main_const_eval_6_0[0]
        main_const_eval_7_0 = main_const_eval_7(device)
        ce_cache["main_const_eval_7"] = main_const_eval_7_0[0]
        main_const_eval_8_0 = main_const_eval_8(
            [weights["model.model.layers.2.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_8"] = main_const_eval_8_0[0]
        main_const_eval_9_0 = main_const_eval_9(
            [weights["model.model.layers.3.mlp.shared_experts.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_9"] = main_const_eval_9_0[0]
        main_const_eval_10_0 = main_const_eval_10(device)
        ce_cache["main_const_eval_10"] = main_const_eval_10_0[0]
        main_const_eval_11_0 = main_const_eval_11(device)
        ce_cache["main_const_eval_11"] = main_const_eval_11_0[0]
        main_const_eval_12_0 = main_const_eval_12(
            [
                weights["model.model.layers.1.self_attn.k_proj.weight"],
                weights["model.model.layers.1.self_attn.v_proj.weight"],
                weights["model.model.layers.1.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_12"] = main_const_eval_12_0[0]
        main_const_eval_13_0 = main_const_eval_13(
            [weights["model.model.layers.3.mlp.mlp.experts.gate_proj"]], device
        )
        ce_cache["main_const_eval_13"] = main_const_eval_13_0[0]
        main_const_eval_14_0 = main_const_eval_14(
            [weights["model.model.layers.3.mlp.mlp.experts.up_proj"]], device
        )
        ce_cache["main_const_eval_14"] = main_const_eval_14_0[0]
        main_const_eval_15_0 = main_const_eval_15(
            [
                weights["model.model.layers.1.self_attn.k_proj.bias"],
                weights["model.model.layers.1.self_attn.v_proj.bias"],
                weights["model.model.layers.1.self_attn.q_proj.bias"],
            ], device
        )
        ce_cache["main_const_eval_15"] = main_const_eval_15_0[0]
        main_const_eval_16_0 = main_const_eval_16(
            [weights["model.model.layers.3.mlp.shared_experts.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_16"] = main_const_eval_16_0[0]
        main_const_eval_17_0 = main_const_eval_17(
            [weights["model.model.layers.3.mlp.mlp.experts.down_proj"]], device
        )
        ce_cache["main_const_eval_17"] = main_const_eval_17_0[0]
        main_const_eval_18_0 = main_const_eval_18(
            [weights["model.model.layers.0.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_18"] = main_const_eval_18_0[0]
        main_const_eval_19_0 = main_const_eval_19(
            [
                weights["model.model.layers.2.self_attn.k_proj.weight"],
                weights["model.model.layers.2.self_attn.v_proj.weight"],
                weights["model.model.layers.2.self_attn.q_proj.weight"],
            ], device
        )
        ce_cache["main_const_eval_19"] = main_const_eval_19_0[0]
        main_const_eval_20_0 = main_const_eval_20([weights["model.lm_head.weight"]], device)
        ce_cache["main_const_eval_20"] = main_const_eval_20_0[0]
        main_const_eval_21_0 = main_const_eval_21(device)
        ce_cache["main_const_eval_21"] = main_const_eval_21_0[0]
        main_const_eval_22_0 = main_const_eval_22(
            [weights["model.model.layers.1.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_22"] = main_const_eval_22_0[0]
        main_const_eval_23_0 = main_const_eval_23(
            [weights["model.model.layers.3.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_23"] = main_const_eval_23_0[0]
        main_const_eval_24_0 = main_const_eval_24(device)
        ce_cache["main_const_eval_24"] = main_const_eval_24_0[0]
        main_const_eval_25_0 = main_const_eval_25(
            [
                weights["model.model.layers.0.self_attn.k_proj.bias"],
                weights["model.model.layers.0.self_attn.v_proj.bias"],
                weights["model.model.layers.0.self_attn.q_proj.bias"],
            ], device
        )
        ce_cache["main_const_eval_25"] = main_const_eval_25_0[0]
        main_const_eval_26_0 = main_const_eval_26(
            [weights["model.model.layers.3.mlp.mlp.router.gate.weight"]], device
        )
        ce_cache["main_const_eval_26"] = main_const_eval_26_0[0]
        main_const_eval_27_0 = main_const_eval_27(
            [weights["model.model.rotary_emb.inv_freq"]], device
        )
        ce_cache["main_const_eval_27"] = main_const_eval_27_0[0]
        main_const_eval_28_0 = main_const_eval_28(
            [weights["model.model.layers.2.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_28"] = main_const_eval_28_0[0]
        main_const_eval_29_0 = main_const_eval_29(device)
        ce_cache["main_const_eval_29"] = main_const_eval_29_0[0]
        main_const_eval_30_0 = main_const_eval_30(device)
        ce_cache["main_const_eval_30"] = main_const_eval_30_0[0]
        main_const_eval_31_0 = main_const_eval_31(device)
        ce_cache["main_const_eval_31"] = main_const_eval_31_0[0]
        main_const_eval_32_0 = main_const_eval_32(
            [weights["model.model.layers.0.mlp.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_32"] = main_const_eval_32_0[0]
        main_const_eval_33_0 = main_const_eval_33(
            [
                weights["model.model.layers.3.self_attn.k_proj.bias"],
                weights["model.model.layers.3.self_attn.v_proj.bias"],
                weights["model.model.layers.3.self_attn.q_proj.bias"],
            ], device
        )
        ce_cache["main_const_eval_33"] = main_const_eval_33_0[0]
        main_const_eval_34_0 = main_const_eval_34(
            [weights["model.model.layers.1.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_34"] = main_const_eval_34_0[0]
        main_const_eval_35_0 = main_const_eval_35(
            [weights["model.model.layers.1.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_35"] = main_const_eval_35_0[0]
        main_const_eval_36_0 = main_const_eval_36(
            [weights["model.model.layers.1.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_36"] = main_const_eval_36_0[0]
        main_const_eval_37_0 = main_const_eval_37(
            [
                weights["model.model.layers.2.self_attn.k_proj.bias"],
                weights["model.model.layers.2.self_attn.v_proj.bias"],
                weights["model.model.layers.2.self_attn.q_proj.bias"],
            ], device
        )
        ce_cache["main_const_eval_37"] = main_const_eval_37_0[0]
        main_const_eval_38_0 = main_const_eval_38(device)
        ce_cache["main_const_eval_38"] = main_const_eval_38_0[0]
        main_const_eval_39_0 = main_const_eval_39(
            [weights["model.model.layers.3.mlp.mlp.expert_mapping"]], device
        )
        ce_cache["main_const_eval_39"] = main_const_eval_39_0[0]
        main_const_eval_40_0 = main_const_eval_40(
            [weights["model.model.layers.0.self_attn.o_proj.weight"]], device
        )
        ce_cache["main_const_eval_40"] = main_const_eval_40_0[0]
        main_const_eval_41_0 = main_const_eval_41(
            [weights["model.model.layers.2.mlp.gate_proj.weight"]], device
        )
        ce_cache["main_const_eval_41"] = main_const_eval_41_0[0]
        main_const_eval_42_0 = main_const_eval_42(
            [weights["model.model.layers.3.mlp.shared_experts.up_proj.weight"]], device
        )
        ce_cache["main_const_eval_42"] = main_const_eval_42_0[0]
        main_const_eval_43_0 = main_const_eval_43(device)
        ce_cache["main_const_eval_43"] = main_const_eval_43_0[0]
        main_const_eval_44_0 = main_const_eval_44(
            [weights["model.model.layers.0.mlp.down_proj.weight"]], device
        )
        ce_cache["main_const_eval_44"] = main_const_eval_44_0[0]
        main_const_eval_45_0 = main_const_eval_45(device)
        ce_cache["main_const_eval_45"] = main_const_eval_45_0[0]
    return ce_cache


def load_activations_for__main():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((4, 8))
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg8.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg9.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg12.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg25.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg26.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg42.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg43.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg46.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg59.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg60.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg63.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
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
    ]



def main():
    load_activations_for__main_0 = load_activations_for__main()
    weights = load_weights_for__main_from_state_dict()
    _main_0 = _main(load_activations_for__main_0, weights)
    return 0


def test_main():
    import model_pt

    exact_pcc = 0.85546875

    activations = load_activations_for__main()
    weights = load_weights_for__main_from_state_dict()
    outputs = _main(activations, weights)

    ttnn_output = [ttnn.from_device(output) for output in outputs]
    golden_output = model_pt.run_pytorch_model()

    # outputs[-1] is the final logits, fully replicated across the 4x8 mesh by
    # the trailing all_gathers (dim 0 over cluster_axis 0, dim 2 over cluster_axis 1).
    # Every device holds an identical copy, so grab a single shard before converting.
    final_output = ttnn.get_device_tensors(ttnn_output[-1])[0]

    pcc = calculate_pcc(ttnn.to_torch(final_output), golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    main()
