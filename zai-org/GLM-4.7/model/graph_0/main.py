import ttnn
import utils


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
        topology=None,
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
        topology=None,
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


ce_cache_forward = {}


def forward(input, device):
    global ce_cache_forward
    ce_cache_forward = consteval_forward(ce_cache_forward, input, device)
    ttnn_to_layout_49 = ttnn.to_layout(
        input[0], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_74 = ttnn.to_device(
        ttnn_to_layout_49,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_50 = ttnn.to_layout(
        input[3], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_75 = ttnn.to_device(
        ttnn_to_layout_50,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_76 = ttnn.to_device(
        input[4],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_77 = ttnn.to_device(
        input[6],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_78 = ttnn.to_device(
        input[8],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_79 = ttnn.to_device(
        input[9],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_80 = ttnn.to_device(
        input[12],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_51 = ttnn.to_layout(
        input[13], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_81 = ttnn.to_device(
        ttnn_to_layout_51,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_52 = ttnn.to_layout(
        input[16], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_82 = ttnn.to_device(
        ttnn_to_layout_52,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_53 = ttnn.to_layout(
        input[19], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_83 = ttnn.to_device(
        ttnn_to_layout_53,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_54 = ttnn.to_layout(
        input[21], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_84 = ttnn.to_device(
        ttnn_to_layout_54,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_85 = ttnn.to_device(
        input[25],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_86 = ttnn.to_device(
        input[26],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_87 = ttnn.to_device(
        input[29],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_55 = ttnn.to_layout(
        input[30], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_88 = ttnn.to_device(
        ttnn_to_layout_55,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_56 = ttnn.to_layout(
        input[33], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_89 = ttnn.to_device(
        ttnn_to_layout_56,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_57 = ttnn.to_layout(
        input[36], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_90 = ttnn.to_device(
        ttnn_to_layout_57,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_58 = ttnn.to_layout(
        input[38], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_91 = ttnn.to_device(
        ttnn_to_layout_58,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_92 = ttnn.to_device(
        input[42],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_93 = ttnn.to_device(
        input[43],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_94 = ttnn.to_device(
        input[46],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_59 = ttnn.to_layout(
        input[47], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_95 = ttnn.to_device(
        ttnn_to_layout_59,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_60 = ttnn.to_layout(
        input[50], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_96 = ttnn.to_device(
        ttnn_to_layout_60,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_61 = ttnn.to_layout(
        input[53], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_97 = ttnn.to_device(
        ttnn_to_layout_61,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_62 = ttnn.to_layout(
        input[55], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_98 = ttnn.to_device(
        ttnn_to_layout_62,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_99 = ttnn.to_device(
        input[59],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_100 = ttnn.to_device(
        input[60],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_device_101 = ttnn.to_device(
        input[63],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_63 = ttnn.to_layout(
        input[65], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_102 = ttnn.to_device(
        ttnn_to_layout_63,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_64 = ttnn.to_layout(
        input[68], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_103 = ttnn.to_device(
        ttnn_to_layout_64,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_65 = ttnn.to_layout(
        input[70], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_104 = ttnn.to_device(
        ttnn_to_layout_65,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_92, False)
    ttnn.deallocate(ttnn_to_device_85, False)
    ttnn.deallocate(ttnn_to_device_78, False)
    var_0 = ce_cache_forward["main_const_eval_10"]
    var_1 = ce_cache_forward["main_const_eval_11"]
    var_2 = ce_cache_forward["main_const_eval_39"]
    ttnn_typecast_29 = ttnn.typecast(
        ttnn_to_device_76,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_76, False)
    ttnn_reshape_9 = ttnn.reshape(
        ttnn_typecast_29,
        [16],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_29, False)
    ttnn_embedding_0 = ttnn.embedding(
        ttnn_reshape_9,
        ce_cache_forward["main_const_eval_0"],
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_9, False)
    ttnn_rms_norm_0 = ttnn.rms_norm(
        ttnn_embedding_0,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_75,
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
    ttnn_linear_0 = ttnn.linear(
        ttnn_rms_norm_0,
        ce_cache_forward["main_const_eval_2"],
        bias=ce_cache_forward["main_const_eval_25"],
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
    ttnn.deallocate(ttnn_rms_norm_0, False)
    ttnn_reshape_10 = ttnn.reshape(
        ttnn_linear_0,
        [16, 1, 1792],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_0, False)
    v_3, v_4, v_5 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_10,
        None,
        num_heads=12,
        num_kv_heads=1,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_10, False)
    ttnn_reshape_11 = ttnn.reshape(
        v_5,
        [1, 16, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_5, False)
    ttnn_rms_norm_1 = ttnn.rms_norm(
        v_3,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_84,
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
    ttnn.deallocate(v_3, False)
    ttnn_slice_0 = ttnn.slice(
        ttnn_rms_norm_1,
        [0, 0, 0, 0],
        [16, 12, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_30 = ttnn.typecast(
        ttnn_to_device_77,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_12 = ttnn.reshape(
        ttnn_typecast_30,
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_30, False)
    ttnn_to_layout_66 = ttnn.to_layout(
        ttnn_reshape_12,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_12, False)
    ttnn_matmul_0 = ttnn.matmul(
        ce_cache_forward["main_const_eval_27"],
        ttnn_to_layout_66,
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
    ttnn.deallocate(ttnn_to_layout_66, False)
    ttnn_reshape_13 = ttnn.reshape(
        ttnn_matmul_0,
        [1, 1, 1, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_concat_8 = ttnn.concat(
        [ttnn_reshape_13, ttnn_reshape_13],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_13, False)
    ttnn_cos_0 = ttnn.cos(
        ttnn_concat_8,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_31 = ttnn.typecast(
        ttnn_cos_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_cos_0, False)
    ttnn_sin_0 = ttnn.sin(
        ttnn_concat_8,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_8, False)
    ttnn_typecast_32 = ttnn.typecast(
        ttnn_sin_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_sin_0, False)
    ttnn_experimental_rotary_embedding_0 = ttnn.experimental.rotary_embedding(
        ttnn_slice_0,
        ttnn_typecast_31,
        ttnn_typecast_32,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_slice_0, False)
    ttnn_slice_1 = ttnn.slice(
        ttnn_experimental_rotary_embedding_0,
        [0, 0, 0, 0],
        [16, 12, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_0, False)
    ttnn_slice_2 = ttnn.slice(
        ttnn_rms_norm_1,
        [0, 0, 0, 64],
        [16, 12, 1, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_1, False)
    ttnn_concat_9 = ttnn.concat(
        [ttnn_slice_1, ttnn_slice_2],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_2, False)
    ttnn.deallocate(ttnn_slice_1, False)
    ttnn_rms_norm_2 = ttnn.rms_norm(
        v_4,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_74,
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
    ttnn.deallocate(v_4, False)
    ttnn_slice_3 = ttnn.slice(
        ttnn_rms_norm_2,
        [0, 0, 0, 0],
        [16, 1, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_experimental_rotary_embedding_1 = ttnn.experimental.rotary_embedding(
        ttnn_slice_3,
        ttnn_typecast_31,
        ttnn_typecast_32,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_slice_3, False)
    ttnn_slice_4 = ttnn.slice(
        ttnn_experimental_rotary_embedding_1,
        [0, 0, 0, 0],
        [16, 1, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_1, False)
    ttnn_slice_5 = ttnn.slice(
        ttnn_rms_norm_2,
        [0, 0, 0, 64],
        [16, 1, 1, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_2, False)
    ttnn_concat_10 = ttnn.concat(
        [ttnn_slice_4, ttnn_slice_5],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_5, False)
    ttnn.deallocate(ttnn_slice_4, False)
    ttnn_reshape_14 = ttnn.reshape(
        ttnn_concat_10,
        [1, 16, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_10, False)
    ttnn_repeat_1 = ttnn.repeat(
        ttnn_to_device_99,
        ttnn.Shape([16]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_memory_config_0 = ttnn.to_memory_config(
        ttnn_reshape_14,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_14, False)
    ttnn.experimental.paged_update_cache(
        ttnn_to_device_79,
        ttnn_to_memory_config_0,
        update_idxs_tensor=ttnn_repeat_1,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_0, False)
    ttnn_reshape_15 = ttnn.reshape(
        ttnn_to_device_99,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_ge_0 = ttnn.ge(
        ttnn_reshape_15,
        ce_cache_forward["main_const_eval_6"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_15, False)
    ttnn_where_0 = ttnn.where(
        ttnn_ge_0,
        ce_cache_forward["main_const_eval_21"],
        ce_cache_forward["main_const_eval_5"],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_ge_0, False)
    ttnn_to_memory_config_1 = ttnn.to_memory_config(
        ttnn_reshape_11,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_11, False)
    ttnn.experimental.paged_update_cache(
        ttnn_to_device_80,
        ttnn_to_memory_config_1,
        update_idxs_tensor=ttnn_repeat_1,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_1, False)
    ttnn_reshape_16 = ttnn.reshape(
        ttnn_concat_9,
        [1, 16, 12, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_9, False)
    ttnn_repeat_2 = ttnn.repeat(
        ttnn_where_0,
        ttnn.Shape([1, 1, 12, 1]),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_where_0, False)
    ttnn_transformer_scaled_dot_product_attention_decode_0 = (
        ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn_reshape_16,
            ttnn_to_device_79,
            ttnn_to_device_80,
            is_causal=False,
            attn_mask=ttnn_repeat_2,
            cur_pos_tensor=None,
            attention_sink=None,
            scale=0.08837890625,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_reshape_16, False)
    ttnn_reshape_17 = ttnn.reshape(
        ttnn_transformer_scaled_dot_product_attention_decode_0,
        [16, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_0, False)
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_reshape_17,
        ce_cache_forward["main_const_eval_40"],
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
    ttnn.deallocate(ttnn_reshape_17, False)
    ttnn_reshape_18 = ttnn.reshape(
        ttnn_matmul_1,
        [1, 1, 16, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_1, False)
    ttnn_reduce_scatter_0 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_18,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_18, False)
    ttnn_reshape_19 = ttnn.reshape(
        ttnn_reduce_scatter_0,
        [16, 640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_0, False)
    ttnn_all_gather_2 = ttnn.all_gather(
        input_tensor=ttnn_reshape_19,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reshape_19, False)
    ttnn_add_0 = ttnn.add(
        ttnn_embedding_0,
        ttnn_all_gather_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_2, False)
    ttnn.deallocate(ttnn_embedding_0, False)
    ttnn_rms_norm_3 = ttnn.rms_norm(
        ttnn_add_0,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_83,
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
    ttnn_matmul_2 = ttnn.matmul(
        ttnn_rms_norm_3,
        ce_cache_forward["main_const_eval_18"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation="silu",
        compute_kernel_config=None,
    )
    ttnn_matmul_3 = ttnn.matmul(
        ttnn_rms_norm_3,
        ce_cache_forward["main_const_eval_32"],
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
    ttnn.deallocate(ttnn_rms_norm_3, False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_matmul_2,
        ttnn_matmul_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_3, False)
    ttnn.deallocate(ttnn_matmul_2, False)
    ttnn_matmul_4 = ttnn.matmul(
        ttnn_multiply_0,
        ce_cache_forward["main_const_eval_44"],
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
    ttnn.deallocate(ttnn_multiply_0, False)
    ttnn_reshape_20 = ttnn.reshape(
        ttnn_matmul_4,
        [1, 1, 16, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_4, False)
    ttnn_reduce_scatter_1 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_20,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_20, False)
    ttnn_reshape_21 = ttnn.reshape(
        ttnn_reduce_scatter_1,
        [16, 640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_1, False)
    ttnn_all_gather_3 = ttnn.all_gather(
        input_tensor=ttnn_reshape_21,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reshape_21, False)
    ttnn_add_1 = ttnn.add(
        ttnn_add_0,
        ttnn_all_gather_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_3, False)
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_rms_norm_4 = ttnn.rms_norm(
        ttnn_add_1,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_82,
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
    ttnn_linear_1 = ttnn.linear(
        ttnn_rms_norm_4,
        ce_cache_forward["main_const_eval_12"],
        bias=ce_cache_forward["main_const_eval_15"],
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
    ttnn.deallocate(ttnn_rms_norm_4, False)
    ttnn_reshape_22 = ttnn.reshape(
        ttnn_linear_1,
        [16, 1, 1792],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_1, False)
    v_6, v_7, v_8 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_22,
        None,
        num_heads=12,
        num_kv_heads=1,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_22, False)
    ttnn_reshape_23 = ttnn.reshape(
        v_8,
        [1, 16, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_8, False)
    ttnn_rms_norm_5 = ttnn.rms_norm(
        v_6,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_91,
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
    ttnn.deallocate(v_6, False)
    ttnn_slice_6 = ttnn.slice(
        ttnn_rms_norm_5,
        [0, 0, 0, 0],
        [16, 12, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_experimental_rotary_embedding_2 = ttnn.experimental.rotary_embedding(
        ttnn_slice_6,
        ttnn_typecast_31,
        ttnn_typecast_32,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_slice_6, False)
    ttnn_slice_7 = ttnn.slice(
        ttnn_experimental_rotary_embedding_2,
        [0, 0, 0, 0],
        [16, 12, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_2, False)
    ttnn_slice_8 = ttnn.slice(
        ttnn_rms_norm_5,
        [0, 0, 0, 64],
        [16, 12, 1, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_5, False)
    ttnn_concat_11 = ttnn.concat(
        [ttnn_slice_7, ttnn_slice_8],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_8, False)
    ttnn.deallocate(ttnn_slice_7, False)
    ttnn_rms_norm_6 = ttnn.rms_norm(
        v_7,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_81,
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
    ttnn.deallocate(v_7, False)
    ttnn_slice_9 = ttnn.slice(
        ttnn_rms_norm_6,
        [0, 0, 0, 0],
        [16, 1, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_experimental_rotary_embedding_3 = ttnn.experimental.rotary_embedding(
        ttnn_slice_9,
        ttnn_typecast_31,
        ttnn_typecast_32,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_slice_9, False)
    ttnn_slice_10 = ttnn.slice(
        ttnn_experimental_rotary_embedding_3,
        [0, 0, 0, 0],
        [16, 1, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_3, False)
    ttnn_slice_11 = ttnn.slice(
        ttnn_rms_norm_6,
        [0, 0, 0, 64],
        [16, 1, 1, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_6, False)
    ttnn_concat_12 = ttnn.concat(
        [ttnn_slice_10, ttnn_slice_11],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_11, False)
    ttnn.deallocate(ttnn_slice_10, False)
    ttnn_reshape_24 = ttnn.reshape(
        ttnn_concat_12,
        [1, 16, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_12, False)
    ttnn_to_memory_config_2 = ttnn.to_memory_config(
        ttnn_reshape_24,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_24, False)
    ttnn.experimental.paged_update_cache(
        ttnn_to_device_86,
        ttnn_to_memory_config_2,
        update_idxs_tensor=ttnn_repeat_1,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_2, False)
    ttnn_to_memory_config_3 = ttnn.to_memory_config(
        ttnn_reshape_23,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_23, False)
    ttnn.experimental.paged_update_cache(
        ttnn_to_device_87,
        ttnn_to_memory_config_3,
        update_idxs_tensor=ttnn_repeat_1,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_3, False)
    ttnn_reshape_25 = ttnn.reshape(
        ttnn_concat_11,
        [1, 16, 12, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_11, False)
    ttnn_transformer_scaled_dot_product_attention_decode_1 = (
        ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn_reshape_25,
            ttnn_to_device_86,
            ttnn_to_device_87,
            is_causal=False,
            attn_mask=ttnn_repeat_2,
            cur_pos_tensor=None,
            attention_sink=None,
            scale=0.08837890625,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_reshape_25, False)
    ttnn_reshape_26 = ttnn.reshape(
        ttnn_transformer_scaled_dot_product_attention_decode_1,
        [16, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_1, False)
    ttnn_matmul_5 = ttnn.matmul(
        ttnn_reshape_26,
        ce_cache_forward["main_const_eval_34"],
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
    ttnn.deallocate(ttnn_reshape_26, False)
    ttnn_reshape_27 = ttnn.reshape(
        ttnn_matmul_5,
        [1, 1, 16, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_5, False)
    ttnn_reduce_scatter_2 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_27,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_27, False)
    ttnn_reshape_28 = ttnn.reshape(
        ttnn_reduce_scatter_2,
        [16, 640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_2, False)
    ttnn_all_gather_4 = ttnn.all_gather(
        input_tensor=ttnn_reshape_28,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reshape_28, False)
    ttnn_add_2 = ttnn.add(
        ttnn_add_1,
        ttnn_all_gather_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_4, False)
    ttnn.deallocate(ttnn_add_1, False)
    ttnn_rms_norm_7 = ttnn.rms_norm(
        ttnn_add_2,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_90,
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
    ttnn_matmul_6 = ttnn.matmul(
        ttnn_rms_norm_7,
        ce_cache_forward["main_const_eval_35"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation="silu",
        compute_kernel_config=None,
    )
    ttnn_matmul_7 = ttnn.matmul(
        ttnn_rms_norm_7,
        ce_cache_forward["main_const_eval_22"],
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
    ttnn.deallocate(ttnn_rms_norm_7, False)
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_matmul_6,
        ttnn_matmul_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_7, False)
    ttnn.deallocate(ttnn_matmul_6, False)
    ttnn_matmul_8 = ttnn.matmul(
        ttnn_multiply_1,
        ce_cache_forward["main_const_eval_36"],
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
    ttnn.deallocate(ttnn_multiply_1, False)
    ttnn_reshape_29 = ttnn.reshape(
        ttnn_matmul_8,
        [1, 1, 16, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_8, False)
    ttnn_reduce_scatter_3 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_29,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_29, False)
    ttnn_reshape_30 = ttnn.reshape(
        ttnn_reduce_scatter_3,
        [16, 640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_3, False)
    ttnn_all_gather_5 = ttnn.all_gather(
        input_tensor=ttnn_reshape_30,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reshape_30, False)
    ttnn_add_3 = ttnn.add(
        ttnn_add_2,
        ttnn_all_gather_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_5, False)
    ttnn.deallocate(ttnn_add_2, False)
    ttnn_rms_norm_8 = ttnn.rms_norm(
        ttnn_add_3,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_89,
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
    ttnn_linear_2 = ttnn.linear(
        ttnn_rms_norm_8,
        ce_cache_forward["main_const_eval_19"],
        bias=ce_cache_forward["main_const_eval_37"],
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
    ttnn.deallocate(ttnn_rms_norm_8, False)
    ttnn_reshape_31 = ttnn.reshape(
        ttnn_linear_2,
        [16, 1, 1792],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_2, False)
    v_9, v_10, v_11 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_31,
        None,
        num_heads=12,
        num_kv_heads=1,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_31, False)
    ttnn_reshape_32 = ttnn.reshape(
        v_11,
        [1, 16, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_11, False)
    ttnn_rms_norm_9 = ttnn.rms_norm(
        v_9,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_98,
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
    ttnn.deallocate(v_9, False)
    ttnn_slice_12 = ttnn.slice(
        ttnn_rms_norm_9,
        [0, 0, 0, 0],
        [16, 12, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_experimental_rotary_embedding_4 = ttnn.experimental.rotary_embedding(
        ttnn_slice_12,
        ttnn_typecast_31,
        ttnn_typecast_32,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_slice_12, False)
    ttnn_slice_13 = ttnn.slice(
        ttnn_experimental_rotary_embedding_4,
        [0, 0, 0, 0],
        [16, 12, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_4, False)
    ttnn_slice_14 = ttnn.slice(
        ttnn_rms_norm_9,
        [0, 0, 0, 64],
        [16, 12, 1, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_9, False)
    ttnn_concat_13 = ttnn.concat(
        [ttnn_slice_13, ttnn_slice_14],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_14, False)
    ttnn.deallocate(ttnn_slice_13, False)
    ttnn_rms_norm_10 = ttnn.rms_norm(
        v_10,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_88,
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
    ttnn.deallocate(v_10, False)
    ttnn_slice_15 = ttnn.slice(
        ttnn_rms_norm_10,
        [0, 0, 0, 0],
        [16, 1, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_experimental_rotary_embedding_5 = ttnn.experimental.rotary_embedding(
        ttnn_slice_15,
        ttnn_typecast_31,
        ttnn_typecast_32,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_slice_15, False)
    ttnn_slice_16 = ttnn.slice(
        ttnn_experimental_rotary_embedding_5,
        [0, 0, 0, 0],
        [16, 1, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_5, False)
    ttnn_slice_17 = ttnn.slice(
        ttnn_rms_norm_10,
        [0, 0, 0, 64],
        [16, 1, 1, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_10, False)
    ttnn_concat_14 = ttnn.concat(
        [ttnn_slice_16, ttnn_slice_17],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_17, False)
    ttnn.deallocate(ttnn_slice_16, False)
    ttnn_reshape_33 = ttnn.reshape(
        ttnn_concat_14,
        [1, 16, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_14, False)
    ttnn_to_memory_config_4 = ttnn.to_memory_config(
        ttnn_reshape_33,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_33, False)
    ttnn.experimental.paged_update_cache(
        ttnn_to_device_93,
        ttnn_to_memory_config_4,
        update_idxs_tensor=ttnn_repeat_1,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_4, False)
    ttnn_to_memory_config_5 = ttnn.to_memory_config(
        ttnn_reshape_32,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_32, False)
    ttnn.experimental.paged_update_cache(
        ttnn_to_device_94,
        ttnn_to_memory_config_5,
        update_idxs_tensor=ttnn_repeat_1,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_5, False)
    ttnn_reshape_34 = ttnn.reshape(
        ttnn_concat_13,
        [1, 16, 12, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_13, False)
    ttnn_transformer_scaled_dot_product_attention_decode_2 = (
        ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn_reshape_34,
            ttnn_to_device_93,
            ttnn_to_device_94,
            is_causal=False,
            attn_mask=ttnn_repeat_2,
            cur_pos_tensor=None,
            attention_sink=None,
            scale=0.08837890625,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_reshape_34, False)
    ttnn_reshape_35 = ttnn.reshape(
        ttnn_transformer_scaled_dot_product_attention_decode_2,
        [16, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_2, False)
    ttnn_matmul_9 = ttnn.matmul(
        ttnn_reshape_35,
        ce_cache_forward["main_const_eval_4"],
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
    ttnn.deallocate(ttnn_reshape_35, False)
    ttnn_reshape_36 = ttnn.reshape(
        ttnn_matmul_9,
        [1, 1, 16, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_9, False)
    ttnn_reduce_scatter_4 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_36,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_36, False)
    ttnn_reshape_37 = ttnn.reshape(
        ttnn_reduce_scatter_4,
        [16, 640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_4, False)
    ttnn_all_gather_6 = ttnn.all_gather(
        input_tensor=ttnn_reshape_37,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reshape_37, False)
    ttnn_add_4 = ttnn.add(
        ttnn_add_3,
        ttnn_all_gather_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_6, False)
    ttnn.deallocate(ttnn_add_3, False)
    ttnn_rms_norm_11 = ttnn.rms_norm(
        ttnn_add_4,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_97,
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
    ttnn_matmul_10 = ttnn.matmul(
        ttnn_rms_norm_11,
        ce_cache_forward["main_const_eval_41"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation="silu",
        compute_kernel_config=None,
    )
    ttnn_matmul_11 = ttnn.matmul(
        ttnn_rms_norm_11,
        ce_cache_forward["main_const_eval_8"],
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
    ttnn.deallocate(ttnn_rms_norm_11, False)
    ttnn_multiply_2 = ttnn.multiply(
        ttnn_matmul_10,
        ttnn_matmul_11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_11, False)
    ttnn.deallocate(ttnn_matmul_10, False)
    ttnn_matmul_12 = ttnn.matmul(
        ttnn_multiply_2,
        ce_cache_forward["main_const_eval_28"],
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
    ttnn.deallocate(ttnn_multiply_2, False)
    ttnn_reshape_38 = ttnn.reshape(
        ttnn_matmul_12,
        [1, 1, 16, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_12, False)
    ttnn_reduce_scatter_5 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_38,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_38, False)
    ttnn_reshape_39 = ttnn.reshape(
        ttnn_reduce_scatter_5,
        [16, 640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_5, False)
    ttnn_all_gather_7 = ttnn.all_gather(
        input_tensor=ttnn_reshape_39,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reshape_39, False)
    ttnn_add_5 = ttnn.add(
        ttnn_add_4,
        ttnn_all_gather_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_7, False)
    ttnn.deallocate(ttnn_add_4, False)
    ttnn_rms_norm_12 = ttnn.rms_norm(
        ttnn_add_5,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_96,
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
    ttnn_linear_3 = ttnn.linear(
        ttnn_rms_norm_12,
        ce_cache_forward["main_const_eval_1"],
        bias=ce_cache_forward["main_const_eval_33"],
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
    ttnn.deallocate(ttnn_rms_norm_12, False)
    ttnn_reshape_40 = ttnn.reshape(
        ttnn_linear_3,
        [16, 1, 1792],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_linear_3, False)
    v_12, v_13, v_14 = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn_reshape_40,
        None,
        num_heads=12,
        num_kv_heads=1,
        transpose_key=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_40, False)
    ttnn_reshape_41 = ttnn.reshape(
        v_14,
        [1, 16, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_14, False)
    ttnn_rms_norm_13 = ttnn.rms_norm(
        v_12,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_104,
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
    ttnn.deallocate(v_12, False)
    ttnn_slice_18 = ttnn.slice(
        ttnn_rms_norm_13,
        [0, 0, 0, 0],
        [16, 12, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_experimental_rotary_embedding_6 = ttnn.experimental.rotary_embedding(
        ttnn_slice_18,
        ttnn_typecast_31,
        ttnn_typecast_32,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_slice_18, False)
    ttnn_slice_19 = ttnn.slice(
        ttnn_experimental_rotary_embedding_6,
        [0, 0, 0, 0],
        [16, 12, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_6, False)
    ttnn_slice_20 = ttnn.slice(
        ttnn_rms_norm_13,
        [0, 0, 0, 64],
        [16, 12, 1, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_13, False)
    ttnn_concat_15 = ttnn.concat(
        [ttnn_slice_19, ttnn_slice_20],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_20, False)
    ttnn.deallocate(ttnn_slice_19, False)
    ttnn_rms_norm_14 = ttnn.rms_norm(
        v_13,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_95,
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
    ttnn.deallocate(v_13, False)
    ttnn_slice_21 = ttnn.slice(
        ttnn_rms_norm_14,
        [0, 0, 0, 0],
        [16, 1, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_experimental_rotary_embedding_7 = ttnn.experimental.rotary_embedding(
        ttnn_slice_21,
        ttnn_typecast_31,
        ttnn_typecast_32,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_slice_21, False)
    ttnn.deallocate(ttnn_typecast_32, False)
    ttnn.deallocate(ttnn_typecast_31, False)
    ttnn_slice_22 = ttnn.slice(
        ttnn_experimental_rotary_embedding_7,
        [0, 0, 0, 0],
        [16, 1, 1, 64],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_experimental_rotary_embedding_7, False)
    ttnn_slice_23 = ttnn.slice(
        ttnn_rms_norm_14,
        [0, 0, 0, 64],
        [16, 1, 1, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_14, False)
    ttnn_concat_16 = ttnn.concat(
        [ttnn_slice_22, ttnn_slice_23],
        3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_23, False)
    ttnn.deallocate(ttnn_slice_22, False)
    ttnn_reshape_42 = ttnn.reshape(
        ttnn_concat_16,
        [1, 16, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_16, False)
    ttnn_to_memory_config_6 = ttnn.to_memory_config(
        ttnn_reshape_42,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_42, False)
    ttnn.experimental.paged_update_cache(
        ttnn_to_device_100,
        ttnn_to_memory_config_6,
        update_idxs_tensor=ttnn_repeat_1,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_6, False)
    ttnn_to_memory_config_7 = ttnn.to_memory_config(
        ttnn_reshape_41,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_41, False)
    ttnn.experimental.paged_update_cache(
        ttnn_to_device_101,
        ttnn_to_memory_config_7,
        update_idxs_tensor=ttnn_repeat_1,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_7, False)
    ttnn.deallocate(ttnn_repeat_1, False)
    ttnn_reshape_43 = ttnn.reshape(
        ttnn_concat_15,
        [1, 16, 12, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_15, False)
    ttnn_transformer_scaled_dot_product_attention_decode_3 = (
        ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn_reshape_43,
            ttnn_to_device_100,
            ttnn_to_device_101,
            is_causal=False,
            attn_mask=ttnn_repeat_2,
            cur_pos_tensor=None,
            attention_sink=None,
            scale=0.08837890625,
            sliding_window_size=None,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
    )
    ttnn.deallocate(ttnn_reshape_43, False)
    ttnn.deallocate(ttnn_repeat_2, False)
    ttnn_reshape_44 = ttnn.reshape(
        ttnn_transformer_scaled_dot_product_attention_decode_3,
        [16, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_transformer_scaled_dot_product_attention_decode_3, False)
    ttnn_matmul_13 = ttnn.matmul(
        ttnn_reshape_44,
        ce_cache_forward["main_const_eval_23"],
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
    ttnn.deallocate(ttnn_reshape_44, False)
    ttnn_reshape_45 = ttnn.reshape(
        ttnn_matmul_13,
        [1, 1, 16, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_13, False)
    ttnn_reduce_scatter_6 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_45,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_45, False)
    ttnn_reshape_46 = ttnn.reshape(
        ttnn_reduce_scatter_6,
        [16, 640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_6, False)
    ttnn_all_gather_8 = ttnn.all_gather(
        input_tensor=ttnn_reshape_46,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reshape_46, False)
    ttnn_add_6 = ttnn.add(
        ttnn_add_5,
        ttnn_all_gather_8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_8, False)
    ttnn.deallocate(ttnn_add_5, False)
    ttnn_reshape_47 = ttnn.reshape(
        ttnn_add_6,
        [16, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_rms_norm_15 = ttnn.rms_norm(
        ttnn_reshape_47,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_103,
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
    ttnn.deallocate(ttnn_reshape_47, False)
    ttnn_reshape_48 = ttnn.reshape(
        ttnn_rms_norm_15,
        [16, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_33 = ttnn.typecast(
        ttnn_reshape_48,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_matmul_14 = ttnn.matmul(
        ttnn_typecast_33,
        ce_cache_forward["main_const_eval_26"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.FLOAT32,
        program_config=None,
        activation="sigmoid",
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_typecast_33, False)
    ttnn_add_7 = ttnn.add(
        ttnn_matmul_14,
        ce_cache_forward["main_const_eval_3"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_49 = ttnn.reshape(
        ttnn_add_7,
        [16, 1, 160],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_34 = ttnn.typecast(
        ttnn_reshape_49,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_49, False)
    v_15, v_16 = ttnn.topk(
        ttnn_typecast_34,
        2,
        -1,
        True,
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_16, False)
    ttnn.deallocate(ttnn_typecast_34, False)
    ttnn_typecast_35 = ttnn.typecast(
        v_15,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_15, False)
    ttnn_sum_0 = ttnn.sum(
        ttnn_typecast_35,
        [2],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_typecast_35, False)
    ttnn_typecast_36 = ttnn.typecast(
        ttnn_sum_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_sum_0, False)
    v_17, v_18 = ttnn.topk(
        ttnn_typecast_36,
        1,
        -1,
        True,
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_17, False)
    ttnn.deallocate(ttnn_typecast_36, False)
    ttnn_typecast_37 = ttnn.typecast(
        v_18,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_18, False)
    ttnn_reshape_50 = ttnn.reshape(
        ttnn_typecast_37,
        [16, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_37, False)
    ttnn_concat_17 = ttnn.concat(
        [ce_cache_forward["main_const_eval_24"], ttnn_reshape_50],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_50, False)
    ttnn_all_gather_9 = ttnn.all_gather(
        input_tensor=ttnn_concat_17,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_concat_17, False)
    ttnn_reshape_51 = ttnn.reshape(
        ttnn_all_gather_9,
        [64, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_9, False)
    ttnn_slice_24 = ttnn.slice(
        ttnn_reshape_51,
        [0, 0],
        [64, 1],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_slice_25 = ttnn.slice(
        ttnn_reshape_51,
        [0, 1],
        [64, 2],
        [1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_51, False)
    ttnn_add_8 = ttnn.add(
        ttnn_slice_24,
        ttnn_slice_25,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_slice_25, False)
    ttnn.deallocate(ttnn_slice_24, False)
    ttnn_reshape_52 = ttnn.reshape(
        ttnn_add_8,
        [64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_8, False)
    ttnn_to_layout_67 = ttnn.to_layout(
        ttnn_reshape_52,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_52, False)
    ttnn_scatter_0 = ttnn.scatter(
        input=ce_cache_forward["main_const_eval_7"],
        dim=0,
        index=ttnn_to_layout_67,
        src=ce_cache_forward["main_const_eval_45"],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_67, False)
    ttnn_to_layout_68 = ttnn.to_layout(
        ttnn_scatter_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_scatter_0, False)
    ttnn_reshape_53 = ttnn.reshape(
        ttnn_to_layout_68,
        [64, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_68, False)
    ttnn_to_layout_69 = ttnn.to_layout(
        ttnn_reshape_53,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_53, False)
    ttnn_mesh_partition_0 = ttnn.mesh_partition(
        input_tensor=ttnn_to_layout_69,
        dim=0,
        cluster_axis=0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_69, False)
    ttnn_repeat_interleave_0 = ttnn.repeat_interleave(
        ttnn_mesh_partition_0,
        160,
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mesh_partition_0, False)
    ttnn_ne_0 = ttnn.ne(
        ttnn_repeat_interleave_0,
        var_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_repeat_interleave_0, False)
    ttnn_typecast_38 = ttnn.typecast(
        ttnn_ne_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_ne_0, False)
    ttnn_where_1 = ttnn.where(
        ttnn_typecast_38,
        ttnn_add_7,
        var_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_38, False)
    ttnn.deallocate(ttnn_add_7, False)
    ttnn_typecast_39 = ttnn.typecast(
        ttnn_where_1,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_where_1, False)
    v_19, v_20 = ttnn.topk(
        ttnn_typecast_39,
        8,
        -1,
        True,
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_19, False)
    ttnn.deallocate(ttnn_typecast_39, False)
    ttnn_typecast_40 = ttnn.typecast(
        v_20,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_20, False)
    ttnn_typecast_41 = ttnn.typecast(
        ttnn_typecast_40,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_54 = ttnn.reshape(
        ttnn_typecast_41,
        [16, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_41, False)
    ttnn_concat_18 = ttnn.concat(
        [ce_cache_forward["main_const_eval_29"], ttnn_reshape_54],
        2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_54, False)
    ttnn_all_gather_10 = ttnn.all_gather(
        input_tensor=ttnn_matmul_14,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_matmul_14, False)
    ttnn_reshape_55 = ttnn.reshape(
        ttnn_all_gather_10,
        [10240, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_10, False)
    ttnn_typecast_42 = ttnn.typecast(
        ttnn_concat_18,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_concat_18, False)
    ttnn_matmul_15 = ttnn.matmul(
        ttnn_typecast_42,
        ce_cache_forward["main_const_eval_43"],
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
    ttnn.deallocate(ttnn_typecast_42, False)
    ttnn_reshape_56 = ttnn.reshape(
        ttnn_matmul_15,
        [128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_15, False)
    ttnn_typecast_43 = ttnn.typecast(
        ttnn_reshape_56,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_56, False)
    ttnn_to_layout_70 = ttnn.to_layout(
        ttnn_typecast_43,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_43, False)
    ttnn_typecast_44 = ttnn.typecast(
        ttnn_reshape_55,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_55, False)
    ttnn_to_layout_71 = ttnn.to_layout(
        ttnn_typecast_44,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_44, False)
    ttnn_embedding_1 = ttnn.embedding(
        ttnn_to_layout_70,
        ttnn_to_layout_71,
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_71, False)
    ttnn.deallocate(ttnn_to_layout_70, False)
    ttnn_typecast_45 = ttnn.typecast(
        ttnn_embedding_1,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_embedding_1, False)
    ttnn_reshape_57 = ttnn.reshape(
        ttnn_typecast_45,
        [16, 8],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_sum_1 = ttnn.sum(
        ttnn_reshape_57,
        [1],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_57, False)
    ttnn_add_9 = ttnn.add(
        ttnn_sum_1,
        ce_cache_forward["main_const_eval_31"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_sum_1, False)
    ttnn_reshape_58 = ttnn.reshape(
        ttnn_add_9,
        [16, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_9, False)
    ttnn_reshape_59 = ttnn.reshape(
        ttnn_typecast_45,
        [16, 1, 8],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_45, False)
    ttnn_divide_0 = ttnn.divide(
        ttnn_reshape_59,
        ttnn_reshape_58,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_59, False)
    ttnn.deallocate(ttnn_reshape_58, False)
    ttnn_multiply_3 = ttnn.multiply(
        ttnn_divide_0,
        ce_cache_forward["main_const_eval_30"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_divide_0, False)
    ttnn_reshape_60 = ttnn.reshape(
        ttnn_typecast_40,
        [16, 8, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_eq_0 = ttnn.eq(
        ttnn_reshape_60,
        ce_cache_forward["main_const_eval_38"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_60, False)
    ttnn_typecast_46 = ttnn.typecast(
        ttnn_eq_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_eq_0, False)
    ttnn_matmul_16 = ttnn.matmul(
        ttnn_multiply_3,
        ttnn_typecast_46,
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
    ttnn.deallocate(ttnn_multiply_3, False)
    ttnn_reshape_61 = ttnn.reshape(
        ttnn_matmul_16,
        [1, 16, 160],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_concat_19 = ttnn.concat(
        [ttnn_reshape_61, ttnn_reshape_61, ttnn_reshape_61, ttnn_reshape_61],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_61, False)
    ttnn_all_gather_11 = ttnn.all_gather(
        input_tensor=ttnn_concat_19,
        dim=1,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_concat_19, False)
    ttnn_reshape_62 = ttnn.reshape(
        ttnn_all_gather_11,
        [1, 1, 256, 160],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_11, False)
    ttnn_reshape_63 = ttnn.reshape(
        ttnn_rms_norm_15,
        [16, 1, 1, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_15, False)
    ttnn_reshape_64 = ttnn.reshape(
        ttnn_typecast_40,
        [16, 1, 1, 8],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_40, False)
    ttnn_all_gather_12 = ttnn.all_gather(
        input_tensor=ttnn_reshape_63,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reshape_63, False)
    ttnn_all_gather_13 = ttnn.all_gather(
        input_tensor=ttnn_reshape_64,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reshape_64, False)
    ttnn_to_layout_72 = ttnn.to_layout(
        ttnn_all_gather_12,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_12, False)
    ttnn_typecast_47 = ttnn.typecast(
        ttnn_all_gather_13,
        ttnn.DataType.UINT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_13, False)
    ttnn_from_device_25 = ttnn.from_device(ttnn_typecast_47)
    ttnn.deallocate(ttnn_typecast_47, False)
    ttnn_to_layout_73 = ttnn.to_layout(
        ttnn_from_device_25, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_25, False)
    ttnn_to_device_105 = ttnn.to_device(
        ttnn_to_layout_73,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_73, False)
    v_21, v_22 = ttnn.all_to_all_dispatch(
        input_tensor=ttnn_to_layout_72,
        expert_indices_tensor=ttnn_to_device_105,
        expert_mapping_tensor=var_2,
        cluster_axis=0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_105, False)
    ttnn.deallocate(ttnn_to_layout_72, False)
    ttnn_to_layout_74 = ttnn.to_layout(
        v_22,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_22, False)
    ttnn_typecast_48 = ttnn.typecast(
        ttnn_to_layout_74,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_74, False)
    ttnn_to_layout_75 = ttnn.to_layout(
        v_21,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_21, False)
    ttnn_reshape_65 = ttnn.reshape(
        ttnn_typecast_48,
        [1, 1, 256, 8],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_48, False)
    ttnn_typecast_49 = ttnn.typecast(
        ttnn_reshape_62,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_62, False)
    ttnn_to_layout_76 = ttnn.to_layout(
        ttnn_typecast_49,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_49, False)
    ttnn_typecast_50 = ttnn.typecast(
        ttnn_reshape_65,
        ttnn.DataType.UINT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_65, False)
    ttnn_from_device_26 = ttnn.from_device(ttnn_typecast_50)
    ttnn.deallocate(ttnn_typecast_50, False)
    ttnn_to_layout_77 = ttnn.to_layout(
        ttnn_from_device_26, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_26, False)
    ttnn_to_device_106 = ttnn.to_device(
        ttnn_to_layout_77,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_77, False)
    v_23, v_24 = ttnn.moe_expert_token_remap(
        topk_tensor=ttnn_to_layout_76,
        expert_mapping_tensor=var_2,
        expert_metadata_tensor=ttnn_to_device_106,
        reduction_size=32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_23, False)
    ttnn.deallocate(ttnn_to_layout_76, False)
    ttnn_to_layout_78 = ttnn.to_layout(
        v_24,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_typecast_51 = ttnn.typecast(
        ttnn_to_layout_78,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_78, False)
    ttnn_add_10 = ttnn.add(
        ttnn_to_device_99,
        var_1,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_99, False)
    ttnn_reshape_66 = ttnn.reshape(
        ttnn_to_layout_75,
        [8, 1, 32, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_75, False)
    ttnn_reshape_67 = ttnn.reshape(
        ttnn_typecast_51,
        [8, 1, 1, 5],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_51, False)
    ttnn_typecast_52 = ttnn.typecast(
        ttnn_reshape_67,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_67, False)
    ttnn_to_layout_79 = ttnn.to_layout(
        ttnn_typecast_52,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_52, False)
    ttnn_sparse_matmul_0 = ttnn.sparse_matmul(
        input_tensor_a=ttnn_reshape_66,
        input_tensor_b=ce_cache_forward["main_const_eval_13"],
        sparsity=ttnn_to_layout_79,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 9),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=6,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=0,
            untilize_out=False,
        ),
        nnz=None,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
    )
    ttnn_reshape_68 = ttnn.reshape(
        ttnn_sparse_matmul_0,
        [8, 5, 32, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_sparse_matmul_0, False)
    ttnn_silu_0 = ttnn.silu(
        ttnn_reshape_68,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_68, False)
    ttnn_sparse_matmul_1 = ttnn.sparse_matmul(
        input_tensor_a=ttnn_reshape_66,
        input_tensor_b=ce_cache_forward["main_const_eval_14"],
        sparsity=ttnn_to_layout_79,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 9),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=6,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=0,
            untilize_out=False,
        ),
        nnz=None,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
    )
    ttnn.deallocate(ttnn_to_layout_79, False)
    ttnn.deallocate(ttnn_reshape_66, False)
    ttnn_reshape_69 = ttnn.reshape(
        ttnn_sparse_matmul_1,
        [8, 5, 32, 1536],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_sparse_matmul_1, False)
    ttnn_multiply_4 = ttnn.multiply(
        ttnn_silu_0,
        ttnn_reshape_69,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_69, False)
    ttnn.deallocate(ttnn_silu_0, False)
    ttnn_typecast_53 = ttnn.typecast(
        v_24,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(v_24, False)
    ttnn_sparse_matmul_2 = ttnn.sparse_matmul(
        input_tensor_a=ttnn_multiply_4,
        input_tensor_b=ce_cache_forward["main_const_eval_17"],
        sparsity=ttnn_typecast_53,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 9),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=20,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=0,
            untilize_out=False,
        ),
        nnz=None,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
    )
    ttnn.deallocate(ttnn_typecast_53, False)
    ttnn.deallocate(ttnn_multiply_4, False)
    ttnn_permute_30 = ttnn.permute(
        ttnn_sparse_matmul_2,
        [1, 0, 2, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_sparse_matmul_2, False)
    ttnn_reshape_70 = ttnn.reshape(
        ttnn_permute_30,
        [5, 1, 256, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_30, False)
    ttnn_to_layout_80 = ttnn.to_layout(
        ttnn_reshape_70,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_70, False)
    ttnn_all_to_all_combine_0 = ttnn.all_to_all_combine(
        input_tensor=ttnn_to_layout_80,
        expert_metadata_tensor=ttnn_to_device_106,
        expert_mapping_tensor=var_2,
        cluster_axis=0,
        output_shard_dim=2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_80, False)
    ttnn.deallocate(ttnn_to_device_106, False)
    ttnn_to_layout_81 = ttnn.to_layout(
        ttnn_all_to_all_combine_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_to_all_combine_0, False)
    ttnn_reduce_scatter_7 = ttnn.reduce_scatter(
        input_tensor=ttnn_to_layout_81,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_to_layout_81, False)
    ttnn_all_gather_14 = ttnn.all_gather(
        input_tensor=ttnn_reduce_scatter_7,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reduce_scatter_7, False)
    ttnn_to_layout_82 = ttnn.to_layout(
        ttnn_all_gather_14,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_14, False)
    ttnn_mesh_partition_1 = ttnn.mesh_partition(
        input_tensor=ttnn_to_layout_82,
        dim=2,
        cluster_axis=0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_82, False)
    ttnn_typecast_54 = ttnn.typecast(
        ttnn_mesh_partition_1,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mesh_partition_1, False)
    ttnn_reshape_71 = ttnn.reshape(
        ttnn_matmul_16,
        [16, 160, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_16, False)
    ttnn_matmul_17 = ttnn.matmul(
        ttnn_typecast_46,
        ttnn_reshape_71,
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
    ttnn.deallocate(ttnn_reshape_71, False)
    ttnn.deallocate(ttnn_typecast_46, False)
    ttnn_reshape_72 = ttnn.reshape(
        ttnn_matmul_17,
        [16, 8],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_17, False)
    ttnn_permute_31 = ttnn.permute(
        ttnn_reshape_72,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_72, False)
    ttnn_reshape_73 = ttnn.reshape(
        ttnn_permute_31,
        [8, 1, 16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_31, False)
    ttnn_multiply_5 = ttnn.multiply(
        ttnn_typecast_54,
        ttnn_reshape_73,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reshape_73, False)
    ttnn.deallocate(ttnn_typecast_54, False)
    ttnn_sum_2 = ttnn.sum(
        ttnn_multiply_5,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_multiply_5, False)
    ttnn_typecast_55 = ttnn.typecast(
        ttnn_sum_2,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_sum_2, False)
    ttnn_reshape_74 = ttnn.reshape(
        ttnn_typecast_55,
        [16, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_55, False)
    ttnn_matmul_18 = ttnn.matmul(
        ttnn_reshape_48,
        ce_cache_forward["main_const_eval_16"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation="silu",
        compute_kernel_config=None,
    )
    ttnn_matmul_19 = ttnn.matmul(
        ttnn_reshape_48,
        ce_cache_forward["main_const_eval_42"],
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
    ttnn.deallocate(ttnn_reshape_48, False)
    ttnn_multiply_6 = ttnn.multiply(
        ttnn_matmul_18,
        ttnn_matmul_19,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_19, False)
    ttnn.deallocate(ttnn_matmul_18, False)
    ttnn_matmul_20 = ttnn.matmul(
        ttnn_multiply_6,
        ce_cache_forward["main_const_eval_9"],
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
    ttnn.deallocate(ttnn_multiply_6, False)
    ttnn_reshape_75 = ttnn.reshape(
        ttnn_matmul_20,
        [1, 1, 16, 5120],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_20, False)
    ttnn_reduce_scatter_8 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_75,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_75, False)
    ttnn_reshape_76 = ttnn.reshape(
        ttnn_reduce_scatter_8,
        [16, 640],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_reduce_scatter_8, False)
    ttnn_all_gather_15 = ttnn.all_gather(
        input_tensor=ttnn_reshape_76,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reshape_76, False)
    ttnn_add_11 = ttnn.add(
        ttnn_reshape_74,
        ttnn_all_gather_15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_all_gather_15, False)
    ttnn.deallocate(ttnn_reshape_74, False)
    ttnn_add_12 = ttnn.add(
        ttnn_add_6,
        ttnn_add_11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_add_11, False)
    ttnn.deallocate(ttnn_add_6, False)
    ttnn_rms_norm_16 = ttnn.rms_norm(
        ttnn_add_12,
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_device_102,
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
    ttnn.deallocate(ttnn_add_12, False)
    ttnn_matmul_21 = ttnn.matmul(
        ttnn_rms_norm_16,
        ce_cache_forward["main_const_eval_20"],
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
    ttnn.deallocate(ttnn_rms_norm_16, False)
    ttnn_reshape_77 = ttnn.reshape(
        ttnn_matmul_21,
        [16, 1, 18944],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_matmul_21, False)
    ttnn_all_gather_16 = ttnn.all_gather(
        input_tensor=ttnn_reshape_77,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_reshape_77, False)
    ttnn_all_gather_17 = ttnn.all_gather(
        input_tensor=ttnn_all_gather_16,
        dim=2,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn.deallocate(ttnn_all_gather_16, False)
    ttnn_to_layout_83 = ttnn.to_layout(
        ttnn_all_gather_17,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mesh_partition_2 = ttnn.mesh_partition(
        input_tensor=ttnn_to_layout_83,
        dim=0,
        cluster_axis=0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_83, False)
    ttnn_argmax_0 = ttnn.argmax(
        ttnn_mesh_partition_2,
        2,
        True,
        sub_core_grids=None,
        use_multicore=True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mesh_partition_2, False)
    ttnn_reshape_78 = ttnn.reshape(
        ttnn_argmax_0,
        [16, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_argmax_0, False)
    ttnn_all_gather_18 = ttnn.all_gather(
        input_tensor=ttnn_reshape_78,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        num_links=None,
        topology=None,
    )
    ttnn_add_13 = ttnn.add(
        ttnn_to_device_77,
        var_1,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_77, False)
    return [
        ttnn_to_device_79,
        ttnn_to_device_80,
        ttnn_add_10,
        ttnn_to_device_86,
        ttnn_to_device_87,
        ttnn_add_10,
        ttnn_to_device_93,
        ttnn_to_device_94,
        ttnn_add_10,
        ttnn_to_device_100,
        ttnn_to_device_101,
        ttnn_add_10,
        ttnn_reshape_78,
        ttnn_all_gather_18,
        ttnn_add_13,
        ttnn_all_gather_17,
    ]


def consteval_forward(ce_cache, input, device):
    if not ce_cache:
        main_const_eval_0_0 = main_const_eval_0([input[5]], device)
        ce_cache["main_const_eval_0"] = main_const_eval_0_0[0]
        main_const_eval_1_0 = main_const_eval_1(
            [input[49], input[62], input[72]], device
        )
        ce_cache["main_const_eval_1"] = main_const_eval_1_0[0]
        main_const_eval_2_0 = main_const_eval_2(
            [input[2], input[11], input[23]], device
        )
        ce_cache["main_const_eval_2"] = main_const_eval_2_0[0]
        main_const_eval_3_0 = main_const_eval_3([input[74]], device)
        ce_cache["main_const_eval_3"] = main_const_eval_3_0[0]
        main_const_eval_4_0 = main_const_eval_4([input[54]], device)
        ce_cache["main_const_eval_4"] = main_const_eval_4_0[0]
        main_const_eval_5_0 = main_const_eval_5(device)
        ce_cache["main_const_eval_5"] = main_const_eval_5_0[0]
        main_const_eval_6_0 = main_const_eval_6(device)
        ce_cache["main_const_eval_6"] = main_const_eval_6_0[0]
        main_const_eval_7_0 = main_const_eval_7(device)
        ce_cache["main_const_eval_7"] = main_const_eval_7_0[0]
        main_const_eval_8_0 = main_const_eval_8([input[52]], device)
        ce_cache["main_const_eval_8"] = main_const_eval_8_0[0]
        main_const_eval_9_0 = main_const_eval_9([input[66]], device)
        ce_cache["main_const_eval_9"] = main_const_eval_9_0[0]
        main_const_eval_10_0 = main_const_eval_10(device)
        ce_cache["main_const_eval_10"] = main_const_eval_10_0[0]
        main_const_eval_11_0 = main_const_eval_11(device)
        ce_cache["main_const_eval_11"] = main_const_eval_11_0[0]
        main_const_eval_12_0 = main_const_eval_12(
            [input[15], input[28], input[40]], device
        )
        ce_cache["main_const_eval_12"] = main_const_eval_12_0[0]
        main_const_eval_13_0 = main_const_eval_13([input[79]], device)
        ce_cache["main_const_eval_13"] = main_const_eval_13_0[0]
        main_const_eval_14_0 = main_const_eval_14([input[78]], device)
        ce_cache["main_const_eval_14"] = main_const_eval_14_0[0]
        main_const_eval_15_0 = main_const_eval_15(
            [input[14], input[27], input[39]], device
        )
        ce_cache["main_const_eval_15"] = main_const_eval_15_0[0]
        main_const_eval_16_0 = main_const_eval_16([input[73]], device)
        ce_cache["main_const_eval_16"] = main_const_eval_16_0[0]
        main_const_eval_17_0 = main_const_eval_17([input[77]], device)
        ce_cache["main_const_eval_17"] = main_const_eval_17_0[0]
        main_const_eval_18_0 = main_const_eval_18([input[24]], device)
        ce_cache["main_const_eval_18"] = main_const_eval_18_0[0]
        main_const_eval_19_0 = main_const_eval_19(
            [input[32], input[45], input[57]], device
        )
        ce_cache["main_const_eval_19"] = main_const_eval_19_0[0]
        main_const_eval_20_0 = main_const_eval_20([input[64]], device)
        ce_cache["main_const_eval_20"] = main_const_eval_20_0[0]
        main_const_eval_21_0 = main_const_eval_21(device)
        ce_cache["main_const_eval_21"] = main_const_eval_21_0[0]
        main_const_eval_22_0 = main_const_eval_22([input[35]], device)
        ce_cache["main_const_eval_22"] = main_const_eval_22_0[0]
        main_const_eval_23_0 = main_const_eval_23([input[69]], device)
        ce_cache["main_const_eval_23"] = main_const_eval_23_0[0]
        main_const_eval_24_0 = main_const_eval_24(device)
        ce_cache["main_const_eval_24"] = main_const_eval_24_0[0]
        main_const_eval_25_0 = main_const_eval_25(
            [input[1], input[10], input[22]], device
        )
        ce_cache["main_const_eval_25"] = main_const_eval_25_0[0]
        main_const_eval_26_0 = main_const_eval_26([input[75]], device)
        ce_cache["main_const_eval_26"] = main_const_eval_26_0[0]
        main_const_eval_27_0 = main_const_eval_27([input[7]], device)
        ce_cache["main_const_eval_27"] = main_const_eval_27_0[0]
        main_const_eval_28_0 = main_const_eval_28([input[51]], device)
        ce_cache["main_const_eval_28"] = main_const_eval_28_0[0]
        main_const_eval_29_0 = main_const_eval_29(device)
        ce_cache["main_const_eval_29"] = main_const_eval_29_0[0]
        main_const_eval_30_0 = main_const_eval_30(device)
        ce_cache["main_const_eval_30"] = main_const_eval_30_0[0]
        main_const_eval_31_0 = main_const_eval_31(device)
        ce_cache["main_const_eval_31"] = main_const_eval_31_0[0]
        main_const_eval_32_0 = main_const_eval_32([input[18]], device)
        ce_cache["main_const_eval_32"] = main_const_eval_32_0[0]
        main_const_eval_33_0 = main_const_eval_33(
            [input[48], input[61], input[71]], device
        )
        ce_cache["main_const_eval_33"] = main_const_eval_33_0[0]
        main_const_eval_34_0 = main_const_eval_34([input[37]], device)
        ce_cache["main_const_eval_34"] = main_const_eval_34_0[0]
        main_const_eval_35_0 = main_const_eval_35([input[41]], device)
        ce_cache["main_const_eval_35"] = main_const_eval_35_0[0]
        main_const_eval_36_0 = main_const_eval_36([input[34]], device)
        ce_cache["main_const_eval_36"] = main_const_eval_36_0[0]
        main_const_eval_37_0 = main_const_eval_37(
            [input[31], input[44], input[56]], device
        )
        ce_cache["main_const_eval_37"] = main_const_eval_37_0[0]
        main_const_eval_38_0 = main_const_eval_38(device)
        ce_cache["main_const_eval_38"] = main_const_eval_38_0[0]
        main_const_eval_39_0 = main_const_eval_39([input[76]], device)
        ce_cache["main_const_eval_39"] = main_const_eval_39_0[0]
        main_const_eval_40_0 = main_const_eval_40([input[20]], device)
        ce_cache["main_const_eval_40"] = main_const_eval_40_0[0]
        main_const_eval_41_0 = main_const_eval_41([input[58]], device)
        ce_cache["main_const_eval_41"] = main_const_eval_41_0[0]
        main_const_eval_42_0 = main_const_eval_42([input[67]], device)
        ce_cache["main_const_eval_42"] = main_const_eval_42_0[0]
        main_const_eval_43_0 = main_const_eval_43(device)
        ce_cache["main_const_eval_43"] = main_const_eval_43_0[0]
        main_const_eval_44_0 = main_const_eval_44([input[17]], device)
        ce_cache["main_const_eval_44"] = main_const_eval_44_0[0]
        main_const_eval_45_0 = main_const_eval_45(device)
        ce_cache["main_const_eval_45"] = main_const_eval_45_0[0]
    return ce_cache
