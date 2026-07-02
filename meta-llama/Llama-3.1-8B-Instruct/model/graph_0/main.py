import time
import ttnn
import utils
import torch
import model_pt
from utils import calculate_pcc
from model_ttnn import ModelTTNN



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
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg7.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg9.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg10.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg20.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg21.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg23.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg24.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg33.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg34.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg36.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_14 = utils.load_tensor(
        "./tensors/arg37.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_15 = utils.load_tensor(
        "./tensors/arg46.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_16 = utils.load_tensor(
        "./tensors/arg47.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg49.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg50.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_19 = utils.load_tensor(
        "./tensors/arg59.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_20 = utils.load_tensor(
        "./tensors/arg60.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_21 = utils.load_tensor(
        "./tensors/arg62.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_22 = utils.load_tensor(
        "./tensors/arg63.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_23 = utils.load_tensor(
        "./tensors/arg72.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg73.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_25 = utils.load_tensor(
        "./tensors/arg75.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_26 = utils.load_tensor(
        "./tensors/arg76.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_27 = utils.load_tensor(
        "./tensors/arg85.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_28 = utils.load_tensor(
        "./tensors/arg86.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg88.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg89.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_31 = utils.load_tensor(
        "./tensors/arg98.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_32 = utils.load_tensor(
        "./tensors/arg99.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_33 = utils.load_tensor(
        "./tensors/arg101.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_34 = utils.load_tensor(
        "./tensors/arg102.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_35 = utils.load_tensor(
        "./tensors/arg111.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg112.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_37 = utils.load_tensor(
        "./tensors/arg114.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_38 = utils.load_tensor(
        "./tensors/arg115.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_39 = utils.load_tensor(
        "./tensors/arg124.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_40 = utils.load_tensor(
        "./tensors/arg125.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_41 = utils.load_tensor(
        "./tensors/arg127.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_42 = utils.load_tensor(
        "./tensors/arg128.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_43 = utils.load_tensor(
        "./tensors/arg137.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_44 = utils.load_tensor(
        "./tensors/arg138.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_45 = utils.load_tensor(
        "./tensors/arg140.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_46 = utils.load_tensor(
        "./tensors/arg141.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_47 = utils.load_tensor(
        "./tensors/arg150.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_48 = utils.load_tensor(
        "./tensors/arg151.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_49 = utils.load_tensor(
        "./tensors/arg153.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_50 = utils.load_tensor(
        "./tensors/arg154.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_51 = utils.load_tensor(
        "./tensors/arg163.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_52 = utils.load_tensor(
        "./tensors/arg164.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_53 = utils.load_tensor(
        "./tensors/arg166.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_54 = utils.load_tensor(
        "./tensors/arg167.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_55 = utils.load_tensor(
        "./tensors/arg176.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_56 = utils.load_tensor(
        "./tensors/arg177.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_57 = utils.load_tensor(
        "./tensors/arg179.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_58 = utils.load_tensor(
        "./tensors/arg180.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_59 = utils.load_tensor(
        "./tensors/arg189.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_60 = utils.load_tensor(
        "./tensors/arg190.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_61 = utils.load_tensor(
        "./tensors/arg192.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_62 = utils.load_tensor(
        "./tensors/arg193.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_63 = utils.load_tensor(
        "./tensors/arg202.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_64 = utils.load_tensor(
        "./tensors/arg203.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_65 = utils.load_tensor(
        "./tensors/arg205.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_66 = utils.load_tensor(
        "./tensors/arg206.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_67 = utils.load_tensor(
        "./tensors/arg215.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_68 = utils.load_tensor(
        "./tensors/arg216.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_69 = utils.load_tensor(
        "./tensors/arg218.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_70 = utils.load_tensor(
        "./tensors/arg219.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_71 = utils.load_tensor(
        "./tensors/arg228.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_72 = utils.load_tensor(
        "./tensors/arg229.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_73 = utils.load_tensor(
        "./tensors/arg231.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_74 = utils.load_tensor(
        "./tensors/arg232.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_75 = utils.load_tensor(
        "./tensors/arg241.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_76 = utils.load_tensor(
        "./tensors/arg242.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_77 = utils.load_tensor(
        "./tensors/arg244.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_78 = utils.load_tensor(
        "./tensors/arg245.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_79 = utils.load_tensor(
        "./tensors/arg254.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_80 = utils.load_tensor(
        "./tensors/arg255.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_81 = utils.load_tensor(
        "./tensors/arg257.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_82 = utils.load_tensor(
        "./tensors/arg258.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_83 = utils.load_tensor(
        "./tensors/arg267.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_84 = utils.load_tensor(
        "./tensors/arg268.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_85 = utils.load_tensor(
        "./tensors/arg270.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_86 = utils.load_tensor(
        "./tensors/arg271.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_87 = utils.load_tensor(
        "./tensors/arg280.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_88 = utils.load_tensor(
        "./tensors/arg281.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_89 = utils.load_tensor(
        "./tensors/arg283.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_90 = utils.load_tensor(
        "./tensors/arg284.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_91 = utils.load_tensor(
        "./tensors/arg293.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_92 = utils.load_tensor(
        "./tensors/arg294.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_93 = utils.load_tensor(
        "./tensors/arg296.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_94 = utils.load_tensor(
        "./tensors/arg297.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_95 = utils.load_tensor(
        "./tensors/arg306.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_96 = utils.load_tensor(
        "./tensors/arg307.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_97 = utils.load_tensor(
        "./tensors/arg309.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_98 = utils.load_tensor(
        "./tensors/arg310.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_99 = utils.load_tensor(
        "./tensors/arg319.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_100 = utils.load_tensor(
        "./tensors/arg320.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_101 = utils.load_tensor(
        "./tensors/arg322.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_102 = utils.load_tensor(
        "./tensors/arg323.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_103 = utils.load_tensor(
        "./tensors/arg332.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_104 = utils.load_tensor(
        "./tensors/arg333.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_105 = utils.load_tensor(
        "./tensors/arg335.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_106 = utils.load_tensor(
        "./tensors/arg336.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_107 = utils.load_tensor(
        "./tensors/arg345.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_108 = utils.load_tensor(
        "./tensors/arg346.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_109 = utils.load_tensor(
        "./tensors/arg348.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_110 = utils.load_tensor(
        "./tensors/arg349.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_111 = utils.load_tensor(
        "./tensors/arg358.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_112 = utils.load_tensor(
        "./tensors/arg359.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_113 = utils.load_tensor(
        "./tensors/arg361.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_114 = utils.load_tensor(
        "./tensors/arg362.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_115 = utils.load_tensor(
        "./tensors/arg371.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_116 = utils.load_tensor(
        "./tensors/arg372.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_117 = utils.load_tensor(
        "./tensors/arg374.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_118 = utils.load_tensor(
        "./tensors/arg375.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_119 = utils.load_tensor(
        "./tensors/arg384.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_120 = utils.load_tensor(
        "./tensors/arg385.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_121 = utils.load_tensor(
        "./tensors/arg387.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_122 = utils.load_tensor(
        "./tensors/arg388.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_123 = utils.load_tensor(
        "./tensors/arg397.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_124 = utils.load_tensor(
        "./tensors/arg398.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_125 = utils.load_tensor(
        "./tensors/arg400.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_126 = utils.load_tensor(
        "./tensors/arg401.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_127 = utils.load_tensor(
        "./tensors/arg410.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_128 = utils.load_tensor(
        "./tensors/arg411.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_129 = utils.load_tensor(
        "./tensors/arg413.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_130 = utils.load_tensor(
        "./tensors/arg414.tensorbin",
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
        utils_load_tensor_49,
        utils_load_tensor_50,
        utils_load_tensor_51,
        utils_load_tensor_52,
        utils_load_tensor_53,
        utils_load_tensor_54,
        utils_load_tensor_55,
        utils_load_tensor_56,
        utils_load_tensor_57,
        utils_load_tensor_58,
        utils_load_tensor_59,
        utils_load_tensor_60,
        utils_load_tensor_61,
        utils_load_tensor_62,
        utils_load_tensor_63,
        utils_load_tensor_64,
        utils_load_tensor_65,
        utils_load_tensor_66,
        utils_load_tensor_67,
        utils_load_tensor_68,
        utils_load_tensor_69,
        utils_load_tensor_70,
        utils_load_tensor_71,
        utils_load_tensor_72,
        utils_load_tensor_73,
        utils_load_tensor_74,
        utils_load_tensor_75,
        utils_load_tensor_76,
        utils_load_tensor_77,
        utils_load_tensor_78,
        utils_load_tensor_79,
        utils_load_tensor_80,
        utils_load_tensor_81,
        utils_load_tensor_82,
        utils_load_tensor_83,
        utils_load_tensor_84,
        utils_load_tensor_85,
        utils_load_tensor_86,
        utils_load_tensor_87,
        utils_load_tensor_88,
        utils_load_tensor_89,
        utils_load_tensor_90,
        utils_load_tensor_91,
        utils_load_tensor_92,
        utils_load_tensor_93,
        utils_load_tensor_94,
        utils_load_tensor_95,
        utils_load_tensor_96,
        utils_load_tensor_97,
        utils_load_tensor_98,
        utils_load_tensor_99,
        utils_load_tensor_100,
        utils_load_tensor_101,
        utils_load_tensor_102,
        utils_load_tensor_103,
        utils_load_tensor_104,
        utils_load_tensor_105,
        utils_load_tensor_106,
        utils_load_tensor_107,
        utils_load_tensor_108,
        utils_load_tensor_109,
        utils_load_tensor_110,
        utils_load_tensor_111,
        utils_load_tensor_112,
        utils_load_tensor_113,
        utils_load_tensor_114,
        utils_load_tensor_115,
        utils_load_tensor_116,
        utils_load_tensor_117,
        utils_load_tensor_118,
        utils_load_tensor_119,
        utils_load_tensor_120,
        utils_load_tensor_121,
        utils_load_tensor_122,
        utils_load_tensor_123,
        utils_load_tensor_124,
        utils_load_tensor_125,
        utils_load_tensor_126,
        utils_load_tensor_127,
        utils_load_tensor_128,
        utils_load_tensor_129,
        utils_load_tensor_130,
    ]


def main():
    device = utils.open_device()
    model = ModelTTNN(device)
    activations = load_activations_for__main(device)
    model(activations)
    return 0


def test_main():
    exact_pcc = 0.9921875

    model = model_pt.load_pytorch_model()
    pytorch_input = model_pt.load_input(model)

    device = utils.open_device()
    dram_interleaved = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )

    def to_int32(t):
        return ttnn.to_device(
            ttnn.from_torch(t, dtype=ttnn.DataType.INT32, layout=ttnn.Layout.ROW_MAJOR),
            device,
            dram_interleaved,
        )

    def to_bf16_tile(t):
        return ttnn.to_device(
            ttnn.from_torch(t, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE),
            device,
            dram_interleaved,
        )

    cache_position = pytorch_input["cache_position"]
    input_ids = pytorch_input["input_ids"]
    layers = pytorch_input["past_key_values"].layers

    # Build activations in the same order as load_activations_for__main():
    # [0] cache_position, [1] input_ids
    activations = [to_int32(cache_position), to_int32(input_ids)]
    # Layer 0: cp, keys, cp, values, cp (extra for attention mask)
    activations.extend([
        to_int32(cache_position),
        to_bf16_tile(layers[0].keys),
        to_int32(cache_position),
        to_bf16_tile(layers[0].values),
        to_int32(cache_position),
    ])
    # Layers 1-31: cp, keys, cp, values
    for layer in layers[1:]:
        activations.extend([
            to_int32(cache_position),
            to_bf16_tile(layer.keys),
            to_int32(cache_position),
            to_bf16_tile(layer.values),
        ])

    model = ModelTTNN(device)
    outputs = model(activations)

    ttnn_output = ttnn.to_torch(ttnn.from_device(outputs[-1]))
    ttnn_output = ttnn_output[:, -1, :]

    golden_output = model_pt.run_pytorch_model()

    pcc = calculate_pcc(ttnn_output, golden_output)
    print(f"\nPCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"

    batch_size = model_pt.BATCH_SIZE
    num_runs = 3
    print(f"\nPerformance ({num_runs} runs, batch_size={batch_size}):")
    for i in range(num_runs):
        start = time.perf_counter()
        model(activations)
        ttnn.synchronize_device(device)
        end = time.perf_counter()
        elapsed = end - start
        tps = batch_size / elapsed
        print(f"  Run {i+1}: {elapsed:.4f}s, {tps:.2f} TPS")


if __name__ == "__main__":
    main()
