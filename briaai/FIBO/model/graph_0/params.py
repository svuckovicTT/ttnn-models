import ttnn
import utils
import model_pt


_main_weights = {}


def load_weights_for__main():
    utils_DeviceGetter_get_device_243 = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )
    global _main_weights
    utils_load_tensor_52 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.proj_out.bias"] = utils_load_tensor_52
    utils_load_tensor_53 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.proj_out.weight"] = utils_load_tensor_53
    utils_load_tensor_54 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.norm_out.linear.bias"] = utils_load_tensor_54
    utils_load_tensor_55 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.norm_out.linear.weight"] = utils_load_tensor_55
    utils_load_tensor_56 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.time_embed.timestep_embedder.linear_2.bias"] = (
        utils_load_tensor_56
    )
    utils_load_tensor_57 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.time_embed.timestep_embedder.linear_2.weight"] = (
        utils_load_tensor_57
    )
    utils_load_tensor_58 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.time_embed.timestep_embedder.linear_1.bias"] = (
        utils_load_tensor_58
    )
    utils_load_tensor_59 = utils.load_tensor(
        "./tensors/arg7.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.time_embed.timestep_embedder.linear_1.weight"] = (
        utils_load_tensor_59
    )
    utils_load_tensor_60 = utils.load_tensor(
        "./tensors/arg9.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.37.proj_out.bias"] = (
        utils_load_tensor_60
    )
    utils_load_tensor_61 = utils.load_tensor(
        "./tensors/arg10.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.37.proj_out.weight"] = (
        utils_load_tensor_61
    )
    utils_load_tensor_62 = utils.load_tensor(
        "./tensors/arg11.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.37.proj_mlp.bias"] = (
        utils_load_tensor_62
    )
    utils_load_tensor_63 = utils.load_tensor(
        "./tensors/arg12.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.37.proj_mlp.weight"] = (
        utils_load_tensor_63
    )
    utils_load_tensor_64 = utils.load_tensor(
        "./tensors/arg13.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.37.norm.linear.bias"] = (
        utils_load_tensor_64
    )
    utils_load_tensor_65 = utils.load_tensor(
        "./tensors/arg14.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.37.norm.linear.weight"] = (
        utils_load_tensor_65
    )
    utils_load_tensor_66 = utils.load_tensor(
        "./tensors/arg15.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.36.proj_out.bias"] = (
        utils_load_tensor_66
    )
    utils_load_tensor_67 = utils.load_tensor(
        "./tensors/arg16.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.36.proj_out.weight"] = (
        utils_load_tensor_67
    )
    utils_load_tensor_68 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.36.proj_mlp.bias"] = (
        utils_load_tensor_68
    )
    utils_load_tensor_69 = utils.load_tensor(
        "./tensors/arg18.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.36.proj_mlp.weight"] = (
        utils_load_tensor_69
    )
    utils_load_tensor_70 = utils.load_tensor(
        "./tensors/arg19.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.36.norm.linear.bias"] = (
        utils_load_tensor_70
    )
    utils_load_tensor_71 = utils.load_tensor(
        "./tensors/arg20.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.36.norm.linear.weight"] = (
        utils_load_tensor_71
    )
    utils_load_tensor_72 = utils.load_tensor(
        "./tensors/arg21.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.35.proj_out.bias"] = (
        utils_load_tensor_72
    )
    utils_load_tensor_73 = utils.load_tensor(
        "./tensors/arg22.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.35.proj_out.weight"] = (
        utils_load_tensor_73
    )
    utils_load_tensor_74 = utils.load_tensor(
        "./tensors/arg23.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.35.proj_mlp.bias"] = (
        utils_load_tensor_74
    )
    utils_load_tensor_75 = utils.load_tensor(
        "./tensors/arg24.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.35.proj_mlp.weight"] = (
        utils_load_tensor_75
    )
    utils_load_tensor_76 = utils.load_tensor(
        "./tensors/arg25.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.35.norm.linear.bias"] = (
        utils_load_tensor_76
    )
    utils_load_tensor_77 = utils.load_tensor(
        "./tensors/arg26.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.35.norm.linear.weight"] = (
        utils_load_tensor_77
    )
    utils_load_tensor_78 = utils.load_tensor(
        "./tensors/arg27.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.34.proj_out.bias"] = (
        utils_load_tensor_78
    )
    utils_load_tensor_79 = utils.load_tensor(
        "./tensors/arg28.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.34.proj_out.weight"] = (
        utils_load_tensor_79
    )
    utils_load_tensor_80 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.34.proj_mlp.bias"] = (
        utils_load_tensor_80
    )
    utils_load_tensor_81 = utils.load_tensor(
        "./tensors/arg30.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.34.proj_mlp.weight"] = (
        utils_load_tensor_81
    )
    utils_load_tensor_82 = utils.load_tensor(
        "./tensors/arg31.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.34.norm.linear.bias"] = (
        utils_load_tensor_82
    )
    utils_load_tensor_83 = utils.load_tensor(
        "./tensors/arg32.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.34.norm.linear.weight"] = (
        utils_load_tensor_83
    )
    utils_load_tensor_84 = utils.load_tensor(
        "./tensors/arg33.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.33.proj_out.bias"] = (
        utils_load_tensor_84
    )
    utils_load_tensor_85 = utils.load_tensor(
        "./tensors/arg34.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.33.proj_out.weight"] = (
        utils_load_tensor_85
    )
    utils_load_tensor_86 = utils.load_tensor(
        "./tensors/arg35.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.33.proj_mlp.bias"] = (
        utils_load_tensor_86
    )
    utils_load_tensor_87 = utils.load_tensor(
        "./tensors/arg36.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.33.proj_mlp.weight"] = (
        utils_load_tensor_87
    )
    utils_load_tensor_88 = utils.load_tensor(
        "./tensors/arg37.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.33.norm.linear.bias"] = (
        utils_load_tensor_88
    )
    utils_load_tensor_89 = utils.load_tensor(
        "./tensors/arg38.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.33.norm.linear.weight"] = (
        utils_load_tensor_89
    )
    utils_load_tensor_90 = utils.load_tensor(
        "./tensors/arg39.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.32.proj_out.bias"] = (
        utils_load_tensor_90
    )
    utils_load_tensor_91 = utils.load_tensor(
        "./tensors/arg40.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.32.proj_out.weight"] = (
        utils_load_tensor_91
    )
    utils_load_tensor_92 = utils.load_tensor(
        "./tensors/arg41.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.32.proj_mlp.bias"] = (
        utils_load_tensor_92
    )
    utils_load_tensor_93 = utils.load_tensor(
        "./tensors/arg42.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.32.proj_mlp.weight"] = (
        utils_load_tensor_93
    )
    utils_load_tensor_94 = utils.load_tensor(
        "./tensors/arg43.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.32.norm.linear.bias"] = (
        utils_load_tensor_94
    )
    utils_load_tensor_95 = utils.load_tensor(
        "./tensors/arg44.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.32.norm.linear.weight"] = (
        utils_load_tensor_95
    )
    utils_load_tensor_96 = utils.load_tensor(
        "./tensors/arg45.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.31.proj_out.bias"] = (
        utils_load_tensor_96
    )
    utils_load_tensor_97 = utils.load_tensor(
        "./tensors/arg46.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.31.proj_out.weight"] = (
        utils_load_tensor_97
    )
    utils_load_tensor_98 = utils.load_tensor(
        "./tensors/arg47.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.31.proj_mlp.bias"] = (
        utils_load_tensor_98
    )
    utils_load_tensor_99 = utils.load_tensor(
        "./tensors/arg48.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.31.proj_mlp.weight"] = (
        utils_load_tensor_99
    )
    utils_load_tensor_100 = utils.load_tensor(
        "./tensors/arg49.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.31.norm.linear.bias"] = (
        utils_load_tensor_100
    )
    utils_load_tensor_101 = utils.load_tensor(
        "./tensors/arg50.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.31.norm.linear.weight"] = (
        utils_load_tensor_101
    )
    utils_load_tensor_102 = utils.load_tensor(
        "./tensors/arg51.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.30.proj_out.bias"] = (
        utils_load_tensor_102
    )
    utils_load_tensor_103 = utils.load_tensor(
        "./tensors/arg52.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.30.proj_out.weight"] = (
        utils_load_tensor_103
    )
    utils_load_tensor_104 = utils.load_tensor(
        "./tensors/arg53.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.30.proj_mlp.bias"] = (
        utils_load_tensor_104
    )
    utils_load_tensor_105 = utils.load_tensor(
        "./tensors/arg54.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.30.proj_mlp.weight"] = (
        utils_load_tensor_105
    )
    utils_load_tensor_106 = utils.load_tensor(
        "./tensors/arg55.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.30.norm.linear.bias"] = (
        utils_load_tensor_106
    )
    utils_load_tensor_107 = utils.load_tensor(
        "./tensors/arg56.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.30.norm.linear.weight"] = (
        utils_load_tensor_107
    )
    utils_load_tensor_108 = utils.load_tensor(
        "./tensors/arg57.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.29.proj_out.bias"] = (
        utils_load_tensor_108
    )
    utils_load_tensor_109 = utils.load_tensor(
        "./tensors/arg58.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.29.proj_out.weight"] = (
        utils_load_tensor_109
    )
    utils_load_tensor_110 = utils.load_tensor(
        "./tensors/arg59.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.29.proj_mlp.bias"] = (
        utils_load_tensor_110
    )
    utils_load_tensor_111 = utils.load_tensor(
        "./tensors/arg60.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.29.proj_mlp.weight"] = (
        utils_load_tensor_111
    )
    utils_load_tensor_112 = utils.load_tensor(
        "./tensors/arg61.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.29.norm.linear.bias"] = (
        utils_load_tensor_112
    )
    utils_load_tensor_113 = utils.load_tensor(
        "./tensors/arg62.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.29.norm.linear.weight"] = (
        utils_load_tensor_113
    )
    utils_load_tensor_114 = utils.load_tensor(
        "./tensors/arg63.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.28.proj_out.bias"] = (
        utils_load_tensor_114
    )
    utils_load_tensor_115 = utils.load_tensor(
        "./tensors/arg64.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.28.proj_out.weight"] = (
        utils_load_tensor_115
    )
    utils_load_tensor_116 = utils.load_tensor(
        "./tensors/arg65.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.28.proj_mlp.bias"] = (
        utils_load_tensor_116
    )
    utils_load_tensor_117 = utils.load_tensor(
        "./tensors/arg66.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.28.proj_mlp.weight"] = (
        utils_load_tensor_117
    )
    utils_load_tensor_118 = utils.load_tensor(
        "./tensors/arg67.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.28.norm.linear.bias"] = (
        utils_load_tensor_118
    )
    utils_load_tensor_119 = utils.load_tensor(
        "./tensors/arg68.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.28.norm.linear.weight"] = (
        utils_load_tensor_119
    )
    utils_load_tensor_120 = utils.load_tensor(
        "./tensors/arg69.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.27.proj_out.bias"] = (
        utils_load_tensor_120
    )
    utils_load_tensor_121 = utils.load_tensor(
        "./tensors/arg70.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.27.proj_out.weight"] = (
        utils_load_tensor_121
    )
    utils_load_tensor_122 = utils.load_tensor(
        "./tensors/arg71.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.27.proj_mlp.bias"] = (
        utils_load_tensor_122
    )
    utils_load_tensor_123 = utils.load_tensor(
        "./tensors/arg72.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.27.proj_mlp.weight"] = (
        utils_load_tensor_123
    )
    utils_load_tensor_124 = utils.load_tensor(
        "./tensors/arg73.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.27.norm.linear.bias"] = (
        utils_load_tensor_124
    )
    utils_load_tensor_125 = utils.load_tensor(
        "./tensors/arg74.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.27.norm.linear.weight"] = (
        utils_load_tensor_125
    )
    utils_load_tensor_126 = utils.load_tensor(
        "./tensors/arg75.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.26.proj_out.bias"] = (
        utils_load_tensor_126
    )
    utils_load_tensor_127 = utils.load_tensor(
        "./tensors/arg76.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.26.proj_out.weight"] = (
        utils_load_tensor_127
    )
    utils_load_tensor_128 = utils.load_tensor(
        "./tensors/arg77.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.26.proj_mlp.bias"] = (
        utils_load_tensor_128
    )
    utils_load_tensor_129 = utils.load_tensor(
        "./tensors/arg78.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.26.proj_mlp.weight"] = (
        utils_load_tensor_129
    )
    utils_load_tensor_130 = utils.load_tensor(
        "./tensors/arg79.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.26.norm.linear.bias"] = (
        utils_load_tensor_130
    )
    utils_load_tensor_131 = utils.load_tensor(
        "./tensors/arg80.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.26.norm.linear.weight"] = (
        utils_load_tensor_131
    )
    utils_load_tensor_132 = utils.load_tensor(
        "./tensors/arg81.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.25.proj_out.bias"] = (
        utils_load_tensor_132
    )
    utils_load_tensor_133 = utils.load_tensor(
        "./tensors/arg82.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.25.proj_out.weight"] = (
        utils_load_tensor_133
    )
    utils_load_tensor_134 = utils.load_tensor(
        "./tensors/arg83.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.25.proj_mlp.bias"] = (
        utils_load_tensor_134
    )
    utils_load_tensor_135 = utils.load_tensor(
        "./tensors/arg84.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.25.proj_mlp.weight"] = (
        utils_load_tensor_135
    )
    utils_load_tensor_136 = utils.load_tensor(
        "./tensors/arg85.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.25.norm.linear.bias"] = (
        utils_load_tensor_136
    )
    utils_load_tensor_137 = utils.load_tensor(
        "./tensors/arg86.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.25.norm.linear.weight"] = (
        utils_load_tensor_137
    )
    utils_load_tensor_138 = utils.load_tensor(
        "./tensors/arg87.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.24.proj_out.bias"] = (
        utils_load_tensor_138
    )
    utils_load_tensor_139 = utils.load_tensor(
        "./tensors/arg88.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.24.proj_out.weight"] = (
        utils_load_tensor_139
    )
    utils_load_tensor_140 = utils.load_tensor(
        "./tensors/arg89.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.24.proj_mlp.bias"] = (
        utils_load_tensor_140
    )
    utils_load_tensor_141 = utils.load_tensor(
        "./tensors/arg90.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.24.proj_mlp.weight"] = (
        utils_load_tensor_141
    )
    utils_load_tensor_142 = utils.load_tensor(
        "./tensors/arg91.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.24.norm.linear.bias"] = (
        utils_load_tensor_142
    )
    utils_load_tensor_143 = utils.load_tensor(
        "./tensors/arg92.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.24.norm.linear.weight"] = (
        utils_load_tensor_143
    )
    utils_load_tensor_144 = utils.load_tensor(
        "./tensors/arg93.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.23.proj_out.bias"] = (
        utils_load_tensor_144
    )
    utils_load_tensor_145 = utils.load_tensor(
        "./tensors/arg94.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.23.proj_out.weight"] = (
        utils_load_tensor_145
    )
    utils_load_tensor_146 = utils.load_tensor(
        "./tensors/arg95.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.23.proj_mlp.bias"] = (
        utils_load_tensor_146
    )
    utils_load_tensor_147 = utils.load_tensor(
        "./tensors/arg96.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.23.proj_mlp.weight"] = (
        utils_load_tensor_147
    )
    utils_load_tensor_148 = utils.load_tensor(
        "./tensors/arg97.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.23.norm.linear.bias"] = (
        utils_load_tensor_148
    )
    utils_load_tensor_149 = utils.load_tensor(
        "./tensors/arg98.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.23.norm.linear.weight"] = (
        utils_load_tensor_149
    )
    utils_load_tensor_150 = utils.load_tensor(
        "./tensors/arg99.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.22.proj_out.bias"] = (
        utils_load_tensor_150
    )
    utils_load_tensor_151 = utils.load_tensor(
        "./tensors/arg100.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.22.proj_out.weight"] = (
        utils_load_tensor_151
    )
    utils_load_tensor_152 = utils.load_tensor(
        "./tensors/arg101.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.22.proj_mlp.bias"] = (
        utils_load_tensor_152
    )
    utils_load_tensor_153 = utils.load_tensor(
        "./tensors/arg102.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.22.proj_mlp.weight"] = (
        utils_load_tensor_153
    )
    utils_load_tensor_154 = utils.load_tensor(
        "./tensors/arg103.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.22.norm.linear.bias"] = (
        utils_load_tensor_154
    )
    utils_load_tensor_155 = utils.load_tensor(
        "./tensors/arg104.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.22.norm.linear.weight"] = (
        utils_load_tensor_155
    )
    utils_load_tensor_156 = utils.load_tensor(
        "./tensors/arg105.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.21.proj_out.bias"] = (
        utils_load_tensor_156
    )
    utils_load_tensor_157 = utils.load_tensor(
        "./tensors/arg106.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.21.proj_out.weight"] = (
        utils_load_tensor_157
    )
    utils_load_tensor_158 = utils.load_tensor(
        "./tensors/arg107.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.21.proj_mlp.bias"] = (
        utils_load_tensor_158
    )
    utils_load_tensor_159 = utils.load_tensor(
        "./tensors/arg108.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.21.proj_mlp.weight"] = (
        utils_load_tensor_159
    )
    utils_load_tensor_160 = utils.load_tensor(
        "./tensors/arg109.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.21.norm.linear.bias"] = (
        utils_load_tensor_160
    )
    utils_load_tensor_161 = utils.load_tensor(
        "./tensors/arg110.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.21.norm.linear.weight"] = (
        utils_load_tensor_161
    )
    utils_load_tensor_162 = utils.load_tensor(
        "./tensors/arg111.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.20.proj_out.bias"] = (
        utils_load_tensor_162
    )
    utils_load_tensor_163 = utils.load_tensor(
        "./tensors/arg112.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.20.proj_out.weight"] = (
        utils_load_tensor_163
    )
    utils_load_tensor_164 = utils.load_tensor(
        "./tensors/arg113.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.20.proj_mlp.bias"] = (
        utils_load_tensor_164
    )
    utils_load_tensor_165 = utils.load_tensor(
        "./tensors/arg114.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.20.proj_mlp.weight"] = (
        utils_load_tensor_165
    )
    utils_load_tensor_166 = utils.load_tensor(
        "./tensors/arg115.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.20.norm.linear.bias"] = (
        utils_load_tensor_166
    )
    utils_load_tensor_167 = utils.load_tensor(
        "./tensors/arg116.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.20.norm.linear.weight"] = (
        utils_load_tensor_167
    )
    utils_load_tensor_168 = utils.load_tensor(
        "./tensors/arg117.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.19.proj_out.bias"] = (
        utils_load_tensor_168
    )
    utils_load_tensor_169 = utils.load_tensor(
        "./tensors/arg118.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.19.proj_out.weight"] = (
        utils_load_tensor_169
    )
    utils_load_tensor_170 = utils.load_tensor(
        "./tensors/arg119.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.19.proj_mlp.bias"] = (
        utils_load_tensor_170
    )
    utils_load_tensor_171 = utils.load_tensor(
        "./tensors/arg120.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.19.proj_mlp.weight"] = (
        utils_load_tensor_171
    )
    utils_load_tensor_172 = utils.load_tensor(
        "./tensors/arg121.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.19.norm.linear.bias"] = (
        utils_load_tensor_172
    )
    utils_load_tensor_173 = utils.load_tensor(
        "./tensors/arg122.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.19.norm.linear.weight"] = (
        utils_load_tensor_173
    )
    utils_load_tensor_174 = utils.load_tensor(
        "./tensors/arg123.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.18.proj_out.bias"] = (
        utils_load_tensor_174
    )
    utils_load_tensor_175 = utils.load_tensor(
        "./tensors/arg124.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.18.proj_out.weight"] = (
        utils_load_tensor_175
    )
    utils_load_tensor_176 = utils.load_tensor(
        "./tensors/arg125.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.18.proj_mlp.bias"] = (
        utils_load_tensor_176
    )
    utils_load_tensor_177 = utils.load_tensor(
        "./tensors/arg126.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.18.proj_mlp.weight"] = (
        utils_load_tensor_177
    )
    utils_load_tensor_178 = utils.load_tensor(
        "./tensors/arg127.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.18.norm.linear.bias"] = (
        utils_load_tensor_178
    )
    utils_load_tensor_179 = utils.load_tensor(
        "./tensors/arg128.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.18.norm.linear.weight"] = (
        utils_load_tensor_179
    )
    utils_load_tensor_180 = utils.load_tensor(
        "./tensors/arg129.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.17.proj_out.bias"] = (
        utils_load_tensor_180
    )
    utils_load_tensor_181 = utils.load_tensor(
        "./tensors/arg130.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.17.proj_out.weight"] = (
        utils_load_tensor_181
    )
    utils_load_tensor_182 = utils.load_tensor(
        "./tensors/arg131.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.17.proj_mlp.bias"] = (
        utils_load_tensor_182
    )
    utils_load_tensor_183 = utils.load_tensor(
        "./tensors/arg132.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.17.proj_mlp.weight"] = (
        utils_load_tensor_183
    )
    utils_load_tensor_184 = utils.load_tensor(
        "./tensors/arg133.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.17.norm.linear.bias"] = (
        utils_load_tensor_184
    )
    utils_load_tensor_185 = utils.load_tensor(
        "./tensors/arg134.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.17.norm.linear.weight"] = (
        utils_load_tensor_185
    )
    utils_load_tensor_186 = utils.load_tensor(
        "./tensors/arg135.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.16.proj_out.bias"] = (
        utils_load_tensor_186
    )
    utils_load_tensor_187 = utils.load_tensor(
        "./tensors/arg136.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.16.proj_out.weight"] = (
        utils_load_tensor_187
    )
    utils_load_tensor_188 = utils.load_tensor(
        "./tensors/arg137.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.16.proj_mlp.bias"] = (
        utils_load_tensor_188
    )
    utils_load_tensor_189 = utils.load_tensor(
        "./tensors/arg138.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.16.proj_mlp.weight"] = (
        utils_load_tensor_189
    )
    utils_load_tensor_190 = utils.load_tensor(
        "./tensors/arg139.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.16.norm.linear.bias"] = (
        utils_load_tensor_190
    )
    utils_load_tensor_191 = utils.load_tensor(
        "./tensors/arg140.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.16.norm.linear.weight"] = (
        utils_load_tensor_191
    )
    utils_load_tensor_192 = utils.load_tensor(
        "./tensors/arg141.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.15.proj_out.bias"] = (
        utils_load_tensor_192
    )
    utils_load_tensor_193 = utils.load_tensor(
        "./tensors/arg142.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.15.proj_out.weight"] = (
        utils_load_tensor_193
    )
    utils_load_tensor_194 = utils.load_tensor(
        "./tensors/arg143.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.15.proj_mlp.bias"] = (
        utils_load_tensor_194
    )
    utils_load_tensor_195 = utils.load_tensor(
        "./tensors/arg144.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.15.proj_mlp.weight"] = (
        utils_load_tensor_195
    )
    utils_load_tensor_196 = utils.load_tensor(
        "./tensors/arg145.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.15.norm.linear.bias"] = (
        utils_load_tensor_196
    )
    utils_load_tensor_197 = utils.load_tensor(
        "./tensors/arg146.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.15.norm.linear.weight"] = (
        utils_load_tensor_197
    )
    utils_load_tensor_198 = utils.load_tensor(
        "./tensors/arg147.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.14.proj_out.bias"] = (
        utils_load_tensor_198
    )
    utils_load_tensor_199 = utils.load_tensor(
        "./tensors/arg148.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.14.proj_out.weight"] = (
        utils_load_tensor_199
    )
    utils_load_tensor_200 = utils.load_tensor(
        "./tensors/arg149.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.14.proj_mlp.bias"] = (
        utils_load_tensor_200
    )
    utils_load_tensor_201 = utils.load_tensor(
        "./tensors/arg150.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.14.proj_mlp.weight"] = (
        utils_load_tensor_201
    )
    utils_load_tensor_202 = utils.load_tensor(
        "./tensors/arg151.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.14.norm.linear.bias"] = (
        utils_load_tensor_202
    )
    utils_load_tensor_203 = utils.load_tensor(
        "./tensors/arg152.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.14.norm.linear.weight"] = (
        utils_load_tensor_203
    )
    utils_load_tensor_204 = utils.load_tensor(
        "./tensors/arg153.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.13.proj_out.bias"] = (
        utils_load_tensor_204
    )
    utils_load_tensor_205 = utils.load_tensor(
        "./tensors/arg154.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.13.proj_out.weight"] = (
        utils_load_tensor_205
    )
    utils_load_tensor_206 = utils.load_tensor(
        "./tensors/arg155.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.13.proj_mlp.bias"] = (
        utils_load_tensor_206
    )
    utils_load_tensor_207 = utils.load_tensor(
        "./tensors/arg156.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.13.proj_mlp.weight"] = (
        utils_load_tensor_207
    )
    utils_load_tensor_208 = utils.load_tensor(
        "./tensors/arg157.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.13.norm.linear.bias"] = (
        utils_load_tensor_208
    )
    utils_load_tensor_209 = utils.load_tensor(
        "./tensors/arg158.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.13.norm.linear.weight"] = (
        utils_load_tensor_209
    )
    utils_load_tensor_210 = utils.load_tensor(
        "./tensors/arg159.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.12.proj_out.bias"] = (
        utils_load_tensor_210
    )
    utils_load_tensor_211 = utils.load_tensor(
        "./tensors/arg160.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.12.proj_out.weight"] = (
        utils_load_tensor_211
    )
    utils_load_tensor_212 = utils.load_tensor(
        "./tensors/arg161.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.12.proj_mlp.bias"] = (
        utils_load_tensor_212
    )
    utils_load_tensor_213 = utils.load_tensor(
        "./tensors/arg162.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.12.proj_mlp.weight"] = (
        utils_load_tensor_213
    )
    utils_load_tensor_214 = utils.load_tensor(
        "./tensors/arg163.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.12.norm.linear.bias"] = (
        utils_load_tensor_214
    )
    utils_load_tensor_215 = utils.load_tensor(
        "./tensors/arg164.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.12.norm.linear.weight"] = (
        utils_load_tensor_215
    )
    utils_load_tensor_216 = utils.load_tensor(
        "./tensors/arg165.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.11.proj_out.bias"] = (
        utils_load_tensor_216
    )
    utils_load_tensor_217 = utils.load_tensor(
        "./tensors/arg166.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.11.proj_out.weight"] = (
        utils_load_tensor_217
    )
    utils_load_tensor_218 = utils.load_tensor(
        "./tensors/arg167.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.11.proj_mlp.bias"] = (
        utils_load_tensor_218
    )
    utils_load_tensor_219 = utils.load_tensor(
        "./tensors/arg168.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.11.proj_mlp.weight"] = (
        utils_load_tensor_219
    )
    utils_load_tensor_220 = utils.load_tensor(
        "./tensors/arg169.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.11.norm.linear.bias"] = (
        utils_load_tensor_220
    )
    utils_load_tensor_221 = utils.load_tensor(
        "./tensors/arg170.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.11.norm.linear.weight"] = (
        utils_load_tensor_221
    )
    utils_load_tensor_222 = utils.load_tensor(
        "./tensors/arg171.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.10.proj_out.bias"] = (
        utils_load_tensor_222
    )
    utils_load_tensor_223 = utils.load_tensor(
        "./tensors/arg172.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.10.proj_out.weight"] = (
        utils_load_tensor_223
    )
    utils_load_tensor_224 = utils.load_tensor(
        "./tensors/arg173.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.10.proj_mlp.bias"] = (
        utils_load_tensor_224
    )
    utils_load_tensor_225 = utils.load_tensor(
        "./tensors/arg174.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.10.proj_mlp.weight"] = (
        utils_load_tensor_225
    )
    utils_load_tensor_226 = utils.load_tensor(
        "./tensors/arg175.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.10.norm.linear.bias"] = (
        utils_load_tensor_226
    )
    utils_load_tensor_227 = utils.load_tensor(
        "./tensors/arg176.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.10.norm.linear.weight"] = (
        utils_load_tensor_227
    )
    utils_load_tensor_228 = utils.load_tensor(
        "./tensors/arg177.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.9.proj_out.bias"] = (
        utils_load_tensor_228
    )
    utils_load_tensor_229 = utils.load_tensor(
        "./tensors/arg178.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.9.proj_out.weight"] = (
        utils_load_tensor_229
    )
    utils_load_tensor_230 = utils.load_tensor(
        "./tensors/arg179.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.9.proj_mlp.bias"] = (
        utils_load_tensor_230
    )
    utils_load_tensor_231 = utils.load_tensor(
        "./tensors/arg180.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.9.proj_mlp.weight"] = (
        utils_load_tensor_231
    )
    utils_load_tensor_232 = utils.load_tensor(
        "./tensors/arg181.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.9.norm.linear.bias"] = (
        utils_load_tensor_232
    )
    utils_load_tensor_233 = utils.load_tensor(
        "./tensors/arg182.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.9.norm.linear.weight"] = (
        utils_load_tensor_233
    )
    utils_load_tensor_234 = utils.load_tensor(
        "./tensors/arg183.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.8.proj_out.bias"] = (
        utils_load_tensor_234
    )
    utils_load_tensor_235 = utils.load_tensor(
        "./tensors/arg184.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.8.proj_out.weight"] = (
        utils_load_tensor_235
    )
    utils_load_tensor_236 = utils.load_tensor(
        "./tensors/arg185.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.8.proj_mlp.bias"] = (
        utils_load_tensor_236
    )
    utils_load_tensor_237 = utils.load_tensor(
        "./tensors/arg186.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.8.proj_mlp.weight"] = (
        utils_load_tensor_237
    )
    utils_load_tensor_238 = utils.load_tensor(
        "./tensors/arg187.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.8.norm.linear.bias"] = (
        utils_load_tensor_238
    )
    utils_load_tensor_239 = utils.load_tensor(
        "./tensors/arg188.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.8.norm.linear.weight"] = (
        utils_load_tensor_239
    )
    utils_load_tensor_240 = utils.load_tensor(
        "./tensors/arg189.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.7.proj_out.bias"] = (
        utils_load_tensor_240
    )
    utils_load_tensor_241 = utils.load_tensor(
        "./tensors/arg190.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.7.proj_out.weight"] = (
        utils_load_tensor_241
    )
    utils_load_tensor_242 = utils.load_tensor(
        "./tensors/arg191.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.7.proj_mlp.bias"] = (
        utils_load_tensor_242
    )
    utils_load_tensor_243 = utils.load_tensor(
        "./tensors/arg192.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.7.proj_mlp.weight"] = (
        utils_load_tensor_243
    )
    utils_load_tensor_244 = utils.load_tensor(
        "./tensors/arg193.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.7.norm.linear.bias"] = (
        utils_load_tensor_244
    )
    utils_load_tensor_245 = utils.load_tensor(
        "./tensors/arg194.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.7.norm.linear.weight"] = (
        utils_load_tensor_245
    )
    utils_load_tensor_246 = utils.load_tensor(
        "./tensors/arg195.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.6.proj_out.bias"] = (
        utils_load_tensor_246
    )
    utils_load_tensor_247 = utils.load_tensor(
        "./tensors/arg196.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.6.proj_out.weight"] = (
        utils_load_tensor_247
    )
    utils_load_tensor_248 = utils.load_tensor(
        "./tensors/arg197.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.6.proj_mlp.bias"] = (
        utils_load_tensor_248
    )
    utils_load_tensor_249 = utils.load_tensor(
        "./tensors/arg198.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.6.proj_mlp.weight"] = (
        utils_load_tensor_249
    )
    utils_load_tensor_250 = utils.load_tensor(
        "./tensors/arg199.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.6.norm.linear.bias"] = (
        utils_load_tensor_250
    )
    utils_load_tensor_251 = utils.load_tensor(
        "./tensors/arg200.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.6.norm.linear.weight"] = (
        utils_load_tensor_251
    )
    utils_load_tensor_252 = utils.load_tensor(
        "./tensors/arg201.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.5.proj_out.bias"] = (
        utils_load_tensor_252
    )
    utils_load_tensor_253 = utils.load_tensor(
        "./tensors/arg202.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.5.proj_out.weight"] = (
        utils_load_tensor_253
    )
    utils_load_tensor_254 = utils.load_tensor(
        "./tensors/arg203.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.5.proj_mlp.bias"] = (
        utils_load_tensor_254
    )
    utils_load_tensor_255 = utils.load_tensor(
        "./tensors/arg204.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.5.proj_mlp.weight"] = (
        utils_load_tensor_255
    )
    utils_load_tensor_256 = utils.load_tensor(
        "./tensors/arg205.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.5.norm.linear.bias"] = (
        utils_load_tensor_256
    )
    utils_load_tensor_257 = utils.load_tensor(
        "./tensors/arg206.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.5.norm.linear.weight"] = (
        utils_load_tensor_257
    )
    utils_load_tensor_258 = utils.load_tensor(
        "./tensors/arg207.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.4.proj_out.bias"] = (
        utils_load_tensor_258
    )
    utils_load_tensor_259 = utils.load_tensor(
        "./tensors/arg208.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.4.proj_out.weight"] = (
        utils_load_tensor_259
    )
    utils_load_tensor_260 = utils.load_tensor(
        "./tensors/arg209.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.4.proj_mlp.bias"] = (
        utils_load_tensor_260
    )
    utils_load_tensor_261 = utils.load_tensor(
        "./tensors/arg210.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.4.proj_mlp.weight"] = (
        utils_load_tensor_261
    )
    utils_load_tensor_262 = utils.load_tensor(
        "./tensors/arg211.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.4.norm.linear.bias"] = (
        utils_load_tensor_262
    )
    utils_load_tensor_263 = utils.load_tensor(
        "./tensors/arg212.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.4.norm.linear.weight"] = (
        utils_load_tensor_263
    )
    utils_load_tensor_264 = utils.load_tensor(
        "./tensors/arg213.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.3.proj_out.bias"] = (
        utils_load_tensor_264
    )
    utils_load_tensor_265 = utils.load_tensor(
        "./tensors/arg214.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.3.proj_out.weight"] = (
        utils_load_tensor_265
    )
    utils_load_tensor_266 = utils.load_tensor(
        "./tensors/arg215.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.3.proj_mlp.bias"] = (
        utils_load_tensor_266
    )
    utils_load_tensor_267 = utils.load_tensor(
        "./tensors/arg216.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.3.proj_mlp.weight"] = (
        utils_load_tensor_267
    )
    utils_load_tensor_268 = utils.load_tensor(
        "./tensors/arg217.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.3.norm.linear.bias"] = (
        utils_load_tensor_268
    )
    utils_load_tensor_269 = utils.load_tensor(
        "./tensors/arg218.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.3.norm.linear.weight"] = (
        utils_load_tensor_269
    )
    utils_load_tensor_270 = utils.load_tensor(
        "./tensors/arg219.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.2.proj_out.bias"] = (
        utils_load_tensor_270
    )
    utils_load_tensor_271 = utils.load_tensor(
        "./tensors/arg220.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.2.proj_out.weight"] = (
        utils_load_tensor_271
    )
    utils_load_tensor_272 = utils.load_tensor(
        "./tensors/arg221.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.2.proj_mlp.bias"] = (
        utils_load_tensor_272
    )
    utils_load_tensor_273 = utils.load_tensor(
        "./tensors/arg222.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.2.proj_mlp.weight"] = (
        utils_load_tensor_273
    )
    utils_load_tensor_274 = utils.load_tensor(
        "./tensors/arg223.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.2.norm.linear.bias"] = (
        utils_load_tensor_274
    )
    utils_load_tensor_275 = utils.load_tensor(
        "./tensors/arg224.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.2.norm.linear.weight"] = (
        utils_load_tensor_275
    )
    utils_load_tensor_276 = utils.load_tensor(
        "./tensors/arg225.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.1.proj_out.bias"] = (
        utils_load_tensor_276
    )
    utils_load_tensor_277 = utils.load_tensor(
        "./tensors/arg226.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.1.proj_out.weight"] = (
        utils_load_tensor_277
    )
    utils_load_tensor_278 = utils.load_tensor(
        "./tensors/arg227.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.1.proj_mlp.bias"] = (
        utils_load_tensor_278
    )
    utils_load_tensor_279 = utils.load_tensor(
        "./tensors/arg228.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.1.proj_mlp.weight"] = (
        utils_load_tensor_279
    )
    utils_load_tensor_280 = utils.load_tensor(
        "./tensors/arg229.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.1.norm.linear.bias"] = (
        utils_load_tensor_280
    )
    utils_load_tensor_281 = utils.load_tensor(
        "./tensors/arg230.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.1.norm.linear.weight"] = (
        utils_load_tensor_281
    )
    utils_load_tensor_282 = utils.load_tensor(
        "./tensors/arg231.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.0.proj_out.bias"] = (
        utils_load_tensor_282
    )
    utils_load_tensor_283 = utils.load_tensor(
        "./tensors/arg232.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.0.proj_out.weight"] = (
        utils_load_tensor_283
    )
    utils_load_tensor_284 = utils.load_tensor(
        "./tensors/arg233.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.0.proj_mlp.bias"] = (
        utils_load_tensor_284
    )
    utils_load_tensor_285 = utils.load_tensor(
        "./tensors/arg234.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.0.proj_mlp.weight"] = (
        utils_load_tensor_285
    )
    utils_load_tensor_286 = utils.load_tensor(
        "./tensors/arg235.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.0.norm.linear.bias"] = (
        utils_load_tensor_286
    )
    utils_load_tensor_287 = utils.load_tensor(
        "./tensors/arg236.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.0.norm.linear.weight"] = (
        utils_load_tensor_287
    )
    utils_load_tensor_288 = utils.load_tensor(
        "./tensors/arg237.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.ff.net.2.bias"] = (
        utils_load_tensor_288
    )
    utils_load_tensor_289 = utils.load_tensor(
        "./tensors/arg238.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.ff.net.2.weight"] = (
        utils_load_tensor_289
    )
    utils_load_tensor_290 = utils.load_tensor(
        "./tensors/arg239.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.ff.net.0.proj.bias"] = (
        utils_load_tensor_290
    )
    utils_load_tensor_291 = utils.load_tensor(
        "./tensors/arg240.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.ff.net.0.proj.weight"] = (
        utils_load_tensor_291
    )
    utils_load_tensor_292 = utils.load_tensor(
        "./tensors/arg241.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.norm1.linear.bias"] = (
        utils_load_tensor_292
    )
    utils_load_tensor_293 = utils.load_tensor(
        "./tensors/arg242.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.norm1.linear.weight"] = (
        utils_load_tensor_293
    )
    utils_load_tensor_294 = utils.load_tensor(
        "./tensors/arg243.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.to_out.0.bias"] = (
        utils_load_tensor_294
    )
    utils_load_tensor_295 = utils.load_tensor(
        "./tensors/arg244.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.attn.to_out.0.weight"] = (
        utils_load_tensor_295
    )
    utils_load_tensor_296 = utils.load_tensor(
        "./tensors/arg245.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.to_v.bias"] = (
        utils_load_tensor_296
    )
    utils_load_tensor_297 = utils.load_tensor(
        "./tensors/arg246.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.to_v.weight"] = (
        utils_load_tensor_297
    )
    utils_load_tensor_298 = utils.load_tensor(
        "./tensors/arg247.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.ff.net.2.bias"] = (
        utils_load_tensor_298
    )
    utils_load_tensor_299 = utils.load_tensor(
        "./tensors/arg248.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.ff.net.2.weight"] = (
        utils_load_tensor_299
    )
    utils_load_tensor_300 = utils.load_tensor(
        "./tensors/arg249.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.ff.net.0.proj.bias"] = (
        utils_load_tensor_300
    )
    utils_load_tensor_301 = utils.load_tensor(
        "./tensors/arg250.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.ff.net.0.proj.weight"] = (
        utils_load_tensor_301
    )
    utils_load_tensor_302 = utils.load_tensor(
        "./tensors/arg251.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.norm1.linear.bias"] = (
        utils_load_tensor_302
    )
    utils_load_tensor_303 = utils.load_tensor(
        "./tensors/arg252.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.norm1.linear.weight"] = (
        utils_load_tensor_303
    )
    utils_load_tensor_304 = utils.load_tensor(
        "./tensors/arg253.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.to_out.0.bias"] = (
        utils_load_tensor_304
    )
    utils_load_tensor_305 = utils.load_tensor(
        "./tensors/arg254.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.attn.to_out.0.weight"] = (
        utils_load_tensor_305
    )
    utils_load_tensor_306 = utils.load_tensor(
        "./tensors/arg255.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.to_v.bias"] = (
        utils_load_tensor_306
    )
    utils_load_tensor_307 = utils.load_tensor(
        "./tensors/arg256.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.to_v.weight"] = (
        utils_load_tensor_307
    )
    utils_load_tensor_308 = utils.load_tensor(
        "./tensors/arg257.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.ff.net.2.bias"] = (
        utils_load_tensor_308
    )
    utils_load_tensor_309 = utils.load_tensor(
        "./tensors/arg258.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.ff.net.2.weight"] = (
        utils_load_tensor_309
    )
    utils_load_tensor_310 = utils.load_tensor(
        "./tensors/arg259.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.ff.net.0.proj.bias"] = (
        utils_load_tensor_310
    )
    utils_load_tensor_311 = utils.load_tensor(
        "./tensors/arg260.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.ff.net.0.proj.weight"] = (
        utils_load_tensor_311
    )
    utils_load_tensor_312 = utils.load_tensor(
        "./tensors/arg261.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.norm1.linear.bias"] = (
        utils_load_tensor_312
    )
    utils_load_tensor_313 = utils.load_tensor(
        "./tensors/arg262.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.norm1.linear.weight"] = (
        utils_load_tensor_313
    )
    utils_load_tensor_314 = utils.load_tensor(
        "./tensors/arg263.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.to_out.0.bias"] = (
        utils_load_tensor_314
    )
    utils_load_tensor_315 = utils.load_tensor(
        "./tensors/arg264.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.attn.to_out.0.weight"] = (
        utils_load_tensor_315
    )
    utils_load_tensor_316 = utils.load_tensor(
        "./tensors/arg265.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.to_v.bias"] = (
        utils_load_tensor_316
    )
    utils_load_tensor_317 = utils.load_tensor(
        "./tensors/arg266.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.to_v.weight"] = (
        utils_load_tensor_317
    )
    utils_load_tensor_318 = utils.load_tensor(
        "./tensors/arg267.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.ff.net.2.bias"] = (
        utils_load_tensor_318
    )
    utils_load_tensor_319 = utils.load_tensor(
        "./tensors/arg268.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.ff.net.2.weight"] = (
        utils_load_tensor_319
    )
    utils_load_tensor_320 = utils.load_tensor(
        "./tensors/arg269.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.ff.net.0.proj.bias"] = (
        utils_load_tensor_320
    )
    utils_load_tensor_321 = utils.load_tensor(
        "./tensors/arg270.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.ff.net.0.proj.weight"] = (
        utils_load_tensor_321
    )
    utils_load_tensor_322 = utils.load_tensor(
        "./tensors/arg271.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.norm1.linear.bias"] = (
        utils_load_tensor_322
    )
    utils_load_tensor_323 = utils.load_tensor(
        "./tensors/arg272.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.norm1.linear.weight"] = (
        utils_load_tensor_323
    )
    utils_load_tensor_324 = utils.load_tensor(
        "./tensors/arg273.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.to_out.0.bias"] = (
        utils_load_tensor_324
    )
    utils_load_tensor_325 = utils.load_tensor(
        "./tensors/arg274.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.attn.to_out.0.weight"] = (
        utils_load_tensor_325
    )
    utils_load_tensor_326 = utils.load_tensor(
        "./tensors/arg275.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.to_v.bias"] = (
        utils_load_tensor_326
    )
    utils_load_tensor_327 = utils.load_tensor(
        "./tensors/arg276.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.to_v.weight"] = (
        utils_load_tensor_327
    )
    utils_load_tensor_328 = utils.load_tensor(
        "./tensors/arg277.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.ff.net.2.bias"] = (
        utils_load_tensor_328
    )
    utils_load_tensor_329 = utils.load_tensor(
        "./tensors/arg278.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.ff.net.2.weight"] = (
        utils_load_tensor_329
    )
    utils_load_tensor_330 = utils.load_tensor(
        "./tensors/arg279.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.ff.net.0.proj.bias"] = (
        utils_load_tensor_330
    )
    utils_load_tensor_331 = utils.load_tensor(
        "./tensors/arg280.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.ff.net.0.proj.weight"] = (
        utils_load_tensor_331
    )
    utils_load_tensor_332 = utils.load_tensor(
        "./tensors/arg281.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.norm1.linear.bias"] = (
        utils_load_tensor_332
    )
    utils_load_tensor_333 = utils.load_tensor(
        "./tensors/arg282.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.norm1.linear.weight"] = (
        utils_load_tensor_333
    )
    utils_load_tensor_334 = utils.load_tensor(
        "./tensors/arg283.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.to_out.0.bias"] = (
        utils_load_tensor_334
    )
    utils_load_tensor_335 = utils.load_tensor(
        "./tensors/arg284.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.attn.to_out.0.weight"] = (
        utils_load_tensor_335
    )
    utils_load_tensor_336 = utils.load_tensor(
        "./tensors/arg285.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.to_v.bias"] = (
        utils_load_tensor_336
    )
    utils_load_tensor_337 = utils.load_tensor(
        "./tensors/arg286.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.to_v.weight"] = (
        utils_load_tensor_337
    )
    utils_load_tensor_338 = utils.load_tensor(
        "./tensors/arg287.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.ff.net.2.bias"] = (
        utils_load_tensor_338
    )
    utils_load_tensor_339 = utils.load_tensor(
        "./tensors/arg288.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.ff.net.2.weight"] = (
        utils_load_tensor_339
    )
    utils_load_tensor_340 = utils.load_tensor(
        "./tensors/arg289.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.ff.net.0.proj.bias"] = (
        utils_load_tensor_340
    )
    utils_load_tensor_341 = utils.load_tensor(
        "./tensors/arg290.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.ff.net.0.proj.weight"] = (
        utils_load_tensor_341
    )
    utils_load_tensor_342 = utils.load_tensor(
        "./tensors/arg291.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.norm1.linear.bias"] = (
        utils_load_tensor_342
    )
    utils_load_tensor_343 = utils.load_tensor(
        "./tensors/arg292.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.norm1.linear.weight"] = (
        utils_load_tensor_343
    )
    utils_load_tensor_344 = utils.load_tensor(
        "./tensors/arg293.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.to_out.0.bias"] = (
        utils_load_tensor_344
    )
    utils_load_tensor_345 = utils.load_tensor(
        "./tensors/arg294.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.attn.to_out.0.weight"] = (
        utils_load_tensor_345
    )
    utils_load_tensor_346 = utils.load_tensor(
        "./tensors/arg295.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.to_v.bias"] = (
        utils_load_tensor_346
    )
    utils_load_tensor_347 = utils.load_tensor(
        "./tensors/arg296.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.to_v.weight"] = (
        utils_load_tensor_347
    )
    utils_load_tensor_348 = utils.load_tensor(
        "./tensors/arg297.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.ff.net.2.bias"] = (
        utils_load_tensor_348
    )
    utils_load_tensor_349 = utils.load_tensor(
        "./tensors/arg298.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.ff.net.2.weight"] = (
        utils_load_tensor_349
    )
    utils_load_tensor_350 = utils.load_tensor(
        "./tensors/arg299.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.ff.net.0.proj.bias"] = (
        utils_load_tensor_350
    )
    utils_load_tensor_351 = utils.load_tensor(
        "./tensors/arg300.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.ff.net.0.proj.weight"] = (
        utils_load_tensor_351
    )
    utils_load_tensor_352 = utils.load_tensor(
        "./tensors/arg301.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.norm1.linear.bias"] = (
        utils_load_tensor_352
    )
    utils_load_tensor_353 = utils.load_tensor(
        "./tensors/arg302.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.norm1.linear.weight"] = (
        utils_load_tensor_353
    )
    utils_load_tensor_354 = utils.load_tensor(
        "./tensors/arg303.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.to_out.0.bias"] = (
        utils_load_tensor_354
    )
    utils_load_tensor_355 = utils.load_tensor(
        "./tensors/arg304.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.attn.to_out.0.weight"] = (
        utils_load_tensor_355
    )
    utils_load_tensor_356 = utils.load_tensor(
        "./tensors/arg305.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.to_v.bias"] = (
        utils_load_tensor_356
    )
    utils_load_tensor_357 = utils.load_tensor(
        "./tensors/arg306.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.to_v.weight"] = (
        utils_load_tensor_357
    )
    utils_load_tensor_358 = utils.load_tensor(
        "./tensors/arg307.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.ff.net.2.bias"] = (
        utils_load_tensor_358
    )
    utils_load_tensor_359 = utils.load_tensor(
        "./tensors/arg308.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.ff.net.2.weight"] = (
        utils_load_tensor_359
    )
    utils_load_tensor_360 = utils.load_tensor(
        "./tensors/arg309.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.ff.net.0.proj.bias"] = (
        utils_load_tensor_360
    )
    utils_load_tensor_361 = utils.load_tensor(
        "./tensors/arg310.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.ff.net.0.proj.weight"] = (
        utils_load_tensor_361
    )
    utils_load_tensor_362 = utils.load_tensor(
        "./tensors/arg311.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.norm1.linear.bias"] = (
        utils_load_tensor_362
    )
    utils_load_tensor_363 = utils.load_tensor(
        "./tensors/arg312.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.norm1.linear.weight"] = (
        utils_load_tensor_363
    )
    utils_load_tensor_364 = utils.load_tensor(
        "./tensors/arg313.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.to_out.0.bias"] = (
        utils_load_tensor_364
    )
    utils_load_tensor_365 = utils.load_tensor(
        "./tensors/arg314.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.attn.to_out.0.weight"] = (
        utils_load_tensor_365
    )
    utils_load_tensor_366 = utils.load_tensor(
        "./tensors/arg315.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.to_v.bias"] = (
        utils_load_tensor_366
    )
    utils_load_tensor_367 = utils.load_tensor(
        "./tensors/arg316.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.to_v.weight"] = (
        utils_load_tensor_367
    )
    utils_load_tensor_368 = utils.load_tensor(
        "./tensors/arg317.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.x_embedder.bias"] = utils_load_tensor_368
    utils_load_tensor_369 = utils.load_tensor(
        "./tensors/arg318.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.x_embedder.weight"] = utils_load_tensor_369
    utils_load_tensor_370 = utils.load_tensor(
        "./tensors/arg320.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.add_v_proj.bias"] = (
        utils_load_tensor_370
    )
    utils_load_tensor_371 = utils.load_tensor(
        "./tensors/arg321.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.add_v_proj.weight"] = (
        utils_load_tensor_371
    )
    utils_load_tensor_372 = utils.load_tensor(
        "./tensors/arg322.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.norm1_context.linear.bias"] = (
        utils_load_tensor_372
    )
    utils_load_tensor_373 = utils.load_tensor(
        "./tensors/arg323.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.norm1_context.linear.weight"] = (
        utils_load_tensor_373
    )
    utils_load_tensor_374 = utils.load_tensor(
        "./tensors/arg324.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.0.linear.weight"] = (
        utils_load_tensor_374
    )
    utils_load_tensor_375 = utils.load_tensor(
        "./tensors/arg326.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.context_embedder.bias"] = utils_load_tensor_375
    utils_load_tensor_376 = utils.load_tensor(
        "./tensors/arg327.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.context_embedder.weight"] = utils_load_tensor_376
    utils_load_tensor_377 = utils.load_tensor(
        "./tensors/arg332.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.attn.norm_k.weight"] = (
        utils_load_tensor_377
    )
    utils_load_tensor_378 = utils.load_tensor(
        "./tensors/arg333.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.to_k.bias"] = (
        utils_load_tensor_378
    )
    utils_load_tensor_379 = utils.load_tensor(
        "./tensors/arg334.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.to_k.weight"] = (
        utils_load_tensor_379
    )
    utils_load_tensor_380 = utils.load_tensor(
        "./tensors/arg335.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.attn.norm_added_k.weight"] = (
        utils_load_tensor_380
    )
    utils_load_tensor_381 = utils.load_tensor(
        "./tensors/arg336.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.add_k_proj.bias"] = (
        utils_load_tensor_381
    )
    utils_load_tensor_382 = utils.load_tensor(
        "./tensors/arg337.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.add_k_proj.weight"] = (
        utils_load_tensor_382
    )
    utils_load_tensor_383 = utils.load_tensor(
        "./tensors/arg338.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.attn.norm_q.weight"] = (
        utils_load_tensor_383
    )
    utils_load_tensor_384 = utils.load_tensor(
        "./tensors/arg339.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.to_q.bias"] = (
        utils_load_tensor_384
    )
    utils_load_tensor_385 = utils.load_tensor(
        "./tensors/arg340.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.to_q.weight"] = (
        utils_load_tensor_385
    )
    utils_load_tensor_386 = utils.load_tensor(
        "./tensors/arg341.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.attn.norm_added_q.weight"] = (
        utils_load_tensor_386
    )
    utils_load_tensor_387 = utils.load_tensor(
        "./tensors/arg342.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.add_q_proj.bias"] = (
        utils_load_tensor_387
    )
    utils_load_tensor_388 = utils.load_tensor(
        "./tensors/arg343.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.add_q_proj.weight"] = (
        utils_load_tensor_388
    )
    utils_load_tensor_389 = utils.load_tensor(
        "./tensors/arg344.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.add_v_proj.bias"] = (
        utils_load_tensor_389
    )
    utils_load_tensor_390 = utils.load_tensor(
        "./tensors/arg345.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.add_v_proj.weight"] = (
        utils_load_tensor_390
    )
    utils_load_tensor_391 = utils.load_tensor(
        "./tensors/arg346.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.norm1_context.linear.bias"] = (
        utils_load_tensor_391
    )
    utils_load_tensor_392 = utils.load_tensor(
        "./tensors/arg347.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.norm1_context.linear.weight"] = (
        utils_load_tensor_392
    )
    utils_load_tensor_393 = utils.load_tensor(
        "./tensors/arg348.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.1.linear.weight"] = (
        utils_load_tensor_393
    )
    utils_load_tensor_394 = utils.load_tensor(
        "./tensors/arg350.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.ff_context.net.2.bias"] = (
        utils_load_tensor_394
    )
    utils_load_tensor_395 = utils.load_tensor(
        "./tensors/arg351.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.ff_context.net.2.weight"] = (
        utils_load_tensor_395
    )
    utils_load_tensor_396 = utils.load_tensor(
        "./tensors/arg352.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.ff_context.net.0.proj.bias"] = (
        utils_load_tensor_396
    )
    utils_load_tensor_397 = utils.load_tensor(
        "./tensors/arg353.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.ff_context.net.0.proj.weight"] = (
        utils_load_tensor_397
    )
    utils_load_tensor_398 = utils.load_tensor(
        "./tensors/arg354.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.0.attn.to_add_out.bias"] = (
        utils_load_tensor_398
    )
    utils_load_tensor_399 = utils.load_tensor(
        "./tensors/arg355.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.0.attn.to_add_out.weight"] = (
        utils_load_tensor_399
    )
    utils_load_tensor_400 = utils.load_tensor(
        "./tensors/arg356.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.attn.norm_k.weight"] = (
        utils_load_tensor_400
    )
    utils_load_tensor_401 = utils.load_tensor(
        "./tensors/arg357.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.to_k.bias"] = (
        utils_load_tensor_401
    )
    utils_load_tensor_402 = utils.load_tensor(
        "./tensors/arg358.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.to_k.weight"] = (
        utils_load_tensor_402
    )
    utils_load_tensor_403 = utils.load_tensor(
        "./tensors/arg359.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.attn.norm_added_k.weight"] = (
        utils_load_tensor_403
    )
    utils_load_tensor_404 = utils.load_tensor(
        "./tensors/arg360.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.add_k_proj.bias"] = (
        utils_load_tensor_404
    )
    utils_load_tensor_405 = utils.load_tensor(
        "./tensors/arg361.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.add_k_proj.weight"] = (
        utils_load_tensor_405
    )
    utils_load_tensor_406 = utils.load_tensor(
        "./tensors/arg362.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.attn.norm_q.weight"] = (
        utils_load_tensor_406
    )
    utils_load_tensor_407 = utils.load_tensor(
        "./tensors/arg363.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.to_q.bias"] = (
        utils_load_tensor_407
    )
    utils_load_tensor_408 = utils.load_tensor(
        "./tensors/arg364.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.to_q.weight"] = (
        utils_load_tensor_408
    )
    utils_load_tensor_409 = utils.load_tensor(
        "./tensors/arg365.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.attn.norm_added_q.weight"] = (
        utils_load_tensor_409
    )
    utils_load_tensor_410 = utils.load_tensor(
        "./tensors/arg366.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.add_q_proj.bias"] = (
        utils_load_tensor_410
    )
    utils_load_tensor_411 = utils.load_tensor(
        "./tensors/arg367.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.add_q_proj.weight"] = (
        utils_load_tensor_411
    )
    utils_load_tensor_412 = utils.load_tensor(
        "./tensors/arg368.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.add_v_proj.bias"] = (
        utils_load_tensor_412
    )
    utils_load_tensor_413 = utils.load_tensor(
        "./tensors/arg369.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.add_v_proj.weight"] = (
        utils_load_tensor_413
    )
    utils_load_tensor_414 = utils.load_tensor(
        "./tensors/arg370.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.norm1_context.linear.bias"] = (
        utils_load_tensor_414
    )
    utils_load_tensor_415 = utils.load_tensor(
        "./tensors/arg371.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.norm1_context.linear.weight"] = (
        utils_load_tensor_415
    )
    utils_load_tensor_416 = utils.load_tensor(
        "./tensors/arg372.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.2.linear.weight"] = (
        utils_load_tensor_416
    )
    utils_load_tensor_417 = utils.load_tensor(
        "./tensors/arg374.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.ff_context.net.2.bias"] = (
        utils_load_tensor_417
    )
    utils_load_tensor_418 = utils.load_tensor(
        "./tensors/arg375.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.ff_context.net.2.weight"] = (
        utils_load_tensor_418
    )
    utils_load_tensor_419 = utils.load_tensor(
        "./tensors/arg376.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.ff_context.net.0.proj.bias"] = (
        utils_load_tensor_419
    )
    utils_load_tensor_420 = utils.load_tensor(
        "./tensors/arg377.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.ff_context.net.0.proj.weight"] = (
        utils_load_tensor_420
    )
    utils_load_tensor_421 = utils.load_tensor(
        "./tensors/arg378.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.1.attn.to_add_out.bias"] = (
        utils_load_tensor_421
    )
    utils_load_tensor_422 = utils.load_tensor(
        "./tensors/arg379.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.1.attn.to_add_out.weight"] = (
        utils_load_tensor_422
    )
    utils_load_tensor_423 = utils.load_tensor(
        "./tensors/arg380.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.attn.norm_k.weight"] = (
        utils_load_tensor_423
    )
    utils_load_tensor_424 = utils.load_tensor(
        "./tensors/arg381.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.to_k.bias"] = (
        utils_load_tensor_424
    )
    utils_load_tensor_425 = utils.load_tensor(
        "./tensors/arg382.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.to_k.weight"] = (
        utils_load_tensor_425
    )
    utils_load_tensor_426 = utils.load_tensor(
        "./tensors/arg383.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.attn.norm_added_k.weight"] = (
        utils_load_tensor_426
    )
    utils_load_tensor_427 = utils.load_tensor(
        "./tensors/arg384.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.add_k_proj.bias"] = (
        utils_load_tensor_427
    )
    utils_load_tensor_428 = utils.load_tensor(
        "./tensors/arg385.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.add_k_proj.weight"] = (
        utils_load_tensor_428
    )
    utils_load_tensor_429 = utils.load_tensor(
        "./tensors/arg386.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.attn.norm_q.weight"] = (
        utils_load_tensor_429
    )
    utils_load_tensor_430 = utils.load_tensor(
        "./tensors/arg387.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.to_q.bias"] = (
        utils_load_tensor_430
    )
    utils_load_tensor_431 = utils.load_tensor(
        "./tensors/arg388.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.to_q.weight"] = (
        utils_load_tensor_431
    )
    utils_load_tensor_432 = utils.load_tensor(
        "./tensors/arg389.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.attn.norm_added_q.weight"] = (
        utils_load_tensor_432
    )
    utils_load_tensor_433 = utils.load_tensor(
        "./tensors/arg390.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.add_q_proj.bias"] = (
        utils_load_tensor_433
    )
    utils_load_tensor_434 = utils.load_tensor(
        "./tensors/arg391.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.add_q_proj.weight"] = (
        utils_load_tensor_434
    )
    utils_load_tensor_435 = utils.load_tensor(
        "./tensors/arg392.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.add_v_proj.bias"] = (
        utils_load_tensor_435
    )
    utils_load_tensor_436 = utils.load_tensor(
        "./tensors/arg393.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.add_v_proj.weight"] = (
        utils_load_tensor_436
    )
    utils_load_tensor_437 = utils.load_tensor(
        "./tensors/arg394.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.norm1_context.linear.bias"] = (
        utils_load_tensor_437
    )
    utils_load_tensor_438 = utils.load_tensor(
        "./tensors/arg395.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.norm1_context.linear.weight"] = (
        utils_load_tensor_438
    )
    utils_load_tensor_439 = utils.load_tensor(
        "./tensors/arg396.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.3.linear.weight"] = (
        utils_load_tensor_439
    )
    utils_load_tensor_440 = utils.load_tensor(
        "./tensors/arg398.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.ff_context.net.2.bias"] = (
        utils_load_tensor_440
    )
    utils_load_tensor_441 = utils.load_tensor(
        "./tensors/arg399.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.ff_context.net.2.weight"] = (
        utils_load_tensor_441
    )
    utils_load_tensor_442 = utils.load_tensor(
        "./tensors/arg400.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.ff_context.net.0.proj.bias"] = (
        utils_load_tensor_442
    )
    utils_load_tensor_443 = utils.load_tensor(
        "./tensors/arg401.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.ff_context.net.0.proj.weight"] = (
        utils_load_tensor_443
    )
    utils_load_tensor_444 = utils.load_tensor(
        "./tensors/arg402.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.2.attn.to_add_out.bias"] = (
        utils_load_tensor_444
    )
    utils_load_tensor_445 = utils.load_tensor(
        "./tensors/arg403.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.2.attn.to_add_out.weight"] = (
        utils_load_tensor_445
    )
    utils_load_tensor_446 = utils.load_tensor(
        "./tensors/arg404.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.attn.norm_k.weight"] = (
        utils_load_tensor_446
    )
    utils_load_tensor_447 = utils.load_tensor(
        "./tensors/arg405.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.to_k.bias"] = (
        utils_load_tensor_447
    )
    utils_load_tensor_448 = utils.load_tensor(
        "./tensors/arg406.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.to_k.weight"] = (
        utils_load_tensor_448
    )
    utils_load_tensor_449 = utils.load_tensor(
        "./tensors/arg407.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.attn.norm_added_k.weight"] = (
        utils_load_tensor_449
    )
    utils_load_tensor_450 = utils.load_tensor(
        "./tensors/arg408.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.add_k_proj.bias"] = (
        utils_load_tensor_450
    )
    utils_load_tensor_451 = utils.load_tensor(
        "./tensors/arg409.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.add_k_proj.weight"] = (
        utils_load_tensor_451
    )
    utils_load_tensor_452 = utils.load_tensor(
        "./tensors/arg410.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.attn.norm_q.weight"] = (
        utils_load_tensor_452
    )
    utils_load_tensor_453 = utils.load_tensor(
        "./tensors/arg411.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.to_q.bias"] = (
        utils_load_tensor_453
    )
    utils_load_tensor_454 = utils.load_tensor(
        "./tensors/arg412.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.to_q.weight"] = (
        utils_load_tensor_454
    )
    utils_load_tensor_455 = utils.load_tensor(
        "./tensors/arg413.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.attn.norm_added_q.weight"] = (
        utils_load_tensor_455
    )
    utils_load_tensor_456 = utils.load_tensor(
        "./tensors/arg414.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.add_q_proj.bias"] = (
        utils_load_tensor_456
    )
    utils_load_tensor_457 = utils.load_tensor(
        "./tensors/arg415.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.add_q_proj.weight"] = (
        utils_load_tensor_457
    )
    utils_load_tensor_458 = utils.load_tensor(
        "./tensors/arg416.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.add_v_proj.bias"] = (
        utils_load_tensor_458
    )
    utils_load_tensor_459 = utils.load_tensor(
        "./tensors/arg417.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.add_v_proj.weight"] = (
        utils_load_tensor_459
    )
    utils_load_tensor_460 = utils.load_tensor(
        "./tensors/arg418.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.norm1_context.linear.bias"] = (
        utils_load_tensor_460
    )
    utils_load_tensor_461 = utils.load_tensor(
        "./tensors/arg419.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.norm1_context.linear.weight"] = (
        utils_load_tensor_461
    )
    utils_load_tensor_462 = utils.load_tensor(
        "./tensors/arg420.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.4.linear.weight"] = (
        utils_load_tensor_462
    )
    utils_load_tensor_463 = utils.load_tensor(
        "./tensors/arg422.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.ff_context.net.2.bias"] = (
        utils_load_tensor_463
    )
    utils_load_tensor_464 = utils.load_tensor(
        "./tensors/arg423.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.ff_context.net.2.weight"] = (
        utils_load_tensor_464
    )
    utils_load_tensor_465 = utils.load_tensor(
        "./tensors/arg424.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.ff_context.net.0.proj.bias"] = (
        utils_load_tensor_465
    )
    utils_load_tensor_466 = utils.load_tensor(
        "./tensors/arg425.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.ff_context.net.0.proj.weight"] = (
        utils_load_tensor_466
    )
    utils_load_tensor_467 = utils.load_tensor(
        "./tensors/arg426.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.3.attn.to_add_out.bias"] = (
        utils_load_tensor_467
    )
    utils_load_tensor_468 = utils.load_tensor(
        "./tensors/arg427.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.3.attn.to_add_out.weight"] = (
        utils_load_tensor_468
    )
    utils_load_tensor_469 = utils.load_tensor(
        "./tensors/arg428.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.attn.norm_k.weight"] = (
        utils_load_tensor_469
    )
    utils_load_tensor_470 = utils.load_tensor(
        "./tensors/arg429.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.to_k.bias"] = (
        utils_load_tensor_470
    )
    utils_load_tensor_471 = utils.load_tensor(
        "./tensors/arg430.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.to_k.weight"] = (
        utils_load_tensor_471
    )
    utils_load_tensor_472 = utils.load_tensor(
        "./tensors/arg431.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.attn.norm_added_k.weight"] = (
        utils_load_tensor_472
    )
    utils_load_tensor_473 = utils.load_tensor(
        "./tensors/arg432.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.add_k_proj.bias"] = (
        utils_load_tensor_473
    )
    utils_load_tensor_474 = utils.load_tensor(
        "./tensors/arg433.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.add_k_proj.weight"] = (
        utils_load_tensor_474
    )
    utils_load_tensor_475 = utils.load_tensor(
        "./tensors/arg434.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.attn.norm_q.weight"] = (
        utils_load_tensor_475
    )
    utils_load_tensor_476 = utils.load_tensor(
        "./tensors/arg435.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.to_q.bias"] = (
        utils_load_tensor_476
    )
    utils_load_tensor_477 = utils.load_tensor(
        "./tensors/arg436.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.to_q.weight"] = (
        utils_load_tensor_477
    )
    utils_load_tensor_478 = utils.load_tensor(
        "./tensors/arg437.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.attn.norm_added_q.weight"] = (
        utils_load_tensor_478
    )
    utils_load_tensor_479 = utils.load_tensor(
        "./tensors/arg438.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.add_q_proj.bias"] = (
        utils_load_tensor_479
    )
    utils_load_tensor_480 = utils.load_tensor(
        "./tensors/arg439.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.add_q_proj.weight"] = (
        utils_load_tensor_480
    )
    utils_load_tensor_481 = utils.load_tensor(
        "./tensors/arg440.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.add_v_proj.bias"] = (
        utils_load_tensor_481
    )
    utils_load_tensor_482 = utils.load_tensor(
        "./tensors/arg441.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.add_v_proj.weight"] = (
        utils_load_tensor_482
    )
    utils_load_tensor_483 = utils.load_tensor(
        "./tensors/arg442.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.norm1_context.linear.bias"] = (
        utils_load_tensor_483
    )
    utils_load_tensor_484 = utils.load_tensor(
        "./tensors/arg443.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.norm1_context.linear.weight"] = (
        utils_load_tensor_484
    )
    utils_load_tensor_485 = utils.load_tensor(
        "./tensors/arg444.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.5.linear.weight"] = (
        utils_load_tensor_485
    )
    utils_load_tensor_486 = utils.load_tensor(
        "./tensors/arg446.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.ff_context.net.2.bias"] = (
        utils_load_tensor_486
    )
    utils_load_tensor_487 = utils.load_tensor(
        "./tensors/arg447.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.ff_context.net.2.weight"] = (
        utils_load_tensor_487
    )
    utils_load_tensor_488 = utils.load_tensor(
        "./tensors/arg448.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.ff_context.net.0.proj.bias"] = (
        utils_load_tensor_488
    )
    utils_load_tensor_489 = utils.load_tensor(
        "./tensors/arg449.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.ff_context.net.0.proj.weight"] = (
        utils_load_tensor_489
    )
    utils_load_tensor_490 = utils.load_tensor(
        "./tensors/arg450.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.4.attn.to_add_out.bias"] = (
        utils_load_tensor_490
    )
    utils_load_tensor_491 = utils.load_tensor(
        "./tensors/arg451.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.4.attn.to_add_out.weight"] = (
        utils_load_tensor_491
    )
    utils_load_tensor_492 = utils.load_tensor(
        "./tensors/arg452.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.attn.norm_k.weight"] = (
        utils_load_tensor_492
    )
    utils_load_tensor_493 = utils.load_tensor(
        "./tensors/arg453.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.to_k.bias"] = (
        utils_load_tensor_493
    )
    utils_load_tensor_494 = utils.load_tensor(
        "./tensors/arg454.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.to_k.weight"] = (
        utils_load_tensor_494
    )
    utils_load_tensor_495 = utils.load_tensor(
        "./tensors/arg455.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.attn.norm_added_k.weight"] = (
        utils_load_tensor_495
    )
    utils_load_tensor_496 = utils.load_tensor(
        "./tensors/arg456.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.add_k_proj.bias"] = (
        utils_load_tensor_496
    )
    utils_load_tensor_497 = utils.load_tensor(
        "./tensors/arg457.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.add_k_proj.weight"] = (
        utils_load_tensor_497
    )
    utils_load_tensor_498 = utils.load_tensor(
        "./tensors/arg458.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.attn.norm_q.weight"] = (
        utils_load_tensor_498
    )
    utils_load_tensor_499 = utils.load_tensor(
        "./tensors/arg459.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.to_q.bias"] = (
        utils_load_tensor_499
    )
    utils_load_tensor_500 = utils.load_tensor(
        "./tensors/arg460.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.to_q.weight"] = (
        utils_load_tensor_500
    )
    utils_load_tensor_501 = utils.load_tensor(
        "./tensors/arg461.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.attn.norm_added_q.weight"] = (
        utils_load_tensor_501
    )
    utils_load_tensor_502 = utils.load_tensor(
        "./tensors/arg462.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.add_q_proj.bias"] = (
        utils_load_tensor_502
    )
    utils_load_tensor_503 = utils.load_tensor(
        "./tensors/arg463.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.add_q_proj.weight"] = (
        utils_load_tensor_503
    )
    utils_load_tensor_504 = utils.load_tensor(
        "./tensors/arg464.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.add_v_proj.bias"] = (
        utils_load_tensor_504
    )
    utils_load_tensor_505 = utils.load_tensor(
        "./tensors/arg465.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.add_v_proj.weight"] = (
        utils_load_tensor_505
    )
    utils_load_tensor_506 = utils.load_tensor(
        "./tensors/arg466.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.norm1_context.linear.bias"] = (
        utils_load_tensor_506
    )
    utils_load_tensor_507 = utils.load_tensor(
        "./tensors/arg467.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.norm1_context.linear.weight"] = (
        utils_load_tensor_507
    )
    utils_load_tensor_508 = utils.load_tensor(
        "./tensors/arg468.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.6.linear.weight"] = (
        utils_load_tensor_508
    )
    utils_load_tensor_509 = utils.load_tensor(
        "./tensors/arg470.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.ff_context.net.2.bias"] = (
        utils_load_tensor_509
    )
    utils_load_tensor_510 = utils.load_tensor(
        "./tensors/arg471.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.ff_context.net.2.weight"] = (
        utils_load_tensor_510
    )
    utils_load_tensor_511 = utils.load_tensor(
        "./tensors/arg472.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.ff_context.net.0.proj.bias"] = (
        utils_load_tensor_511
    )
    utils_load_tensor_512 = utils.load_tensor(
        "./tensors/arg473.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.ff_context.net.0.proj.weight"] = (
        utils_load_tensor_512
    )
    utils_load_tensor_513 = utils.load_tensor(
        "./tensors/arg474.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.5.attn.to_add_out.bias"] = (
        utils_load_tensor_513
    )
    utils_load_tensor_514 = utils.load_tensor(
        "./tensors/arg475.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.5.attn.to_add_out.weight"] = (
        utils_load_tensor_514
    )
    utils_load_tensor_515 = utils.load_tensor(
        "./tensors/arg476.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.attn.norm_k.weight"] = (
        utils_load_tensor_515
    )
    utils_load_tensor_516 = utils.load_tensor(
        "./tensors/arg477.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.to_k.bias"] = (
        utils_load_tensor_516
    )
    utils_load_tensor_517 = utils.load_tensor(
        "./tensors/arg478.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.to_k.weight"] = (
        utils_load_tensor_517
    )
    utils_load_tensor_518 = utils.load_tensor(
        "./tensors/arg479.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.attn.norm_added_k.weight"] = (
        utils_load_tensor_518
    )
    utils_load_tensor_519 = utils.load_tensor(
        "./tensors/arg480.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.add_k_proj.bias"] = (
        utils_load_tensor_519
    )
    utils_load_tensor_520 = utils.load_tensor(
        "./tensors/arg481.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.add_k_proj.weight"] = (
        utils_load_tensor_520
    )
    utils_load_tensor_521 = utils.load_tensor(
        "./tensors/arg482.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.attn.norm_q.weight"] = (
        utils_load_tensor_521
    )
    utils_load_tensor_522 = utils.load_tensor(
        "./tensors/arg483.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.to_q.bias"] = (
        utils_load_tensor_522
    )
    utils_load_tensor_523 = utils.load_tensor(
        "./tensors/arg484.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.to_q.weight"] = (
        utils_load_tensor_523
    )
    utils_load_tensor_524 = utils.load_tensor(
        "./tensors/arg485.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.attn.norm_added_q.weight"] = (
        utils_load_tensor_524
    )
    utils_load_tensor_525 = utils.load_tensor(
        "./tensors/arg486.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.add_q_proj.bias"] = (
        utils_load_tensor_525
    )
    utils_load_tensor_526 = utils.load_tensor(
        "./tensors/arg487.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.add_q_proj.weight"] = (
        utils_load_tensor_526
    )
    utils_load_tensor_527 = utils.load_tensor(
        "./tensors/arg488.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.add_v_proj.bias"] = (
        utils_load_tensor_527
    )
    utils_load_tensor_528 = utils.load_tensor(
        "./tensors/arg489.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.add_v_proj.weight"] = (
        utils_load_tensor_528
    )
    utils_load_tensor_529 = utils.load_tensor(
        "./tensors/arg490.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.norm1_context.linear.bias"] = (
        utils_load_tensor_529
    )
    utils_load_tensor_530 = utils.load_tensor(
        "./tensors/arg491.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.norm1_context.linear.weight"] = (
        utils_load_tensor_530
    )
    utils_load_tensor_531 = utils.load_tensor(
        "./tensors/arg492.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.7.linear.weight"] = (
        utils_load_tensor_531
    )
    utils_load_tensor_532 = utils.load_tensor(
        "./tensors/arg494.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.ff_context.net.2.bias"] = (
        utils_load_tensor_532
    )
    utils_load_tensor_533 = utils.load_tensor(
        "./tensors/arg495.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.ff_context.net.2.weight"] = (
        utils_load_tensor_533
    )
    utils_load_tensor_534 = utils.load_tensor(
        "./tensors/arg496.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.ff_context.net.0.proj.bias"] = (
        utils_load_tensor_534
    )
    utils_load_tensor_535 = utils.load_tensor(
        "./tensors/arg497.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.ff_context.net.0.proj.weight"] = (
        utils_load_tensor_535
    )
    utils_load_tensor_536 = utils.load_tensor(
        "./tensors/arg498.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.6.attn.to_add_out.bias"] = (
        utils_load_tensor_536
    )
    utils_load_tensor_537 = utils.load_tensor(
        "./tensors/arg499.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.6.attn.to_add_out.weight"] = (
        utils_load_tensor_537
    )
    utils_load_tensor_538 = utils.load_tensor(
        "./tensors/arg500.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.attn.norm_k.weight"] = (
        utils_load_tensor_538
    )
    utils_load_tensor_539 = utils.load_tensor(
        "./tensors/arg501.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.to_k.bias"] = (
        utils_load_tensor_539
    )
    utils_load_tensor_540 = utils.load_tensor(
        "./tensors/arg502.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.to_k.weight"] = (
        utils_load_tensor_540
    )
    utils_load_tensor_541 = utils.load_tensor(
        "./tensors/arg503.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.attn.norm_added_k.weight"] = (
        utils_load_tensor_541
    )
    utils_load_tensor_542 = utils.load_tensor(
        "./tensors/arg504.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.add_k_proj.bias"] = (
        utils_load_tensor_542
    )
    utils_load_tensor_543 = utils.load_tensor(
        "./tensors/arg505.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.add_k_proj.weight"] = (
        utils_load_tensor_543
    )
    utils_load_tensor_544 = utils.load_tensor(
        "./tensors/arg506.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.attn.norm_q.weight"] = (
        utils_load_tensor_544
    )
    utils_load_tensor_545 = utils.load_tensor(
        "./tensors/arg507.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.to_q.bias"] = (
        utils_load_tensor_545
    )
    utils_load_tensor_546 = utils.load_tensor(
        "./tensors/arg508.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.to_q.weight"] = (
        utils_load_tensor_546
    )
    utils_load_tensor_547 = utils.load_tensor(
        "./tensors/arg509.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.attn.norm_added_q.weight"] = (
        utils_load_tensor_547
    )
    utils_load_tensor_548 = utils.load_tensor(
        "./tensors/arg510.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.add_q_proj.bias"] = (
        utils_load_tensor_548
    )
    utils_load_tensor_549 = utils.load_tensor(
        "./tensors/arg511.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.add_q_proj.weight"] = (
        utils_load_tensor_549
    )
    utils_load_tensor_550 = utils.load_tensor(
        "./tensors/arg512.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.8.linear.weight"] = (
        utils_load_tensor_550
    )
    utils_load_tensor_551 = utils.load_tensor(
        "./tensors/arg514.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.ff_context.net.2.bias"] = (
        utils_load_tensor_551
    )
    utils_load_tensor_552 = utils.load_tensor(
        "./tensors/arg515.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.ff_context.net.2.weight"] = (
        utils_load_tensor_552
    )
    utils_load_tensor_553 = utils.load_tensor(
        "./tensors/arg516.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.ff_context.net.0.proj.bias"] = (
        utils_load_tensor_553
    )
    utils_load_tensor_554 = utils.load_tensor(
        "./tensors/arg517.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.ff_context.net.0.proj.weight"] = (
        utils_load_tensor_554
    )
    utils_load_tensor_555 = utils.load_tensor(
        "./tensors/arg518.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.transformer_blocks.7.attn.to_add_out.bias"] = (
        utils_load_tensor_555
    )
    utils_load_tensor_556 = utils.load_tensor(
        "./tensors/arg519.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.transformer_blocks.7.attn.to_add_out.weight"] = (
        utils_load_tensor_556
    )
    utils_load_tensor_557 = utils.load_tensor(
        "./tensors/arg520.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.0.attn.to_v.bias"] = (
        utils_load_tensor_557
    )
    utils_load_tensor_558 = utils.load_tensor(
        "./tensors/arg521.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.0.attn.to_v.weight"] = (
        utils_load_tensor_558
    )
    utils_load_tensor_559 = utils.load_tensor(
        "./tensors/arg522.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.0.attn.norm_k.weight"] = (
        utils_load_tensor_559
    )
    utils_load_tensor_560 = utils.load_tensor(
        "./tensors/arg523.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.0.attn.to_k.bias"] = (
        utils_load_tensor_560
    )
    utils_load_tensor_561 = utils.load_tensor(
        "./tensors/arg524.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.0.attn.to_k.weight"] = (
        utils_load_tensor_561
    )
    utils_load_tensor_562 = utils.load_tensor(
        "./tensors/arg525.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.0.attn.norm_q.weight"] = (
        utils_load_tensor_562
    )
    utils_load_tensor_563 = utils.load_tensor(
        "./tensors/arg526.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.0.attn.to_q.bias"] = (
        utils_load_tensor_563
    )
    utils_load_tensor_564 = utils.load_tensor(
        "./tensors/arg527.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.0.attn.to_q.weight"] = (
        utils_load_tensor_564
    )
    utils_load_tensor_565 = utils.load_tensor(
        "./tensors/arg528.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.9.linear.weight"] = (
        utils_load_tensor_565
    )
    utils_load_tensor_566 = utils.load_tensor(
        "./tensors/arg530.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.1.attn.to_v.bias"] = (
        utils_load_tensor_566
    )
    utils_load_tensor_567 = utils.load_tensor(
        "./tensors/arg531.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.1.attn.to_v.weight"] = (
        utils_load_tensor_567
    )
    utils_load_tensor_568 = utils.load_tensor(
        "./tensors/arg532.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.1.attn.norm_k.weight"] = (
        utils_load_tensor_568
    )
    utils_load_tensor_569 = utils.load_tensor(
        "./tensors/arg533.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.1.attn.to_k.bias"] = (
        utils_load_tensor_569
    )
    utils_load_tensor_570 = utils.load_tensor(
        "./tensors/arg534.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.1.attn.to_k.weight"] = (
        utils_load_tensor_570
    )
    utils_load_tensor_571 = utils.load_tensor(
        "./tensors/arg535.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.1.attn.norm_q.weight"] = (
        utils_load_tensor_571
    )
    utils_load_tensor_572 = utils.load_tensor(
        "./tensors/arg536.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.1.attn.to_q.bias"] = (
        utils_load_tensor_572
    )
    utils_load_tensor_573 = utils.load_tensor(
        "./tensors/arg537.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.1.attn.to_q.weight"] = (
        utils_load_tensor_573
    )
    utils_load_tensor_574 = utils.load_tensor(
        "./tensors/arg538.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.10.linear.weight"] = (
        utils_load_tensor_574
    )
    utils_load_tensor_575 = utils.load_tensor(
        "./tensors/arg540.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.2.attn.to_v.bias"] = (
        utils_load_tensor_575
    )
    utils_load_tensor_576 = utils.load_tensor(
        "./tensors/arg541.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.2.attn.to_v.weight"] = (
        utils_load_tensor_576
    )
    utils_load_tensor_577 = utils.load_tensor(
        "./tensors/arg542.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.2.attn.norm_k.weight"] = (
        utils_load_tensor_577
    )
    utils_load_tensor_578 = utils.load_tensor(
        "./tensors/arg543.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.2.attn.to_k.bias"] = (
        utils_load_tensor_578
    )
    utils_load_tensor_579 = utils.load_tensor(
        "./tensors/arg544.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.2.attn.to_k.weight"] = (
        utils_load_tensor_579
    )
    utils_load_tensor_580 = utils.load_tensor(
        "./tensors/arg545.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.2.attn.norm_q.weight"] = (
        utils_load_tensor_580
    )
    utils_load_tensor_581 = utils.load_tensor(
        "./tensors/arg546.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.2.attn.to_q.bias"] = (
        utils_load_tensor_581
    )
    utils_load_tensor_582 = utils.load_tensor(
        "./tensors/arg547.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.2.attn.to_q.weight"] = (
        utils_load_tensor_582
    )
    utils_load_tensor_583 = utils.load_tensor(
        "./tensors/arg548.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.11.linear.weight"] = (
        utils_load_tensor_583
    )
    utils_load_tensor_584 = utils.load_tensor(
        "./tensors/arg550.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.3.attn.to_v.bias"] = (
        utils_load_tensor_584
    )
    utils_load_tensor_585 = utils.load_tensor(
        "./tensors/arg551.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.3.attn.to_v.weight"] = (
        utils_load_tensor_585
    )
    utils_load_tensor_586 = utils.load_tensor(
        "./tensors/arg552.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.3.attn.norm_k.weight"] = (
        utils_load_tensor_586
    )
    utils_load_tensor_587 = utils.load_tensor(
        "./tensors/arg553.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.3.attn.to_k.bias"] = (
        utils_load_tensor_587
    )
    utils_load_tensor_588 = utils.load_tensor(
        "./tensors/arg554.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.3.attn.to_k.weight"] = (
        utils_load_tensor_588
    )
    utils_load_tensor_589 = utils.load_tensor(
        "./tensors/arg555.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.3.attn.norm_q.weight"] = (
        utils_load_tensor_589
    )
    utils_load_tensor_590 = utils.load_tensor(
        "./tensors/arg556.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.3.attn.to_q.bias"] = (
        utils_load_tensor_590
    )
    utils_load_tensor_591 = utils.load_tensor(
        "./tensors/arg557.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.3.attn.to_q.weight"] = (
        utils_load_tensor_591
    )
    utils_load_tensor_592 = utils.load_tensor(
        "./tensors/arg558.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.12.linear.weight"] = (
        utils_load_tensor_592
    )
    utils_load_tensor_593 = utils.load_tensor(
        "./tensors/arg560.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.4.attn.to_v.bias"] = (
        utils_load_tensor_593
    )
    utils_load_tensor_594 = utils.load_tensor(
        "./tensors/arg561.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.4.attn.to_v.weight"] = (
        utils_load_tensor_594
    )
    utils_load_tensor_595 = utils.load_tensor(
        "./tensors/arg562.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.4.attn.norm_k.weight"] = (
        utils_load_tensor_595
    )
    utils_load_tensor_596 = utils.load_tensor(
        "./tensors/arg563.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.4.attn.to_k.bias"] = (
        utils_load_tensor_596
    )
    utils_load_tensor_597 = utils.load_tensor(
        "./tensors/arg564.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.4.attn.to_k.weight"] = (
        utils_load_tensor_597
    )
    utils_load_tensor_598 = utils.load_tensor(
        "./tensors/arg565.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.4.attn.norm_q.weight"] = (
        utils_load_tensor_598
    )
    utils_load_tensor_599 = utils.load_tensor(
        "./tensors/arg566.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.4.attn.to_q.bias"] = (
        utils_load_tensor_599
    )
    utils_load_tensor_600 = utils.load_tensor(
        "./tensors/arg567.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.4.attn.to_q.weight"] = (
        utils_load_tensor_600
    )
    utils_load_tensor_601 = utils.load_tensor(
        "./tensors/arg568.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.13.linear.weight"] = (
        utils_load_tensor_601
    )
    utils_load_tensor_602 = utils.load_tensor(
        "./tensors/arg570.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.5.attn.to_v.bias"] = (
        utils_load_tensor_602
    )
    utils_load_tensor_603 = utils.load_tensor(
        "./tensors/arg571.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.5.attn.to_v.weight"] = (
        utils_load_tensor_603
    )
    utils_load_tensor_604 = utils.load_tensor(
        "./tensors/arg572.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.5.attn.norm_k.weight"] = (
        utils_load_tensor_604
    )
    utils_load_tensor_605 = utils.load_tensor(
        "./tensors/arg573.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.5.attn.to_k.bias"] = (
        utils_load_tensor_605
    )
    utils_load_tensor_606 = utils.load_tensor(
        "./tensors/arg574.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.5.attn.to_k.weight"] = (
        utils_load_tensor_606
    )
    utils_load_tensor_607 = utils.load_tensor(
        "./tensors/arg575.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.5.attn.norm_q.weight"] = (
        utils_load_tensor_607
    )
    utils_load_tensor_608 = utils.load_tensor(
        "./tensors/arg576.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.5.attn.to_q.bias"] = (
        utils_load_tensor_608
    )
    utils_load_tensor_609 = utils.load_tensor(
        "./tensors/arg577.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.5.attn.to_q.weight"] = (
        utils_load_tensor_609
    )
    utils_load_tensor_610 = utils.load_tensor(
        "./tensors/arg578.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.14.linear.weight"] = (
        utils_load_tensor_610
    )
    utils_load_tensor_611 = utils.load_tensor(
        "./tensors/arg580.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.6.attn.to_v.bias"] = (
        utils_load_tensor_611
    )
    utils_load_tensor_612 = utils.load_tensor(
        "./tensors/arg581.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.6.attn.to_v.weight"] = (
        utils_load_tensor_612
    )
    utils_load_tensor_613 = utils.load_tensor(
        "./tensors/arg582.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.6.attn.norm_k.weight"] = (
        utils_load_tensor_613
    )
    utils_load_tensor_614 = utils.load_tensor(
        "./tensors/arg583.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.6.attn.to_k.bias"] = (
        utils_load_tensor_614
    )
    utils_load_tensor_615 = utils.load_tensor(
        "./tensors/arg584.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.6.attn.to_k.weight"] = (
        utils_load_tensor_615
    )
    utils_load_tensor_616 = utils.load_tensor(
        "./tensors/arg585.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.6.attn.norm_q.weight"] = (
        utils_load_tensor_616
    )
    utils_load_tensor_617 = utils.load_tensor(
        "./tensors/arg586.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.6.attn.to_q.bias"] = (
        utils_load_tensor_617
    )
    utils_load_tensor_618 = utils.load_tensor(
        "./tensors/arg587.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.6.attn.to_q.weight"] = (
        utils_load_tensor_618
    )
    utils_load_tensor_619 = utils.load_tensor(
        "./tensors/arg588.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.15.linear.weight"] = (
        utils_load_tensor_619
    )
    utils_load_tensor_620 = utils.load_tensor(
        "./tensors/arg590.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.7.attn.to_v.bias"] = (
        utils_load_tensor_620
    )
    utils_load_tensor_621 = utils.load_tensor(
        "./tensors/arg591.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.7.attn.to_v.weight"] = (
        utils_load_tensor_621
    )
    utils_load_tensor_622 = utils.load_tensor(
        "./tensors/arg592.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.7.attn.norm_k.weight"] = (
        utils_load_tensor_622
    )
    utils_load_tensor_623 = utils.load_tensor(
        "./tensors/arg593.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.7.attn.to_k.bias"] = (
        utils_load_tensor_623
    )
    utils_load_tensor_624 = utils.load_tensor(
        "./tensors/arg594.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.7.attn.to_k.weight"] = (
        utils_load_tensor_624
    )
    utils_load_tensor_625 = utils.load_tensor(
        "./tensors/arg595.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.7.attn.norm_q.weight"] = (
        utils_load_tensor_625
    )
    utils_load_tensor_626 = utils.load_tensor(
        "./tensors/arg596.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.7.attn.to_q.bias"] = (
        utils_load_tensor_626
    )
    utils_load_tensor_627 = utils.load_tensor(
        "./tensors/arg597.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.7.attn.to_q.weight"] = (
        utils_load_tensor_627
    )
    utils_load_tensor_628 = utils.load_tensor(
        "./tensors/arg598.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.16.linear.weight"] = (
        utils_load_tensor_628
    )
    utils_load_tensor_629 = utils.load_tensor(
        "./tensors/arg600.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.8.attn.to_v.bias"] = (
        utils_load_tensor_629
    )
    utils_load_tensor_630 = utils.load_tensor(
        "./tensors/arg601.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.8.attn.to_v.weight"] = (
        utils_load_tensor_630
    )
    utils_load_tensor_631 = utils.load_tensor(
        "./tensors/arg602.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.8.attn.norm_k.weight"] = (
        utils_load_tensor_631
    )
    utils_load_tensor_632 = utils.load_tensor(
        "./tensors/arg603.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.8.attn.to_k.bias"] = (
        utils_load_tensor_632
    )
    utils_load_tensor_633 = utils.load_tensor(
        "./tensors/arg604.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.8.attn.to_k.weight"] = (
        utils_load_tensor_633
    )
    utils_load_tensor_634 = utils.load_tensor(
        "./tensors/arg605.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.8.attn.norm_q.weight"] = (
        utils_load_tensor_634
    )
    utils_load_tensor_635 = utils.load_tensor(
        "./tensors/arg606.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.8.attn.to_q.bias"] = (
        utils_load_tensor_635
    )
    utils_load_tensor_636 = utils.load_tensor(
        "./tensors/arg607.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.8.attn.to_q.weight"] = (
        utils_load_tensor_636
    )
    utils_load_tensor_637 = utils.load_tensor(
        "./tensors/arg608.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.17.linear.weight"] = (
        utils_load_tensor_637
    )
    utils_load_tensor_638 = utils.load_tensor(
        "./tensors/arg610.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.9.attn.to_v.bias"] = (
        utils_load_tensor_638
    )
    utils_load_tensor_639 = utils.load_tensor(
        "./tensors/arg611.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.9.attn.to_v.weight"] = (
        utils_load_tensor_639
    )
    utils_load_tensor_640 = utils.load_tensor(
        "./tensors/arg612.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.9.attn.norm_k.weight"] = (
        utils_load_tensor_640
    )
    utils_load_tensor_641 = utils.load_tensor(
        "./tensors/arg613.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.9.attn.to_k.bias"] = (
        utils_load_tensor_641
    )
    utils_load_tensor_642 = utils.load_tensor(
        "./tensors/arg614.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.9.attn.to_k.weight"] = (
        utils_load_tensor_642
    )
    utils_load_tensor_643 = utils.load_tensor(
        "./tensors/arg615.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.9.attn.norm_q.weight"] = (
        utils_load_tensor_643
    )
    utils_load_tensor_644 = utils.load_tensor(
        "./tensors/arg616.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.9.attn.to_q.bias"] = (
        utils_load_tensor_644
    )
    utils_load_tensor_645 = utils.load_tensor(
        "./tensors/arg617.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.9.attn.to_q.weight"] = (
        utils_load_tensor_645
    )
    utils_load_tensor_646 = utils.load_tensor(
        "./tensors/arg618.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.18.linear.weight"] = (
        utils_load_tensor_646
    )
    utils_load_tensor_647 = utils.load_tensor(
        "./tensors/arg620.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.10.attn.to_v.bias"] = (
        utils_load_tensor_647
    )
    utils_load_tensor_648 = utils.load_tensor(
        "./tensors/arg621.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.10.attn.to_v.weight"] = (
        utils_load_tensor_648
    )
    utils_load_tensor_649 = utils.load_tensor(
        "./tensors/arg622.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.10.attn.norm_k.weight"] = (
        utils_load_tensor_649
    )
    utils_load_tensor_650 = utils.load_tensor(
        "./tensors/arg623.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.10.attn.to_k.bias"] = (
        utils_load_tensor_650
    )
    utils_load_tensor_651 = utils.load_tensor(
        "./tensors/arg624.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.10.attn.to_k.weight"] = (
        utils_load_tensor_651
    )
    utils_load_tensor_652 = utils.load_tensor(
        "./tensors/arg625.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.10.attn.norm_q.weight"] = (
        utils_load_tensor_652
    )
    utils_load_tensor_653 = utils.load_tensor(
        "./tensors/arg626.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.10.attn.to_q.bias"] = (
        utils_load_tensor_653
    )
    utils_load_tensor_654 = utils.load_tensor(
        "./tensors/arg627.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.10.attn.to_q.weight"] = (
        utils_load_tensor_654
    )
    utils_load_tensor_655 = utils.load_tensor(
        "./tensors/arg628.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.19.linear.weight"] = (
        utils_load_tensor_655
    )
    utils_load_tensor_656 = utils.load_tensor(
        "./tensors/arg630.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.11.attn.to_v.bias"] = (
        utils_load_tensor_656
    )
    utils_load_tensor_657 = utils.load_tensor(
        "./tensors/arg631.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.11.attn.to_v.weight"] = (
        utils_load_tensor_657
    )
    utils_load_tensor_658 = utils.load_tensor(
        "./tensors/arg632.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.11.attn.norm_k.weight"] = (
        utils_load_tensor_658
    )
    utils_load_tensor_659 = utils.load_tensor(
        "./tensors/arg633.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.11.attn.to_k.bias"] = (
        utils_load_tensor_659
    )
    utils_load_tensor_660 = utils.load_tensor(
        "./tensors/arg634.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.11.attn.to_k.weight"] = (
        utils_load_tensor_660
    )
    utils_load_tensor_661 = utils.load_tensor(
        "./tensors/arg635.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.11.attn.norm_q.weight"] = (
        utils_load_tensor_661
    )
    utils_load_tensor_662 = utils.load_tensor(
        "./tensors/arg636.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.11.attn.to_q.bias"] = (
        utils_load_tensor_662
    )
    utils_load_tensor_663 = utils.load_tensor(
        "./tensors/arg637.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.11.attn.to_q.weight"] = (
        utils_load_tensor_663
    )
    utils_load_tensor_664 = utils.load_tensor(
        "./tensors/arg638.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.20.linear.weight"] = (
        utils_load_tensor_664
    )
    utils_load_tensor_665 = utils.load_tensor(
        "./tensors/arg640.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.12.attn.to_v.bias"] = (
        utils_load_tensor_665
    )
    utils_load_tensor_666 = utils.load_tensor(
        "./tensors/arg641.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.12.attn.to_v.weight"] = (
        utils_load_tensor_666
    )
    utils_load_tensor_667 = utils.load_tensor(
        "./tensors/arg642.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.12.attn.norm_k.weight"] = (
        utils_load_tensor_667
    )
    utils_load_tensor_668 = utils.load_tensor(
        "./tensors/arg643.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.12.attn.to_k.bias"] = (
        utils_load_tensor_668
    )
    utils_load_tensor_669 = utils.load_tensor(
        "./tensors/arg644.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.12.attn.to_k.weight"] = (
        utils_load_tensor_669
    )
    utils_load_tensor_670 = utils.load_tensor(
        "./tensors/arg645.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.12.attn.norm_q.weight"] = (
        utils_load_tensor_670
    )
    utils_load_tensor_671 = utils.load_tensor(
        "./tensors/arg646.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.12.attn.to_q.bias"] = (
        utils_load_tensor_671
    )
    utils_load_tensor_672 = utils.load_tensor(
        "./tensors/arg647.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.12.attn.to_q.weight"] = (
        utils_load_tensor_672
    )
    utils_load_tensor_673 = utils.load_tensor(
        "./tensors/arg648.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.21.linear.weight"] = (
        utils_load_tensor_673
    )
    utils_load_tensor_674 = utils.load_tensor(
        "./tensors/arg650.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.13.attn.to_v.bias"] = (
        utils_load_tensor_674
    )
    utils_load_tensor_675 = utils.load_tensor(
        "./tensors/arg651.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.13.attn.to_v.weight"] = (
        utils_load_tensor_675
    )
    utils_load_tensor_676 = utils.load_tensor(
        "./tensors/arg652.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.13.attn.norm_k.weight"] = (
        utils_load_tensor_676
    )
    utils_load_tensor_677 = utils.load_tensor(
        "./tensors/arg653.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.13.attn.to_k.bias"] = (
        utils_load_tensor_677
    )
    utils_load_tensor_678 = utils.load_tensor(
        "./tensors/arg654.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.13.attn.to_k.weight"] = (
        utils_load_tensor_678
    )
    utils_load_tensor_679 = utils.load_tensor(
        "./tensors/arg655.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.13.attn.norm_q.weight"] = (
        utils_load_tensor_679
    )
    utils_load_tensor_680 = utils.load_tensor(
        "./tensors/arg656.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.13.attn.to_q.bias"] = (
        utils_load_tensor_680
    )
    utils_load_tensor_681 = utils.load_tensor(
        "./tensors/arg657.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.13.attn.to_q.weight"] = (
        utils_load_tensor_681
    )
    utils_load_tensor_682 = utils.load_tensor(
        "./tensors/arg658.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.22.linear.weight"] = (
        utils_load_tensor_682
    )
    utils_load_tensor_683 = utils.load_tensor(
        "./tensors/arg660.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.14.attn.to_v.bias"] = (
        utils_load_tensor_683
    )
    utils_load_tensor_684 = utils.load_tensor(
        "./tensors/arg661.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.14.attn.to_v.weight"] = (
        utils_load_tensor_684
    )
    utils_load_tensor_685 = utils.load_tensor(
        "./tensors/arg662.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.14.attn.norm_k.weight"] = (
        utils_load_tensor_685
    )
    utils_load_tensor_686 = utils.load_tensor(
        "./tensors/arg663.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.14.attn.to_k.bias"] = (
        utils_load_tensor_686
    )
    utils_load_tensor_687 = utils.load_tensor(
        "./tensors/arg664.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.14.attn.to_k.weight"] = (
        utils_load_tensor_687
    )
    utils_load_tensor_688 = utils.load_tensor(
        "./tensors/arg665.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.14.attn.norm_q.weight"] = (
        utils_load_tensor_688
    )
    utils_load_tensor_689 = utils.load_tensor(
        "./tensors/arg666.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.14.attn.to_q.bias"] = (
        utils_load_tensor_689
    )
    utils_load_tensor_690 = utils.load_tensor(
        "./tensors/arg667.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.14.attn.to_q.weight"] = (
        utils_load_tensor_690
    )
    utils_load_tensor_691 = utils.load_tensor(
        "./tensors/arg668.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.23.linear.weight"] = (
        utils_load_tensor_691
    )
    utils_load_tensor_692 = utils.load_tensor(
        "./tensors/arg670.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.15.attn.to_v.bias"] = (
        utils_load_tensor_692
    )
    utils_load_tensor_693 = utils.load_tensor(
        "./tensors/arg671.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.15.attn.to_v.weight"] = (
        utils_load_tensor_693
    )
    utils_load_tensor_694 = utils.load_tensor(
        "./tensors/arg672.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.15.attn.norm_k.weight"] = (
        utils_load_tensor_694
    )
    utils_load_tensor_695 = utils.load_tensor(
        "./tensors/arg673.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.15.attn.to_k.bias"] = (
        utils_load_tensor_695
    )
    utils_load_tensor_696 = utils.load_tensor(
        "./tensors/arg674.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.15.attn.to_k.weight"] = (
        utils_load_tensor_696
    )
    utils_load_tensor_697 = utils.load_tensor(
        "./tensors/arg675.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.15.attn.norm_q.weight"] = (
        utils_load_tensor_697
    )
    utils_load_tensor_698 = utils.load_tensor(
        "./tensors/arg676.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.15.attn.to_q.bias"] = (
        utils_load_tensor_698
    )
    utils_load_tensor_699 = utils.load_tensor(
        "./tensors/arg677.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.15.attn.to_q.weight"] = (
        utils_load_tensor_699
    )
    utils_load_tensor_700 = utils.load_tensor(
        "./tensors/arg678.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.24.linear.weight"] = (
        utils_load_tensor_700
    )
    utils_load_tensor_701 = utils.load_tensor(
        "./tensors/arg680.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.16.attn.to_v.bias"] = (
        utils_load_tensor_701
    )
    utils_load_tensor_702 = utils.load_tensor(
        "./tensors/arg681.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.16.attn.to_v.weight"] = (
        utils_load_tensor_702
    )
    utils_load_tensor_703 = utils.load_tensor(
        "./tensors/arg682.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.16.attn.norm_k.weight"] = (
        utils_load_tensor_703
    )
    utils_load_tensor_704 = utils.load_tensor(
        "./tensors/arg683.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.16.attn.to_k.bias"] = (
        utils_load_tensor_704
    )
    utils_load_tensor_705 = utils.load_tensor(
        "./tensors/arg684.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.16.attn.to_k.weight"] = (
        utils_load_tensor_705
    )
    utils_load_tensor_706 = utils.load_tensor(
        "./tensors/arg685.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.16.attn.norm_q.weight"] = (
        utils_load_tensor_706
    )
    utils_load_tensor_707 = utils.load_tensor(
        "./tensors/arg686.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.16.attn.to_q.bias"] = (
        utils_load_tensor_707
    )
    utils_load_tensor_708 = utils.load_tensor(
        "./tensors/arg687.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.16.attn.to_q.weight"] = (
        utils_load_tensor_708
    )
    utils_load_tensor_709 = utils.load_tensor(
        "./tensors/arg688.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.25.linear.weight"] = (
        utils_load_tensor_709
    )
    utils_load_tensor_710 = utils.load_tensor(
        "./tensors/arg690.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.17.attn.to_v.bias"] = (
        utils_load_tensor_710
    )
    utils_load_tensor_711 = utils.load_tensor(
        "./tensors/arg691.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.17.attn.to_v.weight"] = (
        utils_load_tensor_711
    )
    utils_load_tensor_712 = utils.load_tensor(
        "./tensors/arg692.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.17.attn.norm_k.weight"] = (
        utils_load_tensor_712
    )
    utils_load_tensor_713 = utils.load_tensor(
        "./tensors/arg693.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.17.attn.to_k.bias"] = (
        utils_load_tensor_713
    )
    utils_load_tensor_714 = utils.load_tensor(
        "./tensors/arg694.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.17.attn.to_k.weight"] = (
        utils_load_tensor_714
    )
    utils_load_tensor_715 = utils.load_tensor(
        "./tensors/arg695.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.17.attn.norm_q.weight"] = (
        utils_load_tensor_715
    )
    utils_load_tensor_716 = utils.load_tensor(
        "./tensors/arg696.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.17.attn.to_q.bias"] = (
        utils_load_tensor_716
    )
    utils_load_tensor_717 = utils.load_tensor(
        "./tensors/arg697.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.17.attn.to_q.weight"] = (
        utils_load_tensor_717
    )
    utils_load_tensor_718 = utils.load_tensor(
        "./tensors/arg698.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.26.linear.weight"] = (
        utils_load_tensor_718
    )
    utils_load_tensor_719 = utils.load_tensor(
        "./tensors/arg700.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.18.attn.to_v.bias"] = (
        utils_load_tensor_719
    )
    utils_load_tensor_720 = utils.load_tensor(
        "./tensors/arg701.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.18.attn.to_v.weight"] = (
        utils_load_tensor_720
    )
    utils_load_tensor_721 = utils.load_tensor(
        "./tensors/arg702.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.18.attn.norm_k.weight"] = (
        utils_load_tensor_721
    )
    utils_load_tensor_722 = utils.load_tensor(
        "./tensors/arg703.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.18.attn.to_k.bias"] = (
        utils_load_tensor_722
    )
    utils_load_tensor_723 = utils.load_tensor(
        "./tensors/arg704.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.18.attn.to_k.weight"] = (
        utils_load_tensor_723
    )
    utils_load_tensor_724 = utils.load_tensor(
        "./tensors/arg705.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.18.attn.norm_q.weight"] = (
        utils_load_tensor_724
    )
    utils_load_tensor_725 = utils.load_tensor(
        "./tensors/arg706.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.18.attn.to_q.bias"] = (
        utils_load_tensor_725
    )
    utils_load_tensor_726 = utils.load_tensor(
        "./tensors/arg707.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.18.attn.to_q.weight"] = (
        utils_load_tensor_726
    )
    utils_load_tensor_727 = utils.load_tensor(
        "./tensors/arg708.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.27.linear.weight"] = (
        utils_load_tensor_727
    )
    utils_load_tensor_728 = utils.load_tensor(
        "./tensors/arg710.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.19.attn.to_v.bias"] = (
        utils_load_tensor_728
    )
    utils_load_tensor_729 = utils.load_tensor(
        "./tensors/arg711.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.19.attn.to_v.weight"] = (
        utils_load_tensor_729
    )
    utils_load_tensor_730 = utils.load_tensor(
        "./tensors/arg712.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.19.attn.norm_k.weight"] = (
        utils_load_tensor_730
    )
    utils_load_tensor_731 = utils.load_tensor(
        "./tensors/arg713.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.19.attn.to_k.bias"] = (
        utils_load_tensor_731
    )
    utils_load_tensor_732 = utils.load_tensor(
        "./tensors/arg714.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.19.attn.to_k.weight"] = (
        utils_load_tensor_732
    )
    utils_load_tensor_733 = utils.load_tensor(
        "./tensors/arg715.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.19.attn.norm_q.weight"] = (
        utils_load_tensor_733
    )
    utils_load_tensor_734 = utils.load_tensor(
        "./tensors/arg716.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.19.attn.to_q.bias"] = (
        utils_load_tensor_734
    )
    utils_load_tensor_735 = utils.load_tensor(
        "./tensors/arg717.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.19.attn.to_q.weight"] = (
        utils_load_tensor_735
    )
    utils_load_tensor_736 = utils.load_tensor(
        "./tensors/arg718.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.28.linear.weight"] = (
        utils_load_tensor_736
    )
    utils_load_tensor_737 = utils.load_tensor(
        "./tensors/arg720.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.20.attn.to_v.bias"] = (
        utils_load_tensor_737
    )
    utils_load_tensor_738 = utils.load_tensor(
        "./tensors/arg721.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.20.attn.to_v.weight"] = (
        utils_load_tensor_738
    )
    utils_load_tensor_739 = utils.load_tensor(
        "./tensors/arg722.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.20.attn.norm_k.weight"] = (
        utils_load_tensor_739
    )
    utils_load_tensor_740 = utils.load_tensor(
        "./tensors/arg723.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.20.attn.to_k.bias"] = (
        utils_load_tensor_740
    )
    utils_load_tensor_741 = utils.load_tensor(
        "./tensors/arg724.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.20.attn.to_k.weight"] = (
        utils_load_tensor_741
    )
    utils_load_tensor_742 = utils.load_tensor(
        "./tensors/arg725.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.20.attn.norm_q.weight"] = (
        utils_load_tensor_742
    )
    utils_load_tensor_743 = utils.load_tensor(
        "./tensors/arg726.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.20.attn.to_q.bias"] = (
        utils_load_tensor_743
    )
    utils_load_tensor_744 = utils.load_tensor(
        "./tensors/arg727.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.20.attn.to_q.weight"] = (
        utils_load_tensor_744
    )
    utils_load_tensor_745 = utils.load_tensor(
        "./tensors/arg728.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.29.linear.weight"] = (
        utils_load_tensor_745
    )
    utils_load_tensor_746 = utils.load_tensor(
        "./tensors/arg730.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.21.attn.to_v.bias"] = (
        utils_load_tensor_746
    )
    utils_load_tensor_747 = utils.load_tensor(
        "./tensors/arg731.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.21.attn.to_v.weight"] = (
        utils_load_tensor_747
    )
    utils_load_tensor_748 = utils.load_tensor(
        "./tensors/arg732.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.21.attn.norm_k.weight"] = (
        utils_load_tensor_748
    )
    utils_load_tensor_749 = utils.load_tensor(
        "./tensors/arg733.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.21.attn.to_k.bias"] = (
        utils_load_tensor_749
    )
    utils_load_tensor_750 = utils.load_tensor(
        "./tensors/arg734.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.21.attn.to_k.weight"] = (
        utils_load_tensor_750
    )
    utils_load_tensor_751 = utils.load_tensor(
        "./tensors/arg735.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.21.attn.norm_q.weight"] = (
        utils_load_tensor_751
    )
    utils_load_tensor_752 = utils.load_tensor(
        "./tensors/arg736.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.21.attn.to_q.bias"] = (
        utils_load_tensor_752
    )
    utils_load_tensor_753 = utils.load_tensor(
        "./tensors/arg737.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.21.attn.to_q.weight"] = (
        utils_load_tensor_753
    )
    utils_load_tensor_754 = utils.load_tensor(
        "./tensors/arg738.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.30.linear.weight"] = (
        utils_load_tensor_754
    )
    utils_load_tensor_755 = utils.load_tensor(
        "./tensors/arg740.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.22.attn.to_v.bias"] = (
        utils_load_tensor_755
    )
    utils_load_tensor_756 = utils.load_tensor(
        "./tensors/arg741.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.22.attn.to_v.weight"] = (
        utils_load_tensor_756
    )
    utils_load_tensor_757 = utils.load_tensor(
        "./tensors/arg742.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.22.attn.norm_k.weight"] = (
        utils_load_tensor_757
    )
    utils_load_tensor_758 = utils.load_tensor(
        "./tensors/arg743.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.22.attn.to_k.bias"] = (
        utils_load_tensor_758
    )
    utils_load_tensor_759 = utils.load_tensor(
        "./tensors/arg744.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.22.attn.to_k.weight"] = (
        utils_load_tensor_759
    )
    utils_load_tensor_760 = utils.load_tensor(
        "./tensors/arg745.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.22.attn.norm_q.weight"] = (
        utils_load_tensor_760
    )
    utils_load_tensor_761 = utils.load_tensor(
        "./tensors/arg746.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.22.attn.to_q.bias"] = (
        utils_load_tensor_761
    )
    utils_load_tensor_762 = utils.load_tensor(
        "./tensors/arg747.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.22.attn.to_q.weight"] = (
        utils_load_tensor_762
    )
    utils_load_tensor_763 = utils.load_tensor(
        "./tensors/arg748.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.31.linear.weight"] = (
        utils_load_tensor_763
    )
    utils_load_tensor_764 = utils.load_tensor(
        "./tensors/arg750.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.23.attn.to_v.bias"] = (
        utils_load_tensor_764
    )
    utils_load_tensor_765 = utils.load_tensor(
        "./tensors/arg751.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.23.attn.to_v.weight"] = (
        utils_load_tensor_765
    )
    utils_load_tensor_766 = utils.load_tensor(
        "./tensors/arg752.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.23.attn.norm_k.weight"] = (
        utils_load_tensor_766
    )
    utils_load_tensor_767 = utils.load_tensor(
        "./tensors/arg753.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.23.attn.to_k.bias"] = (
        utils_load_tensor_767
    )
    utils_load_tensor_768 = utils.load_tensor(
        "./tensors/arg754.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.23.attn.to_k.weight"] = (
        utils_load_tensor_768
    )
    utils_load_tensor_769 = utils.load_tensor(
        "./tensors/arg755.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.23.attn.norm_q.weight"] = (
        utils_load_tensor_769
    )
    utils_load_tensor_770 = utils.load_tensor(
        "./tensors/arg756.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.23.attn.to_q.bias"] = (
        utils_load_tensor_770
    )
    utils_load_tensor_771 = utils.load_tensor(
        "./tensors/arg757.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.23.attn.to_q.weight"] = (
        utils_load_tensor_771
    )
    utils_load_tensor_772 = utils.load_tensor(
        "./tensors/arg758.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.32.linear.weight"] = (
        utils_load_tensor_772
    )
    utils_load_tensor_773 = utils.load_tensor(
        "./tensors/arg760.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.24.attn.to_v.bias"] = (
        utils_load_tensor_773
    )
    utils_load_tensor_774 = utils.load_tensor(
        "./tensors/arg761.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.24.attn.to_v.weight"] = (
        utils_load_tensor_774
    )
    utils_load_tensor_775 = utils.load_tensor(
        "./tensors/arg762.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.24.attn.norm_k.weight"] = (
        utils_load_tensor_775
    )
    utils_load_tensor_776 = utils.load_tensor(
        "./tensors/arg763.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.24.attn.to_k.bias"] = (
        utils_load_tensor_776
    )
    utils_load_tensor_777 = utils.load_tensor(
        "./tensors/arg764.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.24.attn.to_k.weight"] = (
        utils_load_tensor_777
    )
    utils_load_tensor_778 = utils.load_tensor(
        "./tensors/arg765.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.24.attn.norm_q.weight"] = (
        utils_load_tensor_778
    )
    utils_load_tensor_779 = utils.load_tensor(
        "./tensors/arg766.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.24.attn.to_q.bias"] = (
        utils_load_tensor_779
    )
    utils_load_tensor_780 = utils.load_tensor(
        "./tensors/arg767.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.24.attn.to_q.weight"] = (
        utils_load_tensor_780
    )
    utils_load_tensor_781 = utils.load_tensor(
        "./tensors/arg768.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.33.linear.weight"] = (
        utils_load_tensor_781
    )
    utils_load_tensor_782 = utils.load_tensor(
        "./tensors/arg770.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.25.attn.to_v.bias"] = (
        utils_load_tensor_782
    )
    utils_load_tensor_783 = utils.load_tensor(
        "./tensors/arg771.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.25.attn.to_v.weight"] = (
        utils_load_tensor_783
    )
    utils_load_tensor_784 = utils.load_tensor(
        "./tensors/arg772.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.25.attn.norm_k.weight"] = (
        utils_load_tensor_784
    )
    utils_load_tensor_785 = utils.load_tensor(
        "./tensors/arg773.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.25.attn.to_k.bias"] = (
        utils_load_tensor_785
    )
    utils_load_tensor_786 = utils.load_tensor(
        "./tensors/arg774.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.25.attn.to_k.weight"] = (
        utils_load_tensor_786
    )
    utils_load_tensor_787 = utils.load_tensor(
        "./tensors/arg775.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.25.attn.norm_q.weight"] = (
        utils_load_tensor_787
    )
    utils_load_tensor_788 = utils.load_tensor(
        "./tensors/arg776.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.25.attn.to_q.bias"] = (
        utils_load_tensor_788
    )
    utils_load_tensor_789 = utils.load_tensor(
        "./tensors/arg777.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.25.attn.to_q.weight"] = (
        utils_load_tensor_789
    )
    utils_load_tensor_790 = utils.load_tensor(
        "./tensors/arg778.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.34.linear.weight"] = (
        utils_load_tensor_790
    )
    utils_load_tensor_791 = utils.load_tensor(
        "./tensors/arg780.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.26.attn.to_v.bias"] = (
        utils_load_tensor_791
    )
    utils_load_tensor_792 = utils.load_tensor(
        "./tensors/arg781.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.26.attn.to_v.weight"] = (
        utils_load_tensor_792
    )
    utils_load_tensor_793 = utils.load_tensor(
        "./tensors/arg782.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.26.attn.norm_k.weight"] = (
        utils_load_tensor_793
    )
    utils_load_tensor_794 = utils.load_tensor(
        "./tensors/arg783.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.26.attn.to_k.bias"] = (
        utils_load_tensor_794
    )
    utils_load_tensor_795 = utils.load_tensor(
        "./tensors/arg784.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.26.attn.to_k.weight"] = (
        utils_load_tensor_795
    )
    utils_load_tensor_796 = utils.load_tensor(
        "./tensors/arg785.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.26.attn.norm_q.weight"] = (
        utils_load_tensor_796
    )
    utils_load_tensor_797 = utils.load_tensor(
        "./tensors/arg786.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.26.attn.to_q.bias"] = (
        utils_load_tensor_797
    )
    utils_load_tensor_798 = utils.load_tensor(
        "./tensors/arg787.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.26.attn.to_q.weight"] = (
        utils_load_tensor_798
    )
    utils_load_tensor_799 = utils.load_tensor(
        "./tensors/arg788.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.35.linear.weight"] = (
        utils_load_tensor_799
    )
    utils_load_tensor_800 = utils.load_tensor(
        "./tensors/arg790.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.27.attn.to_v.bias"] = (
        utils_load_tensor_800
    )
    utils_load_tensor_801 = utils.load_tensor(
        "./tensors/arg791.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.27.attn.to_v.weight"] = (
        utils_load_tensor_801
    )
    utils_load_tensor_802 = utils.load_tensor(
        "./tensors/arg792.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.27.attn.norm_k.weight"] = (
        utils_load_tensor_802
    )
    utils_load_tensor_803 = utils.load_tensor(
        "./tensors/arg793.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.27.attn.to_k.bias"] = (
        utils_load_tensor_803
    )
    utils_load_tensor_804 = utils.load_tensor(
        "./tensors/arg794.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.27.attn.to_k.weight"] = (
        utils_load_tensor_804
    )
    utils_load_tensor_805 = utils.load_tensor(
        "./tensors/arg795.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.27.attn.norm_q.weight"] = (
        utils_load_tensor_805
    )
    utils_load_tensor_806 = utils.load_tensor(
        "./tensors/arg796.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.27.attn.to_q.bias"] = (
        utils_load_tensor_806
    )
    utils_load_tensor_807 = utils.load_tensor(
        "./tensors/arg797.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.27.attn.to_q.weight"] = (
        utils_load_tensor_807
    )
    utils_load_tensor_808 = utils.load_tensor(
        "./tensors/arg798.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.36.linear.weight"] = (
        utils_load_tensor_808
    )
    utils_load_tensor_809 = utils.load_tensor(
        "./tensors/arg800.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.28.attn.to_v.bias"] = (
        utils_load_tensor_809
    )
    utils_load_tensor_810 = utils.load_tensor(
        "./tensors/arg801.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.28.attn.to_v.weight"] = (
        utils_load_tensor_810
    )
    utils_load_tensor_811 = utils.load_tensor(
        "./tensors/arg802.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.28.attn.norm_k.weight"] = (
        utils_load_tensor_811
    )
    utils_load_tensor_812 = utils.load_tensor(
        "./tensors/arg803.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.28.attn.to_k.bias"] = (
        utils_load_tensor_812
    )
    utils_load_tensor_813 = utils.load_tensor(
        "./tensors/arg804.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.28.attn.to_k.weight"] = (
        utils_load_tensor_813
    )
    utils_load_tensor_814 = utils.load_tensor(
        "./tensors/arg805.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.28.attn.norm_q.weight"] = (
        utils_load_tensor_814
    )
    utils_load_tensor_815 = utils.load_tensor(
        "./tensors/arg806.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.28.attn.to_q.bias"] = (
        utils_load_tensor_815
    )
    utils_load_tensor_816 = utils.load_tensor(
        "./tensors/arg807.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.28.attn.to_q.weight"] = (
        utils_load_tensor_816
    )
    utils_load_tensor_817 = utils.load_tensor(
        "./tensors/arg808.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.37.linear.weight"] = (
        utils_load_tensor_817
    )
    utils_load_tensor_818 = utils.load_tensor(
        "./tensors/arg810.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.29.attn.to_v.bias"] = (
        utils_load_tensor_818
    )
    utils_load_tensor_819 = utils.load_tensor(
        "./tensors/arg811.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.29.attn.to_v.weight"] = (
        utils_load_tensor_819
    )
    utils_load_tensor_820 = utils.load_tensor(
        "./tensors/arg812.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.29.attn.norm_k.weight"] = (
        utils_load_tensor_820
    )
    utils_load_tensor_821 = utils.load_tensor(
        "./tensors/arg813.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.29.attn.to_k.bias"] = (
        utils_load_tensor_821
    )
    utils_load_tensor_822 = utils.load_tensor(
        "./tensors/arg814.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.29.attn.to_k.weight"] = (
        utils_load_tensor_822
    )
    utils_load_tensor_823 = utils.load_tensor(
        "./tensors/arg815.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.29.attn.norm_q.weight"] = (
        utils_load_tensor_823
    )
    utils_load_tensor_824 = utils.load_tensor(
        "./tensors/arg816.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.29.attn.to_q.bias"] = (
        utils_load_tensor_824
    )
    utils_load_tensor_825 = utils.load_tensor(
        "./tensors/arg817.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.29.attn.to_q.weight"] = (
        utils_load_tensor_825
    )
    utils_load_tensor_826 = utils.load_tensor(
        "./tensors/arg818.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.38.linear.weight"] = (
        utils_load_tensor_826
    )
    utils_load_tensor_827 = utils.load_tensor(
        "./tensors/arg820.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.30.attn.to_v.bias"] = (
        utils_load_tensor_827
    )
    utils_load_tensor_828 = utils.load_tensor(
        "./tensors/arg821.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.30.attn.to_v.weight"] = (
        utils_load_tensor_828
    )
    utils_load_tensor_829 = utils.load_tensor(
        "./tensors/arg822.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.30.attn.norm_k.weight"] = (
        utils_load_tensor_829
    )
    utils_load_tensor_830 = utils.load_tensor(
        "./tensors/arg823.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.30.attn.to_k.bias"] = (
        utils_load_tensor_830
    )
    utils_load_tensor_831 = utils.load_tensor(
        "./tensors/arg824.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.30.attn.to_k.weight"] = (
        utils_load_tensor_831
    )
    utils_load_tensor_832 = utils.load_tensor(
        "./tensors/arg825.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.30.attn.norm_q.weight"] = (
        utils_load_tensor_832
    )
    utils_load_tensor_833 = utils.load_tensor(
        "./tensors/arg826.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.30.attn.to_q.bias"] = (
        utils_load_tensor_833
    )
    utils_load_tensor_834 = utils.load_tensor(
        "./tensors/arg827.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.30.attn.to_q.weight"] = (
        utils_load_tensor_834
    )
    utils_load_tensor_835 = utils.load_tensor(
        "./tensors/arg828.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.39.linear.weight"] = (
        utils_load_tensor_835
    )
    utils_load_tensor_836 = utils.load_tensor(
        "./tensors/arg830.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.31.attn.to_v.bias"] = (
        utils_load_tensor_836
    )
    utils_load_tensor_837 = utils.load_tensor(
        "./tensors/arg831.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.31.attn.to_v.weight"] = (
        utils_load_tensor_837
    )
    utils_load_tensor_838 = utils.load_tensor(
        "./tensors/arg832.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.31.attn.norm_k.weight"] = (
        utils_load_tensor_838
    )
    utils_load_tensor_839 = utils.load_tensor(
        "./tensors/arg833.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.31.attn.to_k.bias"] = (
        utils_load_tensor_839
    )
    utils_load_tensor_840 = utils.load_tensor(
        "./tensors/arg834.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.31.attn.to_k.weight"] = (
        utils_load_tensor_840
    )
    utils_load_tensor_841 = utils.load_tensor(
        "./tensors/arg835.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.31.attn.norm_q.weight"] = (
        utils_load_tensor_841
    )
    utils_load_tensor_842 = utils.load_tensor(
        "./tensors/arg836.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.31.attn.to_q.bias"] = (
        utils_load_tensor_842
    )
    utils_load_tensor_843 = utils.load_tensor(
        "./tensors/arg837.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.31.attn.to_q.weight"] = (
        utils_load_tensor_843
    )
    utils_load_tensor_844 = utils.load_tensor(
        "./tensors/arg838.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.40.linear.weight"] = (
        utils_load_tensor_844
    )
    utils_load_tensor_845 = utils.load_tensor(
        "./tensors/arg840.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.32.attn.to_v.bias"] = (
        utils_load_tensor_845
    )
    utils_load_tensor_846 = utils.load_tensor(
        "./tensors/arg841.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.32.attn.to_v.weight"] = (
        utils_load_tensor_846
    )
    utils_load_tensor_847 = utils.load_tensor(
        "./tensors/arg842.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.32.attn.norm_k.weight"] = (
        utils_load_tensor_847
    )
    utils_load_tensor_848 = utils.load_tensor(
        "./tensors/arg843.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.32.attn.to_k.bias"] = (
        utils_load_tensor_848
    )
    utils_load_tensor_849 = utils.load_tensor(
        "./tensors/arg844.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.32.attn.to_k.weight"] = (
        utils_load_tensor_849
    )
    utils_load_tensor_850 = utils.load_tensor(
        "./tensors/arg845.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.32.attn.norm_q.weight"] = (
        utils_load_tensor_850
    )
    utils_load_tensor_851 = utils.load_tensor(
        "./tensors/arg846.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.32.attn.to_q.bias"] = (
        utils_load_tensor_851
    )
    utils_load_tensor_852 = utils.load_tensor(
        "./tensors/arg847.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.32.attn.to_q.weight"] = (
        utils_load_tensor_852
    )
    utils_load_tensor_853 = utils.load_tensor(
        "./tensors/arg848.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.41.linear.weight"] = (
        utils_load_tensor_853
    )
    utils_load_tensor_854 = utils.load_tensor(
        "./tensors/arg850.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.33.attn.to_v.bias"] = (
        utils_load_tensor_854
    )
    utils_load_tensor_855 = utils.load_tensor(
        "./tensors/arg851.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.33.attn.to_v.weight"] = (
        utils_load_tensor_855
    )
    utils_load_tensor_856 = utils.load_tensor(
        "./tensors/arg852.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.33.attn.norm_k.weight"] = (
        utils_load_tensor_856
    )
    utils_load_tensor_857 = utils.load_tensor(
        "./tensors/arg853.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.33.attn.to_k.bias"] = (
        utils_load_tensor_857
    )
    utils_load_tensor_858 = utils.load_tensor(
        "./tensors/arg854.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.33.attn.to_k.weight"] = (
        utils_load_tensor_858
    )
    utils_load_tensor_859 = utils.load_tensor(
        "./tensors/arg855.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.33.attn.norm_q.weight"] = (
        utils_load_tensor_859
    )
    utils_load_tensor_860 = utils.load_tensor(
        "./tensors/arg856.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.33.attn.to_q.bias"] = (
        utils_load_tensor_860
    )
    utils_load_tensor_861 = utils.load_tensor(
        "./tensors/arg857.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.33.attn.to_q.weight"] = (
        utils_load_tensor_861
    )
    utils_load_tensor_862 = utils.load_tensor(
        "./tensors/arg858.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.42.linear.weight"] = (
        utils_load_tensor_862
    )
    utils_load_tensor_863 = utils.load_tensor(
        "./tensors/arg860.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.34.attn.to_v.bias"] = (
        utils_load_tensor_863
    )
    utils_load_tensor_864 = utils.load_tensor(
        "./tensors/arg861.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.34.attn.to_v.weight"] = (
        utils_load_tensor_864
    )
    utils_load_tensor_865 = utils.load_tensor(
        "./tensors/arg862.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.34.attn.norm_k.weight"] = (
        utils_load_tensor_865
    )
    utils_load_tensor_866 = utils.load_tensor(
        "./tensors/arg863.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.34.attn.to_k.bias"] = (
        utils_load_tensor_866
    )
    utils_load_tensor_867 = utils.load_tensor(
        "./tensors/arg864.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.34.attn.to_k.weight"] = (
        utils_load_tensor_867
    )
    utils_load_tensor_868 = utils.load_tensor(
        "./tensors/arg865.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.34.attn.norm_q.weight"] = (
        utils_load_tensor_868
    )
    utils_load_tensor_869 = utils.load_tensor(
        "./tensors/arg866.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.34.attn.to_q.bias"] = (
        utils_load_tensor_869
    )
    utils_load_tensor_870 = utils.load_tensor(
        "./tensors/arg867.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.34.attn.to_q.weight"] = (
        utils_load_tensor_870
    )
    utils_load_tensor_871 = utils.load_tensor(
        "./tensors/arg868.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.43.linear.weight"] = (
        utils_load_tensor_871
    )
    utils_load_tensor_872 = utils.load_tensor(
        "./tensors/arg870.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.35.attn.to_v.bias"] = (
        utils_load_tensor_872
    )
    utils_load_tensor_873 = utils.load_tensor(
        "./tensors/arg871.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.35.attn.to_v.weight"] = (
        utils_load_tensor_873
    )
    utils_load_tensor_874 = utils.load_tensor(
        "./tensors/arg872.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.35.attn.norm_k.weight"] = (
        utils_load_tensor_874
    )
    utils_load_tensor_875 = utils.load_tensor(
        "./tensors/arg873.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.35.attn.to_k.bias"] = (
        utils_load_tensor_875
    )
    utils_load_tensor_876 = utils.load_tensor(
        "./tensors/arg874.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.35.attn.to_k.weight"] = (
        utils_load_tensor_876
    )
    utils_load_tensor_877 = utils.load_tensor(
        "./tensors/arg875.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.35.attn.norm_q.weight"] = (
        utils_load_tensor_877
    )
    utils_load_tensor_878 = utils.load_tensor(
        "./tensors/arg876.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.35.attn.to_q.bias"] = (
        utils_load_tensor_878
    )
    utils_load_tensor_879 = utils.load_tensor(
        "./tensors/arg877.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.35.attn.to_q.weight"] = (
        utils_load_tensor_879
    )
    utils_load_tensor_880 = utils.load_tensor(
        "./tensors/arg878.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.44.linear.weight"] = (
        utils_load_tensor_880
    )
    utils_load_tensor_881 = utils.load_tensor(
        "./tensors/arg880.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.36.attn.to_v.bias"] = (
        utils_load_tensor_881
    )
    utils_load_tensor_882 = utils.load_tensor(
        "./tensors/arg881.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.36.attn.to_v.weight"] = (
        utils_load_tensor_882
    )
    utils_load_tensor_883 = utils.load_tensor(
        "./tensors/arg882.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.36.attn.norm_k.weight"] = (
        utils_load_tensor_883
    )
    utils_load_tensor_884 = utils.load_tensor(
        "./tensors/arg883.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.36.attn.to_k.bias"] = (
        utils_load_tensor_884
    )
    utils_load_tensor_885 = utils.load_tensor(
        "./tensors/arg884.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.36.attn.to_k.weight"] = (
        utils_load_tensor_885
    )
    utils_load_tensor_886 = utils.load_tensor(
        "./tensors/arg885.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.36.attn.norm_q.weight"] = (
        utils_load_tensor_886
    )
    utils_load_tensor_887 = utils.load_tensor(
        "./tensors/arg886.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.36.attn.to_q.bias"] = (
        utils_load_tensor_887
    )
    utils_load_tensor_888 = utils.load_tensor(
        "./tensors/arg887.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.36.attn.to_q.weight"] = (
        utils_load_tensor_888
    )
    utils_load_tensor_889 = utils.load_tensor(
        "./tensors/arg888.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.caption_projection.45.linear.weight"] = (
        utils_load_tensor_889
    )
    utils_load_tensor_890 = utils.load_tensor(
        "./tensors/arg890.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.37.attn.to_v.bias"] = (
        utils_load_tensor_890
    )
    utils_load_tensor_891 = utils.load_tensor(
        "./tensors/arg891.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.37.attn.to_v.weight"] = (
        utils_load_tensor_891
    )
    utils_load_tensor_892 = utils.load_tensor(
        "./tensors/arg892.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.37.attn.norm_k.weight"] = (
        utils_load_tensor_892
    )
    utils_load_tensor_893 = utils.load_tensor(
        "./tensors/arg893.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.37.attn.to_k.bias"] = (
        utils_load_tensor_893
    )
    utils_load_tensor_894 = utils.load_tensor(
        "./tensors/arg894.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.37.attn.to_k.weight"] = (
        utils_load_tensor_894
    )
    utils_load_tensor_895 = utils.load_tensor(
        "./tensors/arg895.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_243,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    _main_weights["transformer.single_transformer_blocks.37.attn.norm_q.weight"] = (
        utils_load_tensor_895
    )
    utils_load_tensor_896 = utils.load_tensor(
        "./tensors/arg896.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.37.attn.to_q.bias"] = (
        utils_load_tensor_896
    )
    utils_load_tensor_897 = utils.load_tensor(
        "./tensors/arg897.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["transformer.single_transformer_blocks.37.attn.to_q.weight"] = (
        utils_load_tensor_897
    )
    return _main_weights



# fmt: off
TILE_ON_DEVICE_WEIGHTS = {
    "transformer.caption_projection.0.linear.weight",
    "transformer.caption_projection.1.linear.weight",
    "transformer.caption_projection.10.linear.weight",
    "transformer.caption_projection.11.linear.weight",
    "transformer.caption_projection.12.linear.weight",
    "transformer.caption_projection.13.linear.weight",
    "transformer.caption_projection.14.linear.weight",
    "transformer.caption_projection.15.linear.weight",
    "transformer.caption_projection.16.linear.weight",
    "transformer.caption_projection.17.linear.weight",
    "transformer.caption_projection.18.linear.weight",
    "transformer.caption_projection.19.linear.weight",
    "transformer.caption_projection.2.linear.weight",
    "transformer.caption_projection.20.linear.weight",
    "transformer.caption_projection.21.linear.weight",
    "transformer.caption_projection.22.linear.weight",
    "transformer.caption_projection.23.linear.weight",
    "transformer.caption_projection.24.linear.weight",
    "transformer.caption_projection.25.linear.weight",
    "transformer.caption_projection.26.linear.weight",
    "transformer.caption_projection.27.linear.weight",
    "transformer.caption_projection.28.linear.weight",
    "transformer.caption_projection.29.linear.weight",
    "transformer.caption_projection.3.linear.weight",
    "transformer.caption_projection.30.linear.weight",
    "transformer.caption_projection.31.linear.weight",
    "transformer.caption_projection.32.linear.weight",
    "transformer.caption_projection.33.linear.weight",
    "transformer.caption_projection.34.linear.weight",
    "transformer.caption_projection.35.linear.weight",
    "transformer.caption_projection.36.linear.weight",
    "transformer.caption_projection.37.linear.weight",
    "transformer.caption_projection.38.linear.weight",
    "transformer.caption_projection.39.linear.weight",
    "transformer.caption_projection.4.linear.weight",
    "transformer.caption_projection.40.linear.weight",
    "transformer.caption_projection.41.linear.weight",
    "transformer.caption_projection.42.linear.weight",
    "transformer.caption_projection.43.linear.weight",
    "transformer.caption_projection.44.linear.weight",
    "transformer.caption_projection.45.linear.weight",
    "transformer.caption_projection.5.linear.weight",
    "transformer.caption_projection.6.linear.weight",
    "transformer.caption_projection.7.linear.weight",
    "transformer.caption_projection.8.linear.weight",
    "transformer.caption_projection.9.linear.weight",
    "transformer.context_embedder.bias",
    "transformer.context_embedder.weight",
    "transformer.proj_out.bias",
    "transformer.proj_out.weight",
    "transformer.single_transformer_blocks.0.attn.norm_k.weight",
    "transformer.single_transformer_blocks.0.attn.norm_q.weight",
    "transformer.single_transformer_blocks.0.proj_out.weight",
    "transformer.single_transformer_blocks.1.attn.norm_k.weight",
    "transformer.single_transformer_blocks.1.attn.norm_q.weight",
    "transformer.single_transformer_blocks.1.proj_out.weight",
    "transformer.single_transformer_blocks.10.attn.norm_k.weight",
    "transformer.single_transformer_blocks.10.attn.norm_q.weight",
    "transformer.single_transformer_blocks.10.proj_out.weight",
    "transformer.single_transformer_blocks.11.attn.norm_k.weight",
    "transformer.single_transformer_blocks.11.attn.norm_q.weight",
    "transformer.single_transformer_blocks.11.proj_out.weight",
    "transformer.single_transformer_blocks.12.attn.norm_k.weight",
    "transformer.single_transformer_blocks.12.attn.norm_q.weight",
    "transformer.single_transformer_blocks.12.proj_out.weight",
    "transformer.single_transformer_blocks.13.attn.norm_k.weight",
    "transformer.single_transformer_blocks.13.attn.norm_q.weight",
    "transformer.single_transformer_blocks.13.proj_out.weight",
    "transformer.single_transformer_blocks.14.attn.norm_k.weight",
    "transformer.single_transformer_blocks.14.attn.norm_q.weight",
    "transformer.single_transformer_blocks.14.proj_out.weight",
    "transformer.single_transformer_blocks.15.attn.norm_k.weight",
    "transformer.single_transformer_blocks.15.attn.norm_q.weight",
    "transformer.single_transformer_blocks.15.proj_out.weight",
    "transformer.single_transformer_blocks.16.attn.norm_k.weight",
    "transformer.single_transformer_blocks.16.attn.norm_q.weight",
    "transformer.single_transformer_blocks.16.proj_out.weight",
    "transformer.single_transformer_blocks.17.attn.norm_k.weight",
    "transformer.single_transformer_blocks.17.attn.norm_q.weight",
    "transformer.single_transformer_blocks.17.proj_out.weight",
    "transformer.single_transformer_blocks.18.attn.norm_k.weight",
    "transformer.single_transformer_blocks.18.attn.norm_q.weight",
    "transformer.single_transformer_blocks.18.proj_out.weight",
    "transformer.single_transformer_blocks.19.attn.norm_k.weight",
    "transformer.single_transformer_blocks.19.attn.norm_q.weight",
    "transformer.single_transformer_blocks.19.proj_out.weight",
    "transformer.single_transformer_blocks.2.attn.norm_k.weight",
    "transformer.single_transformer_blocks.2.attn.norm_q.weight",
    "transformer.single_transformer_blocks.2.proj_out.weight",
    "transformer.single_transformer_blocks.20.attn.norm_k.weight",
    "transformer.single_transformer_blocks.20.attn.norm_q.weight",
    "transformer.single_transformer_blocks.20.proj_out.weight",
    "transformer.single_transformer_blocks.21.attn.norm_k.weight",
    "transformer.single_transformer_blocks.21.attn.norm_q.weight",
    "transformer.single_transformer_blocks.21.proj_out.weight",
    "transformer.single_transformer_blocks.22.attn.norm_k.weight",
    "transformer.single_transformer_blocks.22.attn.norm_q.weight",
    "transformer.single_transformer_blocks.22.proj_out.weight",
    "transformer.single_transformer_blocks.23.attn.norm_k.weight",
    "transformer.single_transformer_blocks.23.attn.norm_q.weight",
    "transformer.single_transformer_blocks.23.proj_out.weight",
    "transformer.single_transformer_blocks.24.attn.norm_k.weight",
    "transformer.single_transformer_blocks.24.attn.norm_q.weight",
    "transformer.single_transformer_blocks.24.proj_out.weight",
    "transformer.single_transformer_blocks.25.attn.norm_k.weight",
    "transformer.single_transformer_blocks.25.attn.norm_q.weight",
    "transformer.single_transformer_blocks.25.proj_out.weight",
    "transformer.single_transformer_blocks.26.attn.norm_k.weight",
    "transformer.single_transformer_blocks.26.attn.norm_q.weight",
    "transformer.single_transformer_blocks.26.proj_out.weight",
    "transformer.single_transformer_blocks.27.attn.norm_k.weight",
    "transformer.single_transformer_blocks.27.attn.norm_q.weight",
    "transformer.single_transformer_blocks.27.proj_out.weight",
    "transformer.single_transformer_blocks.28.attn.norm_k.weight",
    "transformer.single_transformer_blocks.28.attn.norm_q.weight",
    "transformer.single_transformer_blocks.28.proj_out.weight",
    "transformer.single_transformer_blocks.29.attn.norm_k.weight",
    "transformer.single_transformer_blocks.29.attn.norm_q.weight",
    "transformer.single_transformer_blocks.29.proj_out.weight",
    "transformer.single_transformer_blocks.3.attn.norm_k.weight",
    "transformer.single_transformer_blocks.3.attn.norm_q.weight",
    "transformer.single_transformer_blocks.3.proj_out.weight",
    "transformer.single_transformer_blocks.30.attn.norm_k.weight",
    "transformer.single_transformer_blocks.30.attn.norm_q.weight",
    "transformer.single_transformer_blocks.30.proj_out.weight",
    "transformer.single_transformer_blocks.31.attn.norm_k.weight",
    "transformer.single_transformer_blocks.31.attn.norm_q.weight",
    "transformer.single_transformer_blocks.31.proj_out.weight",
    "transformer.single_transformer_blocks.32.attn.norm_k.weight",
    "transformer.single_transformer_blocks.32.attn.norm_q.weight",
    "transformer.single_transformer_blocks.32.proj_out.weight",
    "transformer.single_transformer_blocks.33.attn.norm_k.weight",
    "transformer.single_transformer_blocks.33.attn.norm_q.weight",
    "transformer.single_transformer_blocks.33.proj_out.weight",
    "transformer.single_transformer_blocks.34.attn.norm_k.weight",
    "transformer.single_transformer_blocks.34.attn.norm_q.weight",
    "transformer.single_transformer_blocks.34.proj_out.weight",
    "transformer.single_transformer_blocks.35.attn.norm_k.weight",
    "transformer.single_transformer_blocks.35.attn.norm_q.weight",
    "transformer.single_transformer_blocks.35.proj_out.weight",
    "transformer.single_transformer_blocks.36.attn.norm_k.weight",
    "transformer.single_transformer_blocks.36.attn.norm_q.weight",
    "transformer.single_transformer_blocks.36.proj_out.weight",
    "transformer.single_transformer_blocks.37.attn.norm_k.weight",
    "transformer.single_transformer_blocks.37.attn.norm_q.weight",
    "transformer.single_transformer_blocks.37.proj_out.weight",
    "transformer.single_transformer_blocks.4.attn.norm_k.weight",
    "transformer.single_transformer_blocks.4.attn.norm_q.weight",
    "transformer.single_transformer_blocks.4.proj_out.weight",
    "transformer.single_transformer_blocks.5.attn.norm_k.weight",
    "transformer.single_transformer_blocks.5.attn.norm_q.weight",
    "transformer.single_transformer_blocks.5.proj_out.weight",
    "transformer.single_transformer_blocks.6.attn.norm_k.weight",
    "transformer.single_transformer_blocks.6.attn.norm_q.weight",
    "transformer.single_transformer_blocks.6.proj_out.weight",
    "transformer.single_transformer_blocks.7.attn.norm_k.weight",
    "transformer.single_transformer_blocks.7.attn.norm_q.weight",
    "transformer.single_transformer_blocks.7.proj_out.weight",
    "transformer.single_transformer_blocks.8.attn.norm_k.weight",
    "transformer.single_transformer_blocks.8.attn.norm_q.weight",
    "transformer.single_transformer_blocks.8.proj_out.weight",
    "transformer.single_transformer_blocks.9.attn.norm_k.weight",
    "transformer.single_transformer_blocks.9.attn.norm_q.weight",
    "transformer.single_transformer_blocks.9.proj_out.weight",
    "transformer.transformer_blocks.0.attn.norm_added_k.weight",
    "transformer.transformer_blocks.0.attn.norm_added_q.weight",
    "transformer.transformer_blocks.0.attn.norm_k.weight",
    "transformer.transformer_blocks.0.attn.norm_q.weight",
    "transformer.transformer_blocks.0.attn.to_add_out.weight",
    "transformer.transformer_blocks.0.attn.to_out.0.weight",
    "transformer.transformer_blocks.0.ff.net.0.proj.bias",
    "transformer.transformer_blocks.0.ff.net.0.proj.weight",
    "transformer.transformer_blocks.0.ff.net.2.weight",
    "transformer.transformer_blocks.0.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.0.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.0.ff_context.net.2.weight",
    "transformer.transformer_blocks.1.attn.norm_added_k.weight",
    "transformer.transformer_blocks.1.attn.norm_added_q.weight",
    "transformer.transformer_blocks.1.attn.norm_k.weight",
    "transformer.transformer_blocks.1.attn.norm_q.weight",
    "transformer.transformer_blocks.1.attn.to_add_out.weight",
    "transformer.transformer_blocks.1.attn.to_out.0.weight",
    "transformer.transformer_blocks.1.ff.net.0.proj.bias",
    "transformer.transformer_blocks.1.ff.net.0.proj.weight",
    "transformer.transformer_blocks.1.ff.net.2.weight",
    "transformer.transformer_blocks.1.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.1.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.1.ff_context.net.2.weight",
    "transformer.transformer_blocks.2.attn.norm_added_k.weight",
    "transformer.transformer_blocks.2.attn.norm_added_q.weight",
    "transformer.transformer_blocks.2.attn.norm_k.weight",
    "transformer.transformer_blocks.2.attn.norm_q.weight",
    "transformer.transformer_blocks.2.attn.to_add_out.weight",
    "transformer.transformer_blocks.2.attn.to_out.0.weight",
    "transformer.transformer_blocks.2.ff.net.0.proj.bias",
    "transformer.transformer_blocks.2.ff.net.0.proj.weight",
    "transformer.transformer_blocks.2.ff.net.2.weight",
    "transformer.transformer_blocks.2.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.2.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.2.ff_context.net.2.weight",
    "transformer.transformer_blocks.3.attn.norm_added_k.weight",
    "transformer.transformer_blocks.3.attn.norm_added_q.weight",
    "transformer.transformer_blocks.3.attn.norm_k.weight",
    "transformer.transformer_blocks.3.attn.norm_q.weight",
    "transformer.transformer_blocks.3.attn.to_add_out.weight",
    "transformer.transformer_blocks.3.attn.to_out.0.weight",
    "transformer.transformer_blocks.3.ff.net.0.proj.bias",
    "transformer.transformer_blocks.3.ff.net.0.proj.weight",
    "transformer.transformer_blocks.3.ff.net.2.weight",
    "transformer.transformer_blocks.3.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.3.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.3.ff_context.net.2.weight",
    "transformer.transformer_blocks.4.attn.norm_added_k.weight",
    "transformer.transformer_blocks.4.attn.norm_added_q.weight",
    "transformer.transformer_blocks.4.attn.norm_k.weight",
    "transformer.transformer_blocks.4.attn.norm_q.weight",
    "transformer.transformer_blocks.4.attn.to_add_out.weight",
    "transformer.transformer_blocks.4.attn.to_out.0.weight",
    "transformer.transformer_blocks.4.ff.net.0.proj.bias",
    "transformer.transformer_blocks.4.ff.net.0.proj.weight",
    "transformer.transformer_blocks.4.ff.net.2.weight",
    "transformer.transformer_blocks.4.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.4.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.4.ff_context.net.2.weight",
    "transformer.transformer_blocks.5.attn.norm_added_k.weight",
    "transformer.transformer_blocks.5.attn.norm_added_q.weight",
    "transformer.transformer_blocks.5.attn.norm_k.weight",
    "transformer.transformer_blocks.5.attn.norm_q.weight",
    "transformer.transformer_blocks.5.attn.to_add_out.weight",
    "transformer.transformer_blocks.5.attn.to_out.0.weight",
    "transformer.transformer_blocks.5.ff.net.0.proj.bias",
    "transformer.transformer_blocks.5.ff.net.0.proj.weight",
    "transformer.transformer_blocks.5.ff.net.2.weight",
    "transformer.transformer_blocks.5.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.5.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.5.ff_context.net.2.weight",
    "transformer.transformer_blocks.6.attn.norm_added_k.weight",
    "transformer.transformer_blocks.6.attn.norm_added_q.weight",
    "transformer.transformer_blocks.6.attn.norm_k.weight",
    "transformer.transformer_blocks.6.attn.norm_q.weight",
    "transformer.transformer_blocks.6.attn.to_add_out.weight",
    "transformer.transformer_blocks.6.attn.to_out.0.weight",
    "transformer.transformer_blocks.6.ff.net.0.proj.bias",
    "transformer.transformer_blocks.6.ff.net.0.proj.weight",
    "transformer.transformer_blocks.6.ff.net.2.weight",
    "transformer.transformer_blocks.6.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.6.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.6.ff_context.net.2.weight",
    "transformer.transformer_blocks.7.attn.norm_added_k.weight",
    "transformer.transformer_blocks.7.attn.norm_added_q.weight",
    "transformer.transformer_blocks.7.attn.norm_k.weight",
    "transformer.transformer_blocks.7.attn.norm_q.weight",
    "transformer.transformer_blocks.7.attn.to_add_out.weight",
    "transformer.transformer_blocks.7.attn.to_out.0.weight",
    "transformer.transformer_blocks.7.ff.net.0.proj.bias",
    "transformer.transformer_blocks.7.ff.net.0.proj.weight",
    "transformer.transformer_blocks.7.ff.net.2.weight",
    "transformer.transformer_blocks.7.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.7.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.7.ff_context.net.2.weight",
    "transformer.x_embedder.bias",
    "transformer.x_embedder.weight",
}

ALL_WEIGHTS = [
    "transformer.caption_projection.0.linear.weight",
    "transformer.caption_projection.1.linear.weight",
    "transformer.caption_projection.10.linear.weight",
    "transformer.caption_projection.11.linear.weight",
    "transformer.caption_projection.12.linear.weight",
    "transformer.caption_projection.13.linear.weight",
    "transformer.caption_projection.14.linear.weight",
    "transformer.caption_projection.15.linear.weight",
    "transformer.caption_projection.16.linear.weight",
    "transformer.caption_projection.17.linear.weight",
    "transformer.caption_projection.18.linear.weight",
    "transformer.caption_projection.19.linear.weight",
    "transformer.caption_projection.2.linear.weight",
    "transformer.caption_projection.20.linear.weight",
    "transformer.caption_projection.21.linear.weight",
    "transformer.caption_projection.22.linear.weight",
    "transformer.caption_projection.23.linear.weight",
    "transformer.caption_projection.24.linear.weight",
    "transformer.caption_projection.25.linear.weight",
    "transformer.caption_projection.26.linear.weight",
    "transformer.caption_projection.27.linear.weight",
    "transformer.caption_projection.28.linear.weight",
    "transformer.caption_projection.29.linear.weight",
    "transformer.caption_projection.3.linear.weight",
    "transformer.caption_projection.30.linear.weight",
    "transformer.caption_projection.31.linear.weight",
    "transformer.caption_projection.32.linear.weight",
    "transformer.caption_projection.33.linear.weight",
    "transformer.caption_projection.34.linear.weight",
    "transformer.caption_projection.35.linear.weight",
    "transformer.caption_projection.36.linear.weight",
    "transformer.caption_projection.37.linear.weight",
    "transformer.caption_projection.38.linear.weight",
    "transformer.caption_projection.39.linear.weight",
    "transformer.caption_projection.4.linear.weight",
    "transformer.caption_projection.40.linear.weight",
    "transformer.caption_projection.41.linear.weight",
    "transformer.caption_projection.42.linear.weight",
    "transformer.caption_projection.43.linear.weight",
    "transformer.caption_projection.44.linear.weight",
    "transformer.caption_projection.45.linear.weight",
    "transformer.caption_projection.5.linear.weight",
    "transformer.caption_projection.6.linear.weight",
    "transformer.caption_projection.7.linear.weight",
    "transformer.caption_projection.8.linear.weight",
    "transformer.caption_projection.9.linear.weight",
    "transformer.context_embedder.bias",
    "transformer.context_embedder.weight",
    "transformer.norm_out.linear.bias",
    "transformer.norm_out.linear.weight",
    "transformer.proj_out.bias",
    "transformer.proj_out.weight",
    "transformer.single_transformer_blocks.0.attn.norm_k.weight",
    "transformer.single_transformer_blocks.0.attn.norm_q.weight",
    "transformer.single_transformer_blocks.0.attn.to_k.bias",
    "transformer.single_transformer_blocks.0.attn.to_k.weight",
    "transformer.single_transformer_blocks.0.attn.to_q.bias",
    "transformer.single_transformer_blocks.0.attn.to_q.weight",
    "transformer.single_transformer_blocks.0.attn.to_v.bias",
    "transformer.single_transformer_blocks.0.attn.to_v.weight",
    "transformer.single_transformer_blocks.0.norm.linear.bias",
    "transformer.single_transformer_blocks.0.norm.linear.weight",
    "transformer.single_transformer_blocks.0.proj_mlp.bias",
    "transformer.single_transformer_blocks.0.proj_mlp.weight",
    "transformer.single_transformer_blocks.0.proj_out.bias",
    "transformer.single_transformer_blocks.0.proj_out.weight",
    "transformer.single_transformer_blocks.1.attn.norm_k.weight",
    "transformer.single_transformer_blocks.1.attn.norm_q.weight",
    "transformer.single_transformer_blocks.1.attn.to_k.bias",
    "transformer.single_transformer_blocks.1.attn.to_k.weight",
    "transformer.single_transformer_blocks.1.attn.to_q.bias",
    "transformer.single_transformer_blocks.1.attn.to_q.weight",
    "transformer.single_transformer_blocks.1.attn.to_v.bias",
    "transformer.single_transformer_blocks.1.attn.to_v.weight",
    "transformer.single_transformer_blocks.1.norm.linear.bias",
    "transformer.single_transformer_blocks.1.norm.linear.weight",
    "transformer.single_transformer_blocks.1.proj_mlp.bias",
    "transformer.single_transformer_blocks.1.proj_mlp.weight",
    "transformer.single_transformer_blocks.1.proj_out.bias",
    "transformer.single_transformer_blocks.1.proj_out.weight",
    "transformer.single_transformer_blocks.10.attn.norm_k.weight",
    "transformer.single_transformer_blocks.10.attn.norm_q.weight",
    "transformer.single_transformer_blocks.10.attn.to_k.bias",
    "transformer.single_transformer_blocks.10.attn.to_k.weight",
    "transformer.single_transformer_blocks.10.attn.to_q.bias",
    "transformer.single_transformer_blocks.10.attn.to_q.weight",
    "transformer.single_transformer_blocks.10.attn.to_v.bias",
    "transformer.single_transformer_blocks.10.attn.to_v.weight",
    "transformer.single_transformer_blocks.10.norm.linear.bias",
    "transformer.single_transformer_blocks.10.norm.linear.weight",
    "transformer.single_transformer_blocks.10.proj_mlp.bias",
    "transformer.single_transformer_blocks.10.proj_mlp.weight",
    "transformer.single_transformer_blocks.10.proj_out.bias",
    "transformer.single_transformer_blocks.10.proj_out.weight",
    "transformer.single_transformer_blocks.11.attn.norm_k.weight",
    "transformer.single_transformer_blocks.11.attn.norm_q.weight",
    "transformer.single_transformer_blocks.11.attn.to_k.bias",
    "transformer.single_transformer_blocks.11.attn.to_k.weight",
    "transformer.single_transformer_blocks.11.attn.to_q.bias",
    "transformer.single_transformer_blocks.11.attn.to_q.weight",
    "transformer.single_transformer_blocks.11.attn.to_v.bias",
    "transformer.single_transformer_blocks.11.attn.to_v.weight",
    "transformer.single_transformer_blocks.11.norm.linear.bias",
    "transformer.single_transformer_blocks.11.norm.linear.weight",
    "transformer.single_transformer_blocks.11.proj_mlp.bias",
    "transformer.single_transformer_blocks.11.proj_mlp.weight",
    "transformer.single_transformer_blocks.11.proj_out.bias",
    "transformer.single_transformer_blocks.11.proj_out.weight",
    "transformer.single_transformer_blocks.12.attn.norm_k.weight",
    "transformer.single_transformer_blocks.12.attn.norm_q.weight",
    "transformer.single_transformer_blocks.12.attn.to_k.bias",
    "transformer.single_transformer_blocks.12.attn.to_k.weight",
    "transformer.single_transformer_blocks.12.attn.to_q.bias",
    "transformer.single_transformer_blocks.12.attn.to_q.weight",
    "transformer.single_transformer_blocks.12.attn.to_v.bias",
    "transformer.single_transformer_blocks.12.attn.to_v.weight",
    "transformer.single_transformer_blocks.12.norm.linear.bias",
    "transformer.single_transformer_blocks.12.norm.linear.weight",
    "transformer.single_transformer_blocks.12.proj_mlp.bias",
    "transformer.single_transformer_blocks.12.proj_mlp.weight",
    "transformer.single_transformer_blocks.12.proj_out.bias",
    "transformer.single_transformer_blocks.12.proj_out.weight",
    "transformer.single_transformer_blocks.13.attn.norm_k.weight",
    "transformer.single_transformer_blocks.13.attn.norm_q.weight",
    "transformer.single_transformer_blocks.13.attn.to_k.bias",
    "transformer.single_transformer_blocks.13.attn.to_k.weight",
    "transformer.single_transformer_blocks.13.attn.to_q.bias",
    "transformer.single_transformer_blocks.13.attn.to_q.weight",
    "transformer.single_transformer_blocks.13.attn.to_v.bias",
    "transformer.single_transformer_blocks.13.attn.to_v.weight",
    "transformer.single_transformer_blocks.13.norm.linear.bias",
    "transformer.single_transformer_blocks.13.norm.linear.weight",
    "transformer.single_transformer_blocks.13.proj_mlp.bias",
    "transformer.single_transformer_blocks.13.proj_mlp.weight",
    "transformer.single_transformer_blocks.13.proj_out.bias",
    "transformer.single_transformer_blocks.13.proj_out.weight",
    "transformer.single_transformer_blocks.14.attn.norm_k.weight",
    "transformer.single_transformer_blocks.14.attn.norm_q.weight",
    "transformer.single_transformer_blocks.14.attn.to_k.bias",
    "transformer.single_transformer_blocks.14.attn.to_k.weight",
    "transformer.single_transformer_blocks.14.attn.to_q.bias",
    "transformer.single_transformer_blocks.14.attn.to_q.weight",
    "transformer.single_transformer_blocks.14.attn.to_v.bias",
    "transformer.single_transformer_blocks.14.attn.to_v.weight",
    "transformer.single_transformer_blocks.14.norm.linear.bias",
    "transformer.single_transformer_blocks.14.norm.linear.weight",
    "transformer.single_transformer_blocks.14.proj_mlp.bias",
    "transformer.single_transformer_blocks.14.proj_mlp.weight",
    "transformer.single_transformer_blocks.14.proj_out.bias",
    "transformer.single_transformer_blocks.14.proj_out.weight",
    "transformer.single_transformer_blocks.15.attn.norm_k.weight",
    "transformer.single_transformer_blocks.15.attn.norm_q.weight",
    "transformer.single_transformer_blocks.15.attn.to_k.bias",
    "transformer.single_transformer_blocks.15.attn.to_k.weight",
    "transformer.single_transformer_blocks.15.attn.to_q.bias",
    "transformer.single_transformer_blocks.15.attn.to_q.weight",
    "transformer.single_transformer_blocks.15.attn.to_v.bias",
    "transformer.single_transformer_blocks.15.attn.to_v.weight",
    "transformer.single_transformer_blocks.15.norm.linear.bias",
    "transformer.single_transformer_blocks.15.norm.linear.weight",
    "transformer.single_transformer_blocks.15.proj_mlp.bias",
    "transformer.single_transformer_blocks.15.proj_mlp.weight",
    "transformer.single_transformer_blocks.15.proj_out.bias",
    "transformer.single_transformer_blocks.15.proj_out.weight",
    "transformer.single_transformer_blocks.16.attn.norm_k.weight",
    "transformer.single_transformer_blocks.16.attn.norm_q.weight",
    "transformer.single_transformer_blocks.16.attn.to_k.bias",
    "transformer.single_transformer_blocks.16.attn.to_k.weight",
    "transformer.single_transformer_blocks.16.attn.to_q.bias",
    "transformer.single_transformer_blocks.16.attn.to_q.weight",
    "transformer.single_transformer_blocks.16.attn.to_v.bias",
    "transformer.single_transformer_blocks.16.attn.to_v.weight",
    "transformer.single_transformer_blocks.16.norm.linear.bias",
    "transformer.single_transformer_blocks.16.norm.linear.weight",
    "transformer.single_transformer_blocks.16.proj_mlp.bias",
    "transformer.single_transformer_blocks.16.proj_mlp.weight",
    "transformer.single_transformer_blocks.16.proj_out.bias",
    "transformer.single_transformer_blocks.16.proj_out.weight",
    "transformer.single_transformer_blocks.17.attn.norm_k.weight",
    "transformer.single_transformer_blocks.17.attn.norm_q.weight",
    "transformer.single_transformer_blocks.17.attn.to_k.bias",
    "transformer.single_transformer_blocks.17.attn.to_k.weight",
    "transformer.single_transformer_blocks.17.attn.to_q.bias",
    "transformer.single_transformer_blocks.17.attn.to_q.weight",
    "transformer.single_transformer_blocks.17.attn.to_v.bias",
    "transformer.single_transformer_blocks.17.attn.to_v.weight",
    "transformer.single_transformer_blocks.17.norm.linear.bias",
    "transformer.single_transformer_blocks.17.norm.linear.weight",
    "transformer.single_transformer_blocks.17.proj_mlp.bias",
    "transformer.single_transformer_blocks.17.proj_mlp.weight",
    "transformer.single_transformer_blocks.17.proj_out.bias",
    "transformer.single_transformer_blocks.17.proj_out.weight",
    "transformer.single_transformer_blocks.18.attn.norm_k.weight",
    "transformer.single_transformer_blocks.18.attn.norm_q.weight",
    "transformer.single_transformer_blocks.18.attn.to_k.bias",
    "transformer.single_transformer_blocks.18.attn.to_k.weight",
    "transformer.single_transformer_blocks.18.attn.to_q.bias",
    "transformer.single_transformer_blocks.18.attn.to_q.weight",
    "transformer.single_transformer_blocks.18.attn.to_v.bias",
    "transformer.single_transformer_blocks.18.attn.to_v.weight",
    "transformer.single_transformer_blocks.18.norm.linear.bias",
    "transformer.single_transformer_blocks.18.norm.linear.weight",
    "transformer.single_transformer_blocks.18.proj_mlp.bias",
    "transformer.single_transformer_blocks.18.proj_mlp.weight",
    "transformer.single_transformer_blocks.18.proj_out.bias",
    "transformer.single_transformer_blocks.18.proj_out.weight",
    "transformer.single_transformer_blocks.19.attn.norm_k.weight",
    "transformer.single_transformer_blocks.19.attn.norm_q.weight",
    "transformer.single_transformer_blocks.19.attn.to_k.bias",
    "transformer.single_transformer_blocks.19.attn.to_k.weight",
    "transformer.single_transformer_blocks.19.attn.to_q.bias",
    "transformer.single_transformer_blocks.19.attn.to_q.weight",
    "transformer.single_transformer_blocks.19.attn.to_v.bias",
    "transformer.single_transformer_blocks.19.attn.to_v.weight",
    "transformer.single_transformer_blocks.19.norm.linear.bias",
    "transformer.single_transformer_blocks.19.norm.linear.weight",
    "transformer.single_transformer_blocks.19.proj_mlp.bias",
    "transformer.single_transformer_blocks.19.proj_mlp.weight",
    "transformer.single_transformer_blocks.19.proj_out.bias",
    "transformer.single_transformer_blocks.19.proj_out.weight",
    "transformer.single_transformer_blocks.2.attn.norm_k.weight",
    "transformer.single_transformer_blocks.2.attn.norm_q.weight",
    "transformer.single_transformer_blocks.2.attn.to_k.bias",
    "transformer.single_transformer_blocks.2.attn.to_k.weight",
    "transformer.single_transformer_blocks.2.attn.to_q.bias",
    "transformer.single_transformer_blocks.2.attn.to_q.weight",
    "transformer.single_transformer_blocks.2.attn.to_v.bias",
    "transformer.single_transformer_blocks.2.attn.to_v.weight",
    "transformer.single_transformer_blocks.2.norm.linear.bias",
    "transformer.single_transformer_blocks.2.norm.linear.weight",
    "transformer.single_transformer_blocks.2.proj_mlp.bias",
    "transformer.single_transformer_blocks.2.proj_mlp.weight",
    "transformer.single_transformer_blocks.2.proj_out.bias",
    "transformer.single_transformer_blocks.2.proj_out.weight",
    "transformer.single_transformer_blocks.20.attn.norm_k.weight",
    "transformer.single_transformer_blocks.20.attn.norm_q.weight",
    "transformer.single_transformer_blocks.20.attn.to_k.bias",
    "transformer.single_transformer_blocks.20.attn.to_k.weight",
    "transformer.single_transformer_blocks.20.attn.to_q.bias",
    "transformer.single_transformer_blocks.20.attn.to_q.weight",
    "transformer.single_transformer_blocks.20.attn.to_v.bias",
    "transformer.single_transformer_blocks.20.attn.to_v.weight",
    "transformer.single_transformer_blocks.20.norm.linear.bias",
    "transformer.single_transformer_blocks.20.norm.linear.weight",
    "transformer.single_transformer_blocks.20.proj_mlp.bias",
    "transformer.single_transformer_blocks.20.proj_mlp.weight",
    "transformer.single_transformer_blocks.20.proj_out.bias",
    "transformer.single_transformer_blocks.20.proj_out.weight",
    "transformer.single_transformer_blocks.21.attn.norm_k.weight",
    "transformer.single_transformer_blocks.21.attn.norm_q.weight",
    "transformer.single_transformer_blocks.21.attn.to_k.bias",
    "transformer.single_transformer_blocks.21.attn.to_k.weight",
    "transformer.single_transformer_blocks.21.attn.to_q.bias",
    "transformer.single_transformer_blocks.21.attn.to_q.weight",
    "transformer.single_transformer_blocks.21.attn.to_v.bias",
    "transformer.single_transformer_blocks.21.attn.to_v.weight",
    "transformer.single_transformer_blocks.21.norm.linear.bias",
    "transformer.single_transformer_blocks.21.norm.linear.weight",
    "transformer.single_transformer_blocks.21.proj_mlp.bias",
    "transformer.single_transformer_blocks.21.proj_mlp.weight",
    "transformer.single_transformer_blocks.21.proj_out.bias",
    "transformer.single_transformer_blocks.21.proj_out.weight",
    "transformer.single_transformer_blocks.22.attn.norm_k.weight",
    "transformer.single_transformer_blocks.22.attn.norm_q.weight",
    "transformer.single_transformer_blocks.22.attn.to_k.bias",
    "transformer.single_transformer_blocks.22.attn.to_k.weight",
    "transformer.single_transformer_blocks.22.attn.to_q.bias",
    "transformer.single_transformer_blocks.22.attn.to_q.weight",
    "transformer.single_transformer_blocks.22.attn.to_v.bias",
    "transformer.single_transformer_blocks.22.attn.to_v.weight",
    "transformer.single_transformer_blocks.22.norm.linear.bias",
    "transformer.single_transformer_blocks.22.norm.linear.weight",
    "transformer.single_transformer_blocks.22.proj_mlp.bias",
    "transformer.single_transformer_blocks.22.proj_mlp.weight",
    "transformer.single_transformer_blocks.22.proj_out.bias",
    "transformer.single_transformer_blocks.22.proj_out.weight",
    "transformer.single_transformer_blocks.23.attn.norm_k.weight",
    "transformer.single_transformer_blocks.23.attn.norm_q.weight",
    "transformer.single_transformer_blocks.23.attn.to_k.bias",
    "transformer.single_transformer_blocks.23.attn.to_k.weight",
    "transformer.single_transformer_blocks.23.attn.to_q.bias",
    "transformer.single_transformer_blocks.23.attn.to_q.weight",
    "transformer.single_transformer_blocks.23.attn.to_v.bias",
    "transformer.single_transformer_blocks.23.attn.to_v.weight",
    "transformer.single_transformer_blocks.23.norm.linear.bias",
    "transformer.single_transformer_blocks.23.norm.linear.weight",
    "transformer.single_transformer_blocks.23.proj_mlp.bias",
    "transformer.single_transformer_blocks.23.proj_mlp.weight",
    "transformer.single_transformer_blocks.23.proj_out.bias",
    "transformer.single_transformer_blocks.23.proj_out.weight",
    "transformer.single_transformer_blocks.24.attn.norm_k.weight",
    "transformer.single_transformer_blocks.24.attn.norm_q.weight",
    "transformer.single_transformer_blocks.24.attn.to_k.bias",
    "transformer.single_transformer_blocks.24.attn.to_k.weight",
    "transformer.single_transformer_blocks.24.attn.to_q.bias",
    "transformer.single_transformer_blocks.24.attn.to_q.weight",
    "transformer.single_transformer_blocks.24.attn.to_v.bias",
    "transformer.single_transformer_blocks.24.attn.to_v.weight",
    "transformer.single_transformer_blocks.24.norm.linear.bias",
    "transformer.single_transformer_blocks.24.norm.linear.weight",
    "transformer.single_transformer_blocks.24.proj_mlp.bias",
    "transformer.single_transformer_blocks.24.proj_mlp.weight",
    "transformer.single_transformer_blocks.24.proj_out.bias",
    "transformer.single_transformer_blocks.24.proj_out.weight",
    "transformer.single_transformer_blocks.25.attn.norm_k.weight",
    "transformer.single_transformer_blocks.25.attn.norm_q.weight",
    "transformer.single_transformer_blocks.25.attn.to_k.bias",
    "transformer.single_transformer_blocks.25.attn.to_k.weight",
    "transformer.single_transformer_blocks.25.attn.to_q.bias",
    "transformer.single_transformer_blocks.25.attn.to_q.weight",
    "transformer.single_transformer_blocks.25.attn.to_v.bias",
    "transformer.single_transformer_blocks.25.attn.to_v.weight",
    "transformer.single_transformer_blocks.25.norm.linear.bias",
    "transformer.single_transformer_blocks.25.norm.linear.weight",
    "transformer.single_transformer_blocks.25.proj_mlp.bias",
    "transformer.single_transformer_blocks.25.proj_mlp.weight",
    "transformer.single_transformer_blocks.25.proj_out.bias",
    "transformer.single_transformer_blocks.25.proj_out.weight",
    "transformer.single_transformer_blocks.26.attn.norm_k.weight",
    "transformer.single_transformer_blocks.26.attn.norm_q.weight",
    "transformer.single_transformer_blocks.26.attn.to_k.bias",
    "transformer.single_transformer_blocks.26.attn.to_k.weight",
    "transformer.single_transformer_blocks.26.attn.to_q.bias",
    "transformer.single_transformer_blocks.26.attn.to_q.weight",
    "transformer.single_transformer_blocks.26.attn.to_v.bias",
    "transformer.single_transformer_blocks.26.attn.to_v.weight",
    "transformer.single_transformer_blocks.26.norm.linear.bias",
    "transformer.single_transformer_blocks.26.norm.linear.weight",
    "transformer.single_transformer_blocks.26.proj_mlp.bias",
    "transformer.single_transformer_blocks.26.proj_mlp.weight",
    "transformer.single_transformer_blocks.26.proj_out.bias",
    "transformer.single_transformer_blocks.26.proj_out.weight",
    "transformer.single_transformer_blocks.27.attn.norm_k.weight",
    "transformer.single_transformer_blocks.27.attn.norm_q.weight",
    "transformer.single_transformer_blocks.27.attn.to_k.bias",
    "transformer.single_transformer_blocks.27.attn.to_k.weight",
    "transformer.single_transformer_blocks.27.attn.to_q.bias",
    "transformer.single_transformer_blocks.27.attn.to_q.weight",
    "transformer.single_transformer_blocks.27.attn.to_v.bias",
    "transformer.single_transformer_blocks.27.attn.to_v.weight",
    "transformer.single_transformer_blocks.27.norm.linear.bias",
    "transformer.single_transformer_blocks.27.norm.linear.weight",
    "transformer.single_transformer_blocks.27.proj_mlp.bias",
    "transformer.single_transformer_blocks.27.proj_mlp.weight",
    "transformer.single_transformer_blocks.27.proj_out.bias",
    "transformer.single_transformer_blocks.27.proj_out.weight",
    "transformer.single_transformer_blocks.28.attn.norm_k.weight",
    "transformer.single_transformer_blocks.28.attn.norm_q.weight",
    "transformer.single_transformer_blocks.28.attn.to_k.bias",
    "transformer.single_transformer_blocks.28.attn.to_k.weight",
    "transformer.single_transformer_blocks.28.attn.to_q.bias",
    "transformer.single_transformer_blocks.28.attn.to_q.weight",
    "transformer.single_transformer_blocks.28.attn.to_v.bias",
    "transformer.single_transformer_blocks.28.attn.to_v.weight",
    "transformer.single_transformer_blocks.28.norm.linear.bias",
    "transformer.single_transformer_blocks.28.norm.linear.weight",
    "transformer.single_transformer_blocks.28.proj_mlp.bias",
    "transformer.single_transformer_blocks.28.proj_mlp.weight",
    "transformer.single_transformer_blocks.28.proj_out.bias",
    "transformer.single_transformer_blocks.28.proj_out.weight",
    "transformer.single_transformer_blocks.29.attn.norm_k.weight",
    "transformer.single_transformer_blocks.29.attn.norm_q.weight",
    "transformer.single_transformer_blocks.29.attn.to_k.bias",
    "transformer.single_transformer_blocks.29.attn.to_k.weight",
    "transformer.single_transformer_blocks.29.attn.to_q.bias",
    "transformer.single_transformer_blocks.29.attn.to_q.weight",
    "transformer.single_transformer_blocks.29.attn.to_v.bias",
    "transformer.single_transformer_blocks.29.attn.to_v.weight",
    "transformer.single_transformer_blocks.29.norm.linear.bias",
    "transformer.single_transformer_blocks.29.norm.linear.weight",
    "transformer.single_transformer_blocks.29.proj_mlp.bias",
    "transformer.single_transformer_blocks.29.proj_mlp.weight",
    "transformer.single_transformer_blocks.29.proj_out.bias",
    "transformer.single_transformer_blocks.29.proj_out.weight",
    "transformer.single_transformer_blocks.3.attn.norm_k.weight",
    "transformer.single_transformer_blocks.3.attn.norm_q.weight",
    "transformer.single_transformer_blocks.3.attn.to_k.bias",
    "transformer.single_transformer_blocks.3.attn.to_k.weight",
    "transformer.single_transformer_blocks.3.attn.to_q.bias",
    "transformer.single_transformer_blocks.3.attn.to_q.weight",
    "transformer.single_transformer_blocks.3.attn.to_v.bias",
    "transformer.single_transformer_blocks.3.attn.to_v.weight",
    "transformer.single_transformer_blocks.3.norm.linear.bias",
    "transformer.single_transformer_blocks.3.norm.linear.weight",
    "transformer.single_transformer_blocks.3.proj_mlp.bias",
    "transformer.single_transformer_blocks.3.proj_mlp.weight",
    "transformer.single_transformer_blocks.3.proj_out.bias",
    "transformer.single_transformer_blocks.3.proj_out.weight",
    "transformer.single_transformer_blocks.30.attn.norm_k.weight",
    "transformer.single_transformer_blocks.30.attn.norm_q.weight",
    "transformer.single_transformer_blocks.30.attn.to_k.bias",
    "transformer.single_transformer_blocks.30.attn.to_k.weight",
    "transformer.single_transformer_blocks.30.attn.to_q.bias",
    "transformer.single_transformer_blocks.30.attn.to_q.weight",
    "transformer.single_transformer_blocks.30.attn.to_v.bias",
    "transformer.single_transformer_blocks.30.attn.to_v.weight",
    "transformer.single_transformer_blocks.30.norm.linear.bias",
    "transformer.single_transformer_blocks.30.norm.linear.weight",
    "transformer.single_transformer_blocks.30.proj_mlp.bias",
    "transformer.single_transformer_blocks.30.proj_mlp.weight",
    "transformer.single_transformer_blocks.30.proj_out.bias",
    "transformer.single_transformer_blocks.30.proj_out.weight",
    "transformer.single_transformer_blocks.31.attn.norm_k.weight",
    "transformer.single_transformer_blocks.31.attn.norm_q.weight",
    "transformer.single_transformer_blocks.31.attn.to_k.bias",
    "transformer.single_transformer_blocks.31.attn.to_k.weight",
    "transformer.single_transformer_blocks.31.attn.to_q.bias",
    "transformer.single_transformer_blocks.31.attn.to_q.weight",
    "transformer.single_transformer_blocks.31.attn.to_v.bias",
    "transformer.single_transformer_blocks.31.attn.to_v.weight",
    "transformer.single_transformer_blocks.31.norm.linear.bias",
    "transformer.single_transformer_blocks.31.norm.linear.weight",
    "transformer.single_transformer_blocks.31.proj_mlp.bias",
    "transformer.single_transformer_blocks.31.proj_mlp.weight",
    "transformer.single_transformer_blocks.31.proj_out.bias",
    "transformer.single_transformer_blocks.31.proj_out.weight",
    "transformer.single_transformer_blocks.32.attn.norm_k.weight",
    "transformer.single_transformer_blocks.32.attn.norm_q.weight",
    "transformer.single_transformer_blocks.32.attn.to_k.bias",
    "transformer.single_transformer_blocks.32.attn.to_k.weight",
    "transformer.single_transformer_blocks.32.attn.to_q.bias",
    "transformer.single_transformer_blocks.32.attn.to_q.weight",
    "transformer.single_transformer_blocks.32.attn.to_v.bias",
    "transformer.single_transformer_blocks.32.attn.to_v.weight",
    "transformer.single_transformer_blocks.32.norm.linear.bias",
    "transformer.single_transformer_blocks.32.norm.linear.weight",
    "transformer.single_transformer_blocks.32.proj_mlp.bias",
    "transformer.single_transformer_blocks.32.proj_mlp.weight",
    "transformer.single_transformer_blocks.32.proj_out.bias",
    "transformer.single_transformer_blocks.32.proj_out.weight",
    "transformer.single_transformer_blocks.33.attn.norm_k.weight",
    "transformer.single_transformer_blocks.33.attn.norm_q.weight",
    "transformer.single_transformer_blocks.33.attn.to_k.bias",
    "transformer.single_transformer_blocks.33.attn.to_k.weight",
    "transformer.single_transformer_blocks.33.attn.to_q.bias",
    "transformer.single_transformer_blocks.33.attn.to_q.weight",
    "transformer.single_transformer_blocks.33.attn.to_v.bias",
    "transformer.single_transformer_blocks.33.attn.to_v.weight",
    "transformer.single_transformer_blocks.33.norm.linear.bias",
    "transformer.single_transformer_blocks.33.norm.linear.weight",
    "transformer.single_transformer_blocks.33.proj_mlp.bias",
    "transformer.single_transformer_blocks.33.proj_mlp.weight",
    "transformer.single_transformer_blocks.33.proj_out.bias",
    "transformer.single_transformer_blocks.33.proj_out.weight",
    "transformer.single_transformer_blocks.34.attn.norm_k.weight",
    "transformer.single_transformer_blocks.34.attn.norm_q.weight",
    "transformer.single_transformer_blocks.34.attn.to_k.bias",
    "transformer.single_transformer_blocks.34.attn.to_k.weight",
    "transformer.single_transformer_blocks.34.attn.to_q.bias",
    "transformer.single_transformer_blocks.34.attn.to_q.weight",
    "transformer.single_transformer_blocks.34.attn.to_v.bias",
    "transformer.single_transformer_blocks.34.attn.to_v.weight",
    "transformer.single_transformer_blocks.34.norm.linear.bias",
    "transformer.single_transformer_blocks.34.norm.linear.weight",
    "transformer.single_transformer_blocks.34.proj_mlp.bias",
    "transformer.single_transformer_blocks.34.proj_mlp.weight",
    "transformer.single_transformer_blocks.34.proj_out.bias",
    "transformer.single_transformer_blocks.34.proj_out.weight",
    "transformer.single_transformer_blocks.35.attn.norm_k.weight",
    "transformer.single_transformer_blocks.35.attn.norm_q.weight",
    "transformer.single_transformer_blocks.35.attn.to_k.bias",
    "transformer.single_transformer_blocks.35.attn.to_k.weight",
    "transformer.single_transformer_blocks.35.attn.to_q.bias",
    "transformer.single_transformer_blocks.35.attn.to_q.weight",
    "transformer.single_transformer_blocks.35.attn.to_v.bias",
    "transformer.single_transformer_blocks.35.attn.to_v.weight",
    "transformer.single_transformer_blocks.35.norm.linear.bias",
    "transformer.single_transformer_blocks.35.norm.linear.weight",
    "transformer.single_transformer_blocks.35.proj_mlp.bias",
    "transformer.single_transformer_blocks.35.proj_mlp.weight",
    "transformer.single_transformer_blocks.35.proj_out.bias",
    "transformer.single_transformer_blocks.35.proj_out.weight",
    "transformer.single_transformer_blocks.36.attn.norm_k.weight",
    "transformer.single_transformer_blocks.36.attn.norm_q.weight",
    "transformer.single_transformer_blocks.36.attn.to_k.bias",
    "transformer.single_transformer_blocks.36.attn.to_k.weight",
    "transformer.single_transformer_blocks.36.attn.to_q.bias",
    "transformer.single_transformer_blocks.36.attn.to_q.weight",
    "transformer.single_transformer_blocks.36.attn.to_v.bias",
    "transformer.single_transformer_blocks.36.attn.to_v.weight",
    "transformer.single_transformer_blocks.36.norm.linear.bias",
    "transformer.single_transformer_blocks.36.norm.linear.weight",
    "transformer.single_transformer_blocks.36.proj_mlp.bias",
    "transformer.single_transformer_blocks.36.proj_mlp.weight",
    "transformer.single_transformer_blocks.36.proj_out.bias",
    "transformer.single_transformer_blocks.36.proj_out.weight",
    "transformer.single_transformer_blocks.37.attn.norm_k.weight",
    "transformer.single_transformer_blocks.37.attn.norm_q.weight",
    "transformer.single_transformer_blocks.37.attn.to_k.bias",
    "transformer.single_transformer_blocks.37.attn.to_k.weight",
    "transformer.single_transformer_blocks.37.attn.to_q.bias",
    "transformer.single_transformer_blocks.37.attn.to_q.weight",
    "transformer.single_transformer_blocks.37.attn.to_v.bias",
    "transformer.single_transformer_blocks.37.attn.to_v.weight",
    "transformer.single_transformer_blocks.37.norm.linear.bias",
    "transformer.single_transformer_blocks.37.norm.linear.weight",
    "transformer.single_transformer_blocks.37.proj_mlp.bias",
    "transformer.single_transformer_blocks.37.proj_mlp.weight",
    "transformer.single_transformer_blocks.37.proj_out.bias",
    "transformer.single_transformer_blocks.37.proj_out.weight",
    "transformer.single_transformer_blocks.4.attn.norm_k.weight",
    "transformer.single_transformer_blocks.4.attn.norm_q.weight",
    "transformer.single_transformer_blocks.4.attn.to_k.bias",
    "transformer.single_transformer_blocks.4.attn.to_k.weight",
    "transformer.single_transformer_blocks.4.attn.to_q.bias",
    "transformer.single_transformer_blocks.4.attn.to_q.weight",
    "transformer.single_transformer_blocks.4.attn.to_v.bias",
    "transformer.single_transformer_blocks.4.attn.to_v.weight",
    "transformer.single_transformer_blocks.4.norm.linear.bias",
    "transformer.single_transformer_blocks.4.norm.linear.weight",
    "transformer.single_transformer_blocks.4.proj_mlp.bias",
    "transformer.single_transformer_blocks.4.proj_mlp.weight",
    "transformer.single_transformer_blocks.4.proj_out.bias",
    "transformer.single_transformer_blocks.4.proj_out.weight",
    "transformer.single_transformer_blocks.5.attn.norm_k.weight",
    "transformer.single_transformer_blocks.5.attn.norm_q.weight",
    "transformer.single_transformer_blocks.5.attn.to_k.bias",
    "transformer.single_transformer_blocks.5.attn.to_k.weight",
    "transformer.single_transformer_blocks.5.attn.to_q.bias",
    "transformer.single_transformer_blocks.5.attn.to_q.weight",
    "transformer.single_transformer_blocks.5.attn.to_v.bias",
    "transformer.single_transformer_blocks.5.attn.to_v.weight",
    "transformer.single_transformer_blocks.5.norm.linear.bias",
    "transformer.single_transformer_blocks.5.norm.linear.weight",
    "transformer.single_transformer_blocks.5.proj_mlp.bias",
    "transformer.single_transformer_blocks.5.proj_mlp.weight",
    "transformer.single_transformer_blocks.5.proj_out.bias",
    "transformer.single_transformer_blocks.5.proj_out.weight",
    "transformer.single_transformer_blocks.6.attn.norm_k.weight",
    "transformer.single_transformer_blocks.6.attn.norm_q.weight",
    "transformer.single_transformer_blocks.6.attn.to_k.bias",
    "transformer.single_transformer_blocks.6.attn.to_k.weight",
    "transformer.single_transformer_blocks.6.attn.to_q.bias",
    "transformer.single_transformer_blocks.6.attn.to_q.weight",
    "transformer.single_transformer_blocks.6.attn.to_v.bias",
    "transformer.single_transformer_blocks.6.attn.to_v.weight",
    "transformer.single_transformer_blocks.6.norm.linear.bias",
    "transformer.single_transformer_blocks.6.norm.linear.weight",
    "transformer.single_transformer_blocks.6.proj_mlp.bias",
    "transformer.single_transformer_blocks.6.proj_mlp.weight",
    "transformer.single_transformer_blocks.6.proj_out.bias",
    "transformer.single_transformer_blocks.6.proj_out.weight",
    "transformer.single_transformer_blocks.7.attn.norm_k.weight",
    "transformer.single_transformer_blocks.7.attn.norm_q.weight",
    "transformer.single_transformer_blocks.7.attn.to_k.bias",
    "transformer.single_transformer_blocks.7.attn.to_k.weight",
    "transformer.single_transformer_blocks.7.attn.to_q.bias",
    "transformer.single_transformer_blocks.7.attn.to_q.weight",
    "transformer.single_transformer_blocks.7.attn.to_v.bias",
    "transformer.single_transformer_blocks.7.attn.to_v.weight",
    "transformer.single_transformer_blocks.7.norm.linear.bias",
    "transformer.single_transformer_blocks.7.norm.linear.weight",
    "transformer.single_transformer_blocks.7.proj_mlp.bias",
    "transformer.single_transformer_blocks.7.proj_mlp.weight",
    "transformer.single_transformer_blocks.7.proj_out.bias",
    "transformer.single_transformer_blocks.7.proj_out.weight",
    "transformer.single_transformer_blocks.8.attn.norm_k.weight",
    "transformer.single_transformer_blocks.8.attn.norm_q.weight",
    "transformer.single_transformer_blocks.8.attn.to_k.bias",
    "transformer.single_transformer_blocks.8.attn.to_k.weight",
    "transformer.single_transformer_blocks.8.attn.to_q.bias",
    "transformer.single_transformer_blocks.8.attn.to_q.weight",
    "transformer.single_transformer_blocks.8.attn.to_v.bias",
    "transformer.single_transformer_blocks.8.attn.to_v.weight",
    "transformer.single_transformer_blocks.8.norm.linear.bias",
    "transformer.single_transformer_blocks.8.norm.linear.weight",
    "transformer.single_transformer_blocks.8.proj_mlp.bias",
    "transformer.single_transformer_blocks.8.proj_mlp.weight",
    "transformer.single_transformer_blocks.8.proj_out.bias",
    "transformer.single_transformer_blocks.8.proj_out.weight",
    "transformer.single_transformer_blocks.9.attn.norm_k.weight",
    "transformer.single_transformer_blocks.9.attn.norm_q.weight",
    "transformer.single_transformer_blocks.9.attn.to_k.bias",
    "transformer.single_transformer_blocks.9.attn.to_k.weight",
    "transformer.single_transformer_blocks.9.attn.to_q.bias",
    "transformer.single_transformer_blocks.9.attn.to_q.weight",
    "transformer.single_transformer_blocks.9.attn.to_v.bias",
    "transformer.single_transformer_blocks.9.attn.to_v.weight",
    "transformer.single_transformer_blocks.9.norm.linear.bias",
    "transformer.single_transformer_blocks.9.norm.linear.weight",
    "transformer.single_transformer_blocks.9.proj_mlp.bias",
    "transformer.single_transformer_blocks.9.proj_mlp.weight",
    "transformer.single_transformer_blocks.9.proj_out.bias",
    "transformer.single_transformer_blocks.9.proj_out.weight",
    "transformer.time_embed.timestep_embedder.linear_1.bias",
    "transformer.time_embed.timestep_embedder.linear_1.weight",
    "transformer.time_embed.timestep_embedder.linear_2.bias",
    "transformer.time_embed.timestep_embedder.linear_2.weight",
    "transformer.transformer_blocks.0.attn.add_k_proj.bias",
    "transformer.transformer_blocks.0.attn.add_k_proj.weight",
    "transformer.transformer_blocks.0.attn.add_q_proj.bias",
    "transformer.transformer_blocks.0.attn.add_q_proj.weight",
    "transformer.transformer_blocks.0.attn.add_v_proj.bias",
    "transformer.transformer_blocks.0.attn.add_v_proj.weight",
    "transformer.transformer_blocks.0.attn.norm_added_k.weight",
    "transformer.transformer_blocks.0.attn.norm_added_q.weight",
    "transformer.transformer_blocks.0.attn.norm_k.weight",
    "transformer.transformer_blocks.0.attn.norm_q.weight",
    "transformer.transformer_blocks.0.attn.to_add_out.bias",
    "transformer.transformer_blocks.0.attn.to_add_out.weight",
    "transformer.transformer_blocks.0.attn.to_k.bias",
    "transformer.transformer_blocks.0.attn.to_k.weight",
    "transformer.transformer_blocks.0.attn.to_out.0.bias",
    "transformer.transformer_blocks.0.attn.to_out.0.weight",
    "transformer.transformer_blocks.0.attn.to_q.bias",
    "transformer.transformer_blocks.0.attn.to_q.weight",
    "transformer.transformer_blocks.0.attn.to_v.bias",
    "transformer.transformer_blocks.0.attn.to_v.weight",
    "transformer.transformer_blocks.0.ff.net.0.proj.bias",
    "transformer.transformer_blocks.0.ff.net.0.proj.weight",
    "transformer.transformer_blocks.0.ff.net.2.bias",
    "transformer.transformer_blocks.0.ff.net.2.weight",
    "transformer.transformer_blocks.0.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.0.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.0.ff_context.net.2.bias",
    "transformer.transformer_blocks.0.ff_context.net.2.weight",
    "transformer.transformer_blocks.0.norm1.linear.bias",
    "transformer.transformer_blocks.0.norm1.linear.weight",
    "transformer.transformer_blocks.0.norm1_context.linear.bias",
    "transformer.transformer_blocks.0.norm1_context.linear.weight",
    "transformer.transformer_blocks.1.attn.add_k_proj.bias",
    "transformer.transformer_blocks.1.attn.add_k_proj.weight",
    "transformer.transformer_blocks.1.attn.add_q_proj.bias",
    "transformer.transformer_blocks.1.attn.add_q_proj.weight",
    "transformer.transformer_blocks.1.attn.add_v_proj.bias",
    "transformer.transformer_blocks.1.attn.add_v_proj.weight",
    "transformer.transformer_blocks.1.attn.norm_added_k.weight",
    "transformer.transformer_blocks.1.attn.norm_added_q.weight",
    "transformer.transformer_blocks.1.attn.norm_k.weight",
    "transformer.transformer_blocks.1.attn.norm_q.weight",
    "transformer.transformer_blocks.1.attn.to_add_out.bias",
    "transformer.transformer_blocks.1.attn.to_add_out.weight",
    "transformer.transformer_blocks.1.attn.to_k.bias",
    "transformer.transformer_blocks.1.attn.to_k.weight",
    "transformer.transformer_blocks.1.attn.to_out.0.bias",
    "transformer.transformer_blocks.1.attn.to_out.0.weight",
    "transformer.transformer_blocks.1.attn.to_q.bias",
    "transformer.transformer_blocks.1.attn.to_q.weight",
    "transformer.transformer_blocks.1.attn.to_v.bias",
    "transformer.transformer_blocks.1.attn.to_v.weight",
    "transformer.transformer_blocks.1.ff.net.0.proj.bias",
    "transformer.transformer_blocks.1.ff.net.0.proj.weight",
    "transformer.transformer_blocks.1.ff.net.2.bias",
    "transformer.transformer_blocks.1.ff.net.2.weight",
    "transformer.transformer_blocks.1.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.1.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.1.ff_context.net.2.bias",
    "transformer.transformer_blocks.1.ff_context.net.2.weight",
    "transformer.transformer_blocks.1.norm1.linear.bias",
    "transformer.transformer_blocks.1.norm1.linear.weight",
    "transformer.transformer_blocks.1.norm1_context.linear.bias",
    "transformer.transformer_blocks.1.norm1_context.linear.weight",
    "transformer.transformer_blocks.2.attn.add_k_proj.bias",
    "transformer.transformer_blocks.2.attn.add_k_proj.weight",
    "transformer.transformer_blocks.2.attn.add_q_proj.bias",
    "transformer.transformer_blocks.2.attn.add_q_proj.weight",
    "transformer.transformer_blocks.2.attn.add_v_proj.bias",
    "transformer.transformer_blocks.2.attn.add_v_proj.weight",
    "transformer.transformer_blocks.2.attn.norm_added_k.weight",
    "transformer.transformer_blocks.2.attn.norm_added_q.weight",
    "transformer.transformer_blocks.2.attn.norm_k.weight",
    "transformer.transformer_blocks.2.attn.norm_q.weight",
    "transformer.transformer_blocks.2.attn.to_add_out.bias",
    "transformer.transformer_blocks.2.attn.to_add_out.weight",
    "transformer.transformer_blocks.2.attn.to_k.bias",
    "transformer.transformer_blocks.2.attn.to_k.weight",
    "transformer.transformer_blocks.2.attn.to_out.0.bias",
    "transformer.transformer_blocks.2.attn.to_out.0.weight",
    "transformer.transformer_blocks.2.attn.to_q.bias",
    "transformer.transformer_blocks.2.attn.to_q.weight",
    "transformer.transformer_blocks.2.attn.to_v.bias",
    "transformer.transformer_blocks.2.attn.to_v.weight",
    "transformer.transformer_blocks.2.ff.net.0.proj.bias",
    "transformer.transformer_blocks.2.ff.net.0.proj.weight",
    "transformer.transformer_blocks.2.ff.net.2.bias",
    "transformer.transformer_blocks.2.ff.net.2.weight",
    "transformer.transformer_blocks.2.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.2.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.2.ff_context.net.2.bias",
    "transformer.transformer_blocks.2.ff_context.net.2.weight",
    "transformer.transformer_blocks.2.norm1.linear.bias",
    "transformer.transformer_blocks.2.norm1.linear.weight",
    "transformer.transformer_blocks.2.norm1_context.linear.bias",
    "transformer.transformer_blocks.2.norm1_context.linear.weight",
    "transformer.transformer_blocks.3.attn.add_k_proj.bias",
    "transformer.transformer_blocks.3.attn.add_k_proj.weight",
    "transformer.transformer_blocks.3.attn.add_q_proj.bias",
    "transformer.transformer_blocks.3.attn.add_q_proj.weight",
    "transformer.transformer_blocks.3.attn.add_v_proj.bias",
    "transformer.transformer_blocks.3.attn.add_v_proj.weight",
    "transformer.transformer_blocks.3.attn.norm_added_k.weight",
    "transformer.transformer_blocks.3.attn.norm_added_q.weight",
    "transformer.transformer_blocks.3.attn.norm_k.weight",
    "transformer.transformer_blocks.3.attn.norm_q.weight",
    "transformer.transformer_blocks.3.attn.to_add_out.bias",
    "transformer.transformer_blocks.3.attn.to_add_out.weight",
    "transformer.transformer_blocks.3.attn.to_k.bias",
    "transformer.transformer_blocks.3.attn.to_k.weight",
    "transformer.transformer_blocks.3.attn.to_out.0.bias",
    "transformer.transformer_blocks.3.attn.to_out.0.weight",
    "transformer.transformer_blocks.3.attn.to_q.bias",
    "transformer.transformer_blocks.3.attn.to_q.weight",
    "transformer.transformer_blocks.3.attn.to_v.bias",
    "transformer.transformer_blocks.3.attn.to_v.weight",
    "transformer.transformer_blocks.3.ff.net.0.proj.bias",
    "transformer.transformer_blocks.3.ff.net.0.proj.weight",
    "transformer.transformer_blocks.3.ff.net.2.bias",
    "transformer.transformer_blocks.3.ff.net.2.weight",
    "transformer.transformer_blocks.3.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.3.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.3.ff_context.net.2.bias",
    "transformer.transformer_blocks.3.ff_context.net.2.weight",
    "transformer.transformer_blocks.3.norm1.linear.bias",
    "transformer.transformer_blocks.3.norm1.linear.weight",
    "transformer.transformer_blocks.3.norm1_context.linear.bias",
    "transformer.transformer_blocks.3.norm1_context.linear.weight",
    "transformer.transformer_blocks.4.attn.add_k_proj.bias",
    "transformer.transformer_blocks.4.attn.add_k_proj.weight",
    "transformer.transformer_blocks.4.attn.add_q_proj.bias",
    "transformer.transformer_blocks.4.attn.add_q_proj.weight",
    "transformer.transformer_blocks.4.attn.add_v_proj.bias",
    "transformer.transformer_blocks.4.attn.add_v_proj.weight",
    "transformer.transformer_blocks.4.attn.norm_added_k.weight",
    "transformer.transformer_blocks.4.attn.norm_added_q.weight",
    "transformer.transformer_blocks.4.attn.norm_k.weight",
    "transformer.transformer_blocks.4.attn.norm_q.weight",
    "transformer.transformer_blocks.4.attn.to_add_out.bias",
    "transformer.transformer_blocks.4.attn.to_add_out.weight",
    "transformer.transformer_blocks.4.attn.to_k.bias",
    "transformer.transformer_blocks.4.attn.to_k.weight",
    "transformer.transformer_blocks.4.attn.to_out.0.bias",
    "transformer.transformer_blocks.4.attn.to_out.0.weight",
    "transformer.transformer_blocks.4.attn.to_q.bias",
    "transformer.transformer_blocks.4.attn.to_q.weight",
    "transformer.transformer_blocks.4.attn.to_v.bias",
    "transformer.transformer_blocks.4.attn.to_v.weight",
    "transformer.transformer_blocks.4.ff.net.0.proj.bias",
    "transformer.transformer_blocks.4.ff.net.0.proj.weight",
    "transformer.transformer_blocks.4.ff.net.2.bias",
    "transformer.transformer_blocks.4.ff.net.2.weight",
    "transformer.transformer_blocks.4.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.4.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.4.ff_context.net.2.bias",
    "transformer.transformer_blocks.4.ff_context.net.2.weight",
    "transformer.transformer_blocks.4.norm1.linear.bias",
    "transformer.transformer_blocks.4.norm1.linear.weight",
    "transformer.transformer_blocks.4.norm1_context.linear.bias",
    "transformer.transformer_blocks.4.norm1_context.linear.weight",
    "transformer.transformer_blocks.5.attn.add_k_proj.bias",
    "transformer.transformer_blocks.5.attn.add_k_proj.weight",
    "transformer.transformer_blocks.5.attn.add_q_proj.bias",
    "transformer.transformer_blocks.5.attn.add_q_proj.weight",
    "transformer.transformer_blocks.5.attn.add_v_proj.bias",
    "transformer.transformer_blocks.5.attn.add_v_proj.weight",
    "transformer.transformer_blocks.5.attn.norm_added_k.weight",
    "transformer.transformer_blocks.5.attn.norm_added_q.weight",
    "transformer.transformer_blocks.5.attn.norm_k.weight",
    "transformer.transformer_blocks.5.attn.norm_q.weight",
    "transformer.transformer_blocks.5.attn.to_add_out.bias",
    "transformer.transformer_blocks.5.attn.to_add_out.weight",
    "transformer.transformer_blocks.5.attn.to_k.bias",
    "transformer.transformer_blocks.5.attn.to_k.weight",
    "transformer.transformer_blocks.5.attn.to_out.0.bias",
    "transformer.transformer_blocks.5.attn.to_out.0.weight",
    "transformer.transformer_blocks.5.attn.to_q.bias",
    "transformer.transformer_blocks.5.attn.to_q.weight",
    "transformer.transformer_blocks.5.attn.to_v.bias",
    "transformer.transformer_blocks.5.attn.to_v.weight",
    "transformer.transformer_blocks.5.ff.net.0.proj.bias",
    "transformer.transformer_blocks.5.ff.net.0.proj.weight",
    "transformer.transformer_blocks.5.ff.net.2.bias",
    "transformer.transformer_blocks.5.ff.net.2.weight",
    "transformer.transformer_blocks.5.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.5.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.5.ff_context.net.2.bias",
    "transformer.transformer_blocks.5.ff_context.net.2.weight",
    "transformer.transformer_blocks.5.norm1.linear.bias",
    "transformer.transformer_blocks.5.norm1.linear.weight",
    "transformer.transformer_blocks.5.norm1_context.linear.bias",
    "transformer.transformer_blocks.5.norm1_context.linear.weight",
    "transformer.transformer_blocks.6.attn.add_k_proj.bias",
    "transformer.transformer_blocks.6.attn.add_k_proj.weight",
    "transformer.transformer_blocks.6.attn.add_q_proj.bias",
    "transformer.transformer_blocks.6.attn.add_q_proj.weight",
    "transformer.transformer_blocks.6.attn.add_v_proj.bias",
    "transformer.transformer_blocks.6.attn.add_v_proj.weight",
    "transformer.transformer_blocks.6.attn.norm_added_k.weight",
    "transformer.transformer_blocks.6.attn.norm_added_q.weight",
    "transformer.transformer_blocks.6.attn.norm_k.weight",
    "transformer.transformer_blocks.6.attn.norm_q.weight",
    "transformer.transformer_blocks.6.attn.to_add_out.bias",
    "transformer.transformer_blocks.6.attn.to_add_out.weight",
    "transformer.transformer_blocks.6.attn.to_k.bias",
    "transformer.transformer_blocks.6.attn.to_k.weight",
    "transformer.transformer_blocks.6.attn.to_out.0.bias",
    "transformer.transformer_blocks.6.attn.to_out.0.weight",
    "transformer.transformer_blocks.6.attn.to_q.bias",
    "transformer.transformer_blocks.6.attn.to_q.weight",
    "transformer.transformer_blocks.6.attn.to_v.bias",
    "transformer.transformer_blocks.6.attn.to_v.weight",
    "transformer.transformer_blocks.6.ff.net.0.proj.bias",
    "transformer.transformer_blocks.6.ff.net.0.proj.weight",
    "transformer.transformer_blocks.6.ff.net.2.bias",
    "transformer.transformer_blocks.6.ff.net.2.weight",
    "transformer.transformer_blocks.6.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.6.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.6.ff_context.net.2.bias",
    "transformer.transformer_blocks.6.ff_context.net.2.weight",
    "transformer.transformer_blocks.6.norm1.linear.bias",
    "transformer.transformer_blocks.6.norm1.linear.weight",
    "transformer.transformer_blocks.6.norm1_context.linear.bias",
    "transformer.transformer_blocks.6.norm1_context.linear.weight",
    "transformer.transformer_blocks.7.attn.add_k_proj.bias",
    "transformer.transformer_blocks.7.attn.add_k_proj.weight",
    "transformer.transformer_blocks.7.attn.add_q_proj.bias",
    "transformer.transformer_blocks.7.attn.add_q_proj.weight",
    "transformer.transformer_blocks.7.attn.add_v_proj.bias",
    "transformer.transformer_blocks.7.attn.add_v_proj.weight",
    "transformer.transformer_blocks.7.attn.norm_added_k.weight",
    "transformer.transformer_blocks.7.attn.norm_added_q.weight",
    "transformer.transformer_blocks.7.attn.norm_k.weight",
    "transformer.transformer_blocks.7.attn.norm_q.weight",
    "transformer.transformer_blocks.7.attn.to_add_out.bias",
    "transformer.transformer_blocks.7.attn.to_add_out.weight",
    "transformer.transformer_blocks.7.attn.to_k.bias",
    "transformer.transformer_blocks.7.attn.to_k.weight",
    "transformer.transformer_blocks.7.attn.to_out.0.bias",
    "transformer.transformer_blocks.7.attn.to_out.0.weight",
    "transformer.transformer_blocks.7.attn.to_q.bias",
    "transformer.transformer_blocks.7.attn.to_q.weight",
    "transformer.transformer_blocks.7.attn.to_v.bias",
    "transformer.transformer_blocks.7.attn.to_v.weight",
    "transformer.transformer_blocks.7.ff.net.0.proj.bias",
    "transformer.transformer_blocks.7.ff.net.0.proj.weight",
    "transformer.transformer_blocks.7.ff.net.2.bias",
    "transformer.transformer_blocks.7.ff.net.2.weight",
    "transformer.transformer_blocks.7.ff_context.net.0.proj.bias",
    "transformer.transformer_blocks.7.ff_context.net.0.proj.weight",
    "transformer.transformer_blocks.7.ff_context.net.2.bias",
    "transformer.transformer_blocks.7.ff_context.net.2.weight",
    "transformer.transformer_blocks.7.norm1.linear.bias",
    "transformer.transformer_blocks.7.norm1.linear.weight",
    "transformer.transformer_blocks.7.norm1_context.linear.bias",
    "transformer.transformer_blocks.7.norm1_context.linear.weight",
    "transformer.x_embedder.bias",
    "transformer.x_embedder.weight",
]
# fmt: on


def load_weights_for__main_from_state_dict():
    device = utils.DeviceGetter.get_device(
        (1, 4), fabric_config=ttnn.FabricConfig.FABRIC_1D_RING
    )

    model = model_pt.load_pytorch_model()
    sd = dict(model.state_dict())
    for name, buf in model.named_buffers():
        if name not in sd:
            sd[name] = buf

    weights = {}
    for key in ALL_WEIGHTS:
        pt_tensor = sd[key]
        ttnn_tensor = ttnn.from_torch(pt_tensor)

        if key in TILE_ON_DEVICE_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)

        weights[key] = ttnn_tensor

    return weights


