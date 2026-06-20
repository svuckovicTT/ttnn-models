import ttnn
import ttir_cpu
import torch


_BN_EPS = 1.0013580322265625e-05


def fold_bn_into_conv(running_var, bn_weight, conv_weight, running_mean, bn_bias, out_channels):
    running_var = ttnn.to_torch(running_var)
    bn_weight = ttnn.to_torch(bn_weight)
    conv_weight = ttnn.to_torch(conv_weight)
    running_mean = ttnn.to_torch(running_mean)
    bn_bias = ttnn.to_torch(bn_bias)

    eps = ttir_cpu.full(shape=[1], fill_value=_BN_EPS, dtype=torch.float32)
    scale = ttir_cpu.div(bn_weight, ttir_cpu.sqrt(ttir_cpu.add(running_var, eps)))

    folded_weight = ttir_cpu.multiply(conv_weight, ttir_cpu.reshape(scale, [out_channels, 1, 1, 1]))

    scale_1d = ttir_cpu.permute(ttir_cpu.reshape(scale, [1, out_channels, 1, 1]), [0, 2, 3, 1])
    mean_1d = ttir_cpu.permute(ttir_cpu.reshape(running_mean, [1, out_channels, 1, 1]), [0, 2, 3, 1])
    bias_1d = ttir_cpu.permute(ttir_cpu.reshape(bn_bias, [1, out_channels, 1, 1]), [0, 2, 3, 1])
    folded_bias = ttir_cpu.subtract(bias_1d, ttir_cpu.multiply(mean_1d, scale_1d))

    return ttnn.from_torch(folded_weight), ttnn.from_torch(folded_bias)


def prepare_conv_weight_and_bias(
    folded_weight,
    folded_bias,
    device,
    in_channels,
    out_channels,
    batch_size,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    input_layout,
    input_memory_config,
    conv_config,
    slice_config,
):
    w = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(folded_weight, False)
    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=w,
        input_memory_config=input_memory_config,
        input_layout=input_layout,
        weights_format="OIHW",
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        has_bias=True,
        groups=groups,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=conv_config,
        compute_config=None,
        slice_config=slice_config,
    )
    ttnn.deallocate(w, False)

    b = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(folded_bias, False)
    prepared_bias = ttnn.prepare_conv_bias(
        bias_tensor=b,
        input_memory_config=input_memory_config,
        input_layout=input_layout,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=conv_config,
        compute_config=None,
        slice_config=slice_config,
    )
    ttnn.deallocate(b, False)

    return prepared_weight, prepared_bias


def _height_sharded_mem_cfg(core_ranges, shard_shape):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(core_ranges),
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _block_sharded_mem_cfg(core_ranges, shard_shape):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet(core_ranges),
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


_INTERLEAVED_L1 = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
)

_DRAM_INTERLEAVED = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)

_CONV2D_SLICE_CFG = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0)

_CORE_RANGE_102 = [
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
    ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(2, 9)),
]
_CORE_RANGE_98 = [
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
    ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
]
_CORE_RANGE_48 = [
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
]
_CORE_RANGE_64 = [
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7)),
]
_CORE_RANGE_80 = [
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9)),
]
_CORE_RANGE_56 = [
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
]
_CORE_RANGE_110 = [
    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9)),
]


def _make_conv_config(activation=None, deallocate_activation=True, act_block_h_override=0, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED):
    kwargs = dict(
        weights_dtype=ttnn.DataType.BFLOAT16,
        deallocate_activation=deallocate_activation,
        config_tensors_in_dram=True,
        act_block_h_override=act_block_h_override,
        shard_layout=shard_layout,
        enable_kernel_stride_folding=False,
    )
    if activation is not None:
        kwargs["activation"] = activation
    return ttnn.Conv2dConfig(**kwargs)


_RELU = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)


# Each entry: (layer_prefix, out_channels, in_channels, input_height, input_width,
#               kernel_size, stride, padding, input_layout, input_memory_config,
#               activation, deallocate_activation, act_block_h_override, shard_layout)
CONV_CONSTEVAL_PARAMS = [
    # Conv 0: resnet.embedder.embedder (7x7 initial conv)
    ("resnet.embedder.embedder", 64, 3, 224, 224,
     [7, 7], [2, 2], [3, 3, 3, 3], ttnn.Layout.TILE,
     _INTERLEAVED_L1,
     _RELU, True, 64, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 1: resnet.encoder.stages.0.layers.0.layer.0 (1x1 bottleneck reduce)
    ("resnet.encoder.stages.0.layers.0.layer.0", 64, 64, 56, 56,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.ROW_MAJOR,
     _height_sharded_mem_cfg(_CORE_RANGE_102, [246, 64]),
     _RELU, False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 2: resnet.encoder.stages.0.layers.0.layer.1 (3x3 bottleneck)
    ("resnet.encoder.stages.0.layers.0.layer.1", 64, 64, 56, 56,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 64]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 3: resnet.encoder.stages.0.layers.0.layer.2 (1x1 bottleneck expand, no activation)
    ("resnet.encoder.stages.0.layers.0.layer.2", 256, 64, 56, 56,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 64]),
     None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 4: resnet.encoder.stages.0.layers.0.shortcut (1x1 skip, no activation)
    ("resnet.encoder.stages.0.layers.0.shortcut", 256, 64, 56, 56,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.ROW_MAJOR,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 64]),
     None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 5: resnet.encoder.stages.0.layers.1.layer.0
    ("resnet.encoder.stages.0.layers.1.layer.0", 64, 256, 56, 56,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 256]),
     _RELU, False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 6: resnet.encoder.stages.0.layers.1.layer.1
    ("resnet.encoder.stages.0.layers.1.layer.1", 64, 64, 56, 56,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 64]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 7: resnet.encoder.stages.0.layers.1.layer.2
    ("resnet.encoder.stages.0.layers.1.layer.2", 256, 64, 56, 56,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 64]),
     None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 8: resnet.encoder.stages.0.layers.2.layer.0
    ("resnet.encoder.stages.0.layers.2.layer.0", 64, 256, 56, 56,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 256]),
     _RELU, False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 9: resnet.encoder.stages.0.layers.2.layer.1
    ("resnet.encoder.stages.0.layers.2.layer.1", 64, 64, 56, 56,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 64]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 10: resnet.encoder.stages.0.layers.2.layer.2
    ("resnet.encoder.stages.0.layers.2.layer.2", 256, 64, 56, 56,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 64]),
     None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 11: resnet.encoder.stages.1.layers.0.layer.0
    ("resnet.encoder.stages.1.layers.0.layer.0", 128, 256, 56, 56,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 256]),
     _RELU, False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 12: resnet.encoder.stages.1.layers.0.layer.1
    ("resnet.encoder.stages.1.layers.0.layer.1", 128, 128, 56, 56,
     [3, 3], [2, 2], [1, 1, 1, 1], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 128]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 13: resnet.encoder.stages.1.layers.0.layer.2
    ("resnet.encoder.stages.1.layers.0.layer.2", 512, 128, 28, 28,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 128]),
     None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 14: resnet.encoder.stages.1.layers.0.shortcut
    ("resnet.encoder.stages.1.layers.0.shortcut", 512, 256, 56, 56,
     [1, 1], [2, 2], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [256, 256]),
     None, False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 15: resnet.encoder.stages.1.layers.1.layer.0
    ("resnet.encoder.stages.1.layers.1.layer.0", 128, 512, 28, 28,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 512]),
     _RELU, False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 16: resnet.encoder.stages.1.layers.1.layer.1
    ("resnet.encoder.stages.1.layers.1.layer.1", 128, 128, 28, 28,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 128]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 17: resnet.encoder.stages.1.layers.1.layer.2
    ("resnet.encoder.stages.1.layers.1.layer.2", 512, 128, 28, 28,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 128]),
     None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 18: resnet.encoder.stages.1.layers.2.layer.0
    ("resnet.encoder.stages.1.layers.2.layer.0", 128, 512, 28, 28,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 512]),
     _RELU, False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 19: resnet.encoder.stages.1.layers.2.layer.1
    ("resnet.encoder.stages.1.layers.2.layer.1", 128, 128, 28, 28,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 128]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 20: resnet.encoder.stages.1.layers.2.layer.2
    ("resnet.encoder.stages.1.layers.2.layer.2", 512, 128, 28, 28,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 128]),
     None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 21: resnet.encoder.stages.1.layers.3.layer.0
    ("resnet.encoder.stages.1.layers.3.layer.0", 128, 512, 28, 28,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 512]),
     _RELU, False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 22: resnet.encoder.stages.1.layers.3.layer.1
    ("resnet.encoder.stages.1.layers.3.layer.1", 128, 128, 28, 28,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 128]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 23: resnet.encoder.stages.1.layers.3.layer.2
    ("resnet.encoder.stages.1.layers.3.layer.2", 512, 128, 28, 28,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 128]),
     None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 24: resnet.encoder.stages.2.layers.0.layer.0
    ("resnet.encoder.stages.2.layers.0.layer.0", 256, 512, 28, 28,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 512]),
     _RELU, False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 25: resnet.encoder.stages.2.layers.0.layer.1
    ("resnet.encoder.stages.2.layers.0.layer.1", 256, 256, 28, 28,
     [3, 3], [2, 2], [1, 1, 1, 1], ttnn.Layout.TILE,
     _height_sharded_mem_cfg(_CORE_RANGE_98, [64, 256]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    # Conv 26: resnet.encoder.stages.2.layers.0.layer.2
    ("resnet.encoder.stages.2.layers.0.layer.2", 1024, 256, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_48, [288, 32]),
     None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 27: resnet.encoder.stages.2.layers.0.shortcut
    ("resnet.encoder.stages.2.layers.0.shortcut", 1024, 512, 28, 28,
     [1, 1], [2, 2], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_64, [800, 64]),
     None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 28: resnet.encoder.stages.2.layers.1.layer.0
    ("resnet.encoder.stages.2.layers.1.layer.0", 256, 1024, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 128]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 29: resnet.encoder.stages.2.layers.1.layer.1
    ("resnet.encoder.stages.2.layers.1.layer.1", 256, 256, 14, 14,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 32]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 30: resnet.encoder.stages.2.layers.1.layer.2
    ("resnet.encoder.stages.2.layers.1.layer.2", 1024, 256, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 32]),
     None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 31: resnet.encoder.stages.2.layers.2.layer.0
    ("resnet.encoder.stages.2.layers.2.layer.0", 256, 1024, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 128]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 32: resnet.encoder.stages.2.layers.2.layer.1
    ("resnet.encoder.stages.2.layers.2.layer.1", 256, 256, 14, 14,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 32]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 33: resnet.encoder.stages.2.layers.2.layer.2
    ("resnet.encoder.stages.2.layers.2.layer.2", 1024, 256, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 32]),
     None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 34: resnet.encoder.stages.2.layers.3.layer.0
    ("resnet.encoder.stages.2.layers.3.layer.0", 256, 1024, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 128]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 35: resnet.encoder.stages.2.layers.3.layer.1
    ("resnet.encoder.stages.2.layers.3.layer.1", 256, 256, 14, 14,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 32]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 36: resnet.encoder.stages.2.layers.3.layer.2
    ("resnet.encoder.stages.2.layers.3.layer.2", 1024, 256, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 32]),
     None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 37: resnet.encoder.stages.2.layers.4.layer.0
    ("resnet.encoder.stages.2.layers.4.layer.0", 256, 1024, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 128]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 38: resnet.encoder.stages.2.layers.4.layer.1
    ("resnet.encoder.stages.2.layers.4.layer.1", 256, 256, 14, 14,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 32]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 39: resnet.encoder.stages.2.layers.4.layer.2
    ("resnet.encoder.stages.2.layers.4.layer.2", 1024, 256, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 32]),
     None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 40: resnet.encoder.stages.2.layers.5.layer.0
    ("resnet.encoder.stages.2.layers.5.layer.0", 256, 1024, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 128]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 41: resnet.encoder.stages.2.layers.5.layer.1
    ("resnet.encoder.stages.2.layers.5.layer.1", 256, 256, 14, 14,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 32]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 42: resnet.encoder.stages.2.layers.5.layer.2
    ("resnet.encoder.stages.2.layers.5.layer.2", 1024, 256, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 32]),
     None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 43: resnet.encoder.stages.3.layers.0.layer.0
    ("resnet.encoder.stages.3.layers.0.layer.0", 512, 1024, 14, 14,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 128]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 44: resnet.encoder.stages.3.layers.0.layer.1
    ("resnet.encoder.stages.3.layers.0.layer.1", 512, 512, 14, 14,
     [3, 3], [2, 2], [1, 1, 1, 1], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_80, [160, 64]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 45: resnet.encoder.stages.3.layers.0.layer.2
    ("resnet.encoder.stages.3.layers.0.layer.2", 2048, 512, 7, 7,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_56, [64, 64]),
     None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 46: resnet.encoder.stages.3.layers.0.shortcut
    ("resnet.encoder.stages.3.layers.0.shortcut", 2048, 1024, 14, 14,
     [1, 1], [2, 2], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_110, [160, 96]),
     None, False, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 47: resnet.encoder.stages.3.layers.1.layer.0
    ("resnet.encoder.stages.3.layers.1.layer.0", 512, 2048, 7, 7,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_56, [64, 256]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 48: resnet.encoder.stages.3.layers.1.layer.1
    ("resnet.encoder.stages.3.layers.1.layer.1", 512, 512, 7, 7,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_56, [64, 64]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 49: resnet.encoder.stages.3.layers.1.layer.2
    ("resnet.encoder.stages.3.layers.1.layer.2", 2048, 512, 7, 7,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_56, [64, 64]),
     None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 50: resnet.encoder.stages.3.layers.2.layer.0
    ("resnet.encoder.stages.3.layers.2.layer.0", 512, 2048, 7, 7,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_56, [64, 256]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 51: resnet.encoder.stages.3.layers.2.layer.1
    ("resnet.encoder.stages.3.layers.2.layer.1", 512, 512, 7, 7,
     [3, 3], [1, 1], [1, 1, 1, 1], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_56, [64, 64]),
     _RELU, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    # Conv 52: resnet.encoder.stages.3.layers.2.layer.2
    ("resnet.encoder.stages.3.layers.2.layer.2", 2048, 512, 7, 7,
     [1, 1], [1, 1], [0, 0, 0, 0], ttnn.Layout.TILE,
     _block_sharded_mem_cfg(_CORE_RANGE_56, [64, 64]),
     None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
]


def run_consteval(weights, device):
    for (prefix, out_ch, in_ch, h, w,
         kernel_size, stride, padding, input_layout, input_memory_config,
         activation, dealloc_act, act_block_h, shard_layout) in CONV_CONSTEVAL_PARAMS:

        rv_key = prefix + ".normalization.running_var"
        rm_key = prefix + ".normalization.running_mean"
        bb_key = prefix + ".normalization.bias"
        bw_key = prefix + ".normalization.weight"
        cw_key = prefix + ".convolution.weight"

        running_var = ttnn.typecast(weights[rv_key], ttnn.DataType.FLOAT32, memory_config=None)
        bn_weight = ttnn.typecast(weights[bw_key], ttnn.DataType.FLOAT32, memory_config=None)
        conv_weight = ttnn.typecast(weights[cw_key], ttnn.DataType.FLOAT32, memory_config=None)
        running_mean = ttnn.typecast(weights[rm_key], ttnn.DataType.FLOAT32, memory_config=None)
        bn_bias = ttnn.typecast(weights[bb_key], ttnn.DataType.FLOAT32, memory_config=None)

        folded_weight, folded_bias = fold_bn_into_conv(
            running_var, bn_weight, conv_weight, running_mean, bn_bias, out_ch,
        )

        ttnn.deallocate(running_var, False)
        ttnn.deallocate(bn_weight, False)
        ttnn.deallocate(conv_weight, False)
        ttnn.deallocate(running_mean, False)
        ttnn.deallocate(bn_bias, False)

        conv_config = _make_conv_config(
            activation=activation,
            deallocate_activation=dealloc_act,
            act_block_h_override=act_block_h,
            shard_layout=shard_layout,
        )

        prepared_weight, prepared_bias = prepare_conv_weight_and_bias(
            folded_weight, folded_bias, device,
            in_channels=in_ch, out_channels=out_ch,
            batch_size=8,
            input_height=h, input_width=w,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=[1, 1], groups=1,
            input_layout=input_layout,
            input_memory_config=input_memory_config,
            conv_config=conv_config,
            slice_config=_CONV2D_SLICE_CFG,
        )

        weights[prefix + ".folded_conv_weight"] = prepared_weight
        weights[prefix + ".folded_conv_bias"] = prepared_bias

        del weights[rv_key]
        del weights[rm_key]
        del weights[bb_key]
        del weights[bw_key]
        del weights[cw_key]

    # Classifier bias: typecast -> to_layout(TILE) -> to_device
    bias_f32 = ttnn.typecast(weights["classifier.1.bias"], ttnn.DataType.FLOAT32, memory_config=None)
    bias_tiled = ttnn.to_layout(bias_f32, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(bias_f32, False)
    weights["classifier.1.bias"] = ttnn.to_device(bias_tiled, device=device, memory_config=_DRAM_INTERLEAVED)
    ttnn.deallocate(bias_tiled, False)

    # Classifier weight: typecast -> permute([1,0]) -> to_layout(TILE) -> to_device
    weight_f32 = ttnn.typecast(weights["classifier.1.weight"], ttnn.DataType.FLOAT32, memory_config=None)
    weight_torch = ttnn.to_torch(weight_f32)
    ttnn.deallocate(weight_f32, False)
    weight_permuted = ttnn.from_torch(ttir_cpu.permute(weight_torch, [1, 0]))
    weight_tiled = ttnn.to_layout(weight_permuted, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(weight_permuted, False)
    weights["classifier.1.weight"] = ttnn.to_device(weight_tiled, device=device, memory_config=_DRAM_INTERLEAVED)
    ttnn.deallocate(weight_tiled, False)

    return weights
