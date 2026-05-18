import ttnn
import ttir_cpu
import torch


BN_EPSILON = 1.0e-5


def _fuse_bn_into_conv(weights, prefix, out_channels):
    """Fold the BatchNorm at <prefix>.normalization into the conv at <prefix>.convolution.

    Returns (fused_weight, fused_bias) as ttnn tensors ready for device prep. The bias
    is laid out as NHWC (channels-last) to match ttnn.conv2d's expectation.
    """
    eps = ttir_cpu.full(shape=[1], fill_value=0.000010, dtype=torch.float32)

    running_var = ttnn.to_torch(
        ttnn.typecast(weights[f"{prefix}.normalization.running_var"], ttnn.DataType.FLOAT32, memory_config=None)
    )
    gamma = ttnn.to_torch(
        ttnn.typecast(weights[f"{prefix}.normalization.weight"], ttnn.DataType.FLOAT32, memory_config=None)
    )
    conv_weight = ttnn.to_torch(
        ttnn.typecast(weights[f"{prefix}.convolution.weight"], ttnn.DataType.FLOAT32, memory_config=None)
    )
    running_mean = ttnn.to_torch(
        ttnn.typecast(weights[f"{prefix}.normalization.running_mean"], ttnn.DataType.FLOAT32, memory_config=None)
    )
    beta = ttnn.to_torch(
        ttnn.typecast(weights[f"{prefix}.normalization.bias"], ttnn.DataType.FLOAT32, memory_config=None)
    )

    inv_std = ttir_cpu.div(gamma, ttir_cpu.sqrt(ttir_cpu.add(running_var, eps)))

    fused_weight = ttir_cpu.multiply(
        conv_weight, ttir_cpu.reshape(inv_std, [out_channels, 1, 1, 1])
    )

    inv_std_nhwc = ttir_cpu.permute(
        ttir_cpu.reshape(inv_std, [1, out_channels, 1, 1]), [0, 2, 3, 1]
    )
    mean_nhwc = ttir_cpu.permute(
        ttir_cpu.reshape(running_mean, [1, out_channels, 1, 1]), [0, 2, 3, 1]
    )
    beta_nhwc = ttir_cpu.permute(
        ttir_cpu.reshape(beta, [1, out_channels, 1, 1]), [0, 2, 3, 1]
    )
    fused_bias = ttir_cpu.subtract(
        beta_nhwc, ttir_cpu.multiply(mean_nhwc, inv_std_nhwc)
    )

    return ttnn.from_torch(fused_weight), ttnn.from_torch(fused_bias)


def _prepare_conv_for_device(
    weights,
    device,
    *,
    prefix,
    in_channels,
    out_channels,
    batch_size,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    dilation,
    input_memory_config,
    input_layout,
    conv_config,
):
    """Fuse BN into the conv at <prefix>, prepare weight+bias for ttnn.conv2d, and
    store them back into `weights` as <prefix>.convolution.weight / .convolution.bias.
    """
    fused_weight, fused_bias = _fuse_bn_into_conv(weights, prefix, out_channels)

    typed_weight = ttnn.typecast(fused_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(fused_weight, False)
    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=typed_weight,
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
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=conv_config,
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(typed_weight, False)

    typed_bias = ttnn.typecast(fused_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(fused_bias, False)
    prepared_bias = ttnn.prepare_conv_bias(
        bias_tensor=typed_bias,
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
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=conv_config,
        compute_config=None,
    )
    ttnn.deallocate(typed_bias, False)

    weights[f"{prefix}.convolution.weight"] = prepared_weight
    weights[f"{prefix}.convolution.bias"] = prepared_bias


def _prepare_classifier_bias_for_device(weights, device):
    """Move the classifier bias to device (TILE layout, DRAM)."""
    typed = ttnn.typecast(weights["classifier.1.bias"], ttnn.DataType.FLOAT32, memory_config=None)
    on_host = ttnn.from_torch(ttnn.to_torch(typed))
    ttnn.deallocate(typed, False)
    tiled = ttnn.to_layout(on_host, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(on_host, False)
    on_device = ttnn.to_device(
        tiled,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(tiled, False)
    weights["classifier.1.bias"] = on_device


def _prepare_classifier_weight_for_device(weights, device):
    """Transpose the classifier weight matrix (linear is OI; matmul wants IO) and move to device."""
    typed = ttnn.typecast(weights["classifier.1.weight"], ttnn.DataType.FLOAT32, memory_config=None)
    transposed = ttnn.from_torch(ttir_cpu.permute(ttnn.to_torch(typed), [1, 0]))
    ttnn.deallocate(typed, False)
    tiled = ttnn.to_layout(transposed, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(transposed, False)
    on_device = ttnn.to_device(
        tiled,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(tiled, False)
    weights["classifier.1.weight"] = on_device


def run_consteval(weights, device):
    """Fuse BN folds, transpose/move the classifier matrix, and stage every weight tensor
    onto the device. Each modified entry in `weights` is replaced with a prepared tensor;
    new keys <prefix>.convolution.bias are added for each fused conv-bn block. Mutates and
    returns `weights`.
    """
    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.embedder.embedder',
        in_channels=3,
        out_channels=64,
        batch_size=1,
        input_height=224,
        input_width=224,
        kernel_size=[7, 7],
        stride=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=64,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.0.layers.0.layer.0',
        in_channels=64,
        out_channels=64,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(1, 7)),
                    ]
                ),
                [40, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.0.layers.0.layer.1',
        in_channels=64,
        out_channels=64,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.0.layers.0.layer.2',
        in_channels=64,
        out_channels=256,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.0.layers.0.shortcut',
        in_channels=64,
        out_channels=256,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.0.layers.1.layer.0',
        in_channels=256,
        out_channels=64,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.0.layers.1.layer.1',
        in_channels=64,
        out_channels=64,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.0.layers.1.layer.2',
        in_channels=64,
        out_channels=256,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.0.layers.2.layer.0',
        in_channels=256,
        out_channels=64,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.0.layers.2.layer.1',
        in_channels=64,
        out_channels=64,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.0.layers.2.layer.2',
        in_channels=64,
        out_channels=256,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.0.layer.0',
        in_channels=256,
        out_channels=128,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.0.layer.1',
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.0.layer.2',
        in_channels=128,
        out_channels=512,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 6))]
                ),
                [128, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.0.shortcut',
        in_channels=256,
        out_channels=512,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [352, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.1.layer.0',
        in_channels=512,
        out_channels=128,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.1.layer.1',
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.1.layer.2',
        in_channels=128,
        out_channels=512,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.2.layer.0',
        in_channels=512,
        out_channels=128,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.2.layer.1',
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.2.layer.2',
        in_channels=128,
        out_channels=512,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.3.layer.0',
        in_channels=512,
        out_channels=128,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.3.layer.1',
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.1.layers.3.layer.2',
        in_channels=128,
        out_channels=512,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.0.layer.0',
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.0.layer.1',
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.0.layer.2',
        in_channels=256,
        out_channels=1024,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.0.shortcut',
        in_channels=512,
        out_channels=1024,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.1.layer.0',
        in_channels=1024,
        out_channels=256,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.1.layer.1',
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.1.layer.2',
        in_channels=256,
        out_channels=1024,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.2.layer.0',
        in_channels=1024,
        out_channels=256,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.2.layer.1',
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.2.layer.2',
        in_channels=256,
        out_channels=1024,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.3.layer.0',
        in_channels=1024,
        out_channels=256,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.3.layer.1',
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.3.layer.2',
        in_channels=256,
        out_channels=1024,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.4.layer.0',
        in_channels=1024,
        out_channels=256,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.4.layer.1',
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.4.layer.2',
        in_channels=256,
        out_channels=1024,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.5.layer.0',
        in_channels=1024,
        out_channels=256,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.5.layer.1',
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.2.layers.5.layer.2',
        in_channels=256,
        out_channels=1024,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.3.layers.0.layer.0',
        in_channels=1024,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.3.layers.0.layer.1',
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.3.layers.0.layer.2',
        in_channels=512,
        out_channels=2048,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.3.layers.0.shortcut',
        in_channels=1024,
        out_channels=2048,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.3.layers.1.layer.0',
        in_channels=2048,
        out_channels=512,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1))]
                ),
                [64, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.3.layers.1.layer.1',
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 0)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(4, 1)),
                    ]
                ),
                [64, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.3.layers.1.layer.2',
        in_channels=512,
        out_channels=2048,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 0)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(4, 1)),
                    ]
                ),
                [64, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.3.layers.2.layer.0',
        in_channels=2048,
        out_channels=512,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),
                    ]
                ),
                [64, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.3.layers.2.layer.1',
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 0)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(4, 1)),
                    ]
                ),
                [64, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_conv_for_device(
        weights, device,
        prefix='resnet.encoder.stages.3.layers.2.layer.2',
        in_channels=512,
        out_channels=2048,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 0)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(4, 1)),
                    ]
                ),
                [64, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
    )

    _prepare_classifier_bias_for_device(weights, device)
    _prepare_classifier_weight_for_device(weights, device)

    return weights
