import ttnn
import torch
import ttir_cpu


def fold_bn_into_conv_weights(running_var, running_mean, bn_bias, bn_weight, conv_weight, out_channels):
    """Fold batch normalization parameters into convolution weights and bias."""
    running_var = ttnn.to_torch(running_var)
    running_mean = ttnn.to_torch(running_mean)
    bn_bias = ttnn.to_torch(bn_bias)
    bn_weight = ttnn.to_torch(bn_weight)
    conv_weight = ttnn.to_torch(conv_weight)
    eps = torch.full([1], 1.0013580322265625e-05, dtype=torch.float32)
    scale = bn_weight / torch.sqrt(running_var + eps)
    folded_weight = conv_weight * scale.reshape(out_channels, 1, 1, 1)
    folded_bias = (
        bn_bias.reshape(1, out_channels, 1, 1).permute(0, 2, 3, 1)
        - running_mean.reshape(1, out_channels, 1, 1).permute(0, 2, 3, 1)
        * scale.reshape(1, out_channels, 1, 1).permute(0, 2, 3, 1)
    )
    folded_weight = ttnn.from_torch(folded_weight)
    folded_bias = ttnn.from_torch(folded_bias)
    return folded_weight, folded_bias


def run_consteval(weights, device):
    """Run constant evaluation: fold BN into conv weights and prepare for device."""

    # Layer 0: resnet.embedder.embedder
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.embedder.embedder.normalization.running_var"],
        weights["resnet.embedder.embedder.normalization.running_mean"],
        weights["resnet.embedder.embedder.normalization.bias"],
        weights["resnet.embedder.embedder.normalization.weight"],
        weights["resnet.embedder.embedder.convolution.weight"],
        64,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.embedder.embedder.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=3,
        out_channels=64,
        batch_size=8,
        input_height=224,
        input_width=224,
        kernel_size=[7, 7],
        stride=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=64,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.embedder.embedder.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=3,
        out_channels=64,
        batch_size=8,
        input_height=224,
        input_width=224,
        kernel_size=[7, 7],
        stride=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=64,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 1: resnet.encoder.stages.0.layers.0.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.0.layers.0.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.0.layers.0.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.0.layers.0.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.0.layers.0.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.0.layers.0.layer.0.convolution.weight"],
        64,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.0.layers.0.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(2, 9)),
                    ]
                ),
                [246, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.0.layers.0.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(2, 9)),
                    ]
                ),
                [246, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 2: resnet.encoder.stages.0.layers.0.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.0.layers.0.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.0.layers.0.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.0.layers.0.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.0.layers.0.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.0.layers.0.layer.1.convolution.weight"],
        64,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.0.layers.0.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.0.layers.0.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 3: resnet.encoder.stages.0.layers.0.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.0.layers.0.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.0.layers.0.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.0.layers.0.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.0.layers.0.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.0.layers.0.layer.2.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.0.layers.0.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.0.layers.0.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 4: resnet.encoder.stages.0.layers.0.shortcut
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.0.layers.0.shortcut.normalization.running_var"],
        weights["resnet.encoder.stages.0.layers.0.shortcut.normalization.running_mean"],
        weights["resnet.encoder.stages.0.layers.0.shortcut.normalization.bias"],
        weights["resnet.encoder.stages.0.layers.0.shortcut.normalization.weight"],
        weights["resnet.encoder.stages.0.layers.0.shortcut.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.0.layers.0.shortcut.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.0.layers.0.shortcut.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 5: resnet.encoder.stages.0.layers.1.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.0.layers.1.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.0.layers.1.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.0.layers.1.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.0.layers.1.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.0.layers.1.layer.0.convolution.weight"],
        64,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.0.layers.1.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.0.layers.1.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 6: resnet.encoder.stages.0.layers.1.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.0.layers.1.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.0.layers.1.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.0.layers.1.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.0.layers.1.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.0.layers.1.layer.1.convolution.weight"],
        64,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.0.layers.1.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.0.layers.1.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 7: resnet.encoder.stages.0.layers.1.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.0.layers.1.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.0.layers.1.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.0.layers.1.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.0.layers.1.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.0.layers.1.layer.2.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.0.layers.1.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.0.layers.1.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 8: resnet.encoder.stages.0.layers.2.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.0.layers.2.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.0.layers.2.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.0.layers.2.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.0.layers.2.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.0.layers.2.layer.0.convolution.weight"],
        64,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.0.layers.2.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.0.layers.2.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 9: resnet.encoder.stages.0.layers.2.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.0.layers.2.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.0.layers.2.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.0.layers.2.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.0.layers.2.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.0.layers.2.layer.1.convolution.weight"],
        64,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.0.layers.2.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.0.layers.2.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=64,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 10: resnet.encoder.stages.0.layers.2.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.0.layers.2.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.0.layers.2.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.0.layers.2.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.0.layers.2.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.0.layers.2.layer.2.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.0.layers.2.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.0.layers.2.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=64,
        out_channels=256,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 11: resnet.encoder.stages.1.layers.0.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.0.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.0.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.0.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.0.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.0.layer.0.convolution.weight"],
        128,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.0.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=128,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.0.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=128,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 12: resnet.encoder.stages.1.layers.0.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.0.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.0.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.0.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.0.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.0.layer.1.convolution.weight"],
        128,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.0.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.0.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 13: resnet.encoder.stages.1.layers.0.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.0.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.0.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.0.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.0.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.0.layer.2.convolution.weight"],
        512,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.0.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.0.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 14: resnet.encoder.stages.1.layers.0.shortcut
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.0.shortcut.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.0.shortcut.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.0.shortcut.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.0.shortcut.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.0.shortcut.convolution.weight"],
        512,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.0.shortcut.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [256, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=512,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.0.shortcut.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [256, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=512,
        batch_size=8,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 15: resnet.encoder.stages.1.layers.1.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.1.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.1.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.1.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.1.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.1.layer.0.convolution.weight"],
        128,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.1.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.1.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 16: resnet.encoder.stages.1.layers.1.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.1.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.1.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.1.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.1.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.1.layer.1.convolution.weight"],
        128,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.1.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.1.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 17: resnet.encoder.stages.1.layers.1.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.1.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.1.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.1.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.1.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.1.layer.2.convolution.weight"],
        512,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.1.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.1.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 18: resnet.encoder.stages.1.layers.2.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.2.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.2.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.2.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.2.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.2.layer.0.convolution.weight"],
        128,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.2.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.2.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 19: resnet.encoder.stages.1.layers.2.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.2.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.2.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.2.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.2.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.2.layer.1.convolution.weight"],
        128,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.2.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.2.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 20: resnet.encoder.stages.1.layers.2.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.2.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.2.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.2.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.2.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.2.layer.2.convolution.weight"],
        512,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.2.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.2.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 21: resnet.encoder.stages.1.layers.3.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.3.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.3.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.3.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.3.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.3.layer.0.convolution.weight"],
        128,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.3.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.3.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 22: resnet.encoder.stages.1.layers.3.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.3.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.3.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.3.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.3.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.3.layer.1.convolution.weight"],
        128,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.3.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.3.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=128,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 23: resnet.encoder.stages.1.layers.3.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.1.layers.3.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.1.layers.3.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.1.layers.3.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.1.layers.3.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.1.layers.3.layer.2.convolution.weight"],
        512,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.1.layers.3.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.1.layers.3.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=128,
        out_channels=512,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 24: resnet.encoder.stages.2.layers.0.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.0.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.0.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.0.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.0.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.0.layer.0.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.0.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=256,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.0.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=256,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 25: resnet.encoder.stages.2.layers.0.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.0.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.0.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.0.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.0.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.0.layer.1.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.0.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
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
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.0.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
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
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 26: resnet.encoder.stages.2.layers.0.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.0.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.0.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.0.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.0.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.0.layer.2.convolution.weight"],
        1024,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.0.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [288, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.0.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))]
                ),
                [288, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 27: resnet.encoder.stages.2.layers.0.shortcut
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.0.shortcut.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.0.shortcut.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.0.shortcut.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.0.shortcut.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.0.shortcut.convolution.weight"],
        1024,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.0.shortcut.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=1024,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.0.shortcut.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=1024,
        batch_size=8,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 28: resnet.encoder.stages.2.layers.1.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.1.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.1.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.1.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.1.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.1.layer.0.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.1.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.1.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 29: resnet.encoder.stages.2.layers.1.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.1.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.1.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.1.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.1.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.1.layer.1.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.1.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.1.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 30: resnet.encoder.stages.2.layers.1.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.1.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.1.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.1.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.1.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.1.layer.2.convolution.weight"],
        1024,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.1.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.1.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 31: resnet.encoder.stages.2.layers.2.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.2.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.2.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.2.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.2.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.2.layer.0.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.2.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.2.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 32: resnet.encoder.stages.2.layers.2.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.2.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.2.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.2.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.2.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.2.layer.1.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.2.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.2.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 33: resnet.encoder.stages.2.layers.2.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.2.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.2.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.2.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.2.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.2.layer.2.convolution.weight"],
        1024,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.2.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.2.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 34: resnet.encoder.stages.2.layers.3.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.3.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.3.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.3.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.3.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.3.layer.0.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.3.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.3.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 35: resnet.encoder.stages.2.layers.3.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.3.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.3.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.3.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.3.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.3.layer.1.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.3.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.3.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 36: resnet.encoder.stages.2.layers.3.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.3.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.3.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.3.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.3.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.3.layer.2.convolution.weight"],
        1024,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.3.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.3.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 37: resnet.encoder.stages.2.layers.4.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.4.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.4.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.4.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.4.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.4.layer.0.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.4.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.4.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 38: resnet.encoder.stages.2.layers.4.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.4.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.4.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.4.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.4.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.4.layer.1.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.4.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.4.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 39: resnet.encoder.stages.2.layers.4.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.4.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.4.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.4.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.4.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.4.layer.2.convolution.weight"],
        1024,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.4.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.4.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 40: resnet.encoder.stages.2.layers.5.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.5.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.5.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.5.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.5.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.5.layer.0.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.5.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.5.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 41: resnet.encoder.stages.2.layers.5.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.5.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.5.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.5.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.5.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.5.layer.1.convolution.weight"],
        256,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.5.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.5.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=256,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 42: resnet.encoder.stages.2.layers.5.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.2.layers.5.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.2.layers.5.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.2.layers.5.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.2.layers.5.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.2.layers.5.layer.2.convolution.weight"],
        1024,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.2.layers.5.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.2.layers.5.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=256,
        out_channels=1024,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 43: resnet.encoder.stages.3.layers.0.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.3.layers.0.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.3.layers.0.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.3.layers.0.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.3.layers.0.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.3.layers.0.layer.0.convolution.weight"],
        512,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.3.layers.0.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=512,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.3.layers.0.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=512,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 44: resnet.encoder.stages.3.layers.0.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.3.layers.0.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.3.layers.0.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.3.layers.0.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.3.layers.0.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.3.layers.0.layer.1.convolution.weight"],
        512,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.3.layers.0.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.3.layers.0.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 9))]
                ),
                [160, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 45: resnet.encoder.stages.3.layers.0.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.3.layers.0.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.3.layers.0.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.3.layers.0.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.3.layers.0.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.3.layers.0.layer.2.convolution.weight"],
        2048,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.3.layers.0.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.3.layers.0.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 46: resnet.encoder.stages.3.layers.0.shortcut
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.3.layers.0.shortcut.normalization.running_var"],
        weights["resnet.encoder.stages.3.layers.0.shortcut.normalization.running_mean"],
        weights["resnet.encoder.stages.3.layers.0.shortcut.normalization.bias"],
        weights["resnet.encoder.stages.3.layers.0.shortcut.normalization.weight"],
        weights["resnet.encoder.stages.3.layers.0.shortcut.convolution.weight"],
        2048,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.3.layers.0.shortcut.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))]
                ),
                [160, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=2048,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.3.layers.0.shortcut.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))]
                ),
                [160, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=1024,
        out_channels=2048,
        batch_size=8,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 47: resnet.encoder.stages.3.layers.1.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.3.layers.1.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.3.layers.1.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.3.layers.1.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.3.layers.1.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.3.layers.1.layer.0.convolution.weight"],
        512,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.3.layers.1.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=2048,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.3.layers.1.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=2048,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 48: resnet.encoder.stages.3.layers.1.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.3.layers.1.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.3.layers.1.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.3.layers.1.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.3.layers.1.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.3.layers.1.layer.1.convolution.weight"],
        512,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.3.layers.1.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.3.layers.1.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 49: resnet.encoder.stages.3.layers.1.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.3.layers.1.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.3.layers.1.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.3.layers.1.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.3.layers.1.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.3.layers.1.layer.2.convolution.weight"],
        2048,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.3.layers.1.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.3.layers.1.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 50: resnet.encoder.stages.3.layers.2.layer.0
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.3.layers.2.layer.0.normalization.running_var"],
        weights["resnet.encoder.stages.3.layers.2.layer.0.normalization.running_mean"],
        weights["resnet.encoder.stages.3.layers.2.layer.0.normalization.bias"],
        weights["resnet.encoder.stages.3.layers.2.layer.0.normalization.weight"],
        weights["resnet.encoder.stages.3.layers.2.layer.0.convolution.weight"],
        512,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.3.layers.2.layer.0.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=2048,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.3.layers.2.layer.0.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=2048,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 51: resnet.encoder.stages.3.layers.2.layer.1
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.3.layers.2.layer.1.normalization.running_var"],
        weights["resnet.encoder.stages.3.layers.2.layer.1.normalization.running_mean"],
        weights["resnet.encoder.stages.3.layers.2.layer.1.normalization.bias"],
        weights["resnet.encoder.stages.3.layers.2.layer.1.normalization.weight"],
        weights["resnet.encoder.stages.3.layers.2.layer.1.convolution.weight"],
        512,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.3.layers.2.layer.1.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.3.layers.2.layer.1.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=512,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Layer 52: resnet.encoder.stages.3.layers.2.layer.2
    folded_weight, folded_bias = fold_bn_into_conv_weights(
        weights["resnet.encoder.stages.3.layers.2.layer.2.normalization.running_var"],
        weights["resnet.encoder.stages.3.layers.2.layer.2.normalization.running_mean"],
        weights["resnet.encoder.stages.3.layers.2.layer.2.normalization.bias"],
        weights["resnet.encoder.stages.3.layers.2.layer.2.normalization.weight"],
        weights["resnet.encoder.stages.3.layers.2.layer.2.convolution.weight"],
        2048,
    )
    folded_weight = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    folded_bias = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    weights["resnet.encoder.stages.3.layers.2.layer.2.folded_conv_weight"] = ttnn.prepare_conv_weights(
        weight_tensor=folded_weight,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )
    weights["resnet.encoder.stages.3.layers.2.layer.2.folded_conv_bias"] = ttnn.prepare_conv_bias(
        bias_tensor=folded_bias,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [64, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.TILE,
        in_channels=512,
        out_channels=2048,
        batch_size=8,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    # Classifier bias
    classifier_bias = ttnn.typecast(weights["classifier.1.bias"], ttnn.DataType.FLOAT32, memory_config=None)
    classifier_bias = ttnn.from_torch(ttnn.to_torch(classifier_bias))
    classifier_bias = ttnn.to_layout(classifier_bias, ttnn.Layout.TILE, None, memory_config=None)
    weights["classifier.1.folded_bias"] = ttnn.to_device(
        classifier_bias,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )

    # Classifier weight (transposed)
    classifier_weight = ttnn.typecast(weights["classifier.1.weight"], ttnn.DataType.FLOAT32, memory_config=None)
    classifier_weight_torch = ttnn.to_torch(classifier_weight)
    classifier_weight_transposed = ttir_cpu.permute(classifier_weight_torch, [1, 0])
    classifier_weight = ttnn.from_torch(classifier_weight_transposed)
    classifier_weight = ttnn.to_layout(classifier_weight, ttnn.Layout.TILE, None, memory_config=None)
    weights["classifier.1.folded_weight"] = ttnn.to_device(
        classifier_weight,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )

    return weights
