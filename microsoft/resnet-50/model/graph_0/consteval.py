import ttnn
import torch
import ttir_cpu


BN_EPSILON = 1.0013580322265625e-05

CONV_LAYER_PREFIXES = [
    "resnet.embedder.embedder",
    "resnet.encoder.stages.0.layers.0.layer.0",
    "resnet.encoder.stages.0.layers.0.layer.1",
    "resnet.encoder.stages.0.layers.0.layer.2",
    "resnet.encoder.stages.0.layers.0.shortcut",
    "resnet.encoder.stages.0.layers.1.layer.0",
    "resnet.encoder.stages.0.layers.1.layer.1",
    "resnet.encoder.stages.0.layers.1.layer.2",
    "resnet.encoder.stages.0.layers.2.layer.0",
    "resnet.encoder.stages.0.layers.2.layer.1",
    "resnet.encoder.stages.0.layers.2.layer.2",
    "resnet.encoder.stages.1.layers.0.layer.0",
    "resnet.encoder.stages.1.layers.0.layer.1",
    "resnet.encoder.stages.1.layers.0.layer.2",
    "resnet.encoder.stages.1.layers.0.shortcut",
    "resnet.encoder.stages.1.layers.1.layer.0",
    "resnet.encoder.stages.1.layers.1.layer.1",
    "resnet.encoder.stages.1.layers.1.layer.2",
    "resnet.encoder.stages.1.layers.2.layer.0",
    "resnet.encoder.stages.1.layers.2.layer.1",
    "resnet.encoder.stages.1.layers.2.layer.2",
    "resnet.encoder.stages.1.layers.3.layer.0",
    "resnet.encoder.stages.1.layers.3.layer.1",
    "resnet.encoder.stages.1.layers.3.layer.2",
    "resnet.encoder.stages.2.layers.0.layer.0",
    "resnet.encoder.stages.2.layers.0.layer.1",
    "resnet.encoder.stages.2.layers.0.layer.2",
    "resnet.encoder.stages.2.layers.0.shortcut",
    "resnet.encoder.stages.2.layers.1.layer.0",
    "resnet.encoder.stages.2.layers.1.layer.1",
    "resnet.encoder.stages.2.layers.1.layer.2",
    "resnet.encoder.stages.2.layers.2.layer.0",
    "resnet.encoder.stages.2.layers.2.layer.1",
    "resnet.encoder.stages.2.layers.2.layer.2",
    "resnet.encoder.stages.2.layers.3.layer.0",
    "resnet.encoder.stages.2.layers.3.layer.1",
    "resnet.encoder.stages.2.layers.3.layer.2",
    "resnet.encoder.stages.2.layers.4.layer.0",
    "resnet.encoder.stages.2.layers.4.layer.1",
    "resnet.encoder.stages.2.layers.4.layer.2",
    "resnet.encoder.stages.2.layers.5.layer.0",
    "resnet.encoder.stages.2.layers.5.layer.1",
    "resnet.encoder.stages.2.layers.5.layer.2",
    "resnet.encoder.stages.3.layers.0.layer.0",
    "resnet.encoder.stages.3.layers.0.layer.1",
    "resnet.encoder.stages.3.layers.0.layer.2",
    "resnet.encoder.stages.3.layers.0.shortcut",
    "resnet.encoder.stages.3.layers.1.layer.0",
    "resnet.encoder.stages.3.layers.1.layer.1",
    "resnet.encoder.stages.3.layers.1.layer.2",
    "resnet.encoder.stages.3.layers.2.layer.0",
    "resnet.encoder.stages.3.layers.2.layer.1",
    "resnet.encoder.stages.3.layers.2.layer.2",
]

CONV_CONFIGS = [
    # (in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, has_bias, groups, activation, deallocate_activation, act_block_h_override, shard_layout)
    (3, 64, 8, 224, 224, [7, 7], [2, 2], [3, 3, 3, 3], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 64, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (64, 64, 8, 56, 56, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (64, 64, 8, 56, 56, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (64, 256, 8, 56, 56, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (64, 256, 8, 56, 56, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (256, 64, 8, 56, 56, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (64, 64, 8, 56, 56, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (64, 256, 8, 56, 56, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (256, 64, 8, 56, 56, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (64, 64, 8, 56, 56, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (64, 256, 8, 56, 56, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (256, 128, 8, 56, 56, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (128, 128, 8, 56, 56, [3, 3], [2, 2], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (128, 512, 8, 28, 28, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (256, 512, 8, 56, 56, [1, 1], [2, 2], [0, 0, 0, 0], True, 1, None, False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (512, 128, 8, 28, 28, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (128, 128, 8, 28, 28, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (128, 512, 8, 28, 28, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (512, 128, 8, 28, 28, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (128, 128, 8, 28, 28, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (128, 512, 8, 28, 28, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (512, 128, 8, 28, 28, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (128, 128, 8, 28, 28, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (128, 512, 8, 28, 28, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (512, 256, 8, 28, 28, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), False, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (256, 256, 8, 28, 28, [3, 3], [2, 2], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (256, 1024, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (512, 1024, 8, 28, 28, [1, 1], [2, 2], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (1024, 256, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (256, 256, 8, 14, 14, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (256, 1024, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (1024, 256, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (256, 256, 8, 14, 14, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (256, 1024, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (1024, 256, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (256, 256, 8, 14, 14, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (256, 1024, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (1024, 256, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (256, 256, 8, 14, 14, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (256, 1024, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (1024, 256, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (256, 256, 8, 14, 14, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (256, 1024, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (1024, 512, 8, 14, 14, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (512, 512, 8, 14, 14, [3, 3], [2, 2], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (512, 2048, 8, 7, 7, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (1024, 2048, 8, 14, 14, [1, 1], [2, 2], [0, 0, 0, 0], True, 1, None, False, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (2048, 512, 8, 7, 7, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (512, 512, 8, 7, 7, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (512, 2048, 8, 7, 7, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (2048, 512, 8, 7, 7, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (512, 512, 8, 7, 7, [3, 3], [1, 1], [1, 1, 1, 1], True, 1, ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (512, 2048, 8, 7, 7, [1, 1], [1, 1], [0, 0, 0, 0], True, 1, None, True, 0, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
]


def fold_bn_into_conv(running_var, running_mean, bn_bias, bn_weight, conv_weight, out_channels):
    eps = torch.full([1], BN_EPSILON, dtype=torch.float32)
    scale = torch.div(bn_weight, torch.sqrt(torch.add(running_var, eps)))
    folded_weight = torch.multiply(conv_weight, scale.reshape(out_channels, 1, 1, 1))

    scale_nhwc = scale.reshape(1, out_channels, 1, 1).permute(0, 2, 3, 1)
    mean_nhwc = running_mean.reshape(1, out_channels, 1, 1).permute(0, 2, 3, 1)
    bias_nhwc = bn_bias.reshape(1, out_channels, 1, 1).permute(0, 2, 3, 1)
    folded_bias = torch.subtract(bias_nhwc, torch.multiply(mean_nhwc, scale_nhwc))

    return folded_weight, folded_bias


def prepare_conv_tensors(folded_weight, folded_bias, conv_cfg, device):
    in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, has_bias, groups, activation, deallocate_activation, act_block_h_override, shard_layout = conv_cfg

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.DataType.BFLOAT16,
        activation=activation,
        deallocate_activation=deallocate_activation,
        config_tensors_in_dram=True,
        act_block_h_override=act_block_h_override,
        shard_layout=shard_layout,
        enable_kernel_stride_folding=False,
    )
    l1_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None)
    slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0)

    weight_bf16 = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(folded_weight, False)
    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=weight_bf16,
        input_memory_config=l1_mem,
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=[1, 1],
        has_bias=has_bias,
        groups=groups,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=conv_config,
        compute_config=None,
        slice_config=slice_config,
    )
    ttnn.deallocate(weight_bf16, False)

    bias_bf16 = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(folded_bias, False)
    prepared_bias = ttnn.prepare_conv_bias(
        bias_tensor=bias_bf16,
        input_memory_config=l1_mem,
        input_layout=ttnn.Layout.TILE,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=[1, 1],
        groups=groups,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=conv_config,
        compute_config=None,
        slice_config=slice_config,
    )
    ttnn.deallocate(bias_bf16, False)

    return prepared_weight, prepared_bias


def prepare_classifier_bias(bias_tensor, device):
    t = ttnn.typecast(bias_tensor, ttnn.DataType.FLOAT32, memory_config=None)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.to_device(
        t, device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return t


def prepare_classifier_weight(weight_tensor, device):
    t = ttnn.typecast(weight_tensor, ttnn.DataType.FLOAT32, memory_config=None)
    t_torch = ttnn.to_torch(t)
    t_transposed = ttir_cpu.permute(t_torch, [1, 0])
    t = ttnn.from_torch(t_transposed)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.to_device(
        t, device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return t


def run_consteval(weights, device):
    for i, prefix in enumerate(CONV_LAYER_PREFIXES):
        running_var = ttnn.to_torch(ttnn.typecast(weights[prefix + ".normalization.running_var"], ttnn.DataType.FLOAT32, memory_config=None))
        running_mean = ttnn.to_torch(ttnn.typecast(weights[prefix + ".normalization.running_mean"], ttnn.DataType.FLOAT32, memory_config=None))
        bn_bias = ttnn.to_torch(ttnn.typecast(weights[prefix + ".normalization.bias"], ttnn.DataType.FLOAT32, memory_config=None))
        bn_weight = ttnn.to_torch(ttnn.typecast(weights[prefix + ".normalization.weight"], ttnn.DataType.FLOAT32, memory_config=None))
        conv_weight = ttnn.to_torch(ttnn.typecast(weights[prefix + ".convolution.weight"], ttnn.DataType.FLOAT32, memory_config=None))

        out_channels = CONV_CONFIGS[i][1]
        folded_weight, folded_bias = fold_bn_into_conv(
            running_var, running_mean, bn_bias, bn_weight, conv_weight, out_channels,
        )

        folded_weight_ttnn = ttnn.from_torch(folded_weight)
        folded_bias_ttnn = ttnn.from_torch(folded_bias)

        prepared_weight, prepared_bias = prepare_conv_tensors(
            folded_weight_ttnn, folded_bias_ttnn, CONV_CONFIGS[i], device,
        )

        weights[prefix + ".folded_conv_weight"] = prepared_weight
        weights[prefix + ".folded_conv_bias"] = prepared_bias

    weights["classifier.1.prepared_weight"] = prepare_classifier_weight(weights["classifier.1.weight"], device)
    weights["classifier.1.prepared_bias"] = prepare_classifier_bias(weights["classifier.1.bias"], device)

    return weights
