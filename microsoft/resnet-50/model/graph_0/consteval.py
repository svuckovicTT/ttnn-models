import ttnn
import ttir_cpu
import torch


BN_EPS = 1.0013580322265625e-05


def fold_bn_into_conv(running_var, bn_weight, conv_weight, running_mean, bn_bias):
    running_var = ttnn.to_torch(running_var).float()
    bn_weight = ttnn.to_torch(bn_weight).float()
    conv_weight = ttnn.to_torch(conv_weight).float()
    running_mean = ttnn.to_torch(running_mean).float()
    bn_bias = ttnn.to_torch(bn_bias).float()

    scale = bn_weight / torch.sqrt(running_var + BN_EPS)
    folded_weight = conv_weight * scale.reshape(-1, 1, 1, 1)

    scale_bias = scale.reshape(1, -1, 1, 1).permute(0, 2, 3, 1)
    mean_bias = running_mean.reshape(1, -1, 1, 1).permute(0, 2, 3, 1)
    folded_bias = bn_bias.reshape(1, -1, 1, 1).permute(0, 2, 3, 1) - mean_bias * scale_bias

    return ttnn.from_torch(folded_weight), ttnn.from_torch(folded_bias)


def prepare_conv_weight_and_bias(folded_weight, folded_bias, conv_params, device):
    weight_bf16 = ttnn.typecast(folded_weight, ttnn.DataType.BFLOAT16, memory_config=None)

    activation = conv_params.get("activation")
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.DataType.BFLOAT16,
        **({"activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)} if activation == "relu" else {}),
        deallocate_activation=conv_params["deallocate_activation"],
        config_tensors_in_dram=True,
        act_block_h_override=conv_params.get("act_block_h_override", 0),
        shard_layout=conv_params["shard_layout"],
        enable_kernel_stride_folding=False,
    )

    prepared_weight = ttnn.prepare_conv_weights(
        weight_tensor=weight_bf16,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        input_layout=conv_params.get("input_layout", ttnn.Layout.TILE),
        weights_format="OIHW",
        in_channels=conv_params["in_channels"],
        out_channels=conv_params["out_channels"],
        batch_size=8,
        input_height=conv_params["input_height"],
        input_width=conv_params["input_width"],
        kernel_size=conv_params["kernel_size"],
        stride=conv_params["stride"],
        padding=conv_params["padding"],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=conv_config,
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    bias_bf16 = ttnn.typecast(folded_bias, ttnn.DataType.BFLOAT16, memory_config=None)

    prepared_bias = ttnn.prepare_conv_bias(
        bias_tensor=bias_bf16,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        input_layout=conv_params.get("input_layout", ttnn.Layout.TILE),
        in_channels=conv_params["in_channels"],
        out_channels=conv_params["out_channels"],
        batch_size=8,
        input_height=conv_params["input_height"],
        input_width=conv_params["input_width"],
        kernel_size=conv_params["kernel_size"],
        stride=conv_params["stride"],
        padding=conv_params["padding"],
        dilation=[1, 1],
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=conv_config,
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
    )

    return prepared_weight, prepared_bias


def prepare_classifier_bias(weights, device):
    t = ttnn.typecast(weights["classifier.1.bias"], ttnn.DataType.FLOAT32, memory_config=None)
    t = ttnn.to_torch(t)
    t = ttnn.from_torch(t)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.to_device(t, device=device, memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ))
    return t


def prepare_classifier_weight(weights, device):
    t = ttnn.typecast(weights["classifier.1.weight"], ttnn.DataType.FLOAT32, memory_config=None)
    t = ttnn.to_torch(t)
    t = ttir_cpu.permute(t, [1, 0])
    t = ttnn.from_torch(t)
    t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
    t = ttnn.to_device(t, device=device, memory_config=ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    ))
    return t


CONV_LAYER_PARAMS = [
    {
        "prefix": "resnet.embedder.embedder",
        "in_channels": 3, "out_channels": 64, "input_height": 224, "input_width": 224,
        "kernel_size": [7, 7], "stride": [2, 2], "padding": [3, 3, 3, 3],
        "activation": "relu", "deallocate_activation": True,
        "act_block_h_override": 64, "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.0.layers.0.layer.0",
        "in_channels": 64, "out_channels": 64, "input_height": 56, "input_width": 56,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": False,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "input_layout": ttnn.Layout.ROW_MAJOR,
    },
    {
        "prefix": "resnet.encoder.stages.0.layers.0.layer.1",
        "in_channels": 64, "out_channels": 64, "input_height": 56, "input_width": 56,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.0.layers.0.layer.2",
        "in_channels": 64, "out_channels": 256, "input_height": 56, "input_width": 56,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.0.layers.0.shortcut",
        "in_channels": 64, "out_channels": 256, "input_height": 56, "input_width": 56,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "input_layout": ttnn.Layout.ROW_MAJOR,
    },
    {
        "prefix": "resnet.encoder.stages.0.layers.1.layer.0",
        "in_channels": 256, "out_channels": 64, "input_height": 56, "input_width": 56,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": False,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.0.layers.1.layer.1",
        "in_channels": 64, "out_channels": 64, "input_height": 56, "input_width": 56,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.0.layers.1.layer.2",
        "in_channels": 64, "out_channels": 256, "input_height": 56, "input_width": 56,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.0.layers.2.layer.0",
        "in_channels": 256, "out_channels": 64, "input_height": 56, "input_width": 56,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": False,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.0.layers.2.layer.1",
        "in_channels": 64, "out_channels": 64, "input_height": 56, "input_width": 56,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.0.layers.2.layer.2",
        "in_channels": 64, "out_channels": 256, "input_height": 56, "input_width": 56,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.0.layer.0",
        "in_channels": 256, "out_channels": 128, "input_height": 56, "input_width": 56,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": False,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.0.layer.1",
        "in_channels": 128, "out_channels": 128, "input_height": 56, "input_width": 56,
        "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.0.layer.2",
        "in_channels": 128, "out_channels": 512, "input_height": 28, "input_width": 28,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.0.shortcut",
        "in_channels": 256, "out_channels": 512, "input_height": 56, "input_width": 56,
        "kernel_size": [1, 1], "stride": [2, 2], "padding": [0, 0, 0, 0],
        "deallocate_activation": False,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.1.layer.0",
        "in_channels": 512, "out_channels": 128, "input_height": 28, "input_width": 28,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": False,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.1.layer.1",
        "in_channels": 128, "out_channels": 128, "input_height": 28, "input_width": 28,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.1.layer.2",
        "in_channels": 128, "out_channels": 512, "input_height": 28, "input_width": 28,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.2.layer.0",
        "in_channels": 512, "out_channels": 128, "input_height": 28, "input_width": 28,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": False,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.2.layer.1",
        "in_channels": 128, "out_channels": 128, "input_height": 28, "input_width": 28,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.2.layer.2",
        "in_channels": 128, "out_channels": 512, "input_height": 28, "input_width": 28,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.3.layer.0",
        "in_channels": 512, "out_channels": 128, "input_height": 28, "input_width": 28,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": False,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.3.layer.1",
        "in_channels": 128, "out_channels": 128, "input_height": 28, "input_width": 28,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.1.layers.3.layer.2",
        "in_channels": 128, "out_channels": 512, "input_height": 28, "input_width": 28,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.0.layer.0",
        "in_channels": 512, "out_channels": 256, "input_height": 28, "input_width": 28,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": False,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.0.layer.1",
        "in_channels": 256, "out_channels": 256, "input_height": 28, "input_width": 28,
        "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.0.layer.2",
        "in_channels": 256, "out_channels": 1024, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.0.shortcut",
        "in_channels": 512, "out_channels": 1024, "input_height": 28, "input_width": 28,
        "kernel_size": [1, 1], "stride": [2, 2], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.1.layer.0",
        "in_channels": 1024, "out_channels": 256, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.1.layer.1",
        "in_channels": 256, "out_channels": 256, "input_height": 14, "input_width": 14,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.1.layer.2",
        "in_channels": 256, "out_channels": 1024, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.2.layer.0",
        "in_channels": 1024, "out_channels": 256, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.2.layer.1",
        "in_channels": 256, "out_channels": 256, "input_height": 14, "input_width": 14,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.2.layer.2",
        "in_channels": 256, "out_channels": 1024, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.3.layer.0",
        "in_channels": 1024, "out_channels": 256, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.3.layer.1",
        "in_channels": 256, "out_channels": 256, "input_height": 14, "input_width": 14,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.3.layer.2",
        "in_channels": 256, "out_channels": 1024, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.4.layer.0",
        "in_channels": 1024, "out_channels": 256, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.4.layer.1",
        "in_channels": 256, "out_channels": 256, "input_height": 14, "input_width": 14,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.4.layer.2",
        "in_channels": 256, "out_channels": 1024, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.5.layer.0",
        "in_channels": 1024, "out_channels": 256, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.5.layer.1",
        "in_channels": 256, "out_channels": 256, "input_height": 14, "input_width": 14,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.2.layers.5.layer.2",
        "in_channels": 256, "out_channels": 1024, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.3.layers.0.layer.0",
        "in_channels": 1024, "out_channels": 512, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.3.layers.0.layer.1",
        "in_channels": 512, "out_channels": 512, "input_height": 14, "input_width": 14,
        "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.3.layers.0.layer.2",
        "in_channels": 512, "out_channels": 2048, "input_height": 7, "input_width": 7,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.3.layers.0.shortcut",
        "in_channels": 1024, "out_channels": 2048, "input_height": 14, "input_width": 14,
        "kernel_size": [1, 1], "stride": [2, 2], "padding": [0, 0, 0, 0],
        "deallocate_activation": False,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.3.layers.1.layer.0",
        "in_channels": 2048, "out_channels": 512, "input_height": 7, "input_width": 7,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.3.layers.1.layer.1",
        "in_channels": 512, "out_channels": 512, "input_height": 7, "input_width": 7,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.3.layers.1.layer.2",
        "in_channels": 512, "out_channels": 2048, "input_height": 7, "input_width": 7,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.3.layers.2.layer.0",
        "in_channels": 2048, "out_channels": 512, "input_height": 7, "input_width": 7,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.3.layers.2.layer.1",
        "in_channels": 512, "out_channels": 512, "input_height": 7, "input_width": 7,
        "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1, 1, 1],
        "activation": "relu", "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
    {
        "prefix": "resnet.encoder.stages.3.layers.2.layer.2",
        "in_channels": 512, "out_channels": 2048, "input_height": 7, "input_width": 7,
        "kernel_size": [1, 1], "stride": [1, 1], "padding": [0, 0, 0, 0],
        "deallocate_activation": True,
        "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    },
]


def run_consteval(weights, device):
    for conv_params in CONV_LAYER_PARAMS:
        prefix = conv_params["prefix"]

        running_var = ttnn.typecast(weights[f"{prefix}.normalization.running_var"], ttnn.DataType.FLOAT32, memory_config=None)
        bn_weight = ttnn.typecast(weights[f"{prefix}.normalization.weight"], ttnn.DataType.FLOAT32, memory_config=None)
        conv_weight = ttnn.typecast(weights[f"{prefix}.convolution.weight"], ttnn.DataType.FLOAT32, memory_config=None)
        running_mean = ttnn.typecast(weights[f"{prefix}.normalization.running_mean"], ttnn.DataType.FLOAT32, memory_config=None)
        bn_bias = ttnn.typecast(weights[f"{prefix}.normalization.bias"], ttnn.DataType.FLOAT32, memory_config=None)

        folded_weight, folded_bias = fold_bn_into_conv(running_var, bn_weight, conv_weight, running_mean, bn_bias)

        prepared_weight, prepared_bias = prepare_conv_weight_and_bias(folded_weight, folded_bias, conv_params, device)

        weights[f"{prefix}.convolution.weight"] = prepared_weight
        weights[f"{prefix}.convolution.bias"] = prepared_bias

    weights["classifier.1.bias"] = prepare_classifier_bias(weights, device)
    weights["classifier.1.weight"] = prepare_classifier_weight(weights, device)

    return weights
