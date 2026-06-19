import ttnn
from consteval import consteval__main

ce_cache__main = {}


def _main(device, activations, weights):
    global ce_cache__main
    ce_cache__main = consteval__main(ce_cache__main, weights, device)
    args_0 = activations[0]
    var_0 = ce_cache__main["main_const_eval_0"]
    utils_DeviceGetter_get_device_0 = device
    ttnn_to_layout_0 = ttnn.to_layout(
        args_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(args_0, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_0,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_permute_0,
        [1, 1, 401408, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_reshape_0,
        weight_tensor=var_0[0],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[1],
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
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(2, 9)),
                    ]
                ),
                [992, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_max_pool2d_0 = ttnn.max_pool2d(
        ttnn_conv2d_0,
        8,
        112,
        112,
        64,
        [3, 3],
        [2, 2],
        [1, 1],
        [1, 1],
        ceil_mode=False,
        memory_config=ttnn.MemoryConfig(
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
        applied_shard_scheme=None,
        reallocate_halo_output=False,
        config_tensor_in_dram=True,
    )
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_conv2d_1 = ttnn.conv2d(
        input_tensor=ttnn_max_pool2d_0,
        weight_tensor=var_0[2],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[3],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_2 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_1,
        weight_tensor=var_0[4],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[5],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_3 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_2,
        weight_tensor=var_0[6],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[7],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_to_memory_config_0 = ttnn.to_memory_config(
        ttnn_max_pool2d_0,
        ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_max_pool2d_0, False)
    ttnn_conv2d_4 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_0,
        weight_tensor=var_0[8],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[9],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_0 = ttnn.add(
        ttnn_conv2d_3,
        ttnn_conv2d_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_4, False)
    ttnn.deallocate(ttnn_conv2d_3, False)
    ttnn_relu_0 = ttnn.relu(
        ttnn_add_0,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_conv2d_5 = ttnn.conv2d(
        input_tensor=ttnn_relu_0,
        weight_tensor=var_0[10],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[11],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_6 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_5,
        weight_tensor=var_0[12],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[13],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_7 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_6,
        weight_tensor=var_0[14],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[15],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_1 = ttnn.add(
        ttnn_conv2d_7,
        ttnn_relu_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_7, False)
    ttnn.deallocate(ttnn_relu_0, False)
    ttnn_relu_1 = ttnn.relu(
        ttnn_add_1,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_1, False)
    ttnn_conv2d_8 = ttnn.conv2d(
        input_tensor=ttnn_relu_1,
        weight_tensor=var_0[16],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[17],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_9 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_8,
        weight_tensor=var_0[18],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[19],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_10 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_9,
        weight_tensor=var_0[20],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[21],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_2 = ttnn.add(
        ttnn_conv2d_10,
        ttnn_relu_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_10, False)
    ttnn.deallocate(ttnn_relu_1, False)
    ttnn_relu_2 = ttnn.relu(
        ttnn_add_2,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_2, False)
    ttnn_conv2d_11 = ttnn.conv2d(
        input_tensor=ttnn_relu_2,
        weight_tensor=var_0[22],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[23],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_12 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_11,
        weight_tensor=var_0[24],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[25],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_13 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_12,
        weight_tensor=var_0[26],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[27],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_14 = ttnn.conv2d(
        input_tensor=ttnn_relu_2,
        weight_tensor=var_0[28],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[29],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_relu_2, False)
    ttnn_add_3 = ttnn.add(
        ttnn_conv2d_13,
        ttnn_conv2d_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_14, False)
    ttnn.deallocate(ttnn_conv2d_13, False)
    ttnn_relu_3 = ttnn.relu(
        ttnn_add_3,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_3, False)
    ttnn_conv2d_15 = ttnn.conv2d(
        input_tensor=ttnn_relu_3,
        weight_tensor=var_0[30],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[31],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_16 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_15,
        weight_tensor=var_0[32],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[33],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_17 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_16,
        weight_tensor=var_0[34],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[35],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_4 = ttnn.add(
        ttnn_conv2d_17,
        ttnn_relu_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_17, False)
    ttnn.deallocate(ttnn_relu_3, False)
    ttnn_relu_4 = ttnn.relu(
        ttnn_add_4,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_4, False)
    ttnn_conv2d_18 = ttnn.conv2d(
        input_tensor=ttnn_relu_4,
        weight_tensor=var_0[36],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[37],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_19 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_18,
        weight_tensor=var_0[38],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[39],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_20 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_19,
        weight_tensor=var_0[40],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[41],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_5 = ttnn.add(
        ttnn_conv2d_20,
        ttnn_relu_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_20, False)
    ttnn.deallocate(ttnn_relu_4, False)
    ttnn_relu_5 = ttnn.relu(
        ttnn_add_5,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_5, False)
    ttnn_conv2d_21 = ttnn.conv2d(
        input_tensor=ttnn_relu_5,
        weight_tensor=var_0[42],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[43],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_22 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_21,
        weight_tensor=var_0[44],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[45],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_23 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_22,
        weight_tensor=var_0[46],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[47],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_6 = ttnn.add(
        ttnn_conv2d_23,
        ttnn_relu_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_23, False)
    ttnn.deallocate(ttnn_relu_5, False)
    ttnn_relu_6 = ttnn.relu(
        ttnn_add_6,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_6, False)
    ttnn_conv2d_24 = ttnn.conv2d(
        input_tensor=ttnn_relu_6,
        weight_tensor=var_0[48],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[49],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_25 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_24,
        weight_tensor=var_0[50],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[51],
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
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 3)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(4, 4)),
                    ]
                ),
                [32, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_1 = ttnn.to_memory_config(
        ttnn_conv2d_25,
        ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_25, False)
    ttnn_conv2d_26 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_1,
        weight_tensor=var_0[52],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[53],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_to_memory_config_2 = ttnn.to_memory_config(
        ttnn_relu_6,
        ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_relu_6, False)
    ttnn_conv2d_27 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_2,
        weight_tensor=var_0[54],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[55],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_7 = ttnn.add(
        ttnn_conv2d_26,
        ttnn_conv2d_27,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_27, False)
    ttnn.deallocate(ttnn_conv2d_26, False)
    ttnn_relu_7 = ttnn.relu(
        ttnn_add_7,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_7, False)
    ttnn_to_memory_config_3 = ttnn.to_memory_config(
        ttnn_relu_7,
        ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_28 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_3,
        weight_tensor=var_0[56],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[57],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_29 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_28,
        weight_tensor=var_0[58],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[59],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_30 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_29,
        weight_tensor=var_0[60],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[61],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_8 = ttnn.add(
        ttnn_conv2d_30,
        ttnn_relu_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_30, False)
    ttnn.deallocate(ttnn_relu_7, False)
    ttnn_relu_8 = ttnn.relu(
        ttnn_add_8,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_8, False)
    ttnn_to_memory_config_4 = ttnn.to_memory_config(
        ttnn_relu_8,
        ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_31 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_4,
        weight_tensor=var_0[62],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[63],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_32 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_31,
        weight_tensor=var_0[64],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[65],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_33 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_32,
        weight_tensor=var_0[66],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[67],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_9 = ttnn.add(
        ttnn_conv2d_33,
        ttnn_relu_8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_33, False)
    ttnn.deallocate(ttnn_relu_8, False)
    ttnn_relu_9 = ttnn.relu(
        ttnn_add_9,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_9, False)
    ttnn_to_memory_config_5 = ttnn.to_memory_config(
        ttnn_relu_9,
        ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_34 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_5,
        weight_tensor=var_0[68],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[69],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_35 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_34,
        weight_tensor=var_0[70],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[71],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_36 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_35,
        weight_tensor=var_0[72],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[73],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_10 = ttnn.add(
        ttnn_conv2d_36,
        ttnn_relu_9,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_36, False)
    ttnn.deallocate(ttnn_relu_9, False)
    ttnn_relu_10 = ttnn.relu(
        ttnn_add_10,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_10, False)
    ttnn_to_memory_config_6 = ttnn.to_memory_config(
        ttnn_relu_10,
        ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_37 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_6,
        weight_tensor=var_0[74],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[75],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_38 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_37,
        weight_tensor=var_0[76],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[77],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_39 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_38,
        weight_tensor=var_0[78],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[79],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_11 = ttnn.add(
        ttnn_conv2d_39,
        ttnn_relu_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_39, False)
    ttnn.deallocate(ttnn_relu_10, False)
    ttnn_relu_11 = ttnn.relu(
        ttnn_add_11,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_11, False)
    ttnn_to_memory_config_7 = ttnn.to_memory_config(
        ttnn_relu_11,
        ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_40 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_7,
        weight_tensor=var_0[80],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[81],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_41 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_40,
        weight_tensor=var_0[82],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[83],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_42 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_41,
        weight_tensor=var_0[84],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[85],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_12 = ttnn.add(
        ttnn_conv2d_42,
        ttnn_relu_11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_42, False)
    ttnn.deallocate(ttnn_relu_11, False)
    ttnn_relu_12 = ttnn.relu(
        ttnn_add_12,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_12, False)
    ttnn_to_memory_config_8 = ttnn.to_memory_config(
        ttnn_relu_12,
        ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_43 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_8,
        weight_tensor=var_0[86],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[87],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_44 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_43,
        weight_tensor=var_0[88],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[89],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_45 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_44,
        weight_tensor=var_0[90],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[91],
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
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_conv2d_46 = ttnn.conv2d(
        input_tensor=ttnn_relu_12,
        weight_tensor=var_0[92],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[93],
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
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_relu_12, False)
    ttnn_add_13 = ttnn.add(
        ttnn_conv2d_45,
        ttnn_conv2d_46,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_conv2d_46, False)
    ttnn.deallocate(ttnn_conv2d_45, False)
    ttnn_relu_13 = ttnn.relu(
        ttnn_add_13,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_add_13, False)
    ttnn_to_memory_config_9 = ttnn.to_memory_config(
        ttnn_relu_13,
        ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_47 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_9,
        weight_tensor=var_0[94],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[95],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_48 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_47,
        weight_tensor=var_0[96],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[97],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_49 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_48,
        weight_tensor=var_0[98],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[99],
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
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_14 = ttnn.add(
        ttnn_conv2d_49,
        ttnn_relu_13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_conv2d_49, False)
    ttnn.deallocate(ttnn_relu_13, False)
    ttnn_relu_14 = ttnn.relu(
        ttnn_add_14,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_add_14, False)
    ttnn_to_memory_config_10 = ttnn.to_memory_config(
        ttnn_relu_14,
        ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_50 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_10,
        weight_tensor=var_0[100],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[101],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_51 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_50,
        weight_tensor=var_0[102],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[103],
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
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_52 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_51,
        weight_tensor=var_0[104],
        device=utils_DeviceGetter_get_device_0,
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
        dtype=ttnn.DataType.BFLOAT16,
        bias_tensor=var_0[105],
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
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_15 = ttnn.add(
        ttnn_conv2d_52,
        ttnn_relu_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_conv2d_52, False)
    ttnn.deallocate(ttnn_relu_14, False)
    ttnn_relu_15 = ttnn.relu(
        ttnn_add_15,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_add_15, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_relu_15,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [64, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_relu_15, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_typecast_0,
        [8, 1, 49, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_mean_0 = ttnn.mean(
        ttnn_reshape_1,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7))]
                ),
                [32, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_mean_0,
        [8, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_mean_0, False)
    ttnn_linear_0 = ttnn.linear(
        ttnn_reshape_2,
        ce_cache__main["main_const_eval_2"],
        bias=ce_cache__main["main_const_eval_1"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                    ]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        dtype=ttnn.DataType.FLOAT32,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 3),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=0,
            untilize_out=False,
        ),
        activation=None,
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_linear_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(9, 2)),
                    ]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_linear_0, False)
    ttnn_to_memory_config_11 = ttnn.to_memory_config(
        ttnn_typecast_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    return [ttnn_to_memory_config_11]
