import torch
import ttnn
import utils
from consteval import consteval__main
from model_pt import run as run_golden

ce_cache__main = {}


def calculate_pcc(x, y):
    # This function calculates the PCC between two torch tensors

    # Assert both are torch tensors
    assert isinstance(x, torch.Tensor), "x must be a torch tensor"
    assert isinstance(y, torch.Tensor), "y must be a torch tensor"

    if x.shape != y.shape:
        raise ValueError(
            f"Shapes of x and y must be the same, but got {x.shape} and {y.shape}"
        )

    # Calculate PCC
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    return float("nan") if denom == 0 else ((vx @ vy) / denom).item()


def _main(activations, weights):
    global ce_cache__main
    ce_cache__main = consteval__main(ce_cache__main, weights)
    args_0 = activations[0]
    var_0 = ce_cache__main["main_const_eval_0"]
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
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
        [1, 1, 50176, 3],
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
        batch_size=1,
        input_height=224,
        input_width=224,
        kernel_size=[7, 7],
        stride=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        groups=1,
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
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_max_pool2d_0 = ttnn.max_pool2d(
        ttnn_conv2d_0,
        1,
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
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 7)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(9, 8)),
                    ]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        applied_shard_scheme=None,
        reallocate_halo_output=False,
        config_tensor_in_dram=True,
    )
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_to_memory_config_0 = ttnn.to_memory_config(
        ttnn_max_pool2d_0,
        ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_1 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_0,
        weight_tensor=var_0[2],
        device=utils_DeviceGetter_get_device_0,
        in_channels=64,
        out_channels=64,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[3],
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
                [32, 64],
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
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
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
                [32, 64],
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
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                [32, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_conv2d_4 = ttnn.conv2d(
        input_tensor=ttnn_max_pool2d_0,
        weight_tensor=var_0[8],
        device=utils_DeviceGetter_get_device_0,
        in_channels=64,
        out_channels=256,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[9],
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
                [32, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_max_pool2d_0, False)
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
                [32, 256],
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
                [32, 256],
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
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                [32, 64],
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
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
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
                [32, 64],
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
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                [32, 256],
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
                [32, 256],
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
                [32, 256],
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
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                [32, 64],
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
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
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
                [32, 64],
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
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                [32, 256],
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
                [32, 256],
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
                [32, 256],
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
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                [32, 128],
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
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
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
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(2, 2)),
                    ]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_1 = ttnn.to_memory_config(
        ttnn_conv2d_12,
        ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_12, False)
    ttnn_conv2d_13 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_1,
        weight_tensor=var_0[26],
        device=utils_DeviceGetter_get_device_0,
        in_channels=128,
        out_channels=512,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[27],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_to_memory_config_2 = ttnn.to_memory_config(
        ttnn_relu_2,
        ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_relu_2, False)
    ttnn_conv2d_14 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_2,
        weight_tensor=var_0[28],
        device=utils_DeviceGetter_get_device_0,
        in_channels=256,
        out_channels=512,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[29],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_3 = ttnn.add(
        ttnn_conv2d_13,
        ttnn_conv2d_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_14, False)
    ttnn.deallocate(ttnn_conv2d_13, False)
    ttnn_relu_3 = ttnn.relu(
        ttnn_add_3,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_3, False)
    ttnn_conv2d_15 = ttnn.conv2d(
        input_tensor=ttnn_relu_3,
        weight_tensor=var_0[30],
        device=utils_DeviceGetter_get_device_0,
        in_channels=512,
        out_channels=128,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[31],
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
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
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[33],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
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
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[35],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_4 = ttnn.add(
        ttnn_conv2d_17,
        ttnn_relu_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_17, False)
    ttnn.deallocate(ttnn_relu_3, False)
    ttnn_relu_4 = ttnn.relu(
        ttnn_add_4,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_4, False)
    ttnn_conv2d_18 = ttnn.conv2d(
        input_tensor=ttnn_relu_4,
        weight_tensor=var_0[36],
        device=utils_DeviceGetter_get_device_0,
        in_channels=512,
        out_channels=128,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[37],
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
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
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[39],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
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
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[41],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_5 = ttnn.add(
        ttnn_conv2d_20,
        ttnn_relu_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_20, False)
    ttnn.deallocate(ttnn_relu_4, False)
    ttnn_relu_5 = ttnn.relu(
        ttnn_add_5,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_5, False)
    ttnn_conv2d_21 = ttnn.conv2d(
        input_tensor=ttnn_relu_5,
        weight_tensor=var_0[42],
        device=utils_DeviceGetter_get_device_0,
        in_channels=512,
        out_channels=128,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[43],
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
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
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[45],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 8))]
                ),
                [96, 32],
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
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[47],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_add_6 = ttnn.add(
        ttnn_conv2d_23,
        ttnn_relu_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_23, False)
    ttnn.deallocate(ttnn_relu_5, False)
    ttnn_relu_6 = ttnn.relu(
        ttnn_add_6,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_6, False)
    ttnn_conv2d_24 = ttnn.conv2d(
        input_tensor=ttnn_relu_6,
        weight_tensor=var_0[48],
        device=utils_DeviceGetter_get_device_0,
        in_channels=512,
        out_channels=256,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[49],
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 8))]
                ),
                [96, 32],
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
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[51],
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
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_conv2d_26 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_25,
        weight_tensor=var_0[52],
        device=utils_DeviceGetter_get_device_0,
        in_channels=256,
        out_channels=1024,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn_conv2d_27 = ttnn.conv2d(
        input_tensor=ttnn_relu_6,
        weight_tensor=var_0[54],
        device=utils_DeviceGetter_get_device_0,
        in_channels=512,
        out_channels=1024,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[55],
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
                [32, 96],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_relu_6, False)
    ttnn_add_7 = ttnn.add(
        ttnn_conv2d_26,
        ttnn_conv2d_27,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 32],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 6))]
                ),
                [32, 96],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 128],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]
                ),
                [32, 64],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))]
                ),
                [32, 64],
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
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1))]
                ),
                [32, 192],
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
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1))]
                ),
                [32, 192],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1))]
                ),
                [32, 192],
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
                    [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 1))]
                ),
                [32, 192],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_add_13, False)
    ttnn_to_memory_config_9 = ttnn.to_memory_config(
        ttnn_relu_13,
        ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_47 = ttnn.conv2d(
        input_tensor=ttnn_to_memory_config_9,
        weight_tensor=var_0[94],
        device=utils_DeviceGetter_get_device_0,
        in_channels=2048,
        out_channels=512,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[95],
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_48 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_47,
        weight_tensor=var_0[96],
        device=utils_DeviceGetter_get_device_0,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[97],
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_49 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_48,
        weight_tensor=var_0[98],
        device=utils_DeviceGetter_get_device_0,
        in_channels=512,
        out_channels=2048,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[99],
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_14 = ttnn.add(
        ttnn_conv2d_49,
        ttnn_relu_13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_49, False)
    ttnn.deallocate(ttnn_relu_13, False)
    ttnn_relu_14 = ttnn.relu(
        ttnn_add_14,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_14, False)
    ttnn_conv2d_50 = ttnn.conv2d(
        input_tensor=ttnn_relu_14,
        weight_tensor=var_0[100],
        device=utils_DeviceGetter_get_device_0,
        in_channels=2048,
        out_channels=512,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[101],
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_51 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_50,
        weight_tensor=var_0[102],
        device=utils_DeviceGetter_get_device_0,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[103],
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_conv2d_52 = ttnn.conv2d(
        input_tensor=ttnn_conv2d_51,
        weight_tensor=var_0[104],
        device=utils_DeviceGetter_get_device_0,
        in_channels=512,
        out_channels=2048,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=var_0[105],
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn_add_15 = ttnn.add(
        ttnn_conv2d_52,
        ttnn_relu_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_conv2d_52, False)
    ttnn.deallocate(ttnn_relu_14, False)
    ttnn_relu_15 = ttnn.relu(
        ttnn_add_15,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_add_15, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_relu_15,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
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
    )
    ttnn.deallocate(ttnn_relu_15, False)
    ttnn_mean_0 = ttnn.mean(
        ttnn_typecast_0,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 4)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(8, 5)),
                    ]
                ),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        compute_kernel_config=None,
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_to_memory_config_10 = ttnn.to_memory_config(
        ttnn_mean_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_mean_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_to_memory_config_10,
        [1, 2048],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
    )
    ttnn.deallocate(ttnn_to_memory_config_10, False)
    ttnn_linear_0 = ttnn.linear(
        ttnn_reshape_1,
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
    ttnn.deallocate(ttnn_reshape_1, False)
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


def load_activations_for__main():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg27.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [utils_load_tensor_0]


_main_weights = {}


def load_weights_for__main():
    global _main_weights
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["classifier.1.bias"] = utils_load_tensor_1
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["classifier.1.weight"] = utils_load_tensor_2
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.0.shortcut.normalization.running_var"
    ] = utils_load_tensor_3
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.0.shortcut.normalization.running_mean"
    ] = utils_load_tensor_4
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.shortcut.normalization.bias"] = (
        utils_load_tensor_5
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.shortcut.normalization.weight"] = (
        utils_load_tensor_6
    )
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.shortcut.convolution.weight"] = (
        utils_load_tensor_7
    )
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg7.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.0.shortcut.normalization.running_var"
    ] = utils_load_tensor_8
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg8.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.0.shortcut.normalization.running_mean"
    ] = utils_load_tensor_9
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg9.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.shortcut.normalization.bias"] = (
        utils_load_tensor_10
    )
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg10.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.shortcut.normalization.weight"] = (
        utils_load_tensor_11
    )
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg11.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.shortcut.convolution.weight"] = (
        utils_load_tensor_12
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg12.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.0.shortcut.normalization.running_var"
    ] = utils_load_tensor_13
    utils_load_tensor_14 = utils.load_tensor(
        "./tensors/arg13.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.0.shortcut.normalization.running_mean"
    ] = utils_load_tensor_14
    utils_load_tensor_15 = utils.load_tensor(
        "./tensors/arg14.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.shortcut.normalization.bias"] = (
        utils_load_tensor_15
    )
    utils_load_tensor_16 = utils.load_tensor(
        "./tensors/arg15.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.shortcut.normalization.weight"] = (
        utils_load_tensor_16
    )
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg16.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.shortcut.convolution.weight"] = (
        utils_load_tensor_17
    )
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.0.shortcut.normalization.running_var"
    ] = utils_load_tensor_18
    utils_load_tensor_19 = utils.load_tensor(
        "./tensors/arg18.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.0.shortcut.normalization.running_mean"
    ] = utils_load_tensor_19
    utils_load_tensor_20 = utils.load_tensor(
        "./tensors/arg19.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.shortcut.normalization.bias"] = (
        utils_load_tensor_20
    )
    utils_load_tensor_21 = utils.load_tensor(
        "./tensors/arg20.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.shortcut.normalization.weight"] = (
        utils_load_tensor_21
    )
    utils_load_tensor_22 = utils.load_tensor(
        "./tensors/arg21.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.shortcut.convolution.weight"] = (
        utils_load_tensor_22
    )
    utils_load_tensor_23 = utils.load_tensor(
        "./tensors/arg22.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.embedder.embedder.normalization.running_var"] = (
        utils_load_tensor_23
    )
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg23.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.embedder.embedder.normalization.running_mean"] = (
        utils_load_tensor_24
    )
    utils_load_tensor_25 = utils.load_tensor(
        "./tensors/arg24.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.embedder.embedder.normalization.bias"] = utils_load_tensor_25
    utils_load_tensor_26 = utils.load_tensor(
        "./tensors/arg25.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.embedder.embedder.normalization.weight"] = (
        utils_load_tensor_26
    )
    utils_load_tensor_27 = utils.load_tensor(
        "./tensors/arg26.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.embedder.embedder.convolution.weight"] = utils_load_tensor_27
    utils_load_tensor_28 = utils.load_tensor(
        "./tensors/arg28.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.0.layer.2.normalization.running_var"
    ] = utils_load_tensor_28
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.0.layer.2.normalization.running_mean"
    ] = utils_load_tensor_29
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg30.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.layer.2.normalization.bias"] = (
        utils_load_tensor_30
    )
    utils_load_tensor_31 = utils.load_tensor(
        "./tensors/arg31.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.layer.2.normalization.weight"] = (
        utils_load_tensor_31
    )
    utils_load_tensor_32 = utils.load_tensor(
        "./tensors/arg32.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.layer.2.convolution.weight"] = (
        utils_load_tensor_32
    )
    utils_load_tensor_33 = utils.load_tensor(
        "./tensors/arg33.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.0.layer.1.normalization.running_var"
    ] = utils_load_tensor_33
    utils_load_tensor_34 = utils.load_tensor(
        "./tensors/arg34.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.0.layer.1.normalization.running_mean"
    ] = utils_load_tensor_34
    utils_load_tensor_35 = utils.load_tensor(
        "./tensors/arg35.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.layer.1.normalization.bias"] = (
        utils_load_tensor_35
    )
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg36.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.layer.1.normalization.weight"] = (
        utils_load_tensor_36
    )
    utils_load_tensor_37 = utils.load_tensor(
        "./tensors/arg37.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.layer.1.convolution.weight"] = (
        utils_load_tensor_37
    )
    utils_load_tensor_38 = utils.load_tensor(
        "./tensors/arg38.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.0.layer.0.normalization.running_var"
    ] = utils_load_tensor_38
    utils_load_tensor_39 = utils.load_tensor(
        "./tensors/arg39.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.0.layer.0.normalization.running_mean"
    ] = utils_load_tensor_39
    utils_load_tensor_40 = utils.load_tensor(
        "./tensors/arg40.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.layer.0.normalization.bias"] = (
        utils_load_tensor_40
    )
    utils_load_tensor_41 = utils.load_tensor(
        "./tensors/arg41.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.layer.0.normalization.weight"] = (
        utils_load_tensor_41
    )
    utils_load_tensor_42 = utils.load_tensor(
        "./tensors/arg42.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.0.layer.0.convolution.weight"] = (
        utils_load_tensor_42
    )
    utils_load_tensor_43 = utils.load_tensor(
        "./tensors/arg43.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.1.layer.2.normalization.running_var"
    ] = utils_load_tensor_43
    utils_load_tensor_44 = utils.load_tensor(
        "./tensors/arg44.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.1.layer.2.normalization.running_mean"
    ] = utils_load_tensor_44
    utils_load_tensor_45 = utils.load_tensor(
        "./tensors/arg45.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.1.layer.2.normalization.bias"] = (
        utils_load_tensor_45
    )
    utils_load_tensor_46 = utils.load_tensor(
        "./tensors/arg46.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.1.layer.2.normalization.weight"] = (
        utils_load_tensor_46
    )
    utils_load_tensor_47 = utils.load_tensor(
        "./tensors/arg47.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.1.layer.2.convolution.weight"] = (
        utils_load_tensor_47
    )
    utils_load_tensor_48 = utils.load_tensor(
        "./tensors/arg48.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.1.layer.1.normalization.running_var"
    ] = utils_load_tensor_48
    utils_load_tensor_49 = utils.load_tensor(
        "./tensors/arg49.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.1.layer.1.normalization.running_mean"
    ] = utils_load_tensor_49
    utils_load_tensor_50 = utils.load_tensor(
        "./tensors/arg50.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.1.layer.1.normalization.bias"] = (
        utils_load_tensor_50
    )
    utils_load_tensor_51 = utils.load_tensor(
        "./tensors/arg51.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.1.layer.1.normalization.weight"] = (
        utils_load_tensor_51
    )
    utils_load_tensor_52 = utils.load_tensor(
        "./tensors/arg52.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.1.layer.1.convolution.weight"] = (
        utils_load_tensor_52
    )
    utils_load_tensor_53 = utils.load_tensor(
        "./tensors/arg53.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.1.layer.0.normalization.running_var"
    ] = utils_load_tensor_53
    utils_load_tensor_54 = utils.load_tensor(
        "./tensors/arg54.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.1.layer.0.normalization.running_mean"
    ] = utils_load_tensor_54
    utils_load_tensor_55 = utils.load_tensor(
        "./tensors/arg55.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.1.layer.0.normalization.bias"] = (
        utils_load_tensor_55
    )
    utils_load_tensor_56 = utils.load_tensor(
        "./tensors/arg56.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.1.layer.0.normalization.weight"] = (
        utils_load_tensor_56
    )
    utils_load_tensor_57 = utils.load_tensor(
        "./tensors/arg57.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.1.layer.0.convolution.weight"] = (
        utils_load_tensor_57
    )
    utils_load_tensor_58 = utils.load_tensor(
        "./tensors/arg58.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.2.layer.2.normalization.running_var"
    ] = utils_load_tensor_58
    utils_load_tensor_59 = utils.load_tensor(
        "./tensors/arg59.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.2.layer.2.normalization.running_mean"
    ] = utils_load_tensor_59
    utils_load_tensor_60 = utils.load_tensor(
        "./tensors/arg60.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.2.layer.2.normalization.bias"] = (
        utils_load_tensor_60
    )
    utils_load_tensor_61 = utils.load_tensor(
        "./tensors/arg61.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.2.layer.2.normalization.weight"] = (
        utils_load_tensor_61
    )
    utils_load_tensor_62 = utils.load_tensor(
        "./tensors/arg62.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.2.layer.2.convolution.weight"] = (
        utils_load_tensor_62
    )
    utils_load_tensor_63 = utils.load_tensor(
        "./tensors/arg63.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.2.layer.1.normalization.running_var"
    ] = utils_load_tensor_63
    utils_load_tensor_64 = utils.load_tensor(
        "./tensors/arg64.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.2.layer.1.normalization.running_mean"
    ] = utils_load_tensor_64
    utils_load_tensor_65 = utils.load_tensor(
        "./tensors/arg65.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.2.layer.1.normalization.bias"] = (
        utils_load_tensor_65
    )
    utils_load_tensor_66 = utils.load_tensor(
        "./tensors/arg66.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.2.layer.1.normalization.weight"] = (
        utils_load_tensor_66
    )
    utils_load_tensor_67 = utils.load_tensor(
        "./tensors/arg67.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.2.layer.1.convolution.weight"] = (
        utils_load_tensor_67
    )
    utils_load_tensor_68 = utils.load_tensor(
        "./tensors/arg68.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.2.layer.0.normalization.running_var"
    ] = utils_load_tensor_68
    utils_load_tensor_69 = utils.load_tensor(
        "./tensors/arg69.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.0.layers.2.layer.0.normalization.running_mean"
    ] = utils_load_tensor_69
    utils_load_tensor_70 = utils.load_tensor(
        "./tensors/arg70.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.2.layer.0.normalization.bias"] = (
        utils_load_tensor_70
    )
    utils_load_tensor_71 = utils.load_tensor(
        "./tensors/arg71.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.2.layer.0.normalization.weight"] = (
        utils_load_tensor_71
    )
    utils_load_tensor_72 = utils.load_tensor(
        "./tensors/arg72.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.0.layers.2.layer.0.convolution.weight"] = (
        utils_load_tensor_72
    )
    utils_load_tensor_73 = utils.load_tensor(
        "./tensors/arg73.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.0.layer.2.normalization.running_var"
    ] = utils_load_tensor_73
    utils_load_tensor_74 = utils.load_tensor(
        "./tensors/arg74.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.0.layer.2.normalization.running_mean"
    ] = utils_load_tensor_74
    utils_load_tensor_75 = utils.load_tensor(
        "./tensors/arg75.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.layer.2.normalization.bias"] = (
        utils_load_tensor_75
    )
    utils_load_tensor_76 = utils.load_tensor(
        "./tensors/arg76.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.layer.2.normalization.weight"] = (
        utils_load_tensor_76
    )
    utils_load_tensor_77 = utils.load_tensor(
        "./tensors/arg77.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.layer.2.convolution.weight"] = (
        utils_load_tensor_77
    )
    utils_load_tensor_78 = utils.load_tensor(
        "./tensors/arg78.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.0.layer.1.normalization.running_var"
    ] = utils_load_tensor_78
    utils_load_tensor_79 = utils.load_tensor(
        "./tensors/arg79.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.0.layer.1.normalization.running_mean"
    ] = utils_load_tensor_79
    utils_load_tensor_80 = utils.load_tensor(
        "./tensors/arg80.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.layer.1.normalization.bias"] = (
        utils_load_tensor_80
    )
    utils_load_tensor_81 = utils.load_tensor(
        "./tensors/arg81.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.layer.1.normalization.weight"] = (
        utils_load_tensor_81
    )
    utils_load_tensor_82 = utils.load_tensor(
        "./tensors/arg82.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.layer.1.convolution.weight"] = (
        utils_load_tensor_82
    )
    utils_load_tensor_83 = utils.load_tensor(
        "./tensors/arg83.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.0.layer.0.normalization.running_var"
    ] = utils_load_tensor_83
    utils_load_tensor_84 = utils.load_tensor(
        "./tensors/arg84.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.0.layer.0.normalization.running_mean"
    ] = utils_load_tensor_84
    utils_load_tensor_85 = utils.load_tensor(
        "./tensors/arg85.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.layer.0.normalization.bias"] = (
        utils_load_tensor_85
    )
    utils_load_tensor_86 = utils.load_tensor(
        "./tensors/arg86.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.layer.0.normalization.weight"] = (
        utils_load_tensor_86
    )
    utils_load_tensor_87 = utils.load_tensor(
        "./tensors/arg87.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.0.layer.0.convolution.weight"] = (
        utils_load_tensor_87
    )
    utils_load_tensor_88 = utils.load_tensor(
        "./tensors/arg88.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.1.layer.2.normalization.running_var"
    ] = utils_load_tensor_88
    utils_load_tensor_89 = utils.load_tensor(
        "./tensors/arg89.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.1.layer.2.normalization.running_mean"
    ] = utils_load_tensor_89
    utils_load_tensor_90 = utils.load_tensor(
        "./tensors/arg90.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.1.layer.2.normalization.bias"] = (
        utils_load_tensor_90
    )
    utils_load_tensor_91 = utils.load_tensor(
        "./tensors/arg91.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.1.layer.2.normalization.weight"] = (
        utils_load_tensor_91
    )
    utils_load_tensor_92 = utils.load_tensor(
        "./tensors/arg92.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.1.layer.2.convolution.weight"] = (
        utils_load_tensor_92
    )
    utils_load_tensor_93 = utils.load_tensor(
        "./tensors/arg93.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.1.layer.1.normalization.running_var"
    ] = utils_load_tensor_93
    utils_load_tensor_94 = utils.load_tensor(
        "./tensors/arg94.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.1.layer.1.normalization.running_mean"
    ] = utils_load_tensor_94
    utils_load_tensor_95 = utils.load_tensor(
        "./tensors/arg95.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.1.layer.1.normalization.bias"] = (
        utils_load_tensor_95
    )
    utils_load_tensor_96 = utils.load_tensor(
        "./tensors/arg96.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.1.layer.1.normalization.weight"] = (
        utils_load_tensor_96
    )
    utils_load_tensor_97 = utils.load_tensor(
        "./tensors/arg97.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.1.layer.1.convolution.weight"] = (
        utils_load_tensor_97
    )
    utils_load_tensor_98 = utils.load_tensor(
        "./tensors/arg98.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.1.layer.0.normalization.running_var"
    ] = utils_load_tensor_98
    utils_load_tensor_99 = utils.load_tensor(
        "./tensors/arg99.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.1.layer.0.normalization.running_mean"
    ] = utils_load_tensor_99
    utils_load_tensor_100 = utils.load_tensor(
        "./tensors/arg100.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.1.layer.0.normalization.bias"] = (
        utils_load_tensor_100
    )
    utils_load_tensor_101 = utils.load_tensor(
        "./tensors/arg101.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.1.layer.0.normalization.weight"] = (
        utils_load_tensor_101
    )
    utils_load_tensor_102 = utils.load_tensor(
        "./tensors/arg102.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.1.layer.0.convolution.weight"] = (
        utils_load_tensor_102
    )
    utils_load_tensor_103 = utils.load_tensor(
        "./tensors/arg103.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.2.layer.2.normalization.running_var"
    ] = utils_load_tensor_103
    utils_load_tensor_104 = utils.load_tensor(
        "./tensors/arg104.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.2.layer.2.normalization.running_mean"
    ] = utils_load_tensor_104
    utils_load_tensor_105 = utils.load_tensor(
        "./tensors/arg105.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.2.layer.2.normalization.bias"] = (
        utils_load_tensor_105
    )
    utils_load_tensor_106 = utils.load_tensor(
        "./tensors/arg106.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.2.layer.2.normalization.weight"] = (
        utils_load_tensor_106
    )
    utils_load_tensor_107 = utils.load_tensor(
        "./tensors/arg107.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.2.layer.2.convolution.weight"] = (
        utils_load_tensor_107
    )
    utils_load_tensor_108 = utils.load_tensor(
        "./tensors/arg108.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.2.layer.1.normalization.running_var"
    ] = utils_load_tensor_108
    utils_load_tensor_109 = utils.load_tensor(
        "./tensors/arg109.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.2.layer.1.normalization.running_mean"
    ] = utils_load_tensor_109
    utils_load_tensor_110 = utils.load_tensor(
        "./tensors/arg110.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.2.layer.1.normalization.bias"] = (
        utils_load_tensor_110
    )
    utils_load_tensor_111 = utils.load_tensor(
        "./tensors/arg111.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.2.layer.1.normalization.weight"] = (
        utils_load_tensor_111
    )
    utils_load_tensor_112 = utils.load_tensor(
        "./tensors/arg112.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.2.layer.1.convolution.weight"] = (
        utils_load_tensor_112
    )
    utils_load_tensor_113 = utils.load_tensor(
        "./tensors/arg113.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.2.layer.0.normalization.running_var"
    ] = utils_load_tensor_113
    utils_load_tensor_114 = utils.load_tensor(
        "./tensors/arg114.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.2.layer.0.normalization.running_mean"
    ] = utils_load_tensor_114
    utils_load_tensor_115 = utils.load_tensor(
        "./tensors/arg115.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.2.layer.0.normalization.bias"] = (
        utils_load_tensor_115
    )
    utils_load_tensor_116 = utils.load_tensor(
        "./tensors/arg116.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.2.layer.0.normalization.weight"] = (
        utils_load_tensor_116
    )
    utils_load_tensor_117 = utils.load_tensor(
        "./tensors/arg117.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.2.layer.0.convolution.weight"] = (
        utils_load_tensor_117
    )
    utils_load_tensor_118 = utils.load_tensor(
        "./tensors/arg118.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.3.layer.2.normalization.running_var"
    ] = utils_load_tensor_118
    utils_load_tensor_119 = utils.load_tensor(
        "./tensors/arg119.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.3.layer.2.normalization.running_mean"
    ] = utils_load_tensor_119
    utils_load_tensor_120 = utils.load_tensor(
        "./tensors/arg120.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.3.layer.2.normalization.bias"] = (
        utils_load_tensor_120
    )
    utils_load_tensor_121 = utils.load_tensor(
        "./tensors/arg121.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.3.layer.2.normalization.weight"] = (
        utils_load_tensor_121
    )
    utils_load_tensor_122 = utils.load_tensor(
        "./tensors/arg122.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.3.layer.2.convolution.weight"] = (
        utils_load_tensor_122
    )
    utils_load_tensor_123 = utils.load_tensor(
        "./tensors/arg123.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.3.layer.1.normalization.running_var"
    ] = utils_load_tensor_123
    utils_load_tensor_124 = utils.load_tensor(
        "./tensors/arg124.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.3.layer.1.normalization.running_mean"
    ] = utils_load_tensor_124
    utils_load_tensor_125 = utils.load_tensor(
        "./tensors/arg125.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.3.layer.1.normalization.bias"] = (
        utils_load_tensor_125
    )
    utils_load_tensor_126 = utils.load_tensor(
        "./tensors/arg126.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.3.layer.1.normalization.weight"] = (
        utils_load_tensor_126
    )
    utils_load_tensor_127 = utils.load_tensor(
        "./tensors/arg127.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.3.layer.1.convolution.weight"] = (
        utils_load_tensor_127
    )
    utils_load_tensor_128 = utils.load_tensor(
        "./tensors/arg128.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.3.layer.0.normalization.running_var"
    ] = utils_load_tensor_128
    utils_load_tensor_129 = utils.load_tensor(
        "./tensors/arg129.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.1.layers.3.layer.0.normalization.running_mean"
    ] = utils_load_tensor_129
    utils_load_tensor_130 = utils.load_tensor(
        "./tensors/arg130.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.3.layer.0.normalization.bias"] = (
        utils_load_tensor_130
    )
    utils_load_tensor_131 = utils.load_tensor(
        "./tensors/arg131.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.3.layer.0.normalization.weight"] = (
        utils_load_tensor_131
    )
    utils_load_tensor_132 = utils.load_tensor(
        "./tensors/arg132.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.1.layers.3.layer.0.convolution.weight"] = (
        utils_load_tensor_132
    )
    utils_load_tensor_133 = utils.load_tensor(
        "./tensors/arg133.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.0.layer.2.normalization.running_var"
    ] = utils_load_tensor_133
    utils_load_tensor_134 = utils.load_tensor(
        "./tensors/arg134.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.0.layer.2.normalization.running_mean"
    ] = utils_load_tensor_134
    utils_load_tensor_135 = utils.load_tensor(
        "./tensors/arg135.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.layer.2.normalization.bias"] = (
        utils_load_tensor_135
    )
    utils_load_tensor_136 = utils.load_tensor(
        "./tensors/arg136.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.layer.2.normalization.weight"] = (
        utils_load_tensor_136
    )
    utils_load_tensor_137 = utils.load_tensor(
        "./tensors/arg137.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.layer.2.convolution.weight"] = (
        utils_load_tensor_137
    )
    utils_load_tensor_138 = utils.load_tensor(
        "./tensors/arg138.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.0.layer.1.normalization.running_var"
    ] = utils_load_tensor_138
    utils_load_tensor_139 = utils.load_tensor(
        "./tensors/arg139.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.0.layer.1.normalization.running_mean"
    ] = utils_load_tensor_139
    utils_load_tensor_140 = utils.load_tensor(
        "./tensors/arg140.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.layer.1.normalization.bias"] = (
        utils_load_tensor_140
    )
    utils_load_tensor_141 = utils.load_tensor(
        "./tensors/arg141.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.layer.1.normalization.weight"] = (
        utils_load_tensor_141
    )
    utils_load_tensor_142 = utils.load_tensor(
        "./tensors/arg142.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.layer.1.convolution.weight"] = (
        utils_load_tensor_142
    )
    utils_load_tensor_143 = utils.load_tensor(
        "./tensors/arg143.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.0.layer.0.normalization.running_var"
    ] = utils_load_tensor_143
    utils_load_tensor_144 = utils.load_tensor(
        "./tensors/arg144.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.0.layer.0.normalization.running_mean"
    ] = utils_load_tensor_144
    utils_load_tensor_145 = utils.load_tensor(
        "./tensors/arg145.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.layer.0.normalization.bias"] = (
        utils_load_tensor_145
    )
    utils_load_tensor_146 = utils.load_tensor(
        "./tensors/arg146.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.layer.0.normalization.weight"] = (
        utils_load_tensor_146
    )
    utils_load_tensor_147 = utils.load_tensor(
        "./tensors/arg147.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.0.layer.0.convolution.weight"] = (
        utils_load_tensor_147
    )
    utils_load_tensor_148 = utils.load_tensor(
        "./tensors/arg148.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.1.layer.2.normalization.running_var"
    ] = utils_load_tensor_148
    utils_load_tensor_149 = utils.load_tensor(
        "./tensors/arg149.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.1.layer.2.normalization.running_mean"
    ] = utils_load_tensor_149
    utils_load_tensor_150 = utils.load_tensor(
        "./tensors/arg150.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.1.layer.2.normalization.bias"] = (
        utils_load_tensor_150
    )
    utils_load_tensor_151 = utils.load_tensor(
        "./tensors/arg151.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.1.layer.2.normalization.weight"] = (
        utils_load_tensor_151
    )
    utils_load_tensor_152 = utils.load_tensor(
        "./tensors/arg152.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.1.layer.2.convolution.weight"] = (
        utils_load_tensor_152
    )
    utils_load_tensor_153 = utils.load_tensor(
        "./tensors/arg153.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.running_var"
    ] = utils_load_tensor_153
    utils_load_tensor_154 = utils.load_tensor(
        "./tensors/arg154.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.1.layer.1.normalization.running_mean"
    ] = utils_load_tensor_154
    utils_load_tensor_155 = utils.load_tensor(
        "./tensors/arg155.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.1.layer.1.normalization.bias"] = (
        utils_load_tensor_155
    )
    utils_load_tensor_156 = utils.load_tensor(
        "./tensors/arg156.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.1.layer.1.normalization.weight"] = (
        utils_load_tensor_156
    )
    utils_load_tensor_157 = utils.load_tensor(
        "./tensors/arg157.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.1.layer.1.convolution.weight"] = (
        utils_load_tensor_157
    )
    utils_load_tensor_158 = utils.load_tensor(
        "./tensors/arg158.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.1.layer.0.normalization.running_var"
    ] = utils_load_tensor_158
    utils_load_tensor_159 = utils.load_tensor(
        "./tensors/arg159.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.1.layer.0.normalization.running_mean"
    ] = utils_load_tensor_159
    utils_load_tensor_160 = utils.load_tensor(
        "./tensors/arg160.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.1.layer.0.normalization.bias"] = (
        utils_load_tensor_160
    )
    utils_load_tensor_161 = utils.load_tensor(
        "./tensors/arg161.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.1.layer.0.normalization.weight"] = (
        utils_load_tensor_161
    )
    utils_load_tensor_162 = utils.load_tensor(
        "./tensors/arg162.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.1.layer.0.convolution.weight"] = (
        utils_load_tensor_162
    )
    utils_load_tensor_163 = utils.load_tensor(
        "./tensors/arg163.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.2.layer.2.normalization.running_var"
    ] = utils_load_tensor_163
    utils_load_tensor_164 = utils.load_tensor(
        "./tensors/arg164.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.2.layer.2.normalization.running_mean"
    ] = utils_load_tensor_164
    utils_load_tensor_165 = utils.load_tensor(
        "./tensors/arg165.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.2.layer.2.normalization.bias"] = (
        utils_load_tensor_165
    )
    utils_load_tensor_166 = utils.load_tensor(
        "./tensors/arg166.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.2.layer.2.normalization.weight"] = (
        utils_load_tensor_166
    )
    utils_load_tensor_167 = utils.load_tensor(
        "./tensors/arg167.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.2.layer.2.convolution.weight"] = (
        utils_load_tensor_167
    )
    utils_load_tensor_168 = utils.load_tensor(
        "./tensors/arg168.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.2.layer.1.normalization.running_var"
    ] = utils_load_tensor_168
    utils_load_tensor_169 = utils.load_tensor(
        "./tensors/arg169.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.2.layer.1.normalization.running_mean"
    ] = utils_load_tensor_169
    utils_load_tensor_170 = utils.load_tensor(
        "./tensors/arg170.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.2.layer.1.normalization.bias"] = (
        utils_load_tensor_170
    )
    utils_load_tensor_171 = utils.load_tensor(
        "./tensors/arg171.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.2.layer.1.normalization.weight"] = (
        utils_load_tensor_171
    )
    utils_load_tensor_172 = utils.load_tensor(
        "./tensors/arg172.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.2.layer.1.convolution.weight"] = (
        utils_load_tensor_172
    )
    utils_load_tensor_173 = utils.load_tensor(
        "./tensors/arg173.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.2.layer.0.normalization.running_var"
    ] = utils_load_tensor_173
    utils_load_tensor_174 = utils.load_tensor(
        "./tensors/arg174.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.2.layer.0.normalization.running_mean"
    ] = utils_load_tensor_174
    utils_load_tensor_175 = utils.load_tensor(
        "./tensors/arg175.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.2.layer.0.normalization.bias"] = (
        utils_load_tensor_175
    )
    utils_load_tensor_176 = utils.load_tensor(
        "./tensors/arg176.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.2.layer.0.normalization.weight"] = (
        utils_load_tensor_176
    )
    utils_load_tensor_177 = utils.load_tensor(
        "./tensors/arg177.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.2.layer.0.convolution.weight"] = (
        utils_load_tensor_177
    )
    utils_load_tensor_178 = utils.load_tensor(
        "./tensors/arg178.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.3.layer.2.normalization.running_var"
    ] = utils_load_tensor_178
    utils_load_tensor_179 = utils.load_tensor(
        "./tensors/arg179.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.3.layer.2.normalization.running_mean"
    ] = utils_load_tensor_179
    utils_load_tensor_180 = utils.load_tensor(
        "./tensors/arg180.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.3.layer.2.normalization.bias"] = (
        utils_load_tensor_180
    )
    utils_load_tensor_181 = utils.load_tensor(
        "./tensors/arg181.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.3.layer.2.normalization.weight"] = (
        utils_load_tensor_181
    )
    utils_load_tensor_182 = utils.load_tensor(
        "./tensors/arg182.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.3.layer.2.convolution.weight"] = (
        utils_load_tensor_182
    )
    utils_load_tensor_183 = utils.load_tensor(
        "./tensors/arg183.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.3.layer.1.normalization.running_var"
    ] = utils_load_tensor_183
    utils_load_tensor_184 = utils.load_tensor(
        "./tensors/arg184.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.3.layer.1.normalization.running_mean"
    ] = utils_load_tensor_184
    utils_load_tensor_185 = utils.load_tensor(
        "./tensors/arg185.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.3.layer.1.normalization.bias"] = (
        utils_load_tensor_185
    )
    utils_load_tensor_186 = utils.load_tensor(
        "./tensors/arg186.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.3.layer.1.normalization.weight"] = (
        utils_load_tensor_186
    )
    utils_load_tensor_187 = utils.load_tensor(
        "./tensors/arg187.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.3.layer.1.convolution.weight"] = (
        utils_load_tensor_187
    )
    utils_load_tensor_188 = utils.load_tensor(
        "./tensors/arg188.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.3.layer.0.normalization.running_var"
    ] = utils_load_tensor_188
    utils_load_tensor_189 = utils.load_tensor(
        "./tensors/arg189.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.3.layer.0.normalization.running_mean"
    ] = utils_load_tensor_189
    utils_load_tensor_190 = utils.load_tensor(
        "./tensors/arg190.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.3.layer.0.normalization.bias"] = (
        utils_load_tensor_190
    )
    utils_load_tensor_191 = utils.load_tensor(
        "./tensors/arg191.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.3.layer.0.normalization.weight"] = (
        utils_load_tensor_191
    )
    utils_load_tensor_192 = utils.load_tensor(
        "./tensors/arg192.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.3.layer.0.convolution.weight"] = (
        utils_load_tensor_192
    )
    utils_load_tensor_193 = utils.load_tensor(
        "./tensors/arg193.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.4.layer.2.normalization.running_var"
    ] = utils_load_tensor_193
    utils_load_tensor_194 = utils.load_tensor(
        "./tensors/arg194.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.4.layer.2.normalization.running_mean"
    ] = utils_load_tensor_194
    utils_load_tensor_195 = utils.load_tensor(
        "./tensors/arg195.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.4.layer.2.normalization.bias"] = (
        utils_load_tensor_195
    )
    utils_load_tensor_196 = utils.load_tensor(
        "./tensors/arg196.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.4.layer.2.normalization.weight"] = (
        utils_load_tensor_196
    )
    utils_load_tensor_197 = utils.load_tensor(
        "./tensors/arg197.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.4.layer.2.convolution.weight"] = (
        utils_load_tensor_197
    )
    utils_load_tensor_198 = utils.load_tensor(
        "./tensors/arg198.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.4.layer.1.normalization.running_var"
    ] = utils_load_tensor_198
    utils_load_tensor_199 = utils.load_tensor(
        "./tensors/arg199.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.4.layer.1.normalization.running_mean"
    ] = utils_load_tensor_199
    utils_load_tensor_200 = utils.load_tensor(
        "./tensors/arg200.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.4.layer.1.normalization.bias"] = (
        utils_load_tensor_200
    )
    utils_load_tensor_201 = utils.load_tensor(
        "./tensors/arg201.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.4.layer.1.normalization.weight"] = (
        utils_load_tensor_201
    )
    utils_load_tensor_202 = utils.load_tensor(
        "./tensors/arg202.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.4.layer.1.convolution.weight"] = (
        utils_load_tensor_202
    )
    utils_load_tensor_203 = utils.load_tensor(
        "./tensors/arg203.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.4.layer.0.normalization.running_var"
    ] = utils_load_tensor_203
    utils_load_tensor_204 = utils.load_tensor(
        "./tensors/arg204.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.4.layer.0.normalization.running_mean"
    ] = utils_load_tensor_204
    utils_load_tensor_205 = utils.load_tensor(
        "./tensors/arg205.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.4.layer.0.normalization.bias"] = (
        utils_load_tensor_205
    )
    utils_load_tensor_206 = utils.load_tensor(
        "./tensors/arg206.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.4.layer.0.normalization.weight"] = (
        utils_load_tensor_206
    )
    utils_load_tensor_207 = utils.load_tensor(
        "./tensors/arg207.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.4.layer.0.convolution.weight"] = (
        utils_load_tensor_207
    )
    utils_load_tensor_208 = utils.load_tensor(
        "./tensors/arg208.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.5.layer.2.normalization.running_var"
    ] = utils_load_tensor_208
    utils_load_tensor_209 = utils.load_tensor(
        "./tensors/arg209.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.5.layer.2.normalization.running_mean"
    ] = utils_load_tensor_209
    utils_load_tensor_210 = utils.load_tensor(
        "./tensors/arg210.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.5.layer.2.normalization.bias"] = (
        utils_load_tensor_210
    )
    utils_load_tensor_211 = utils.load_tensor(
        "./tensors/arg211.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.5.layer.2.normalization.weight"] = (
        utils_load_tensor_211
    )
    utils_load_tensor_212 = utils.load_tensor(
        "./tensors/arg212.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.5.layer.2.convolution.weight"] = (
        utils_load_tensor_212
    )
    utils_load_tensor_213 = utils.load_tensor(
        "./tensors/arg213.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.5.layer.1.normalization.running_var"
    ] = utils_load_tensor_213
    utils_load_tensor_214 = utils.load_tensor(
        "./tensors/arg214.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.5.layer.1.normalization.running_mean"
    ] = utils_load_tensor_214
    utils_load_tensor_215 = utils.load_tensor(
        "./tensors/arg215.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.5.layer.1.normalization.bias"] = (
        utils_load_tensor_215
    )
    utils_load_tensor_216 = utils.load_tensor(
        "./tensors/arg216.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.5.layer.1.normalization.weight"] = (
        utils_load_tensor_216
    )
    utils_load_tensor_217 = utils.load_tensor(
        "./tensors/arg217.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.5.layer.1.convolution.weight"] = (
        utils_load_tensor_217
    )
    utils_load_tensor_218 = utils.load_tensor(
        "./tensors/arg218.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.5.layer.0.normalization.running_var"
    ] = utils_load_tensor_218
    utils_load_tensor_219 = utils.load_tensor(
        "./tensors/arg219.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.2.layers.5.layer.0.normalization.running_mean"
    ] = utils_load_tensor_219
    utils_load_tensor_220 = utils.load_tensor(
        "./tensors/arg220.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.5.layer.0.normalization.bias"] = (
        utils_load_tensor_220
    )
    utils_load_tensor_221 = utils.load_tensor(
        "./tensors/arg221.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.5.layer.0.normalization.weight"] = (
        utils_load_tensor_221
    )
    utils_load_tensor_222 = utils.load_tensor(
        "./tensors/arg222.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.2.layers.5.layer.0.convolution.weight"] = (
        utils_load_tensor_222
    )
    utils_load_tensor_223 = utils.load_tensor(
        "./tensors/arg223.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.0.layer.2.normalization.running_var"
    ] = utils_load_tensor_223
    utils_load_tensor_224 = utils.load_tensor(
        "./tensors/arg224.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.0.layer.2.normalization.running_mean"
    ] = utils_load_tensor_224
    utils_load_tensor_225 = utils.load_tensor(
        "./tensors/arg225.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.layer.2.normalization.bias"] = (
        utils_load_tensor_225
    )
    utils_load_tensor_226 = utils.load_tensor(
        "./tensors/arg226.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.layer.2.normalization.weight"] = (
        utils_load_tensor_226
    )
    utils_load_tensor_227 = utils.load_tensor(
        "./tensors/arg227.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.layer.2.convolution.weight"] = (
        utils_load_tensor_227
    )
    utils_load_tensor_228 = utils.load_tensor(
        "./tensors/arg228.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.0.layer.1.normalization.running_var"
    ] = utils_load_tensor_228
    utils_load_tensor_229 = utils.load_tensor(
        "./tensors/arg229.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.0.layer.1.normalization.running_mean"
    ] = utils_load_tensor_229
    utils_load_tensor_230 = utils.load_tensor(
        "./tensors/arg230.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.layer.1.normalization.bias"] = (
        utils_load_tensor_230
    )
    utils_load_tensor_231 = utils.load_tensor(
        "./tensors/arg231.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.layer.1.normalization.weight"] = (
        utils_load_tensor_231
    )
    utils_load_tensor_232 = utils.load_tensor(
        "./tensors/arg232.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.layer.1.convolution.weight"] = (
        utils_load_tensor_232
    )
    utils_load_tensor_233 = utils.load_tensor(
        "./tensors/arg233.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.0.layer.0.normalization.running_var"
    ] = utils_load_tensor_233
    utils_load_tensor_234 = utils.load_tensor(
        "./tensors/arg234.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.0.layer.0.normalization.running_mean"
    ] = utils_load_tensor_234
    utils_load_tensor_235 = utils.load_tensor(
        "./tensors/arg235.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.layer.0.normalization.bias"] = (
        utils_load_tensor_235
    )
    utils_load_tensor_236 = utils.load_tensor(
        "./tensors/arg236.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.layer.0.normalization.weight"] = (
        utils_load_tensor_236
    )
    utils_load_tensor_237 = utils.load_tensor(
        "./tensors/arg237.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.0.layer.0.convolution.weight"] = (
        utils_load_tensor_237
    )
    utils_load_tensor_238 = utils.load_tensor(
        "./tensors/arg238.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.1.layer.2.normalization.running_var"
    ] = utils_load_tensor_238
    utils_load_tensor_239 = utils.load_tensor(
        "./tensors/arg239.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.1.layer.2.normalization.running_mean"
    ] = utils_load_tensor_239
    utils_load_tensor_240 = utils.load_tensor(
        "./tensors/arg240.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.1.layer.2.normalization.bias"] = (
        utils_load_tensor_240
    )
    utils_load_tensor_241 = utils.load_tensor(
        "./tensors/arg241.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.1.layer.2.normalization.weight"] = (
        utils_load_tensor_241
    )
    utils_load_tensor_242 = utils.load_tensor(
        "./tensors/arg242.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.1.layer.2.convolution.weight"] = (
        utils_load_tensor_242
    )
    utils_load_tensor_243 = utils.load_tensor(
        "./tensors/arg243.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.1.layer.1.normalization.running_var"
    ] = utils_load_tensor_243
    utils_load_tensor_244 = utils.load_tensor(
        "./tensors/arg244.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.1.layer.1.normalization.running_mean"
    ] = utils_load_tensor_244
    utils_load_tensor_245 = utils.load_tensor(
        "./tensors/arg245.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.1.layer.1.normalization.bias"] = (
        utils_load_tensor_245
    )
    utils_load_tensor_246 = utils.load_tensor(
        "./tensors/arg246.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.1.layer.1.normalization.weight"] = (
        utils_load_tensor_246
    )
    utils_load_tensor_247 = utils.load_tensor(
        "./tensors/arg247.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.1.layer.1.convolution.weight"] = (
        utils_load_tensor_247
    )
    utils_load_tensor_248 = utils.load_tensor(
        "./tensors/arg248.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.1.layer.0.normalization.running_var"
    ] = utils_load_tensor_248
    utils_load_tensor_249 = utils.load_tensor(
        "./tensors/arg249.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.1.layer.0.normalization.running_mean"
    ] = utils_load_tensor_249
    utils_load_tensor_250 = utils.load_tensor(
        "./tensors/arg250.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.1.layer.0.normalization.bias"] = (
        utils_load_tensor_250
    )
    utils_load_tensor_251 = utils.load_tensor(
        "./tensors/arg251.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.1.layer.0.normalization.weight"] = (
        utils_load_tensor_251
    )
    utils_load_tensor_252 = utils.load_tensor(
        "./tensors/arg252.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.1.layer.0.convolution.weight"] = (
        utils_load_tensor_252
    )
    utils_load_tensor_253 = utils.load_tensor(
        "./tensors/arg253.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.2.layer.2.normalization.running_var"
    ] = utils_load_tensor_253
    utils_load_tensor_254 = utils.load_tensor(
        "./tensors/arg254.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.2.layer.2.normalization.running_mean"
    ] = utils_load_tensor_254
    utils_load_tensor_255 = utils.load_tensor(
        "./tensors/arg255.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.2.layer.2.normalization.bias"] = (
        utils_load_tensor_255
    )
    utils_load_tensor_256 = utils.load_tensor(
        "./tensors/arg256.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.2.layer.2.normalization.weight"] = (
        utils_load_tensor_256
    )
    utils_load_tensor_257 = utils.load_tensor(
        "./tensors/arg257.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.2.layer.2.convolution.weight"] = (
        utils_load_tensor_257
    )
    utils_load_tensor_258 = utils.load_tensor(
        "./tensors/arg258.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.2.layer.1.normalization.running_var"
    ] = utils_load_tensor_258
    utils_load_tensor_259 = utils.load_tensor(
        "./tensors/arg259.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.2.layer.1.normalization.running_mean"
    ] = utils_load_tensor_259
    utils_load_tensor_260 = utils.load_tensor(
        "./tensors/arg260.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.2.layer.1.normalization.bias"] = (
        utils_load_tensor_260
    )
    utils_load_tensor_261 = utils.load_tensor(
        "./tensors/arg261.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.2.layer.1.normalization.weight"] = (
        utils_load_tensor_261
    )
    utils_load_tensor_262 = utils.load_tensor(
        "./tensors/arg262.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.2.layer.1.convolution.weight"] = (
        utils_load_tensor_262
    )
    utils_load_tensor_263 = utils.load_tensor(
        "./tensors/arg263.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.2.layer.0.normalization.running_var"
    ] = utils_load_tensor_263
    utils_load_tensor_264 = utils.load_tensor(
        "./tensors/arg264.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights[
        "resnet.encoder.stages.3.layers.2.layer.0.normalization.running_mean"
    ] = utils_load_tensor_264
    utils_load_tensor_265 = utils.load_tensor(
        "./tensors/arg265.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.2.layer.0.normalization.bias"] = (
        utils_load_tensor_265
    )
    utils_load_tensor_266 = utils.load_tensor(
        "./tensors/arg266.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.2.layer.0.normalization.weight"] = (
        utils_load_tensor_266
    )
    utils_load_tensor_267 = utils.load_tensor(
        "./tensors/arg267.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["resnet.encoder.stages.3.layers.2.layer.0.convolution.weight"] = (
        utils_load_tensor_267
    )
    return _main_weights


def main():
    load_activations_for__main_0 = load_activations_for__main()
    load_weights_for__main_0 = load_weights_for__main()
    _main_0 = _main(load_activations_for__main_0, load_weights_for__main_0)

    ttnn_output = ttnn.to_torch(_main_0[0]).reshape(1, 1000).to(torch.float32)
    golden_output = run_golden().reshape(1, 1000).to(torch.float32)

    pcc = calculate_pcc(ttnn_output, golden_output)
    print(f"PCC: {pcc}")
    assert pcc >= 0.99, f"PCC {pcc} is below threshold 0.99"
    return 0


def test_main():
    return 0


if __name__ == "__main__":
    main()
