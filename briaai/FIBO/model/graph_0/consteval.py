import ttnn

DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _tile(device, t):
    t = ttnn.to_device(t, device=device, memory_config=DRAM)
    return ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=DRAM)


def _reshape(device, t, shape):
    return [ttnn.reshape(_tile(device, t), shape, memory_config=DRAM)][0]


def _reshape_cast_reshape(device, t, shape1, dtype, shape2):
    x = ttnn.reshape(_tile(device, t), shape1, memory_config=DRAM)
    x = ttnn.typecast(x, dtype, memory_config=DRAM)
    return ttnn.reshape(x, shape2, memory_config=DRAM)


def _tile_concat(device, tensors, dim):
    prepared = [_tile(device, t) for t in tensors]
    return ttnn.concat(prepared, dim, memory_config=DRAM)


def _tile_permute_concat(device, tensors, perm, dim):
    prepared = [ttnn.permute(_tile(device, t), perm, memory_config=DRAM, pad_value=0.0) for t in tensors]
    return ttnn.concat(prepared, dim, memory_config=DRAM)


def main_const_eval_0(device):
    ttnn_Tensor_0 = ttnn.Tensor(
        [
            1.0,
            0.9305720329284668,
            0.86596429347991943,
            0.80584216117858887,
            0.74989420175552368,
            0.69783055782318115,
            0.64938163757324219,
            0.60429638624191284,
            0.56234133243560791,
            0.52329909801483154,
            0.48696750402450562,
            0.45315837860107422,
            0.42169648408889771,
            0.39241895079612732,
            0.36517414450645447,
            0.33982083201408386,
            0.31622776389122009,
            0.29427272081375122,
            0.27384194731712341,
            0.25482964515686035,
            0.23713734745979309,
            0.22067341208457947,
            0.20535250008106232,
            0.19109529256820679,
            0.17782793939113617,
            0.16548170149326324,
            0.15399263799190521,
            0.14330124855041504,
            0.13335214555263519,
            0.12409376353025436,
            0.11547819525003433,
            0.10746076703071594,
            0.099999994039535522,
            0.093057207763195038,
            0.086596429347991943,
            0.080584220588207245,
            0.07498941570520401,
            0.069783061742782593,
            0.064938157796859741,
            0.060429636389017105,
            0.056234125047922134,
            0.052329909056425095,
            0.048696756362915039,
            0.045315831899642944,
            0.04216964915394783,
            0.039241891354322433,
            0.036517411470413208,
            0.033982079476118088,
            0.03162277489900589,
            0.029427273198962212,
            0.027384193614125252,
            0.025482967495918274,
            0.023713734000921249,
            0.022067340090870857,
            0.020535247400403023,
            0.019109528511762619,
            0.017782794311642647,
            0.016548173502087593,
            0.015399262309074402,
            0.014330124482512474,
            0.013335213996469975,
            0.01240937877446413,
            0.011547816917300224,
            0.010746076703071594,
            0.0099999997764825821,
            0.0093057211488485336,
            0.0086596440523862839,
            0.0080584203824400902,
            0.0074989409185945988,
            0.006978305522352457,
            0.0064938166178762913,
            0.0060429619625210762,
            0.0056234123185276985,
            0.0052329907193779945,
            0.0048696752637624741,
            0.0045315842144191265,
            0.0042169638909399509,
            0.0039241891354322433,
            0.0036517411936074495,
            0.0033982084132730961,
            0.0031622766982764006,
            0.002942726481705904,
            0.002738419221714139,
            0.0025482967030256987,
            0.0023713738191872835,
            0.002206733450293541,
            0.0020535246003419161,
            0.0019109528511762619,
            0.0017782794311642647,
            0.0016548173734918237,
            0.0015399261610582471,
            0.001433012424968183,
            0.0013335214462131262,
            0.0012409378541633487,
            0.0011547816684469581,
            0.0010746076004579663,
            0.00099999993108212948,
            0.00093057204503566027,
            0.0008659643935970962,
            0.00080584199167788029,
            0.00074989406857639551,
            0.00069783051731064916,
            0.00064938166178762913,
            0.00060429621953517199,
            0.00056234118528664112,
            0.00052329903701320291,
            0.00048696750309318304,
            0.00045315839815884829,
            0.00042169637163169682,
            0.00039241890772245824,
            0.00036517408443614841,
            0.00033982083550654352,
            0.00031622781534679234,
            0.00029427278786897659,
            0.00027384204440750182,
            0.00025482953060418367,
            0.00023713726841378957,
            0.00022067333338782191,
            0.0002053524658549577,
            0.00019109527056571096,
            0.00017782794020604342,
            0.00016548173152841628,
            0.00015399268886540085,
            0.00014330129488371313,
            0.00013335207768250257,
            0.00012409372720867395,
            0.00011547815665835515,
            0.00010746075713541359,
        ],
        [1, 128],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_Tensor_0]


def main_const_eval_95(device, arg):
    ttnn_to_device_216 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_216 = ttnn.to_layout(
        ttnn_to_device_216,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_216, False)
    ttnn_typecast_19 = ttnn.typecast(
        ttnn_to_layout_216,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_216, False)
    return [ttnn_typecast_19]


def main_const_eval_107(device, arg):
    ttnn_to_device_238 = ttnn.to_device(
        arg[54],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_238 = ttnn.to_layout(
        ttnn_to_device_238,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_238, False)
    ttnn_to_device_239 = ttnn.to_device(
        arg[53],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_239 = ttnn.to_layout(
        ttnn_to_device_239,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_239, False)
    ttnn_to_device_240 = ttnn.to_device(
        arg[52],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_240 = ttnn.to_layout(
        ttnn_to_device_240,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_240, False)
    ttnn_to_device_241 = ttnn.to_device(
        arg[51],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_241 = ttnn.to_layout(
        ttnn_to_device_241,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_241, False)
    ttnn_to_device_242 = ttnn.to_device(
        arg[50],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_242 = ttnn.to_layout(
        ttnn_to_device_242,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_242, False)
    ttnn_to_device_243 = ttnn.to_device(
        arg[49],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_243 = ttnn.to_layout(
        ttnn_to_device_243,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_243, False)
    ttnn_to_device_244 = ttnn.to_device(
        arg[48],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_244 = ttnn.to_layout(
        ttnn_to_device_244,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_244, False)
    ttnn_to_device_245 = ttnn.to_device(
        arg[47],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_245 = ttnn.to_layout(
        ttnn_to_device_245,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_245, False)
    ttnn_to_device_246 = ttnn.to_device(
        arg[46],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_246 = ttnn.to_layout(
        ttnn_to_device_246,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_246, False)
    ttnn_to_device_247 = ttnn.to_device(
        arg[45],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_247 = ttnn.to_layout(
        ttnn_to_device_247,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_247, False)
    ttnn_to_device_248 = ttnn.to_device(
        arg[44],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_248 = ttnn.to_layout(
        ttnn_to_device_248,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_248, False)
    ttnn_to_device_249 = ttnn.to_device(
        arg[43],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_249 = ttnn.to_layout(
        ttnn_to_device_249,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_249, False)
    ttnn_to_device_250 = ttnn.to_device(
        arg[42],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_250 = ttnn.to_layout(
        ttnn_to_device_250,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_250, False)
    ttnn_to_device_251 = ttnn.to_device(
        arg[41],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_251 = ttnn.to_layout(
        ttnn_to_device_251,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_251, False)
    ttnn_to_device_252 = ttnn.to_device(
        arg[40],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_252 = ttnn.to_layout(
        ttnn_to_device_252,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_252, False)
    ttnn_to_device_253 = ttnn.to_device(
        arg[39],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_253 = ttnn.to_layout(
        ttnn_to_device_253,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_253, False)
    ttnn_to_device_254 = ttnn.to_device(
        arg[38],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_254 = ttnn.to_layout(
        ttnn_to_device_254,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_254, False)
    ttnn_to_device_255 = ttnn.to_device(
        arg[37],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_255 = ttnn.to_layout(
        ttnn_to_device_255,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_255, False)
    ttnn_to_device_256 = ttnn.to_device(
        arg[36],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_256 = ttnn.to_layout(
        ttnn_to_device_256,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_256, False)
    ttnn_to_device_257 = ttnn.to_device(
        arg[35],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_257 = ttnn.to_layout(
        ttnn_to_device_257,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_257, False)
    ttnn_to_device_258 = ttnn.to_device(
        arg[34],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_258 = ttnn.to_layout(
        ttnn_to_device_258,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_258, False)
    ttnn_to_device_259 = ttnn.to_device(
        arg[33],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_259 = ttnn.to_layout(
        ttnn_to_device_259,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_259, False)
    ttnn_to_device_260 = ttnn.to_device(
        arg[32],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_260 = ttnn.to_layout(
        ttnn_to_device_260,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_260, False)
    ttnn_to_device_261 = ttnn.to_device(
        arg[31],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_261 = ttnn.to_layout(
        ttnn_to_device_261,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_261, False)
    ttnn_to_device_262 = ttnn.to_device(
        arg[30],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_262 = ttnn.to_layout(
        ttnn_to_device_262,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_262, False)
    ttnn_to_device_263 = ttnn.to_device(
        arg[29],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_263 = ttnn.to_layout(
        ttnn_to_device_263,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_263, False)
    ttnn_to_device_264 = ttnn.to_device(
        arg[28],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_264 = ttnn.to_layout(
        ttnn_to_device_264,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_264, False)
    ttnn_to_device_265 = ttnn.to_device(
        arg[27],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_265 = ttnn.to_layout(
        ttnn_to_device_265,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_265, False)
    ttnn_to_device_266 = ttnn.to_device(
        arg[26],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_266 = ttnn.to_layout(
        ttnn_to_device_266,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_266, False)
    ttnn_to_device_267 = ttnn.to_device(
        arg[25],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_267 = ttnn.to_layout(
        ttnn_to_device_267,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_267, False)
    ttnn_to_device_268 = ttnn.to_device(
        arg[24],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_268 = ttnn.to_layout(
        ttnn_to_device_268,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_268, False)
    ttnn_to_device_269 = ttnn.to_device(
        arg[23],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_269 = ttnn.to_layout(
        ttnn_to_device_269,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_269, False)
    ttnn_to_device_270 = ttnn.to_device(
        arg[22],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_270 = ttnn.to_layout(
        ttnn_to_device_270,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_270, False)
    ttnn_to_device_271 = ttnn.to_device(
        arg[21],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_271 = ttnn.to_layout(
        ttnn_to_device_271,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_271, False)
    ttnn_to_device_272 = ttnn.to_device(
        arg[20],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_272 = ttnn.to_layout(
        ttnn_to_device_272,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_272, False)
    ttnn_to_device_273 = ttnn.to_device(
        arg[19],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_273 = ttnn.to_layout(
        ttnn_to_device_273,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_273, False)
    ttnn_to_device_274 = ttnn.to_device(
        arg[18],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_274 = ttnn.to_layout(
        ttnn_to_device_274,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_274, False)
    ttnn_to_device_275 = ttnn.to_device(
        arg[17],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_275 = ttnn.to_layout(
        ttnn_to_device_275,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_275, False)
    ttnn_to_device_276 = ttnn.to_device(
        arg[16],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_276 = ttnn.to_layout(
        ttnn_to_device_276,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_276, False)
    ttnn_to_device_277 = ttnn.to_device(
        arg[15],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_277 = ttnn.to_layout(
        ttnn_to_device_277,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_277, False)
    ttnn_to_device_278 = ttnn.to_device(
        arg[14],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_278 = ttnn.to_layout(
        ttnn_to_device_278,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_278, False)
    ttnn_to_device_279 = ttnn.to_device(
        arg[13],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_279 = ttnn.to_layout(
        ttnn_to_device_279,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_279, False)
    ttnn_to_device_280 = ttnn.to_device(
        arg[12],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_280 = ttnn.to_layout(
        ttnn_to_device_280,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_280, False)
    ttnn_to_device_281 = ttnn.to_device(
        arg[11],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_281 = ttnn.to_layout(
        ttnn_to_device_281,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_281, False)
    ttnn_to_device_282 = ttnn.to_device(
        arg[10],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_282 = ttnn.to_layout(
        ttnn_to_device_282,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_282, False)
    ttnn_to_device_283 = ttnn.to_device(
        arg[9],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_283 = ttnn.to_layout(
        ttnn_to_device_283,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_283, False)
    ttnn_to_device_284 = ttnn.to_device(
        arg[8],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_284 = ttnn.to_layout(
        ttnn_to_device_284,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_284, False)
    ttnn_to_device_285 = ttnn.to_device(
        arg[7],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_285 = ttnn.to_layout(
        ttnn_to_device_285,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_285, False)
    ttnn_to_device_286 = ttnn.to_device(
        arg[6],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_286 = ttnn.to_layout(
        ttnn_to_device_286,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_286, False)
    ttnn_to_device_287 = ttnn.to_device(
        arg[5],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_287 = ttnn.to_layout(
        ttnn_to_device_287,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_287, False)
    ttnn_to_device_288 = ttnn.to_device(
        arg[4],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_288 = ttnn.to_layout(
        ttnn_to_device_288,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_288, False)
    ttnn_to_device_289 = ttnn.to_device(
        arg[3],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_289 = ttnn.to_layout(
        ttnn_to_device_289,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_289, False)
    ttnn_to_device_290 = ttnn.to_device(
        arg[2],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_290 = ttnn.to_layout(
        ttnn_to_device_290,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_290, False)
    ttnn_to_device_291 = ttnn.to_device(
        arg[1],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_291 = ttnn.to_layout(
        ttnn_to_device_291,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_291, False)
    ttnn_to_device_292 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_292 = ttnn.to_layout(
        ttnn_to_device_292,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_292, False)
    ttnn_permute_90 = ttnn.permute(
        ttnn_to_layout_292,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_292, False)
    ttnn_typecast_23 = ttnn.typecast(
        ttnn_permute_90,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_90, False)
    ttnn_permute_91 = ttnn.permute(
        ttnn_to_layout_291,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_291, False)
    ttnn_typecast_24 = ttnn.typecast(
        ttnn_permute_91,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_91, False)
    ttnn_permute_92 = ttnn.permute(
        ttnn_to_layout_290,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_290, False)
    ttnn_typecast_25 = ttnn.typecast(
        ttnn_permute_92,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_92, False)
    ttnn_permute_93 = ttnn.permute(
        ttnn_to_layout_289,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_289, False)
    ttnn_typecast_26 = ttnn.typecast(
        ttnn_permute_93,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_93, False)
    ttnn_permute_94 = ttnn.permute(
        ttnn_to_layout_288,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_288, False)
    ttnn_typecast_27 = ttnn.typecast(
        ttnn_permute_94,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_94, False)
    ttnn_permute_95 = ttnn.permute(
        ttnn_to_layout_287,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_287, False)
    ttnn_typecast_28 = ttnn.typecast(
        ttnn_permute_95,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_95, False)
    ttnn_permute_96 = ttnn.permute(
        ttnn_to_layout_286,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_286, False)
    ttnn_typecast_29 = ttnn.typecast(
        ttnn_permute_96,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_96, False)
    ttnn_permute_97 = ttnn.permute(
        ttnn_to_layout_285,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_285, False)
    ttnn_typecast_30 = ttnn.typecast(
        ttnn_permute_97,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_97, False)
    ttnn_permute_98 = ttnn.permute(
        ttnn_to_layout_284,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_284, False)
    ttnn_typecast_31 = ttnn.typecast(
        ttnn_permute_98,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_98, False)
    ttnn_permute_99 = ttnn.permute(
        ttnn_to_layout_283,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_283, False)
    ttnn_typecast_32 = ttnn.typecast(
        ttnn_permute_99,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_99, False)
    ttnn_permute_100 = ttnn.permute(
        ttnn_to_layout_282,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_282, False)
    ttnn_typecast_33 = ttnn.typecast(
        ttnn_permute_100,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_100, False)
    ttnn_permute_101 = ttnn.permute(
        ttnn_to_layout_281,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_281, False)
    ttnn_typecast_34 = ttnn.typecast(
        ttnn_permute_101,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_101, False)
    ttnn_permute_102 = ttnn.permute(
        ttnn_to_layout_280,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_280, False)
    ttnn_typecast_35 = ttnn.typecast(
        ttnn_permute_102,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_102, False)
    ttnn_permute_103 = ttnn.permute(
        ttnn_to_layout_279,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_279, False)
    ttnn_typecast_36 = ttnn.typecast(
        ttnn_permute_103,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_103, False)
    ttnn_permute_104 = ttnn.permute(
        ttnn_to_layout_278,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_278, False)
    ttnn_typecast_37 = ttnn.typecast(
        ttnn_permute_104,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_104, False)
    ttnn_permute_105 = ttnn.permute(
        ttnn_to_layout_277,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_277, False)
    ttnn_typecast_38 = ttnn.typecast(
        ttnn_permute_105,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_105, False)
    ttnn_permute_106 = ttnn.permute(
        ttnn_to_layout_276,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_276, False)
    ttnn_typecast_39 = ttnn.typecast(
        ttnn_permute_106,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_106, False)
    ttnn_permute_107 = ttnn.permute(
        ttnn_to_layout_275,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_275, False)
    ttnn_typecast_40 = ttnn.typecast(
        ttnn_permute_107,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_107, False)
    ttnn_permute_108 = ttnn.permute(
        ttnn_to_layout_274,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_274, False)
    ttnn_typecast_41 = ttnn.typecast(
        ttnn_permute_108,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_108, False)
    ttnn_permute_109 = ttnn.permute(
        ttnn_to_layout_273,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_273, False)
    ttnn_typecast_42 = ttnn.typecast(
        ttnn_permute_109,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_109, False)
    ttnn_permute_110 = ttnn.permute(
        ttnn_to_layout_272,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_272, False)
    ttnn_typecast_43 = ttnn.typecast(
        ttnn_permute_110,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_110, False)
    ttnn_permute_111 = ttnn.permute(
        ttnn_to_layout_271,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_271, False)
    ttnn_typecast_44 = ttnn.typecast(
        ttnn_permute_111,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_111, False)
    ttnn_permute_112 = ttnn.permute(
        ttnn_to_layout_270,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_270, False)
    ttnn_typecast_45 = ttnn.typecast(
        ttnn_permute_112,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_112, False)
    ttnn_permute_113 = ttnn.permute(
        ttnn_to_layout_269,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_269, False)
    ttnn_typecast_46 = ttnn.typecast(
        ttnn_permute_113,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_113, False)
    ttnn_permute_114 = ttnn.permute(
        ttnn_to_layout_268,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_268, False)
    ttnn_typecast_47 = ttnn.typecast(
        ttnn_permute_114,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_114, False)
    ttnn_permute_115 = ttnn.permute(
        ttnn_to_layout_267,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_267, False)
    ttnn_typecast_48 = ttnn.typecast(
        ttnn_permute_115,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_115, False)
    ttnn_permute_116 = ttnn.permute(
        ttnn_to_layout_266,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_266, False)
    ttnn_typecast_49 = ttnn.typecast(
        ttnn_permute_116,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_116, False)
    ttnn_permute_117 = ttnn.permute(
        ttnn_to_layout_265,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_265, False)
    ttnn_typecast_50 = ttnn.typecast(
        ttnn_permute_117,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_117, False)
    ttnn_permute_118 = ttnn.permute(
        ttnn_to_layout_264,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_264, False)
    ttnn_typecast_51 = ttnn.typecast(
        ttnn_permute_118,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_118, False)
    ttnn_permute_119 = ttnn.permute(
        ttnn_to_layout_263,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_263, False)
    ttnn_typecast_52 = ttnn.typecast(
        ttnn_permute_119,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_119, False)
    ttnn_permute_120 = ttnn.permute(
        ttnn_to_layout_262,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_262, False)
    ttnn_typecast_53 = ttnn.typecast(
        ttnn_permute_120,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_120, False)
    ttnn_permute_121 = ttnn.permute(
        ttnn_to_layout_261,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_261, False)
    ttnn_typecast_54 = ttnn.typecast(
        ttnn_permute_121,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_121, False)
    ttnn_permute_122 = ttnn.permute(
        ttnn_to_layout_260,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_260, False)
    ttnn_typecast_55 = ttnn.typecast(
        ttnn_permute_122,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_122, False)
    ttnn_permute_123 = ttnn.permute(
        ttnn_to_layout_259,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_259, False)
    ttnn_typecast_56 = ttnn.typecast(
        ttnn_permute_123,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_123, False)
    ttnn_permute_124 = ttnn.permute(
        ttnn_to_layout_258,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_258, False)
    ttnn_typecast_57 = ttnn.typecast(
        ttnn_permute_124,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_124, False)
    ttnn_permute_125 = ttnn.permute(
        ttnn_to_layout_257,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_257, False)
    ttnn_typecast_58 = ttnn.typecast(
        ttnn_permute_125,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_125, False)
    ttnn_permute_126 = ttnn.permute(
        ttnn_to_layout_256,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_256, False)
    ttnn_typecast_59 = ttnn.typecast(
        ttnn_permute_126,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_126, False)
    ttnn_permute_127 = ttnn.permute(
        ttnn_to_layout_255,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_255, False)
    ttnn_typecast_60 = ttnn.typecast(
        ttnn_permute_127,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_127, False)
    ttnn_permute_128 = ttnn.permute(
        ttnn_to_layout_254,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_254, False)
    ttnn_typecast_61 = ttnn.typecast(
        ttnn_permute_128,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_128, False)
    ttnn_permute_129 = ttnn.permute(
        ttnn_to_layout_253,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_253, False)
    ttnn_typecast_62 = ttnn.typecast(
        ttnn_permute_129,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_129, False)
    ttnn_permute_130 = ttnn.permute(
        ttnn_to_layout_238,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_238, False)
    ttnn_typecast_63 = ttnn.typecast(
        ttnn_permute_130,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_130, False)
    ttnn_permute_131 = ttnn.permute(
        ttnn_to_layout_252,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_252, False)
    ttnn_typecast_64 = ttnn.typecast(
        ttnn_permute_131,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_131, False)
    ttnn_permute_132 = ttnn.permute(
        ttnn_to_layout_239,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_239, False)
    ttnn_typecast_65 = ttnn.typecast(
        ttnn_permute_132,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_132, False)
    ttnn_permute_133 = ttnn.permute(
        ttnn_to_layout_251,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_251, False)
    ttnn_typecast_66 = ttnn.typecast(
        ttnn_permute_133,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_133, False)
    ttnn_permute_134 = ttnn.permute(
        ttnn_to_layout_240,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_240, False)
    ttnn_typecast_67 = ttnn.typecast(
        ttnn_permute_134,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_134, False)
    ttnn_permute_135 = ttnn.permute(
        ttnn_to_layout_250,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_250, False)
    ttnn_typecast_68 = ttnn.typecast(
        ttnn_permute_135,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_135, False)
    ttnn_permute_136 = ttnn.permute(
        ttnn_to_layout_241,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_241, False)
    ttnn_typecast_69 = ttnn.typecast(
        ttnn_permute_136,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_136, False)
    ttnn_permute_137 = ttnn.permute(
        ttnn_to_layout_249,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_249, False)
    ttnn_typecast_70 = ttnn.typecast(
        ttnn_permute_137,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_137, False)
    ttnn_permute_138 = ttnn.permute(
        ttnn_to_layout_242,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_242, False)
    ttnn_typecast_71 = ttnn.typecast(
        ttnn_permute_138,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_138, False)
    ttnn_permute_139 = ttnn.permute(
        ttnn_to_layout_248,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_248, False)
    ttnn_typecast_72 = ttnn.typecast(
        ttnn_permute_139,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_139, False)
    ttnn_permute_140 = ttnn.permute(
        ttnn_to_layout_243,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_243, False)
    ttnn_typecast_73 = ttnn.typecast(
        ttnn_permute_140,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_140, False)
    ttnn_permute_141 = ttnn.permute(
        ttnn_to_layout_247,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_247, False)
    ttnn_typecast_74 = ttnn.typecast(
        ttnn_permute_141,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_141, False)
    ttnn_permute_142 = ttnn.permute(
        ttnn_to_layout_244,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_244, False)
    ttnn_typecast_75 = ttnn.typecast(
        ttnn_permute_142,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_142, False)
    ttnn_permute_143 = ttnn.permute(
        ttnn_to_layout_246,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_246, False)
    ttnn_typecast_76 = ttnn.typecast(
        ttnn_permute_143,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_143, False)
    ttnn_permute_144 = ttnn.permute(
        ttnn_to_layout_245,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_245, False)
    ttnn_typecast_77 = ttnn.typecast(
        ttnn_permute_144,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_144, False)
    ttnn_concat_50 = ttnn.concat(
        [
            ttnn_typecast_23,
            ttnn_typecast_24,
            ttnn_typecast_25,
            ttnn_typecast_26,
            ttnn_typecast_27,
            ttnn_typecast_28,
            ttnn_typecast_29,
            ttnn_typecast_30,
            ttnn_typecast_31,
            ttnn_typecast_32,
            ttnn_typecast_33,
            ttnn_typecast_34,
            ttnn_typecast_35,
            ttnn_typecast_36,
            ttnn_typecast_37,
            ttnn_typecast_38,
            ttnn_typecast_39,
            ttnn_typecast_40,
            ttnn_typecast_41,
            ttnn_typecast_42,
            ttnn_typecast_43,
            ttnn_typecast_44,
            ttnn_typecast_45,
            ttnn_typecast_46,
            ttnn_typecast_47,
            ttnn_typecast_48,
            ttnn_typecast_49,
            ttnn_typecast_50,
            ttnn_typecast_51,
            ttnn_typecast_52,
            ttnn_typecast_53,
            ttnn_typecast_54,
            ttnn_typecast_55,
            ttnn_typecast_56,
            ttnn_typecast_57,
            ttnn_typecast_58,
            ttnn_typecast_59,
            ttnn_typecast_60,
            ttnn_typecast_61,
            ttnn_typecast_62,
            ttnn_typecast_63,
            ttnn_typecast_64,
            ttnn_typecast_65,
            ttnn_typecast_66,
            ttnn_typecast_67,
            ttnn_typecast_68,
            ttnn_typecast_69,
            ttnn_typecast_70,
            ttnn_typecast_71,
            ttnn_typecast_72,
            ttnn_typecast_73,
            ttnn_typecast_74,
            ttnn_typecast_75,
            ttnn_typecast_76,
            ttnn_typecast_77,
        ],
        1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_typecast_77, False)
    ttnn.deallocate(ttnn_typecast_76, False)
    ttnn.deallocate(ttnn_typecast_75, False)
    ttnn.deallocate(ttnn_typecast_74, False)
    ttnn.deallocate(ttnn_typecast_73, False)
    ttnn.deallocate(ttnn_typecast_72, False)
    ttnn.deallocate(ttnn_typecast_71, False)
    ttnn.deallocate(ttnn_typecast_70, False)
    ttnn.deallocate(ttnn_typecast_69, False)
    ttnn.deallocate(ttnn_typecast_68, False)
    ttnn.deallocate(ttnn_typecast_67, False)
    ttnn.deallocate(ttnn_typecast_66, False)
    ttnn.deallocate(ttnn_typecast_65, False)
    ttnn.deallocate(ttnn_typecast_64, False)
    ttnn.deallocate(ttnn_typecast_63, False)
    ttnn.deallocate(ttnn_typecast_62, False)
    ttnn.deallocate(ttnn_typecast_61, False)
    ttnn.deallocate(ttnn_typecast_60, False)
    ttnn.deallocate(ttnn_typecast_59, False)
    ttnn.deallocate(ttnn_typecast_58, False)
    ttnn.deallocate(ttnn_typecast_57, False)
    ttnn.deallocate(ttnn_typecast_56, False)
    ttnn.deallocate(ttnn_typecast_55, False)
    ttnn.deallocate(ttnn_typecast_54, False)
    ttnn.deallocate(ttnn_typecast_53, False)
    ttnn.deallocate(ttnn_typecast_52, False)
    ttnn.deallocate(ttnn_typecast_51, False)
    ttnn.deallocate(ttnn_typecast_50, False)
    ttnn.deallocate(ttnn_typecast_49, False)
    ttnn.deallocate(ttnn_typecast_48, False)
    ttnn.deallocate(ttnn_typecast_47, False)
    ttnn.deallocate(ttnn_typecast_46, False)
    ttnn.deallocate(ttnn_typecast_45, False)
    ttnn.deallocate(ttnn_typecast_44, False)
    ttnn.deallocate(ttnn_typecast_43, False)
    ttnn.deallocate(ttnn_typecast_42, False)
    ttnn.deallocate(ttnn_typecast_41, False)
    ttnn.deallocate(ttnn_typecast_40, False)
    ttnn.deallocate(ttnn_typecast_39, False)
    ttnn.deallocate(ttnn_typecast_38, False)
    ttnn.deallocate(ttnn_typecast_37, False)
    ttnn.deallocate(ttnn_typecast_36, False)
    ttnn.deallocate(ttnn_typecast_35, False)
    ttnn.deallocate(ttnn_typecast_34, False)
    ttnn.deallocate(ttnn_typecast_33, False)
    ttnn.deallocate(ttnn_typecast_32, False)
    ttnn.deallocate(ttnn_typecast_31, False)
    ttnn.deallocate(ttnn_typecast_30, False)
    ttnn.deallocate(ttnn_typecast_29, False)
    ttnn.deallocate(ttnn_typecast_28, False)
    ttnn.deallocate(ttnn_typecast_27, False)
    ttnn.deallocate(ttnn_typecast_26, False)
    ttnn.deallocate(ttnn_typecast_25, False)
    ttnn.deallocate(ttnn_typecast_24, False)
    ttnn.deallocate(ttnn_typecast_23, False)
    return [ttnn_concat_50]


def main_const_eval_168(device, arg):
    ttnn_to_device_437 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_to_layout_437 = ttnn.to_layout(
        ttnn_to_device_437,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_437, False)
    ttnn_permute_201 = ttnn.permute(
        ttnn_to_layout_437,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_437, False)
    ttnn_typecast_93 = ttnn.typecast(
        ttnn_permute_201,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_201, False)
    return [ttnn_typecast_93]


def main_const_eval_176(device):
    ttnn_Tensor_1 = ttnn.Tensor(
        [
            1.0,
            0.31622776389122009,
            0.10000000149011612,
            0.031622778624296188,
            0.0099999997764825821,
            0.0031622778624296188,
            0.0010000000474974513,
            0.00031622778624296188,
        ],
        [1, 8],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_Tensor_1]


def main_const_eval_187(device, arg):
    ttnn_to_device_464 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mesh_partition_0 = ttnn.mesh_partition(
        input_tensor=ttnn_to_device_464,
        dim=0,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_464, False)
    ttnn_to_layout_464 = ttnn.to_layout(
        ttnn_mesh_partition_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mesh_partition_0, False)
    ttnn_permute_206 = ttnn.permute(
        ttnn_to_layout_464,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_464, False)
    ttnn_typecast_100 = ttnn.typecast(
        ttnn_permute_206,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_206, False)
    return [ttnn_typecast_100]


def main_const_eval_199(device, arg):
    ttnn_to_device_487 = ttnn.to_device(
        arg[0],
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_mesh_partition_1 = ttnn.mesh_partition(
        input_tensor=ttnn_to_device_487,
        dim=0,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_487, False)
    ttnn_to_layout_487 = ttnn.to_layout(
        ttnn_mesh_partition_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_mesh_partition_1, False)
    ttnn_typecast_103 = ttnn.typecast(
        ttnn_to_layout_487,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_487, False)
    return [ttnn_typecast_103]


def main_const_eval_233(device):
    ttnn_Tensor_2 = ttnn.Tensor(
        [
            1.0,
            0.71968567371368408,
            0.51794743537902832,
            0.3727593719959259,
            0.26826956868171692,
            0.19306977093219757,
            0.13894954323768616,
            0.10000000149011612,
            0.071968555450439453,
            0.05179474875330925,
            0.037275936454534531,
            0.026826959103345871,
            0.019306976348161697,
            0.013894956558942795,
            0.0099999997764825821,
            0.0071968580596148968,
            0.0051794731989502907,
            0.0037275934591889381,
            0.002682696096599102,
            0.001930698286741972,
            0.0013894952135160565,
            0.0010000000474974513,
            0.00071968580596148968,
            0.00051794730825349689,
            0.00037275932845659554,
            0.00026826959219761193,
            0.00019306980539113283,
            0.00013894953008275479,
        ],
        [1, 28],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_Tensor_2]


def main_const_eval_238(device):
    ttnn_ones_0 = ttnn.ones(
        shape=ttnn.Shape([1, 1, 1]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    return [ttnn_ones_0]


def run_consteval(weights, device):
    weights['consteval.const_0'] = main_const_eval_0(device)[0]
    weights['transformer.single_transformer_blocks.10.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.10.attn.to_q.bias'], weights['transformer.single_transformer_blocks.10.attn.to_k.bias'], weights['transformer.single_transformer_blocks.10.attn.to_v.bias'], weights['transformer.single_transformer_blocks.10.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.1.attn.fused_to_q_to_k_to_v.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.1.attn.to_q.weight'], weights['transformer.transformer_blocks.1.attn.to_k.weight'], weights['transformer.transformer_blocks.1.attn.to_v.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.3.ff_context.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.3.ff_context.net.2.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.22.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.22.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.transformer_blocks.6.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.6.attn.add_q_proj.bias'], weights['transformer.transformer_blocks.6.attn.add_k_proj.bias'], weights['transformer.transformer_blocks.6.attn.add_v_proj.bias']], 0)
    weights['transformer.single_transformer_blocks.16.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.16.attn.to_q.weight'], weights['transformer.single_transformer_blocks.16.attn.to_k.weight'], weights['transformer.single_transformer_blocks.16.attn.to_v.weight'], weights['transformer.single_transformer_blocks.16.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.37.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.37.attn.to_q.bias'], weights['transformer.single_transformer_blocks.37.attn.to_k.bias'], weights['transformer.single_transformer_blocks.37.attn.to_v.bias'], weights['transformer.single_transformer_blocks.37.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.0.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.0.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.3.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.3.attn.add_q_proj.bias'], weights['transformer.transformer_blocks.3.attn.add_k_proj.bias'], weights['transformer.transformer_blocks.3.attn.add_v_proj.bias']], 0)
    weights['transformer.transformer_blocks.5.norm1.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.5.norm1.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.transformer_blocks.7.attn.to_add_out.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.7.attn.to_add_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.12.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.12.attn.to_q.weight'], weights['transformer.single_transformer_blocks.12.attn.to_k.weight'], weights['transformer.single_transformer_blocks.12.attn.to_v.weight'], weights['transformer.single_transformer_blocks.12.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.4.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.4.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.6.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.6.attn.to_q.bias'], weights['transformer.single_transformer_blocks.6.attn.to_k.bias'], weights['transformer.single_transformer_blocks.6.attn.to_v.bias'], weights['transformer.single_transformer_blocks.6.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.30.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.30.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.23.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.23.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.7.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.7.attn.add_q_proj.weight'], weights['transformer.transformer_blocks.7.attn.add_k_proj.weight'], weights['transformer.transformer_blocks.7.attn.add_v_proj.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.6.ff_context.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.6.ff_context.net.2.bias'], [1, 3072])
    weights['transformer.transformer_blocks.0.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.0.attn.add_q_proj.bias'], weights['transformer.transformer_blocks.0.attn.add_k_proj.bias'], weights['transformer.transformer_blocks.0.attn.add_v_proj.bias']], 0)
    weights['transformer.single_transformer_blocks.8.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.8.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.20.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.20.attn.to_q.weight'], weights['transformer.single_transformer_blocks.20.attn.to_k.weight'], weights['transformer.single_transformer_blocks.20.attn.to_v.weight'], weights['transformer.single_transformer_blocks.20.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.7.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.7.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.transformer_blocks.0.ff_context.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.0.ff_context.net.2.bias'], [1, 3072])
    weights['transformer.transformer_blocks.3.norm1_context.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.3.norm1_context.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.31.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.31.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.18.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.18.attn.to_q.bias'], weights['transformer.single_transformer_blocks.18.attn.to_k.bias'], weights['transformer.single_transformer_blocks.18.attn.to_v.bias'], weights['transformer.single_transformer_blocks.18.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.35.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.35.attn.to_q.weight'], weights['transformer.single_transformer_blocks.35.attn.to_k.weight'], weights['transformer.single_transformer_blocks.35.attn.to_v.weight'], weights['transformer.single_transformer_blocks.35.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.7.attn.fused_to_q_to_k_to_v.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.7.attn.to_q.weight'], weights['transformer.transformer_blocks.7.attn.to_k.weight'], weights['transformer.transformer_blocks.7.attn.to_v.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.1.attn.to_out.0.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.1.attn.to_out.0.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.33.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.33.attn.to_q.bias'], weights['transformer.single_transformer_blocks.33.attn.to_k.bias'], weights['transformer.single_transformer_blocks.33.attn.to_v.bias'], weights['transformer.single_transformer_blocks.33.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.37.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.37.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.16.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.16.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.15.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.15.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.3.attn.fused_to_q_to_k_to_v.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.3.attn.to_q.bias'], weights['transformer.transformer_blocks.3.attn.to_k.bias'], weights['transformer.transformer_blocks.3.attn.to_v.bias']], 0)
    weights['transformer.transformer_blocks.2.ff.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.2.ff.net.2.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.14.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.14.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.34.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.34.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.norm_out.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.norm_out.linear.bias'], [1, 1, 6144], ttnn.DataType.FLOAT32, [1, 6144])
    weights['transformer.transformer_blocks.1.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.1.attn.add_q_proj.weight'], weights['transformer.transformer_blocks.1.attn.add_k_proj.weight'], weights['transformer.transformer_blocks.1.attn.add_v_proj.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.4.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.4.attn.add_q_proj.weight'], weights['transformer.transformer_blocks.4.attn.add_k_proj.weight'], weights['transformer.transformer_blocks.4.attn.add_v_proj.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.4.attn.to_add_out.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.4.attn.to_add_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.1.attn.to_add_out.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.1.attn.to_add_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.32.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.32.attn.to_q.weight'], weights['transformer.single_transformer_blocks.32.attn.to_k.weight'], weights['transformer.single_transformer_blocks.32.attn.to_v.weight'], weights['transformer.single_transformer_blocks.32.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.4.attn.fused_to_q_to_k_to_v.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.4.attn.to_q.weight'], weights['transformer.transformer_blocks.4.attn.to_k.weight'], weights['transformer.transformer_blocks.4.attn.to_v.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.36.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.36.attn.to_q.weight'], weights['transformer.single_transformer_blocks.36.attn.to_k.weight'], weights['transformer.single_transformer_blocks.36.attn.to_v.weight'], weights['transformer.single_transformer_blocks.36.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.19.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.19.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.21.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.21.attn.to_q.bias'], weights['transformer.single_transformer_blocks.21.attn.to_k.bias'], weights['transformer.single_transformer_blocks.21.attn.to_v.bias'], weights['transformer.single_transformer_blocks.21.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.9.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.9.attn.to_q.weight'], weights['transformer.single_transformer_blocks.9.attn.to_k.weight'], weights['transformer.single_transformer_blocks.9.attn.to_v.weight'], weights['transformer.single_transformer_blocks.9.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.2.norm1.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.2.norm1.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.18.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.18.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.transformer_blocks.2.attn.fused_to_q_to_k_to_v.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.2.attn.to_q.bias'], weights['transformer.transformer_blocks.2.attn.to_k.bias'], weights['transformer.transformer_blocks.2.attn.to_v.bias']], 0)
    weights['transformer.single_transformer_blocks.35.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.35.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.6.attn.fused_to_q_to_k_to_v.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.6.attn.to_q.bias'], weights['transformer.transformer_blocks.6.attn.to_k.bias'], weights['transformer.transformer_blocks.6.attn.to_v.bias']], 0)
    weights['transformer.single_transformer_blocks.17.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.17.attn.to_q.bias'], weights['transformer.single_transformer_blocks.17.attn.to_k.bias'], weights['transformer.single_transformer_blocks.17.attn.to_v.bias'], weights['transformer.single_transformer_blocks.17.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.14.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.14.attn.to_q.bias'], weights['transformer.single_transformer_blocks.14.attn.to_k.bias'], weights['transformer.single_transformer_blocks.14.attn.to_v.bias'], weights['transformer.single_transformer_blocks.14.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.6.norm1_context.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.6.norm1_context.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.19.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.19.attn.to_q.weight'], weights['transformer.single_transformer_blocks.19.attn.to_k.weight'], weights['transformer.single_transformer_blocks.19.attn.to_v.weight'], weights['transformer.single_transformer_blocks.19.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.4.attn.to_out.0.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.4.attn.to_out.0.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.12.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.12.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.11.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.11.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.36.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.36.attn.to_q.bias'], weights['transformer.single_transformer_blocks.36.attn.to_k.bias'], weights['transformer.single_transformer_blocks.36.attn.to_v.bias'], weights['transformer.single_transformer_blocks.36.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.17.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.17.attn.to_q.weight'], weights['transformer.single_transformer_blocks.17.attn.to_k.weight'], weights['transformer.single_transformer_blocks.17.attn.to_v.weight'], weights['transformer.single_transformer_blocks.17.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.6.ff.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.6.ff.net.2.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.9.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.9.attn.to_q.bias'], weights['transformer.single_transformer_blocks.9.attn.to_k.bias'], weights['transformer.single_transformer_blocks.9.attn.to_v.bias'], weights['transformer.single_transformer_blocks.9.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.1.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.1.attn.add_q_proj.bias'], weights['transformer.transformer_blocks.1.attn.add_k_proj.bias'], weights['transformer.transformer_blocks.1.attn.add_v_proj.bias']], 0)
    weights['transformer.single_transformer_blocks.25.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.25.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.28.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.28.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.34.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.34.attn.to_q.bias'], weights['transformer.single_transformer_blocks.34.attn.to_k.bias'], weights['transformer.single_transformer_blocks.34.attn.to_v.bias'], weights['transformer.single_transformer_blocks.34.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.0.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.0.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.transformer_blocks.5.attn.to_add_out.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.5.attn.to_add_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.0.attn.fused_to_q_to_k_to_v.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.0.attn.to_q.weight'], weights['transformer.transformer_blocks.0.attn.to_k.weight'], weights['transformer.transformer_blocks.0.attn.to_v.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.19.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.19.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.transformer_blocks.5.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.5.attn.add_q_proj.weight'], weights['transformer.transformer_blocks.5.attn.add_k_proj.weight'], weights['transformer.transformer_blocks.5.attn.add_v_proj.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.7.ff.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.7.ff.net.2.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.34.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.34.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.3.ff.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.3.ff.net.2.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.3.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.3.attn.to_q.bias'], weights['transformer.single_transformer_blocks.3.attn.to_k.bias'], weights['transformer.single_transformer_blocks.3.attn.to_v.bias'], weights['transformer.single_transformer_blocks.3.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.7.norm1.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.7.norm1.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.23.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.23.attn.to_q.weight'], weights['transformer.single_transformer_blocks.23.attn.to_k.weight'], weights['transformer.single_transformer_blocks.23.attn.to_v.weight'], weights['transformer.single_transformer_blocks.23.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.5.norm1_context.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.5.norm1_context.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.15.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.15.attn.to_q.bias'], weights['transformer.single_transformer_blocks.15.attn.to_k.bias'], weights['transformer.single_transformer_blocks.15.attn.to_v.bias'], weights['transformer.single_transformer_blocks.15.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.30.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.30.attn.to_q.bias'], weights['transformer.single_transformer_blocks.30.attn.to_k.bias'], weights['transformer.single_transformer_blocks.30.attn.to_v.bias'], weights['transformer.single_transformer_blocks.30.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.26.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.26.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.27.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.27.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.transformer_blocks.2.ff_context.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.2.ff_context.net.2.bias'], [1, 3072])
    weights['transformer.transformer_blocks.3.attn.fused_to_q_to_k_to_v.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.3.attn.to_q.weight'], weights['transformer.transformer_blocks.3.attn.to_k.weight'], weights['transformer.transformer_blocks.3.attn.to_v.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.1.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.1.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.25.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.25.attn.to_q.weight'], weights['transformer.single_transformer_blocks.25.attn.to_k.weight'], weights['transformer.single_transformer_blocks.25.attn.to_v.weight'], weights['transformer.single_transformer_blocks.25.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.28.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.28.attn.to_q.bias'], weights['transformer.single_transformer_blocks.28.attn.to_k.bias'], weights['transformer.single_transformer_blocks.28.attn.to_v.bias'], weights['transformer.single_transformer_blocks.28.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.5.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.5.attn.to_q.weight'], weights['transformer.single_transformer_blocks.5.attn.to_k.weight'], weights['transformer.single_transformer_blocks.5.attn.to_v.weight'], weights['transformer.single_transformer_blocks.5.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.3.attn.to_out.0.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.3.attn.to_out.0.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.23.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.23.attn.to_q.bias'], weights['transformer.single_transformer_blocks.23.attn.to_k.bias'], weights['transformer.single_transformer_blocks.23.attn.to_v.bias'], weights['transformer.single_transformer_blocks.23.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.4.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.4.attn.to_q.weight'], weights['transformer.single_transformer_blocks.4.attn.to_k.weight'], weights['transformer.single_transformer_blocks.4.attn.to_v.weight'], weights['transformer.single_transformer_blocks.4.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.30.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.30.attn.to_q.weight'], weights['transformer.single_transformer_blocks.30.attn.to_k.weight'], weights['transformer.single_transformer_blocks.30.attn.to_v.weight'], weights['transformer.single_transformer_blocks.30.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.time_embed.timestep_embedder.linear_1.bias.fused.transformer.time_embed.timestep_embedder.linear_1.bias'] = main_const_eval_95(device, [weights['transformer.time_embed.timestep_embedder.linear_1.bias']])[0]
    weights['transformer.transformer_blocks.5.ff.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.5.ff.net.2.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.12.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.12.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.31.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.31.attn.to_q.weight'], weights['transformer.single_transformer_blocks.31.attn.to_k.weight'], weights['transformer.single_transformer_blocks.31.attn.to_v.weight'], weights['transformer.single_transformer_blocks.31.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.22.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.22.attn.to_q.bias'], weights['transformer.single_transformer_blocks.22.attn.to_k.bias'], weights['transformer.single_transformer_blocks.22.attn.to_v.bias'], weights['transformer.single_transformer_blocks.22.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.5.attn.fused_to_q_to_k_to_v.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.5.attn.to_q.bias'], weights['transformer.transformer_blocks.5.attn.to_k.bias'], weights['transformer.transformer_blocks.5.attn.to_v.bias']], 0)
    weights['transformer.single_transformer_blocks.13.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.13.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.20.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.20.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.6.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.6.attn.add_q_proj.weight'], weights['transformer.transformer_blocks.6.attn.add_k_proj.weight'], weights['transformer.transformer_blocks.6.attn.add_v_proj.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.13.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.13.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.33.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.33.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.transformer_blocks.6.attn.to_add_out.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.6.attn.to_add_out.bias'], [1, 3072])
    weights['transformer.fused_norm_out_single_transformer_blocks_37_norm_36_35_34_33_32_31_30_29_28_27_26_25_24_23_22_21_20_19_18_17_16_15_14_13_12_11_10_9_8_7_6_5_4_3_2_1_0_transformer_blocks_norm1_norm1_context.linear.weight'] = main_const_eval_107(device, [weights['transformer.norm_out.linear.weight'], weights['transformer.single_transformer_blocks.37.norm.linear.weight'], weights['transformer.single_transformer_blocks.36.norm.linear.weight'], weights['transformer.single_transformer_blocks.35.norm.linear.weight'], weights['transformer.single_transformer_blocks.34.norm.linear.weight'], weights['transformer.single_transformer_blocks.33.norm.linear.weight'], weights['transformer.single_transformer_blocks.32.norm.linear.weight'], weights['transformer.single_transformer_blocks.31.norm.linear.weight'], weights['transformer.single_transformer_blocks.30.norm.linear.weight'], weights['transformer.single_transformer_blocks.29.norm.linear.weight'], weights['transformer.single_transformer_blocks.28.norm.linear.weight'], weights['transformer.single_transformer_blocks.27.norm.linear.weight'], weights['transformer.single_transformer_blocks.26.norm.linear.weight'], weights['transformer.single_transformer_blocks.25.norm.linear.weight'], weights['transformer.single_transformer_blocks.24.norm.linear.weight'], weights['transformer.single_transformer_blocks.23.norm.linear.weight'], weights['transformer.single_transformer_blocks.22.norm.linear.weight'], weights['transformer.single_transformer_blocks.21.norm.linear.weight'], weights['transformer.single_transformer_blocks.20.norm.linear.weight'], weights['transformer.single_transformer_blocks.19.norm.linear.weight'], weights['transformer.single_transformer_blocks.18.norm.linear.weight'], weights['transformer.single_transformer_blocks.17.norm.linear.weight'], weights['transformer.single_transformer_blocks.16.norm.linear.weight'], weights['transformer.single_transformer_blocks.15.norm.linear.weight'], weights['transformer.single_transformer_blocks.14.norm.linear.weight'], weights['transformer.single_transformer_blocks.13.norm.linear.weight'], weights['transformer.single_transformer_blocks.12.norm.linear.weight'], weights['transformer.single_transformer_blocks.11.norm.linear.weight'], weights['transformer.single_transformer_blocks.10.norm.linear.weight'], weights['transformer.single_transformer_blocks.9.norm.linear.weight'], weights['transformer.single_transformer_blocks.8.norm.linear.weight'], weights['transformer.single_transformer_blocks.7.norm.linear.weight'], weights['transformer.single_transformer_blocks.6.norm.linear.weight'], weights['transformer.single_transformer_blocks.5.norm.linear.weight'], weights['transformer.single_transformer_blocks.4.norm.linear.weight'], weights['transformer.single_transformer_blocks.3.norm.linear.weight'], weights['transformer.single_transformer_blocks.2.norm.linear.weight'], weights['transformer.single_transformer_blocks.1.norm.linear.weight'], weights['transformer.single_transformer_blocks.0.norm.linear.weight'], weights['transformer.transformer_blocks.7.norm1.linear.weight'], weights['transformer.transformer_blocks.6.norm1.linear.weight'], weights['transformer.transformer_blocks.5.norm1.linear.weight'], weights['transformer.transformer_blocks.4.norm1.linear.weight'], weights['transformer.transformer_blocks.3.norm1.linear.weight'], weights['transformer.transformer_blocks.2.norm1.linear.weight'], weights['transformer.transformer_blocks.1.norm1.linear.weight'], weights['transformer.transformer_blocks.0.norm1.linear.weight'], weights['transformer.transformer_blocks.0.norm1_context.linear.weight'], weights['transformer.transformer_blocks.1.norm1_context.linear.weight'], weights['transformer.transformer_blocks.2.norm1_context.linear.weight'], weights['transformer.transformer_blocks.3.norm1_context.linear.weight'], weights['transformer.transformer_blocks.4.norm1_context.linear.weight'], weights['transformer.transformer_blocks.5.norm1_context.linear.weight'], weights['transformer.transformer_blocks.6.norm1_context.linear.weight'], weights['transformer.transformer_blocks.7.norm1_context.linear.weight']])[0]
    weights['transformer.single_transformer_blocks.14.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.14.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.1.ff_context.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.1.ff_context.net.2.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.29.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.29.attn.to_q.bias'], weights['transformer.single_transformer_blocks.29.attn.to_k.bias'], weights['transformer.single_transformer_blocks.29.attn.to_v.bias'], weights['transformer.single_transformer_blocks.29.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.4.norm1_context.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.4.norm1_context.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.24.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.24.attn.to_q.weight'], weights['transformer.single_transformer_blocks.24.attn.to_k.weight'], weights['transformer.single_transformer_blocks.24.attn.to_v.weight'], weights['transformer.single_transformer_blocks.24.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.4.ff.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.4.ff.net.2.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.11.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.11.attn.to_q.weight'], weights['transformer.single_transformer_blocks.11.attn.to_k.weight'], weights['transformer.single_transformer_blocks.11.attn.to_v.weight'], weights['transformer.single_transformer_blocks.11.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.2.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.2.attn.to_q.bias'], weights['transformer.single_transformer_blocks.2.attn.to_k.bias'], weights['transformer.single_transformer_blocks.2.attn.to_v.bias'], weights['transformer.single_transformer_blocks.2.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.2.attn.to_out.0.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.2.attn.to_out.0.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.7.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.7.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.27.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.27.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.2.attn.fused_to_q_to_k_to_v.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.2.attn.to_q.weight'], weights['transformer.transformer_blocks.2.attn.to_k.weight'], weights['transformer.transformer_blocks.2.attn.to_v.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.6.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.6.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.26.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.26.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.24.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.24.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.35.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.35.attn.to_q.bias'], weights['transformer.single_transformer_blocks.35.attn.to_k.bias'], weights['transformer.single_transformer_blocks.35.attn.to_v.bias'], weights['transformer.single_transformer_blocks.35.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.7.norm1_context.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.7.norm1_context.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.18.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.18.attn.to_q.weight'], weights['transformer.single_transformer_blocks.18.attn.to_k.weight'], weights['transformer.single_transformer_blocks.18.attn.to_v.weight'], weights['transformer.single_transformer_blocks.18.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.1.ff.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.1.ff.net.2.bias'], [1, 3072])
    weights['transformer.transformer_blocks.4.attn.fused_to_q_to_k_to_v.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.4.attn.to_q.bias'], weights['transformer.transformer_blocks.4.attn.to_k.bias'], weights['transformer.transformer_blocks.4.attn.to_v.bias']], 0)
    weights['transformer.single_transformer_blocks.12.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.12.attn.to_q.bias'], weights['transformer.single_transformer_blocks.12.attn.to_k.bias'], weights['transformer.single_transformer_blocks.12.attn.to_v.bias'], weights['transformer.single_transformer_blocks.12.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.3.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.3.attn.add_q_proj.weight'], weights['transformer.transformer_blocks.3.attn.add_k_proj.weight'], weights['transformer.transformer_blocks.3.attn.add_v_proj.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.33.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.33.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.28.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.28.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.transformer_blocks.0.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.0.attn.add_q_proj.weight'], weights['transformer.transformer_blocks.0.attn.add_k_proj.weight'], weights['transformer.transformer_blocks.0.attn.add_v_proj.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.31.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.31.attn.to_q.bias'], weights['transformer.single_transformer_blocks.31.attn.to_k.bias'], weights['transformer.single_transformer_blocks.31.attn.to_v.bias'], weights['transformer.single_transformer_blocks.31.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.0.attn.to_add_out.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.0.attn.to_add_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.22.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.22.attn.to_q.weight'], weights['transformer.single_transformer_blocks.22.attn.to_k.weight'], weights['transformer.single_transformer_blocks.22.attn.to_v.weight'], weights['transformer.single_transformer_blocks.22.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.37.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.37.attn.to_q.weight'], weights['transformer.single_transformer_blocks.37.attn.to_k.weight'], weights['transformer.single_transformer_blocks.37.attn.to_v.weight'], weights['transformer.single_transformer_blocks.37.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.29.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.29.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.10.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.10.attn.to_q.weight'], weights['transformer.single_transformer_blocks.10.attn.to_k.weight'], weights['transformer.single_transformer_blocks.10.attn.to_v.weight'], weights['transformer.single_transformer_blocks.10.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.16.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.16.attn.to_q.bias'], weights['transformer.single_transformer_blocks.16.attn.to_k.bias'], weights['transformer.single_transformer_blocks.16.attn.to_v.bias'], weights['transformer.single_transformer_blocks.16.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.1.norm1.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.1.norm1.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.32.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.32.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.transformer_blocks.1.attn.fused_to_q_to_k_to_v.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.1.attn.to_q.bias'], weights['transformer.transformer_blocks.1.attn.to_k.bias'], weights['transformer.transformer_blocks.1.attn.to_v.bias']], 0)
    weights['transformer.transformer_blocks.7.attn.fused_to_q_to_k_to_v.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.7.attn.to_q.bias'], weights['transformer.transformer_blocks.7.attn.to_k.bias'], weights['transformer.transformer_blocks.7.attn.to_v.bias']], 0)
    weights['transformer.single_transformer_blocks.5.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.5.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.20.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.20.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.transformer_blocks.5.attn.fused_to_q_to_k_to_v.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.5.attn.to_q.weight'], weights['transformer.transformer_blocks.5.attn.to_k.weight'], weights['transformer.transformer_blocks.5.attn.to_v.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.1.norm1_context.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.1.norm1_context.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.6.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.6.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.21.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.21.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.8.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.8.attn.to_q.bias'], weights['transformer.single_transformer_blocks.8.attn.to_k.bias'], weights['transformer.single_transformer_blocks.8.attn.to_v.bias'], weights['transformer.single_transformer_blocks.8.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.5.attn.to_out.0.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.5.attn.to_out.0.bias'], [1, 3072])
    weights['transformer.transformer_blocks.7.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.7.attn.add_q_proj.bias'], weights['transformer.transformer_blocks.7.attn.add_k_proj.bias'], weights['transformer.transformer_blocks.7.attn.add_v_proj.bias']], 0)
    weights['transformer.single_transformer_blocks.29.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.29.attn.to_q.weight'], weights['transformer.single_transformer_blocks.29.attn.to_k.weight'], weights['transformer.single_transformer_blocks.29.attn.to_v.weight'], weights['transformer.single_transformer_blocks.29.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.3.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.3.attn.to_q.weight'], weights['transformer.single_transformer_blocks.3.attn.to_k.weight'], weights['transformer.single_transformer_blocks.3.attn.to_v.weight'], weights['transformer.single_transformer_blocks.3.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.35.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.35.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.18.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.18.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.0.norm1_context.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.0.norm1_context.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.24.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.24.attn.to_q.bias'], weights['transformer.single_transformer_blocks.24.attn.to_k.bias'], weights['transformer.single_transformer_blocks.24.attn.to_v.bias'], weights['transformer.single_transformer_blocks.24.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.2.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.2.attn.add_q_proj.bias'], weights['transformer.transformer_blocks.2.attn.add_k_proj.bias'], weights['transformer.transformer_blocks.2.attn.add_v_proj.bias']], 0)
    weights['transformer.single_transformer_blocks.6.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.6.attn.to_q.weight'], weights['transformer.single_transformer_blocks.6.attn.to_k.weight'], weights['transformer.single_transformer_blocks.6.attn.to_v.weight'], weights['transformer.single_transformer_blocks.6.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.33.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.33.attn.to_q.weight'], weights['transformer.single_transformer_blocks.33.attn.to_k.weight'], weights['transformer.single_transformer_blocks.33.attn.to_v.weight'], weights['transformer.single_transformer_blocks.33.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.0.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.0.attn.to_q.bias'], weights['transformer.single_transformer_blocks.0.attn.to_k.bias'], weights['transformer.single_transformer_blocks.0.attn.to_v.bias'], weights['transformer.single_transformer_blocks.0.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.8.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.8.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.20.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.20.attn.to_q.bias'], weights['transformer.single_transformer_blocks.20.attn.to_k.bias'], weights['transformer.single_transformer_blocks.20.attn.to_v.bias'], weights['transformer.single_transformer_blocks.20.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.2.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.2.attn.to_q.weight'], weights['transformer.single_transformer_blocks.2.attn.to_k.weight'], weights['transformer.single_transformer_blocks.2.attn.to_v.weight'], weights['transformer.single_transformer_blocks.2.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.1.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.1.attn.to_q.bias'], weights['transformer.single_transformer_blocks.1.attn.to_k.bias'], weights['transformer.single_transformer_blocks.1.attn.to_v.bias'], weights['transformer.single_transformer_blocks.1.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.4.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.4.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.time_embed.timestep_embedder.linear_1.weight.fused.transformer.time_embed.timestep_embedder.linear_1.weight'] = main_const_eval_168(device, [weights['transformer.time_embed.timestep_embedder.linear_1.weight']])[0]
    weights['transformer.single_transformer_blocks.22.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.22.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.4.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.4.attn.to_q.bias'], weights['transformer.single_transformer_blocks.4.attn.to_k.bias'], weights['transformer.single_transformer_blocks.4.attn.to_v.bias'], weights['transformer.single_transformer_blocks.4.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.6.norm1.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.6.norm1.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.transformer_blocks.7.ff_context.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.7.ff_context.net.2.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.25.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.25.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.26.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.26.attn.to_q.weight'], weights['transformer.single_transformer_blocks.26.attn.to_k.weight'], weights['transformer.single_transformer_blocks.26.attn.to_v.weight'], weights['transformer.single_transformer_blocks.26.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.27.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.27.attn.to_q.bias'], weights['transformer.single_transformer_blocks.27.attn.to_k.bias'], weights['transformer.single_transformer_blocks.27.attn.to_v.bias'], weights['transformer.single_transformer_blocks.27.proj_mlp.bias']], 0)
    weights['consteval.const_176'] = main_const_eval_176(device)[0]
    weights['transformer.single_transformer_blocks.1.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.1.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.32.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.32.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.0.attn.to_out.0.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.0.attn.to_out.0.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.2.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.2.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.21.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.21.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.31.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.31.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.30.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.30.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.2.norm1_context.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.2.norm1_context.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.23.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.23.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.5.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.5.proj_out.bias'], [1, 3072])
    weights['transformer.time_embed.timestep_embedder.linear_2.weight.fused.transformer.time_embed.timestep_embedder.linear_2.weight'] = main_const_eval_187(device, [weights['transformer.time_embed.timestep_embedder.linear_2.weight']])[0]
    weights['transformer.single_transformer_blocks.3.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.3.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.7.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.7.attn.to_q.bias'], weights['transformer.single_transformer_blocks.7.attn.to_k.bias'], weights['transformer.single_transformer_blocks.7.attn.to_v.bias'], weights['transformer.single_transformer_blocks.7.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.13.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.13.attn.to_q.weight'], weights['transformer.single_transformer_blocks.13.attn.to_k.weight'], weights['transformer.single_transformer_blocks.13.attn.to_v.weight'], weights['transformer.single_transformer_blocks.13.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.0.attn.fused_to_q_to_k_to_v.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.0.attn.to_q.bias'], weights['transformer.transformer_blocks.0.attn.to_k.bias'], weights['transformer.transformer_blocks.0.attn.to_v.bias']], 0)
    weights['transformer.transformer_blocks.3.norm1.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.3.norm1.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.single_transformer_blocks.11.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.11.attn.to_q.bias'], weights['transformer.single_transformer_blocks.11.attn.to_k.bias'], weights['transformer.single_transformer_blocks.11.attn.to_v.bias'], weights['transformer.single_transformer_blocks.11.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.9.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.9.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.2.attn.to_add_out.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.2.attn.to_add_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.24.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.24.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.17.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.17.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.36.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.36.proj_out.bias'], [1, 3072])
    weights['transformer.time_embed.timestep_embedder.linear_2.bias.fused.transformer.time_embed.timestep_embedder.linear_2.bias'] = main_const_eval_199(device, [weights['transformer.time_embed.timestep_embedder.linear_2.bias']])[0]
    weights['transformer.single_transformer_blocks.11.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.11.proj_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.5.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.5.attn.to_q.bias'], weights['transformer.single_transformer_blocks.5.attn.to_k.bias'], weights['transformer.single_transformer_blocks.5.attn.to_v.bias'], weights['transformer.single_transformer_blocks.5.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.4.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.4.attn.add_q_proj.bias'], weights['transformer.transformer_blocks.4.attn.add_k_proj.bias'], weights['transformer.transformer_blocks.4.attn.add_v_proj.bias']], 0)
    weights['transformer.single_transformer_blocks.15.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.15.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.13.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.13.attn.to_q.bias'], weights['transformer.single_transformer_blocks.13.attn.to_k.bias'], weights['transformer.single_transformer_blocks.13.attn.to_v.bias'], weights['transformer.single_transformer_blocks.13.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.0.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.0.attn.to_q.weight'], weights['transformer.single_transformer_blocks.0.attn.to_k.weight'], weights['transformer.single_transformer_blocks.0.attn.to_v.weight'], weights['transformer.single_transformer_blocks.0.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.2.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.2.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.15.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.15.attn.to_q.weight'], weights['transformer.single_transformer_blocks.15.attn.to_k.weight'], weights['transformer.single_transformer_blocks.15.attn.to_v.weight'], weights['transformer.single_transformer_blocks.15.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.0.norm1.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.0.norm1.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.transformer_blocks.4.ff_context.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.4.ff_context.net.2.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.17.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.17.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.4.norm1.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.transformer_blocks.4.norm1.linear.bias'], [1, 1, 18432], ttnn.DataType.FLOAT32, [1, 18432])
    weights['transformer.transformer_blocks.0.ff.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.0.ff.net.2.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.28.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.28.attn.to_q.weight'], weights['transformer.single_transformer_blocks.28.attn.to_k.weight'], weights['transformer.single_transformer_blocks.28.attn.to_v.weight'], weights['transformer.single_transformer_blocks.28.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.10.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.10.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.1.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.1.attn.to_q.weight'], weights['transformer.single_transformer_blocks.1.attn.to_k.weight'], weights['transformer.single_transformer_blocks.1.attn.to_v.weight'], weights['transformer.single_transformer_blocks.1.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.7.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.7.attn.to_q.weight'], weights['transformer.single_transformer_blocks.7.attn.to_k.weight'], weights['transformer.single_transformer_blocks.7.attn.to_v.weight'], weights['transformer.single_transformer_blocks.7.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.26.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.26.attn.to_q.bias'], weights['transformer.single_transformer_blocks.26.attn.to_k.bias'], weights['transformer.single_transformer_blocks.26.attn.to_v.bias'], weights['transformer.single_transformer_blocks.26.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.8.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.8.attn.to_q.weight'], weights['transformer.single_transformer_blocks.8.attn.to_k.weight'], weights['transformer.single_transformer_blocks.8.attn.to_v.weight'], weights['transformer.single_transformer_blocks.8.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.25.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.25.attn.to_q.bias'], weights['transformer.single_transformer_blocks.25.attn.to_k.bias'], weights['transformer.single_transformer_blocks.25.attn.to_v.bias'], weights['transformer.single_transformer_blocks.25.proj_mlp.bias']], 0)
    weights['transformer.transformer_blocks.3.attn.to_add_out.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.3.attn.to_add_out.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.9.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.9.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.27.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.27.attn.to_q.weight'], weights['transformer.single_transformer_blocks.27.attn.to_k.weight'], weights['transformer.single_transformer_blocks.27.attn.to_v.weight'], weights['transformer.single_transformer_blocks.27.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.10.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.10.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.2.attn.fused_add_q_proj_add_k_proj_add_v_proj.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.2.attn.add_q_proj.weight'], weights['transformer.transformer_blocks.2.attn.add_k_proj.weight'], weights['transformer.transformer_blocks.2.attn.add_v_proj.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.6.attn.to_out.0.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.6.attn.to_out.0.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.3.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.3.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.29.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.29.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.37.proj_out.bias.reshaped'] = _reshape(device, weights['transformer.single_transformer_blocks.37.proj_out.bias'], [1, 3072])
    weights['transformer.transformer_blocks.5.attn.fused_add_q_proj_add_k_proj_add_v_proj.bias'] = _tile_concat(device, [weights['transformer.transformer_blocks.5.attn.add_q_proj.bias'], weights['transformer.transformer_blocks.5.attn.add_k_proj.bias'], weights['transformer.transformer_blocks.5.attn.add_v_proj.bias']], 0)
    weights['transformer.single_transformer_blocks.32.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.32.attn.to_q.bias'], weights['transformer.single_transformer_blocks.32.attn.to_k.bias'], weights['transformer.single_transformer_blocks.32.attn.to_v.bias'], weights['transformer.single_transformer_blocks.32.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.21.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.21.attn.to_q.weight'], weights['transformer.single_transformer_blocks.21.attn.to_k.weight'], weights['transformer.single_transformer_blocks.21.attn.to_v.weight'], weights['transformer.single_transformer_blocks.21.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.5.ff_context.net.2.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.5.ff_context.net.2.bias'], [1, 3072])
    weights['consteval.const_233'] = main_const_eval_233(device)[0]
    weights['transformer.single_transformer_blocks.19.fused_attn_to_q_to_k_to_v_proj_mlp.bias'] = _tile_concat(device, [weights['transformer.single_transformer_blocks.19.attn.to_q.bias'], weights['transformer.single_transformer_blocks.19.attn.to_k.bias'], weights['transformer.single_transformer_blocks.19.attn.to_v.bias'], weights['transformer.single_transformer_blocks.19.proj_mlp.bias']], 0)
    weights['transformer.single_transformer_blocks.34.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.34.attn.to_q.weight'], weights['transformer.single_transformer_blocks.34.attn.to_k.weight'], weights['transformer.single_transformer_blocks.34.attn.to_v.weight'], weights['transformer.single_transformer_blocks.34.proj_mlp.weight']], [1, 0], 1)
    weights['transformer.transformer_blocks.7.attn.to_out.0.bias.reshaped'] = _reshape(device, weights['transformer.transformer_blocks.7.attn.to_out.0.bias'], [1, 3072])
    weights['transformer.single_transformer_blocks.14.fused_attn_to_q_to_k_to_v_proj_mlp.weight'] = _tile_permute_concat(device, [weights['transformer.single_transformer_blocks.14.attn.to_q.weight'], weights['transformer.single_transformer_blocks.14.attn.to_k.weight'], weights['transformer.single_transformer_blocks.14.attn.to_v.weight'], weights['transformer.single_transformer_blocks.14.proj_mlp.weight']], [1, 0], 1)
    weights['consteval.const_238'] = main_const_eval_238(device)[0]
    weights['transformer.transformer_blocks.6.attn.fused_to_q_to_k_to_v.weight'] = _tile_permute_concat(device, [weights['transformer.transformer_blocks.6.attn.to_q.weight'], weights['transformer.transformer_blocks.6.attn.to_k.weight'], weights['transformer.transformer_blocks.6.attn.to_v.weight']], [1, 0], 1)
    weights['transformer.single_transformer_blocks.16.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.16.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    weights['transformer.single_transformer_blocks.36.norm.linear.bias.f32'] = _reshape_cast_reshape(device, weights['transformer.single_transformer_blocks.36.norm.linear.bias'], [1, 1, 9216], ttnn.DataType.FLOAT32, [1, 9216])
    return weights
