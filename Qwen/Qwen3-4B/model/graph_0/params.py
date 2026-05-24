# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

import utils


_LAYER_IDXS = range(36)

# Weights kept on host as ROW_MAJOR + BFLOAT16 (embedding + Q/K/V projections).
_HOST_BFLOAT16_WEIGHTS = {"L__self___model_embed_tokens.weight"} | {
    f"L__self___model_layers_{i}_self_attn_{p}_proj.weight"
    for i in _LAYER_IDXS
    for p in ("q", "k", "v")
}

# Weight kept on host as ROW_MAJOR + FLOAT32 (rotary inv_freq).
_HOST_FLOAT32_WEIGHTS = {"L__self___model_rotary_emb_inv_freq"}

# Weights placed on device as TILE + BFLOAT16 in DRAM
# (norms, MLP gate/up/down, attention output, lm_head, model norm).
_DEVICE_TILE_BFLOAT16_WEIGHTS = (
    {"L__self___lm_head.weight", "L__self___model_norm_weight"}
    | {
        f"L__self___model_layers_{i}_{name}_weight"
        for i in _LAYER_IDXS
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn_q_norm",
            "self_attn_k_norm",
        )
    }
    | {
        f"L__self___model_layers_{i}_{name}.weight"
        for i in _LAYER_IDXS
        for name in (
            "self_attn_o_proj",
            "mlp_gate_proj",
            "mlp_up_proj",
            "mlp_down_proj",
        )
    }
)

_ALL_WEIGHTS = (
    _HOST_BFLOAT16_WEIGHTS | _HOST_FLOAT32_WEIGHTS | _DEVICE_TILE_BFLOAT16_WEIGHTS
)


# Maps FX-style weight key to its on-disk tensor index (./tensors/arg<N>.tensorbin).
# Index 2 is reserved for the activation tensor and is omitted here.
_WEIGHT_KEY_BY_ARG_INDEX = {
    0: "L__self___model_layers_0_self_attn_v_proj.weight",
    1: "L__self___model_layers_0_input_layernorm_weight",
    3: "L__self___model_embed_tokens.weight",
    4: "L__self___model_rotary_emb_inv_freq",
    5: "L__self___model_layers_0_self_attn_k_norm_weight",
    6: "L__self___model_layers_0_self_attn_k_proj.weight",
    7: "L__self___model_layers_1_self_attn_v_proj.weight",
    8: "L__self___model_layers_1_input_layernorm_weight",
    9: "L__self___model_layers_0_mlp_down_proj.weight",
    10: "L__self___model_layers_0_mlp_up_proj.weight",
    11: "L__self___model_layers_0_post_attention_layernorm_weight",
    12: "L__self___model_layers_0_self_attn_o_proj.weight",
    13: "L__self___model_layers_0_self_attn_q_norm_weight",
    14: "L__self___model_layers_0_self_attn_q_proj.weight",
    15: "L__self___model_layers_0_mlp_gate_proj.weight",
    16: "L__self___model_layers_1_self_attn_k_norm_weight",
    17: "L__self___model_layers_1_self_attn_k_proj.weight",
    18: "L__self___model_layers_2_self_attn_v_proj.weight",
    19: "L__self___model_layers_2_input_layernorm_weight",
    20: "L__self___model_layers_1_mlp_down_proj.weight",
    21: "L__self___model_layers_1_mlp_up_proj.weight",
    22: "L__self___model_layers_1_post_attention_layernorm_weight",
    23: "L__self___model_layers_1_self_attn_o_proj.weight",
    24: "L__self___model_layers_1_self_attn_q_norm_weight",
    25: "L__self___model_layers_1_self_attn_q_proj.weight",
    26: "L__self___model_layers_1_mlp_gate_proj.weight",
    27: "L__self___model_layers_2_self_attn_k_norm_weight",
    28: "L__self___model_layers_2_self_attn_k_proj.weight",
    29: "L__self___model_layers_3_self_attn_v_proj.weight",
    30: "L__self___model_layers_3_input_layernorm_weight",
    31: "L__self___model_layers_2_mlp_down_proj.weight",
    32: "L__self___model_layers_2_mlp_up_proj.weight",
    33: "L__self___model_layers_2_post_attention_layernorm_weight",
    34: "L__self___model_layers_2_self_attn_o_proj.weight",
    35: "L__self___model_layers_2_self_attn_q_norm_weight",
    36: "L__self___model_layers_2_self_attn_q_proj.weight",
    37: "L__self___model_layers_2_mlp_gate_proj.weight",
    38: "L__self___model_layers_3_self_attn_k_norm_weight",
    39: "L__self___model_layers_3_self_attn_k_proj.weight",
    40: "L__self___model_layers_4_self_attn_v_proj.weight",
    41: "L__self___model_layers_4_input_layernorm_weight",
    42: "L__self___model_layers_3_mlp_down_proj.weight",
    43: "L__self___model_layers_3_mlp_up_proj.weight",
    44: "L__self___model_layers_3_post_attention_layernorm_weight",
    45: "L__self___model_layers_3_self_attn_o_proj.weight",
    46: "L__self___model_layers_3_self_attn_q_norm_weight",
    47: "L__self___model_layers_3_self_attn_q_proj.weight",
    48: "L__self___model_layers_3_mlp_gate_proj.weight",
    49: "L__self___model_layers_4_self_attn_k_norm_weight",
    50: "L__self___model_layers_4_self_attn_k_proj.weight",
    51: "L__self___model_layers_5_self_attn_v_proj.weight",
    52: "L__self___model_layers_5_input_layernorm_weight",
    53: "L__self___model_layers_4_mlp_down_proj.weight",
    54: "L__self___model_layers_4_mlp_up_proj.weight",
    55: "L__self___model_layers_4_post_attention_layernorm_weight",
    56: "L__self___model_layers_4_self_attn_o_proj.weight",
    57: "L__self___model_layers_4_self_attn_q_norm_weight",
    58: "L__self___model_layers_4_self_attn_q_proj.weight",
    59: "L__self___model_layers_4_mlp_gate_proj.weight",
    60: "L__self___model_layers_5_self_attn_k_norm_weight",
    61: "L__self___model_layers_5_self_attn_k_proj.weight",
    62: "L__self___model_layers_6_self_attn_v_proj.weight",
    63: "L__self___model_layers_6_input_layernorm_weight",
    64: "L__self___model_layers_5_mlp_down_proj.weight",
    65: "L__self___model_layers_5_mlp_up_proj.weight",
    66: "L__self___model_layers_5_post_attention_layernorm_weight",
    67: "L__self___model_layers_5_self_attn_o_proj.weight",
    68: "L__self___model_layers_5_self_attn_q_norm_weight",
    69: "L__self___model_layers_5_self_attn_q_proj.weight",
    70: "L__self___model_layers_5_mlp_gate_proj.weight",
    71: "L__self___model_layers_6_self_attn_k_norm_weight",
    72: "L__self___model_layers_6_self_attn_k_proj.weight",
    73: "L__self___model_layers_7_self_attn_v_proj.weight",
    74: "L__self___model_layers_7_input_layernorm_weight",
    75: "L__self___model_layers_6_mlp_down_proj.weight",
    76: "L__self___model_layers_6_mlp_up_proj.weight",
    77: "L__self___model_layers_6_post_attention_layernorm_weight",
    78: "L__self___model_layers_6_self_attn_o_proj.weight",
    79: "L__self___model_layers_6_self_attn_q_norm_weight",
    80: "L__self___model_layers_6_self_attn_q_proj.weight",
    81: "L__self___model_layers_6_mlp_gate_proj.weight",
    82: "L__self___model_layers_7_self_attn_k_norm_weight",
    83: "L__self___model_layers_7_self_attn_k_proj.weight",
    84: "L__self___model_layers_8_self_attn_v_proj.weight",
    85: "L__self___model_layers_8_input_layernorm_weight",
    86: "L__self___model_layers_7_mlp_down_proj.weight",
    87: "L__self___model_layers_7_mlp_up_proj.weight",
    88: "L__self___model_layers_7_post_attention_layernorm_weight",
    89: "L__self___model_layers_7_self_attn_o_proj.weight",
    90: "L__self___model_layers_7_self_attn_q_norm_weight",
    91: "L__self___model_layers_7_self_attn_q_proj.weight",
    92: "L__self___model_layers_7_mlp_gate_proj.weight",
    93: "L__self___model_layers_8_self_attn_k_norm_weight",
    94: "L__self___model_layers_8_self_attn_k_proj.weight",
    95: "L__self___model_layers_9_self_attn_v_proj.weight",
    96: "L__self___model_layers_9_input_layernorm_weight",
    97: "L__self___model_layers_8_mlp_down_proj.weight",
    98: "L__self___model_layers_8_mlp_up_proj.weight",
    99: "L__self___model_layers_8_post_attention_layernorm_weight",
    100: "L__self___model_layers_8_self_attn_o_proj.weight",
    101: "L__self___model_layers_8_self_attn_q_norm_weight",
    102: "L__self___model_layers_8_self_attn_q_proj.weight",
    103: "L__self___model_layers_8_mlp_gate_proj.weight",
    104: "L__self___model_layers_9_self_attn_k_norm_weight",
    105: "L__self___model_layers_9_self_attn_k_proj.weight",
    106: "L__self___model_layers_10_self_attn_v_proj.weight",
    107: "L__self___model_layers_10_input_layernorm_weight",
    108: "L__self___model_layers_9_mlp_down_proj.weight",
    109: "L__self___model_layers_9_mlp_up_proj.weight",
    110: "L__self___model_layers_9_post_attention_layernorm_weight",
    111: "L__self___model_layers_9_self_attn_o_proj.weight",
    112: "L__self___model_layers_9_self_attn_q_norm_weight",
    113: "L__self___model_layers_9_self_attn_q_proj.weight",
    114: "L__self___model_layers_9_mlp_gate_proj.weight",
    115: "L__self___model_layers_10_self_attn_k_norm_weight",
    116: "L__self___model_layers_10_self_attn_k_proj.weight",
    117: "L__self___model_layers_11_self_attn_v_proj.weight",
    118: "L__self___model_layers_11_input_layernorm_weight",
    119: "L__self___model_layers_10_mlp_down_proj.weight",
    120: "L__self___model_layers_10_mlp_up_proj.weight",
    121: "L__self___model_layers_10_post_attention_layernorm_weight",
    122: "L__self___model_layers_10_self_attn_o_proj.weight",
    123: "L__self___model_layers_10_self_attn_q_norm_weight",
    124: "L__self___model_layers_10_self_attn_q_proj.weight",
    125: "L__self___model_layers_10_mlp_gate_proj.weight",
    126: "L__self___model_layers_11_self_attn_k_norm_weight",
    127: "L__self___model_layers_11_self_attn_k_proj.weight",
    128: "L__self___model_layers_12_self_attn_v_proj.weight",
    129: "L__self___model_layers_12_input_layernorm_weight",
    130: "L__self___model_layers_11_mlp_down_proj.weight",
    131: "L__self___model_layers_11_mlp_up_proj.weight",
    132: "L__self___model_layers_11_post_attention_layernorm_weight",
    133: "L__self___model_layers_11_self_attn_o_proj.weight",
    134: "L__self___model_layers_11_self_attn_q_norm_weight",
    135: "L__self___model_layers_11_self_attn_q_proj.weight",
    136: "L__self___model_layers_11_mlp_gate_proj.weight",
    137: "L__self___model_layers_12_self_attn_k_norm_weight",
    138: "L__self___model_layers_12_self_attn_k_proj.weight",
    139: "L__self___model_layers_13_self_attn_v_proj.weight",
    140: "L__self___model_layers_13_input_layernorm_weight",
    141: "L__self___model_layers_12_mlp_down_proj.weight",
    142: "L__self___model_layers_12_mlp_up_proj.weight",
    143: "L__self___model_layers_12_post_attention_layernorm_weight",
    144: "L__self___model_layers_12_self_attn_o_proj.weight",
    145: "L__self___model_layers_12_self_attn_q_norm_weight",
    146: "L__self___model_layers_12_self_attn_q_proj.weight",
    147: "L__self___model_layers_12_mlp_gate_proj.weight",
    148: "L__self___model_layers_13_self_attn_k_norm_weight",
    149: "L__self___model_layers_13_self_attn_k_proj.weight",
    150: "L__self___model_layers_14_self_attn_v_proj.weight",
    151: "L__self___model_layers_14_input_layernorm_weight",
    152: "L__self___model_layers_13_mlp_down_proj.weight",
    153: "L__self___model_layers_13_mlp_up_proj.weight",
    154: "L__self___model_layers_13_post_attention_layernorm_weight",
    155: "L__self___model_layers_13_self_attn_o_proj.weight",
    156: "L__self___model_layers_13_self_attn_q_norm_weight",
    157: "L__self___model_layers_13_self_attn_q_proj.weight",
    158: "L__self___model_layers_13_mlp_gate_proj.weight",
    159: "L__self___model_layers_14_self_attn_k_norm_weight",
    160: "L__self___model_layers_14_self_attn_k_proj.weight",
    161: "L__self___model_layers_15_self_attn_v_proj.weight",
    162: "L__self___model_layers_15_input_layernorm_weight",
    163: "L__self___model_layers_14_mlp_down_proj.weight",
    164: "L__self___model_layers_14_mlp_up_proj.weight",
    165: "L__self___model_layers_14_post_attention_layernorm_weight",
    166: "L__self___model_layers_14_self_attn_o_proj.weight",
    167: "L__self___model_layers_14_self_attn_q_norm_weight",
    168: "L__self___model_layers_14_self_attn_q_proj.weight",
    169: "L__self___model_layers_14_mlp_gate_proj.weight",
    170: "L__self___model_layers_15_self_attn_k_norm_weight",
    171: "L__self___model_layers_15_self_attn_k_proj.weight",
    172: "L__self___model_layers_16_self_attn_v_proj.weight",
    173: "L__self___model_layers_16_input_layernorm_weight",
    174: "L__self___model_layers_15_mlp_down_proj.weight",
    175: "L__self___model_layers_15_mlp_up_proj.weight",
    176: "L__self___model_layers_15_post_attention_layernorm_weight",
    177: "L__self___model_layers_15_self_attn_o_proj.weight",
    178: "L__self___model_layers_15_self_attn_q_norm_weight",
    179: "L__self___model_layers_15_self_attn_q_proj.weight",
    180: "L__self___model_layers_15_mlp_gate_proj.weight",
    181: "L__self___model_layers_16_self_attn_k_norm_weight",
    182: "L__self___model_layers_16_self_attn_k_proj.weight",
    183: "L__self___model_layers_17_self_attn_v_proj.weight",
    184: "L__self___model_layers_17_input_layernorm_weight",
    185: "L__self___model_layers_16_mlp_down_proj.weight",
    186: "L__self___model_layers_16_mlp_up_proj.weight",
    187: "L__self___model_layers_16_post_attention_layernorm_weight",
    188: "L__self___model_layers_16_self_attn_o_proj.weight",
    189: "L__self___model_layers_16_self_attn_q_norm_weight",
    190: "L__self___model_layers_16_self_attn_q_proj.weight",
    191: "L__self___model_layers_16_mlp_gate_proj.weight",
    192: "L__self___model_layers_17_self_attn_k_norm_weight",
    193: "L__self___model_layers_17_self_attn_k_proj.weight",
    194: "L__self___model_layers_18_self_attn_v_proj.weight",
    195: "L__self___model_layers_18_input_layernorm_weight",
    196: "L__self___model_layers_17_mlp_down_proj.weight",
    197: "L__self___model_layers_17_mlp_up_proj.weight",
    198: "L__self___model_layers_17_post_attention_layernorm_weight",
    199: "L__self___model_layers_17_self_attn_o_proj.weight",
    200: "L__self___model_layers_17_self_attn_q_norm_weight",
    201: "L__self___model_layers_17_self_attn_q_proj.weight",
    202: "L__self___model_layers_17_mlp_gate_proj.weight",
    203: "L__self___model_layers_18_self_attn_k_norm_weight",
    204: "L__self___model_layers_18_self_attn_k_proj.weight",
    205: "L__self___model_layers_19_self_attn_v_proj.weight",
    206: "L__self___model_layers_19_input_layernorm_weight",
    207: "L__self___model_layers_18_mlp_down_proj.weight",
    208: "L__self___model_layers_18_mlp_up_proj.weight",
    209: "L__self___model_layers_18_post_attention_layernorm_weight",
    210: "L__self___model_layers_18_self_attn_o_proj.weight",
    211: "L__self___model_layers_18_self_attn_q_norm_weight",
    212: "L__self___model_layers_18_self_attn_q_proj.weight",
    213: "L__self___model_layers_18_mlp_gate_proj.weight",
    214: "L__self___model_layers_19_self_attn_k_norm_weight",
    215: "L__self___model_layers_19_self_attn_k_proj.weight",
    216: "L__self___model_layers_20_self_attn_v_proj.weight",
    217: "L__self___model_layers_20_input_layernorm_weight",
    218: "L__self___model_layers_19_mlp_down_proj.weight",
    219: "L__self___model_layers_19_mlp_up_proj.weight",
    220: "L__self___model_layers_19_post_attention_layernorm_weight",
    221: "L__self___model_layers_19_self_attn_o_proj.weight",
    222: "L__self___model_layers_19_self_attn_q_norm_weight",
    223: "L__self___model_layers_19_self_attn_q_proj.weight",
    224: "L__self___model_layers_19_mlp_gate_proj.weight",
    225: "L__self___model_layers_20_self_attn_k_norm_weight",
    226: "L__self___model_layers_20_self_attn_k_proj.weight",
    227: "L__self___model_layers_21_self_attn_v_proj.weight",
    228: "L__self___model_layers_21_input_layernorm_weight",
    229: "L__self___model_layers_20_mlp_down_proj.weight",
    230: "L__self___model_layers_20_mlp_up_proj.weight",
    231: "L__self___model_layers_20_post_attention_layernorm_weight",
    232: "L__self___model_layers_20_self_attn_o_proj.weight",
    233: "L__self___model_layers_20_self_attn_q_norm_weight",
    234: "L__self___model_layers_20_self_attn_q_proj.weight",
    235: "L__self___model_layers_20_mlp_gate_proj.weight",
    236: "L__self___model_layers_21_self_attn_k_norm_weight",
    237: "L__self___model_layers_21_self_attn_k_proj.weight",
    238: "L__self___model_layers_22_self_attn_v_proj.weight",
    239: "L__self___model_layers_22_input_layernorm_weight",
    240: "L__self___model_layers_21_mlp_down_proj.weight",
    241: "L__self___model_layers_21_mlp_up_proj.weight",
    242: "L__self___model_layers_21_post_attention_layernorm_weight",
    243: "L__self___model_layers_21_self_attn_o_proj.weight",
    244: "L__self___model_layers_21_self_attn_q_norm_weight",
    245: "L__self___model_layers_21_self_attn_q_proj.weight",
    246: "L__self___model_layers_21_mlp_gate_proj.weight",
    247: "L__self___model_layers_22_self_attn_k_norm_weight",
    248: "L__self___model_layers_22_self_attn_k_proj.weight",
    249: "L__self___model_layers_23_self_attn_v_proj.weight",
    250: "L__self___model_layers_23_input_layernorm_weight",
    251: "L__self___model_layers_22_mlp_down_proj.weight",
    252: "L__self___model_layers_22_mlp_up_proj.weight",
    253: "L__self___model_layers_22_post_attention_layernorm_weight",
    254: "L__self___model_layers_22_self_attn_o_proj.weight",
    255: "L__self___model_layers_22_self_attn_q_norm_weight",
    256: "L__self___model_layers_22_self_attn_q_proj.weight",
    257: "L__self___model_layers_22_mlp_gate_proj.weight",
    258: "L__self___model_layers_23_self_attn_k_norm_weight",
    259: "L__self___model_layers_23_self_attn_k_proj.weight",
    260: "L__self___model_layers_24_self_attn_v_proj.weight",
    261: "L__self___model_layers_24_input_layernorm_weight",
    262: "L__self___model_layers_23_mlp_down_proj.weight",
    263: "L__self___model_layers_23_mlp_up_proj.weight",
    264: "L__self___model_layers_23_post_attention_layernorm_weight",
    265: "L__self___model_layers_23_self_attn_o_proj.weight",
    266: "L__self___model_layers_23_self_attn_q_norm_weight",
    267: "L__self___model_layers_23_self_attn_q_proj.weight",
    268: "L__self___model_layers_23_mlp_gate_proj.weight",
    269: "L__self___model_layers_24_self_attn_k_norm_weight",
    270: "L__self___model_layers_24_self_attn_k_proj.weight",
    271: "L__self___model_layers_25_self_attn_v_proj.weight",
    272: "L__self___model_layers_25_input_layernorm_weight",
    273: "L__self___model_layers_24_mlp_down_proj.weight",
    274: "L__self___model_layers_24_mlp_up_proj.weight",
    275: "L__self___model_layers_24_post_attention_layernorm_weight",
    276: "L__self___model_layers_24_self_attn_o_proj.weight",
    277: "L__self___model_layers_24_self_attn_q_norm_weight",
    278: "L__self___model_layers_24_self_attn_q_proj.weight",
    279: "L__self___model_layers_24_mlp_gate_proj.weight",
    280: "L__self___model_layers_25_self_attn_k_norm_weight",
    281: "L__self___model_layers_25_self_attn_k_proj.weight",
    282: "L__self___model_layers_26_self_attn_v_proj.weight",
    283: "L__self___model_layers_26_input_layernorm_weight",
    284: "L__self___model_layers_25_mlp_down_proj.weight",
    285: "L__self___model_layers_25_mlp_up_proj.weight",
    286: "L__self___model_layers_25_post_attention_layernorm_weight",
    287: "L__self___model_layers_25_self_attn_o_proj.weight",
    288: "L__self___model_layers_25_self_attn_q_norm_weight",
    289: "L__self___model_layers_25_self_attn_q_proj.weight",
    290: "L__self___model_layers_25_mlp_gate_proj.weight",
    291: "L__self___model_layers_26_self_attn_k_norm_weight",
    292: "L__self___model_layers_26_self_attn_k_proj.weight",
    293: "L__self___model_layers_27_self_attn_v_proj.weight",
    294: "L__self___model_layers_27_input_layernorm_weight",
    295: "L__self___model_layers_26_mlp_down_proj.weight",
    296: "L__self___model_layers_26_mlp_up_proj.weight",
    297: "L__self___model_layers_26_post_attention_layernorm_weight",
    298: "L__self___model_layers_26_self_attn_o_proj.weight",
    299: "L__self___model_layers_26_self_attn_q_norm_weight",
    300: "L__self___model_layers_26_self_attn_q_proj.weight",
    301: "L__self___model_layers_26_mlp_gate_proj.weight",
    302: "L__self___model_layers_27_self_attn_k_norm_weight",
    303: "L__self___model_layers_27_self_attn_k_proj.weight",
    304: "L__self___model_layers_28_self_attn_v_proj.weight",
    305: "L__self___model_layers_28_input_layernorm_weight",
    306: "L__self___model_layers_27_mlp_down_proj.weight",
    307: "L__self___model_layers_27_mlp_up_proj.weight",
    308: "L__self___model_layers_27_post_attention_layernorm_weight",
    309: "L__self___model_layers_27_self_attn_o_proj.weight",
    310: "L__self___model_layers_27_self_attn_q_norm_weight",
    311: "L__self___model_layers_27_self_attn_q_proj.weight",
    312: "L__self___model_layers_27_mlp_gate_proj.weight",
    313: "L__self___model_layers_28_self_attn_k_norm_weight",
    314: "L__self___model_layers_28_self_attn_k_proj.weight",
    315: "L__self___model_layers_29_self_attn_v_proj.weight",
    316: "L__self___model_layers_29_input_layernorm_weight",
    317: "L__self___model_layers_28_mlp_down_proj.weight",
    318: "L__self___model_layers_28_mlp_up_proj.weight",
    319: "L__self___model_layers_28_post_attention_layernorm_weight",
    320: "L__self___model_layers_28_self_attn_o_proj.weight",
    321: "L__self___model_layers_28_self_attn_q_norm_weight",
    322: "L__self___model_layers_28_self_attn_q_proj.weight",
    323: "L__self___model_layers_28_mlp_gate_proj.weight",
    324: "L__self___model_layers_29_self_attn_k_norm_weight",
    325: "L__self___model_layers_29_self_attn_k_proj.weight",
    326: "L__self___model_layers_30_self_attn_v_proj.weight",
    327: "L__self___model_layers_30_input_layernorm_weight",
    328: "L__self___model_layers_29_mlp_down_proj.weight",
    329: "L__self___model_layers_29_mlp_up_proj.weight",
    330: "L__self___model_layers_29_post_attention_layernorm_weight",
    331: "L__self___model_layers_29_self_attn_o_proj.weight",
    332: "L__self___model_layers_29_self_attn_q_norm_weight",
    333: "L__self___model_layers_29_self_attn_q_proj.weight",
    334: "L__self___model_layers_29_mlp_gate_proj.weight",
    335: "L__self___model_layers_30_self_attn_k_norm_weight",
    336: "L__self___model_layers_30_self_attn_k_proj.weight",
    337: "L__self___model_layers_31_self_attn_v_proj.weight",
    338: "L__self___model_layers_31_input_layernorm_weight",
    339: "L__self___model_layers_30_mlp_down_proj.weight",
    340: "L__self___model_layers_30_mlp_up_proj.weight",
    341: "L__self___model_layers_30_post_attention_layernorm_weight",
    342: "L__self___model_layers_30_self_attn_o_proj.weight",
    343: "L__self___model_layers_30_self_attn_q_norm_weight",
    344: "L__self___model_layers_30_self_attn_q_proj.weight",
    345: "L__self___model_layers_30_mlp_gate_proj.weight",
    346: "L__self___model_layers_31_self_attn_k_norm_weight",
    347: "L__self___model_layers_31_self_attn_k_proj.weight",
    348: "L__self___model_layers_32_self_attn_v_proj.weight",
    349: "L__self___model_layers_32_input_layernorm_weight",
    350: "L__self___model_layers_31_mlp_down_proj.weight",
    351: "L__self___model_layers_31_mlp_up_proj.weight",
    352: "L__self___model_layers_31_post_attention_layernorm_weight",
    353: "L__self___model_layers_31_self_attn_o_proj.weight",
    354: "L__self___model_layers_31_self_attn_q_norm_weight",
    355: "L__self___model_layers_31_self_attn_q_proj.weight",
    356: "L__self___model_layers_31_mlp_gate_proj.weight",
    357: "L__self___model_layers_32_self_attn_k_norm_weight",
    358: "L__self___model_layers_32_self_attn_k_proj.weight",
    359: "L__self___model_layers_33_self_attn_v_proj.weight",
    360: "L__self___model_layers_33_input_layernorm_weight",
    361: "L__self___model_layers_32_mlp_down_proj.weight",
    362: "L__self___model_layers_32_mlp_up_proj.weight",
    363: "L__self___model_layers_32_post_attention_layernorm_weight",
    364: "L__self___model_layers_32_self_attn_o_proj.weight",
    365: "L__self___model_layers_32_self_attn_q_norm_weight",
    366: "L__self___model_layers_32_self_attn_q_proj.weight",
    367: "L__self___model_layers_32_mlp_gate_proj.weight",
    368: "L__self___model_layers_33_self_attn_k_norm_weight",
    369: "L__self___model_layers_33_self_attn_k_proj.weight",
    370: "L__self___model_layers_34_self_attn_v_proj.weight",
    371: "L__self___model_layers_34_input_layernorm_weight",
    372: "L__self___model_layers_33_mlp_down_proj.weight",
    373: "L__self___model_layers_33_mlp_up_proj.weight",
    374: "L__self___model_layers_33_post_attention_layernorm_weight",
    375: "L__self___model_layers_33_self_attn_o_proj.weight",
    376: "L__self___model_layers_33_self_attn_q_norm_weight",
    377: "L__self___model_layers_33_self_attn_q_proj.weight",
    378: "L__self___model_layers_33_mlp_gate_proj.weight",
    379: "L__self___model_layers_34_self_attn_k_norm_weight",
    380: "L__self___model_layers_34_self_attn_k_proj.weight",
    381: "L__self___model_layers_35_self_attn_v_proj.weight",
    382: "L__self___model_layers_35_input_layernorm_weight",
    383: "L__self___model_layers_34_mlp_down_proj.weight",
    384: "L__self___model_layers_34_mlp_up_proj.weight",
    385: "L__self___model_layers_34_post_attention_layernorm_weight",
    386: "L__self___model_layers_34_self_attn_o_proj.weight",
    387: "L__self___model_layers_34_self_attn_q_norm_weight",
    388: "L__self___model_layers_34_self_attn_q_proj.weight",
    389: "L__self___model_layers_34_mlp_gate_proj.weight",
    390: "L__self___model_layers_35_self_attn_k_norm_weight",
    391: "L__self___model_layers_35_self_attn_k_proj.weight",
    392: "L__self___lm_head.weight",
    393: "L__self___model_norm_weight",
    394: "L__self___model_layers_35_mlp_down_proj.weight",
    395: "L__self___model_layers_35_mlp_up_proj.weight",
    396: "L__self___model_layers_35_post_attention_layernorm_weight",
    397: "L__self___model_layers_35_self_attn_o_proj.weight",
    398: "L__self___model_layers_35_self_attn_q_norm_weight",
    399: "L__self___model_layers_35_self_attn_q_proj.weight",
    400: "L__self___model_layers_35_mlp_gate_proj.weight",
}


def load_weights_for__main():
    device = utils.DeviceGetter.get_device((1, 1))
    weights = {}
    for arg_idx, key in _WEIGHT_KEY_BY_ARG_INDEX.items():
        path = f"./tensors/arg{arg_idx}.tensorbin"
        if key in _HOST_BFLOAT16_WEIGHTS:
            tensor = utils.load_tensor(
                path, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, None, None
            )
        elif key in _HOST_FLOAT32_WEIGHTS:
            tensor = utils.load_tensor(
                path, ttnn.Layout.ROW_MAJOR, ttnn.DataType.FLOAT32, None, None
            )
        elif key in _DEVICE_TILE_BFLOAT16_WEIGHTS:
            tensor = utils.load_tensor(
                path,
                ttnn.Layout.TILE,
                ttnn.DataType.BFLOAT16,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            raise KeyError(f"Unknown weight key: {key}")
        weights[key] = tensor
    return weights


def _fx_key_to_state_dict_key(key: str) -> str:
    """Translate an FX-traced parameter key (``L__self___...``) into its PyTorch state_dict key."""
    name = key.removeprefix("L__self___")
    if name == "lm_head.weight":
        return "lm_head.weight"
    if name == "model_norm_weight":
        return "model.norm.weight"
    if name == "model_embed_tokens.weight":
        return "model.embed_tokens.weight"
    if name == "model_rotary_emb_inv_freq":
        return "model.rotary_emb.inv_freq"

    assert name.startswith("model_layers_"), key
    rest = name[len("model_layers_") :]
    sep = rest.index("_")
    layer_idx = rest[:sep]
    sub = rest[sep + 1 :]

    if sub == "input_layernorm_weight":
        return f"model.layers.{layer_idx}.input_layernorm.weight"
    if sub == "post_attention_layernorm_weight":
        return f"model.layers.{layer_idx}.post_attention_layernorm.weight"
    if sub.startswith("self_attn_") and sub.endswith("_proj.weight"):
        proj = sub[len("self_attn_") : -len("_proj.weight")]
        return f"model.layers.{layer_idx}.self_attn.{proj}_proj.weight"
    if sub.startswith("self_attn_") and sub.endswith("_norm_weight"):
        norm = sub[len("self_attn_") : -len("_norm_weight")]
        return f"model.layers.{layer_idx}.self_attn.{norm}_norm.weight"
    if sub.startswith("mlp_") and sub.endswith("_proj.weight"):
        proj = sub[len("mlp_") : -len("_proj.weight")]
        return f"model.layers.{layer_idx}.mlp.{proj}_proj.weight"

    raise KeyError(f"Unrecognized weight key: {key}")


def load_weights_for__main_from_state_dict(state_dict):
    device = utils.DeviceGetter.get_device((1, 1))

    weights = {}
    for key in _ALL_WEIGHTS:
        pt_tensor = state_dict[_fx_key_to_state_dict_key(key)]
        ttnn_tensor = ttnn.from_torch(pt_tensor)

        if key in _HOST_BFLOAT16_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)

        if key in _HOST_FLOAT32_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.ROW_MAJOR)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.FLOAT32)

        if key in _DEVICE_TILE_BFLOAT16_WEIGHTS:
            ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.Layout.TILE)
            ttnn_tensor = ttnn.to_dtype(ttnn_tensor, ttnn.DataType.BFLOAT16)
            ttnn_tensor = ttnn.to_device(ttnn_tensor, device, ttnn.DRAM_MEMORY_CONFIG)

        weights[key] = ttnn_tensor

    return weights
