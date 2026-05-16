import ttnn
import ttir_cpu
import torch


def cpu_hoisted_const_eval_3d7508d3(
    arg_0,
    arg_1,
    arg_2,
    arg_3,
    arg_4,
    arg_5,
    arg_6,
    arg_7,
    arg_8,
    arg_9,
    arg_10,
    arg_11,
    arg_12,
    arg_13,
    arg_14,
    arg_15,
    arg_16,
    arg_17,
    arg_18,
    arg_19,
    arg_20,
    arg_21,
    arg_22,
    arg_23,
    arg_24,
    arg_25,
    arg_26,
    arg_27,
    arg_28,
    arg_29,
    arg_30,
    arg_31,
    arg_32,
    arg_33,
    arg_34,
    arg_35,
    arg_36,
    arg_37,
    arg_38,
    arg_39,
    arg_40,
    arg_41,
    arg_42,
    arg_43,
    arg_44,
    arg_45,
    arg_46,
    arg_47,
    arg_48,
    arg_49,
    arg_50,
    arg_51,
    arg_52,
    arg_53,
    arg_54,
    arg_55,
    arg_56,
    arg_57,
    arg_58,
    arg_59,
    arg_60,
    arg_61,
    arg_62,
    arg_63,
    arg_64,
    arg_65,
    arg_66,
    arg_67,
    arg_68,
    arg_69,
    arg_70,
    arg_71,
    arg_72,
    arg_73,
    arg_74,
    arg_75,
    arg_76,
    arg_77,
    arg_78,
    arg_79,
    arg_80,
    arg_81,
    arg_82,
    arg_83,
    arg_84,
    arg_85,
    arg_86,
    arg_87,
    arg_88,
    arg_89,
    arg_90,
    arg_91,
    arg_92,
    arg_93,
    arg_94,
    arg_95,
    arg_96,
    arg_97,
    arg_98,
    arg_99,
    arg_100,
    arg_101,
    arg_102,
    arg_103,
    arg_104,
    arg_105,
    arg_106,
    arg_107,
    arg_108,
    arg_109,
    arg_110,
    arg_111,
    arg_112,
    arg_113,
    arg_114,
    arg_115,
    arg_116,
    arg_117,
    arg_118,
    arg_119,
    arg_120,
    arg_121,
    arg_122,
    arg_123,
    arg_124,
    arg_125,
    arg_126,
    arg_127,
    arg_128,
    arg_129,
    arg_130,
    arg_131,
    arg_132,
    arg_133,
    arg_134,
    arg_135,
    arg_136,
    arg_137,
    arg_138,
    arg_139,
    arg_140,
    arg_141,
    arg_142,
    arg_143,
    arg_144,
    arg_145,
    arg_146,
    arg_147,
    arg_148,
    arg_149,
    arg_150,
    arg_151,
    arg_152,
    arg_153,
    arg_154,
    arg_155,
    arg_156,
    arg_157,
    arg_158,
    arg_159,
    arg_160,
    arg_161,
    arg_162,
    arg_163,
    arg_164,
    arg_165,
    arg_166,
    arg_167,
    arg_168,
    arg_169,
    arg_170,
    arg_171,
    arg_172,
    arg_173,
    arg_174,
    arg_175,
    arg_176,
    arg_177,
    arg_178,
    arg_179,
    arg_180,
    arg_181,
    arg_182,
    arg_183,
    arg_184,
    arg_185,
    arg_186,
    arg_187,
    arg_188,
    arg_189,
    arg_190,
    arg_191,
    arg_192,
    arg_193,
    arg_194,
    arg_195,
    arg_196,
    arg_197,
    arg_198,
    arg_199,
    arg_200,
    arg_201,
    arg_202,
    arg_203,
    arg_204,
    arg_205,
    arg_206,
    arg_207,
    arg_208,
    arg_209,
    arg_210,
    arg_211,
    arg_212,
    arg_213,
    arg_214,
    arg_215,
    arg_216,
    arg_217,
    arg_218,
    arg_219,
    arg_220,
    arg_221,
    arg_222,
    arg_223,
    arg_224,
    arg_225,
    arg_226,
    arg_227,
    arg_228,
    arg_229,
    arg_230,
    arg_231,
    arg_232,
    arg_233,
    arg_234,
    arg_235,
    arg_236,
    arg_237,
    arg_238,
    arg_239,
    arg_240,
    arg_241,
    arg_242,
    arg_243,
    arg_244,
    arg_245,
    arg_246,
    arg_247,
    arg_248,
    arg_249,
    arg_250,
    arg_251,
    arg_252,
    arg_253,
    arg_254,
    arg_255,
    arg_256,
    arg_257,
    arg_258,
    arg_259,
    arg_260,
    arg_261,
    arg_262,
    arg_263,
    arg_264,
):
    ttnn_to_torch_0 = ttnn.to_torch(arg_0)
    ttnn_to_torch_1 = ttnn.to_torch(arg_1)
    ttnn_to_torch_2 = ttnn.to_torch(arg_2)
    ttnn_to_torch_3 = ttnn.to_torch(arg_3)
    ttnn_to_torch_4 = ttnn.to_torch(arg_4)
    ttnn_to_torch_5 = ttnn.to_torch(arg_5)
    ttnn_to_torch_6 = ttnn.to_torch(arg_6)
    ttnn_to_torch_7 = ttnn.to_torch(arg_7)
    ttnn_to_torch_8 = ttnn.to_torch(arg_8)
    ttnn_to_torch_9 = ttnn.to_torch(arg_9)
    ttnn_to_torch_10 = ttnn.to_torch(arg_10)
    ttnn_to_torch_11 = ttnn.to_torch(arg_11)
    ttnn_to_torch_12 = ttnn.to_torch(arg_12)
    ttnn_to_torch_13 = ttnn.to_torch(arg_13)
    ttnn_to_torch_14 = ttnn.to_torch(arg_14)
    ttnn_to_torch_15 = ttnn.to_torch(arg_15)
    ttnn_to_torch_16 = ttnn.to_torch(arg_16)
    ttnn_to_torch_17 = ttnn.to_torch(arg_17)
    ttnn_to_torch_18 = ttnn.to_torch(arg_18)
    ttnn_to_torch_19 = ttnn.to_torch(arg_19)
    ttnn_to_torch_20 = ttnn.to_torch(arg_20)
    ttnn_to_torch_21 = ttnn.to_torch(arg_21)
    ttnn_to_torch_22 = ttnn.to_torch(arg_22)
    ttnn_to_torch_23 = ttnn.to_torch(arg_23)
    ttnn_to_torch_24 = ttnn.to_torch(arg_24)
    ttnn_to_torch_25 = ttnn.to_torch(arg_25)
    ttnn_to_torch_26 = ttnn.to_torch(arg_26)
    ttnn_to_torch_27 = ttnn.to_torch(arg_27)
    ttnn_to_torch_28 = ttnn.to_torch(arg_28)
    ttnn_to_torch_29 = ttnn.to_torch(arg_29)
    ttnn_to_torch_30 = ttnn.to_torch(arg_30)
    ttnn_to_torch_31 = ttnn.to_torch(arg_31)
    ttnn_to_torch_32 = ttnn.to_torch(arg_32)
    ttnn_to_torch_33 = ttnn.to_torch(arg_33)
    ttnn_to_torch_34 = ttnn.to_torch(arg_34)
    ttnn_to_torch_35 = ttnn.to_torch(arg_35)
    ttnn_to_torch_36 = ttnn.to_torch(arg_36)
    ttnn_to_torch_37 = ttnn.to_torch(arg_37)
    ttnn_to_torch_38 = ttnn.to_torch(arg_38)
    ttnn_to_torch_39 = ttnn.to_torch(arg_39)
    ttnn_to_torch_40 = ttnn.to_torch(arg_40)
    ttnn_to_torch_41 = ttnn.to_torch(arg_41)
    ttnn_to_torch_42 = ttnn.to_torch(arg_42)
    ttnn_to_torch_43 = ttnn.to_torch(arg_43)
    ttnn_to_torch_44 = ttnn.to_torch(arg_44)
    ttnn_to_torch_45 = ttnn.to_torch(arg_45)
    ttnn_to_torch_46 = ttnn.to_torch(arg_46)
    ttnn_to_torch_47 = ttnn.to_torch(arg_47)
    ttnn_to_torch_48 = ttnn.to_torch(arg_48)
    ttnn_to_torch_49 = ttnn.to_torch(arg_49)
    ttnn_to_torch_50 = ttnn.to_torch(arg_50)
    ttnn_to_torch_51 = ttnn.to_torch(arg_51)
    ttnn_to_torch_52 = ttnn.to_torch(arg_52)
    ttnn_to_torch_53 = ttnn.to_torch(arg_53)
    ttnn_to_torch_54 = ttnn.to_torch(arg_54)
    ttnn_to_torch_55 = ttnn.to_torch(arg_55)
    ttnn_to_torch_56 = ttnn.to_torch(arg_56)
    ttnn_to_torch_57 = ttnn.to_torch(arg_57)
    ttnn_to_torch_58 = ttnn.to_torch(arg_58)
    ttnn_to_torch_59 = ttnn.to_torch(arg_59)
    ttnn_to_torch_60 = ttnn.to_torch(arg_60)
    ttnn_to_torch_61 = ttnn.to_torch(arg_61)
    ttnn_to_torch_62 = ttnn.to_torch(arg_62)
    ttnn_to_torch_63 = ttnn.to_torch(arg_63)
    ttnn_to_torch_64 = ttnn.to_torch(arg_64)
    ttnn_to_torch_65 = ttnn.to_torch(arg_65)
    ttnn_to_torch_66 = ttnn.to_torch(arg_66)
    ttnn_to_torch_67 = ttnn.to_torch(arg_67)
    ttnn_to_torch_68 = ttnn.to_torch(arg_68)
    ttnn_to_torch_69 = ttnn.to_torch(arg_69)
    ttnn_to_torch_70 = ttnn.to_torch(arg_70)
    ttnn_to_torch_71 = ttnn.to_torch(arg_71)
    ttnn_to_torch_72 = ttnn.to_torch(arg_72)
    ttnn_to_torch_73 = ttnn.to_torch(arg_73)
    ttnn_to_torch_74 = ttnn.to_torch(arg_74)
    ttnn_to_torch_75 = ttnn.to_torch(arg_75)
    ttnn_to_torch_76 = ttnn.to_torch(arg_76)
    ttnn_to_torch_77 = ttnn.to_torch(arg_77)
    ttnn_to_torch_78 = ttnn.to_torch(arg_78)
    ttnn_to_torch_79 = ttnn.to_torch(arg_79)
    ttnn_to_torch_80 = ttnn.to_torch(arg_80)
    ttnn_to_torch_81 = ttnn.to_torch(arg_81)
    ttnn_to_torch_82 = ttnn.to_torch(arg_82)
    ttnn_to_torch_83 = ttnn.to_torch(arg_83)
    ttnn_to_torch_84 = ttnn.to_torch(arg_84)
    ttnn_to_torch_85 = ttnn.to_torch(arg_85)
    ttnn_to_torch_86 = ttnn.to_torch(arg_86)
    ttnn_to_torch_87 = ttnn.to_torch(arg_87)
    ttnn_to_torch_88 = ttnn.to_torch(arg_88)
    ttnn_to_torch_89 = ttnn.to_torch(arg_89)
    ttnn_to_torch_90 = ttnn.to_torch(arg_90)
    ttnn_to_torch_91 = ttnn.to_torch(arg_91)
    ttnn_to_torch_92 = ttnn.to_torch(arg_92)
    ttnn_to_torch_93 = ttnn.to_torch(arg_93)
    ttnn_to_torch_94 = ttnn.to_torch(arg_94)
    ttnn_to_torch_95 = ttnn.to_torch(arg_95)
    ttnn_to_torch_96 = ttnn.to_torch(arg_96)
    ttnn_to_torch_97 = ttnn.to_torch(arg_97)
    ttnn_to_torch_98 = ttnn.to_torch(arg_98)
    ttnn_to_torch_99 = ttnn.to_torch(arg_99)
    ttnn_to_torch_100 = ttnn.to_torch(arg_100)
    ttnn_to_torch_101 = ttnn.to_torch(arg_101)
    ttnn_to_torch_102 = ttnn.to_torch(arg_102)
    ttnn_to_torch_103 = ttnn.to_torch(arg_103)
    ttnn_to_torch_104 = ttnn.to_torch(arg_104)
    ttnn_to_torch_105 = ttnn.to_torch(arg_105)
    ttnn_to_torch_106 = ttnn.to_torch(arg_106)
    ttnn_to_torch_107 = ttnn.to_torch(arg_107)
    ttnn_to_torch_108 = ttnn.to_torch(arg_108)
    ttnn_to_torch_109 = ttnn.to_torch(arg_109)
    ttnn_to_torch_110 = ttnn.to_torch(arg_110)
    ttnn_to_torch_111 = ttnn.to_torch(arg_111)
    ttnn_to_torch_112 = ttnn.to_torch(arg_112)
    ttnn_to_torch_113 = ttnn.to_torch(arg_113)
    ttnn_to_torch_114 = ttnn.to_torch(arg_114)
    ttnn_to_torch_115 = ttnn.to_torch(arg_115)
    ttnn_to_torch_116 = ttnn.to_torch(arg_116)
    ttnn_to_torch_117 = ttnn.to_torch(arg_117)
    ttnn_to_torch_118 = ttnn.to_torch(arg_118)
    ttnn_to_torch_119 = ttnn.to_torch(arg_119)
    ttnn_to_torch_120 = ttnn.to_torch(arg_120)
    ttnn_to_torch_121 = ttnn.to_torch(arg_121)
    ttnn_to_torch_122 = ttnn.to_torch(arg_122)
    ttnn_to_torch_123 = ttnn.to_torch(arg_123)
    ttnn_to_torch_124 = ttnn.to_torch(arg_124)
    ttnn_to_torch_125 = ttnn.to_torch(arg_125)
    ttnn_to_torch_126 = ttnn.to_torch(arg_126)
    ttnn_to_torch_127 = ttnn.to_torch(arg_127)
    ttnn_to_torch_128 = ttnn.to_torch(arg_128)
    ttnn_to_torch_129 = ttnn.to_torch(arg_129)
    ttnn_to_torch_130 = ttnn.to_torch(arg_130)
    ttnn_to_torch_131 = ttnn.to_torch(arg_131)
    ttnn_to_torch_132 = ttnn.to_torch(arg_132)
    ttnn_to_torch_133 = ttnn.to_torch(arg_133)
    ttnn_to_torch_134 = ttnn.to_torch(arg_134)
    ttnn_to_torch_135 = ttnn.to_torch(arg_135)
    ttnn_to_torch_136 = ttnn.to_torch(arg_136)
    ttnn_to_torch_137 = ttnn.to_torch(arg_137)
    ttnn_to_torch_138 = ttnn.to_torch(arg_138)
    ttnn_to_torch_139 = ttnn.to_torch(arg_139)
    ttnn_to_torch_140 = ttnn.to_torch(arg_140)
    ttnn_to_torch_141 = ttnn.to_torch(arg_141)
    ttnn_to_torch_142 = ttnn.to_torch(arg_142)
    ttnn_to_torch_143 = ttnn.to_torch(arg_143)
    ttnn_to_torch_144 = ttnn.to_torch(arg_144)
    ttnn_to_torch_145 = ttnn.to_torch(arg_145)
    ttnn_to_torch_146 = ttnn.to_torch(arg_146)
    ttnn_to_torch_147 = ttnn.to_torch(arg_147)
    ttnn_to_torch_148 = ttnn.to_torch(arg_148)
    ttnn_to_torch_149 = ttnn.to_torch(arg_149)
    ttnn_to_torch_150 = ttnn.to_torch(arg_150)
    ttnn_to_torch_151 = ttnn.to_torch(arg_151)
    ttnn_to_torch_152 = ttnn.to_torch(arg_152)
    ttnn_to_torch_153 = ttnn.to_torch(arg_153)
    ttnn_to_torch_154 = ttnn.to_torch(arg_154)
    ttnn_to_torch_155 = ttnn.to_torch(arg_155)
    ttnn_to_torch_156 = ttnn.to_torch(arg_156)
    ttnn_to_torch_157 = ttnn.to_torch(arg_157)
    ttnn_to_torch_158 = ttnn.to_torch(arg_158)
    ttnn_to_torch_159 = ttnn.to_torch(arg_159)
    ttnn_to_torch_160 = ttnn.to_torch(arg_160)
    ttnn_to_torch_161 = ttnn.to_torch(arg_161)
    ttnn_to_torch_162 = ttnn.to_torch(arg_162)
    ttnn_to_torch_163 = ttnn.to_torch(arg_163)
    ttnn_to_torch_164 = ttnn.to_torch(arg_164)
    ttnn_to_torch_165 = ttnn.to_torch(arg_165)
    ttnn_to_torch_166 = ttnn.to_torch(arg_166)
    ttnn_to_torch_167 = ttnn.to_torch(arg_167)
    ttnn_to_torch_168 = ttnn.to_torch(arg_168)
    ttnn_to_torch_169 = ttnn.to_torch(arg_169)
    ttnn_to_torch_170 = ttnn.to_torch(arg_170)
    ttnn_to_torch_171 = ttnn.to_torch(arg_171)
    ttnn_to_torch_172 = ttnn.to_torch(arg_172)
    ttnn_to_torch_173 = ttnn.to_torch(arg_173)
    ttnn_to_torch_174 = ttnn.to_torch(arg_174)
    ttnn_to_torch_175 = ttnn.to_torch(arg_175)
    ttnn_to_torch_176 = ttnn.to_torch(arg_176)
    ttnn_to_torch_177 = ttnn.to_torch(arg_177)
    ttnn_to_torch_178 = ttnn.to_torch(arg_178)
    ttnn_to_torch_179 = ttnn.to_torch(arg_179)
    ttnn_to_torch_180 = ttnn.to_torch(arg_180)
    ttnn_to_torch_181 = ttnn.to_torch(arg_181)
    ttnn_to_torch_182 = ttnn.to_torch(arg_182)
    ttnn_to_torch_183 = ttnn.to_torch(arg_183)
    ttnn_to_torch_184 = ttnn.to_torch(arg_184)
    ttnn_to_torch_185 = ttnn.to_torch(arg_185)
    ttnn_to_torch_186 = ttnn.to_torch(arg_186)
    ttnn_to_torch_187 = ttnn.to_torch(arg_187)
    ttnn_to_torch_188 = ttnn.to_torch(arg_188)
    ttnn_to_torch_189 = ttnn.to_torch(arg_189)
    ttnn_to_torch_190 = ttnn.to_torch(arg_190)
    ttnn_to_torch_191 = ttnn.to_torch(arg_191)
    ttnn_to_torch_192 = ttnn.to_torch(arg_192)
    ttnn_to_torch_193 = ttnn.to_torch(arg_193)
    ttnn_to_torch_194 = ttnn.to_torch(arg_194)
    ttnn_to_torch_195 = ttnn.to_torch(arg_195)
    ttnn_to_torch_196 = ttnn.to_torch(arg_196)
    ttnn_to_torch_197 = ttnn.to_torch(arg_197)
    ttnn_to_torch_198 = ttnn.to_torch(arg_198)
    ttnn_to_torch_199 = ttnn.to_torch(arg_199)
    ttnn_to_torch_200 = ttnn.to_torch(arg_200)
    ttnn_to_torch_201 = ttnn.to_torch(arg_201)
    ttnn_to_torch_202 = ttnn.to_torch(arg_202)
    ttnn_to_torch_203 = ttnn.to_torch(arg_203)
    ttnn_to_torch_204 = ttnn.to_torch(arg_204)
    ttnn_to_torch_205 = ttnn.to_torch(arg_205)
    ttnn_to_torch_206 = ttnn.to_torch(arg_206)
    ttnn_to_torch_207 = ttnn.to_torch(arg_207)
    ttnn_to_torch_208 = ttnn.to_torch(arg_208)
    ttnn_to_torch_209 = ttnn.to_torch(arg_209)
    ttnn_to_torch_210 = ttnn.to_torch(arg_210)
    ttnn_to_torch_211 = ttnn.to_torch(arg_211)
    ttnn_to_torch_212 = ttnn.to_torch(arg_212)
    ttnn_to_torch_213 = ttnn.to_torch(arg_213)
    ttnn_to_torch_214 = ttnn.to_torch(arg_214)
    ttnn_to_torch_215 = ttnn.to_torch(arg_215)
    ttnn_to_torch_216 = ttnn.to_torch(arg_216)
    ttnn_to_torch_217 = ttnn.to_torch(arg_217)
    ttnn_to_torch_218 = ttnn.to_torch(arg_218)
    ttnn_to_torch_219 = ttnn.to_torch(arg_219)
    ttnn_to_torch_220 = ttnn.to_torch(arg_220)
    ttnn_to_torch_221 = ttnn.to_torch(arg_221)
    ttnn_to_torch_222 = ttnn.to_torch(arg_222)
    ttnn_to_torch_223 = ttnn.to_torch(arg_223)
    ttnn_to_torch_224 = ttnn.to_torch(arg_224)
    ttnn_to_torch_225 = ttnn.to_torch(arg_225)
    ttnn_to_torch_226 = ttnn.to_torch(arg_226)
    ttnn_to_torch_227 = ttnn.to_torch(arg_227)
    ttnn_to_torch_228 = ttnn.to_torch(arg_228)
    ttnn_to_torch_229 = ttnn.to_torch(arg_229)
    ttnn_to_torch_230 = ttnn.to_torch(arg_230)
    ttnn_to_torch_231 = ttnn.to_torch(arg_231)
    ttnn_to_torch_232 = ttnn.to_torch(arg_232)
    ttnn_to_torch_233 = ttnn.to_torch(arg_233)
    ttnn_to_torch_234 = ttnn.to_torch(arg_234)
    ttnn_to_torch_235 = ttnn.to_torch(arg_235)
    ttnn_to_torch_236 = ttnn.to_torch(arg_236)
    ttnn_to_torch_237 = ttnn.to_torch(arg_237)
    ttnn_to_torch_238 = ttnn.to_torch(arg_238)
    ttnn_to_torch_239 = ttnn.to_torch(arg_239)
    ttnn_to_torch_240 = ttnn.to_torch(arg_240)
    ttnn_to_torch_241 = ttnn.to_torch(arg_241)
    ttnn_to_torch_242 = ttnn.to_torch(arg_242)
    ttnn_to_torch_243 = ttnn.to_torch(arg_243)
    ttnn_to_torch_244 = ttnn.to_torch(arg_244)
    ttnn_to_torch_245 = ttnn.to_torch(arg_245)
    ttnn_to_torch_246 = ttnn.to_torch(arg_246)
    ttnn_to_torch_247 = ttnn.to_torch(arg_247)
    ttnn_to_torch_248 = ttnn.to_torch(arg_248)
    ttnn_to_torch_249 = ttnn.to_torch(arg_249)
    ttnn_to_torch_250 = ttnn.to_torch(arg_250)
    ttnn_to_torch_251 = ttnn.to_torch(arg_251)
    ttnn_to_torch_252 = ttnn.to_torch(arg_252)
    ttnn_to_torch_253 = ttnn.to_torch(arg_253)
    ttnn_to_torch_254 = ttnn.to_torch(arg_254)
    ttnn_to_torch_255 = ttnn.to_torch(arg_255)
    ttnn_to_torch_256 = ttnn.to_torch(arg_256)
    ttnn_to_torch_257 = ttnn.to_torch(arg_257)
    ttnn_to_torch_258 = ttnn.to_torch(arg_258)
    ttnn_to_torch_259 = ttnn.to_torch(arg_259)
    ttnn_to_torch_260 = ttnn.to_torch(arg_260)
    ttnn_to_torch_261 = ttnn.to_torch(arg_261)
    ttnn_to_torch_262 = ttnn.to_torch(arg_262)
    ttnn_to_torch_263 = ttnn.to_torch(arg_263)
    ttnn_to_torch_264 = ttnn.to_torch(arg_264)
    ttir_cpu_full_0 = ttir_cpu.full(shape=[1], fill_value=0.000010, dtype=torch.float32)
    ttir_cpu_add_0 = ttir_cpu.add(ttnn_to_torch_0, ttir_cpu_full_0)
    ttir_cpu_sqrt_0 = ttir_cpu.sqrt(ttir_cpu_add_0)
    ttir_cpu_div_0 = ttir_cpu.div(ttnn_to_torch_1, ttir_cpu_sqrt_0)
    ttir_cpu_reshape_0 = ttir_cpu.reshape(ttir_cpu_div_0, [64, 1, 1, 1])
    ttir_cpu_multiply_0 = ttir_cpu.multiply(ttnn_to_torch_2, ttir_cpu_reshape_0)
    ttir_cpu_reshape_1 = ttir_cpu.reshape(ttnn_to_torch_3, [1, 64, 1, 1])
    ttir_cpu_reshape_2 = ttir_cpu.reshape(ttir_cpu_div_0, [1, 64, 1, 1])
    ttir_cpu_permute_0 = ttir_cpu.permute(ttir_cpu_reshape_1, [0, 2, 3, 1])
    ttir_cpu_permute_1 = ttir_cpu.permute(ttir_cpu_reshape_2, [0, 2, 3, 1])
    ttir_cpu_multiply_1 = ttir_cpu.multiply(ttir_cpu_permute_0, ttir_cpu_permute_1)
    ttir_cpu_reshape_3 = ttir_cpu.reshape(ttnn_to_torch_4, [1, 64, 1, 1])
    ttir_cpu_permute_2 = ttir_cpu.permute(ttir_cpu_reshape_3, [0, 2, 3, 1])
    ttir_cpu_subtract_0 = ttir_cpu.subtract(ttir_cpu_permute_2, ttir_cpu_multiply_1)
    ttir_cpu_add_1 = ttir_cpu.add(ttnn_to_torch_5, ttir_cpu_full_0)
    ttir_cpu_sqrt_1 = ttir_cpu.sqrt(ttir_cpu_add_1)
    ttir_cpu_div_1 = ttir_cpu.div(ttnn_to_torch_6, ttir_cpu_sqrt_1)
    ttir_cpu_reshape_4 = ttir_cpu.reshape(ttir_cpu_div_1, [64, 1, 1, 1])
    ttir_cpu_multiply_2 = ttir_cpu.multiply(ttnn_to_torch_7, ttir_cpu_reshape_4)
    ttir_cpu_reshape_5 = ttir_cpu.reshape(ttnn_to_torch_8, [1, 64, 1, 1])
    ttir_cpu_reshape_6 = ttir_cpu.reshape(ttir_cpu_div_1, [1, 64, 1, 1])
    ttir_cpu_permute_3 = ttir_cpu.permute(ttir_cpu_reshape_5, [0, 2, 3, 1])
    ttir_cpu_permute_4 = ttir_cpu.permute(ttir_cpu_reshape_6, [0, 2, 3, 1])
    ttir_cpu_multiply_3 = ttir_cpu.multiply(ttir_cpu_permute_3, ttir_cpu_permute_4)
    ttir_cpu_reshape_7 = ttir_cpu.reshape(ttnn_to_torch_9, [1, 64, 1, 1])
    ttir_cpu_permute_5 = ttir_cpu.permute(ttir_cpu_reshape_7, [0, 2, 3, 1])
    ttir_cpu_subtract_1 = ttir_cpu.subtract(ttir_cpu_permute_5, ttir_cpu_multiply_3)
    ttir_cpu_add_2 = ttir_cpu.add(ttnn_to_torch_10, ttir_cpu_full_0)
    ttir_cpu_sqrt_2 = ttir_cpu.sqrt(ttir_cpu_add_2)
    ttir_cpu_div_2 = ttir_cpu.div(ttnn_to_torch_11, ttir_cpu_sqrt_2)
    ttir_cpu_reshape_8 = ttir_cpu.reshape(ttir_cpu_div_2, [64, 1, 1, 1])
    ttir_cpu_multiply_4 = ttir_cpu.multiply(ttnn_to_torch_12, ttir_cpu_reshape_8)
    ttir_cpu_reshape_9 = ttir_cpu.reshape(ttnn_to_torch_13, [1, 64, 1, 1])
    ttir_cpu_reshape_10 = ttir_cpu.reshape(ttir_cpu_div_2, [1, 64, 1, 1])
    ttir_cpu_permute_6 = ttir_cpu.permute(ttir_cpu_reshape_9, [0, 2, 3, 1])
    ttir_cpu_permute_7 = ttir_cpu.permute(ttir_cpu_reshape_10, [0, 2, 3, 1])
    ttir_cpu_multiply_5 = ttir_cpu.multiply(ttir_cpu_permute_6, ttir_cpu_permute_7)
    ttir_cpu_reshape_11 = ttir_cpu.reshape(ttnn_to_torch_14, [1, 64, 1, 1])
    ttir_cpu_permute_8 = ttir_cpu.permute(ttir_cpu_reshape_11, [0, 2, 3, 1])
    ttir_cpu_subtract_2 = ttir_cpu.subtract(ttir_cpu_permute_8, ttir_cpu_multiply_5)
    ttir_cpu_add_3 = ttir_cpu.add(ttnn_to_torch_15, ttir_cpu_full_0)
    ttir_cpu_sqrt_3 = ttir_cpu.sqrt(ttir_cpu_add_3)
    ttir_cpu_div_3 = ttir_cpu.div(ttnn_to_torch_16, ttir_cpu_sqrt_3)
    ttir_cpu_reshape_12 = ttir_cpu.reshape(ttir_cpu_div_3, [256, 1, 1, 1])
    ttir_cpu_multiply_6 = ttir_cpu.multiply(ttnn_to_torch_17, ttir_cpu_reshape_12)
    ttir_cpu_reshape_13 = ttir_cpu.reshape(ttnn_to_torch_18, [1, 256, 1, 1])
    ttir_cpu_reshape_14 = ttir_cpu.reshape(ttir_cpu_div_3, [1, 256, 1, 1])
    ttir_cpu_permute_9 = ttir_cpu.permute(ttir_cpu_reshape_13, [0, 2, 3, 1])
    ttir_cpu_permute_10 = ttir_cpu.permute(ttir_cpu_reshape_14, [0, 2, 3, 1])
    ttir_cpu_multiply_7 = ttir_cpu.multiply(ttir_cpu_permute_9, ttir_cpu_permute_10)
    ttir_cpu_reshape_15 = ttir_cpu.reshape(ttnn_to_torch_19, [1, 256, 1, 1])
    ttir_cpu_permute_11 = ttir_cpu.permute(ttir_cpu_reshape_15, [0, 2, 3, 1])
    ttir_cpu_subtract_3 = ttir_cpu.subtract(ttir_cpu_permute_11, ttir_cpu_multiply_7)
    ttir_cpu_add_4 = ttir_cpu.add(ttnn_to_torch_20, ttir_cpu_full_0)
    ttir_cpu_sqrt_4 = ttir_cpu.sqrt(ttir_cpu_add_4)
    ttir_cpu_div_4 = ttir_cpu.div(ttnn_to_torch_21, ttir_cpu_sqrt_4)
    ttir_cpu_reshape_16 = ttir_cpu.reshape(ttir_cpu_div_4, [256, 1, 1, 1])
    ttir_cpu_multiply_8 = ttir_cpu.multiply(ttnn_to_torch_22, ttir_cpu_reshape_16)
    ttir_cpu_reshape_17 = ttir_cpu.reshape(ttnn_to_torch_23, [1, 256, 1, 1])
    ttir_cpu_reshape_18 = ttir_cpu.reshape(ttir_cpu_div_4, [1, 256, 1, 1])
    ttir_cpu_permute_12 = ttir_cpu.permute(ttir_cpu_reshape_17, [0, 2, 3, 1])
    ttir_cpu_permute_13 = ttir_cpu.permute(ttir_cpu_reshape_18, [0, 2, 3, 1])
    ttir_cpu_multiply_9 = ttir_cpu.multiply(ttir_cpu_permute_12, ttir_cpu_permute_13)
    ttir_cpu_reshape_19 = ttir_cpu.reshape(ttnn_to_torch_24, [1, 256, 1, 1])
    ttir_cpu_permute_14 = ttir_cpu.permute(ttir_cpu_reshape_19, [0, 2, 3, 1])
    ttir_cpu_subtract_4 = ttir_cpu.subtract(ttir_cpu_permute_14, ttir_cpu_multiply_9)
    ttir_cpu_add_5 = ttir_cpu.add(ttnn_to_torch_25, ttir_cpu_full_0)
    ttir_cpu_sqrt_5 = ttir_cpu.sqrt(ttir_cpu_add_5)
    ttir_cpu_div_5 = ttir_cpu.div(ttnn_to_torch_26, ttir_cpu_sqrt_5)
    ttir_cpu_reshape_20 = ttir_cpu.reshape(ttir_cpu_div_5, [64, 1, 1, 1])
    ttir_cpu_multiply_10 = ttir_cpu.multiply(ttnn_to_torch_27, ttir_cpu_reshape_20)
    ttir_cpu_reshape_21 = ttir_cpu.reshape(ttnn_to_torch_28, [1, 64, 1, 1])
    ttir_cpu_reshape_22 = ttir_cpu.reshape(ttir_cpu_div_5, [1, 64, 1, 1])
    ttir_cpu_permute_15 = ttir_cpu.permute(ttir_cpu_reshape_21, [0, 2, 3, 1])
    ttir_cpu_permute_16 = ttir_cpu.permute(ttir_cpu_reshape_22, [0, 2, 3, 1])
    ttir_cpu_multiply_11 = ttir_cpu.multiply(ttir_cpu_permute_15, ttir_cpu_permute_16)
    ttir_cpu_reshape_23 = ttir_cpu.reshape(ttnn_to_torch_29, [1, 64, 1, 1])
    ttir_cpu_permute_17 = ttir_cpu.permute(ttir_cpu_reshape_23, [0, 2, 3, 1])
    ttir_cpu_subtract_5 = ttir_cpu.subtract(ttir_cpu_permute_17, ttir_cpu_multiply_11)
    ttir_cpu_add_6 = ttir_cpu.add(ttnn_to_torch_30, ttir_cpu_full_0)
    ttir_cpu_sqrt_6 = ttir_cpu.sqrt(ttir_cpu_add_6)
    ttir_cpu_div_6 = ttir_cpu.div(ttnn_to_torch_31, ttir_cpu_sqrt_6)
    ttir_cpu_reshape_24 = ttir_cpu.reshape(ttir_cpu_div_6, [64, 1, 1, 1])
    ttir_cpu_multiply_12 = ttir_cpu.multiply(ttnn_to_torch_32, ttir_cpu_reshape_24)
    ttir_cpu_reshape_25 = ttir_cpu.reshape(ttnn_to_torch_33, [1, 64, 1, 1])
    ttir_cpu_reshape_26 = ttir_cpu.reshape(ttir_cpu_div_6, [1, 64, 1, 1])
    ttir_cpu_permute_18 = ttir_cpu.permute(ttir_cpu_reshape_25, [0, 2, 3, 1])
    ttir_cpu_permute_19 = ttir_cpu.permute(ttir_cpu_reshape_26, [0, 2, 3, 1])
    ttir_cpu_multiply_13 = ttir_cpu.multiply(ttir_cpu_permute_18, ttir_cpu_permute_19)
    ttir_cpu_reshape_27 = ttir_cpu.reshape(ttnn_to_torch_34, [1, 64, 1, 1])
    ttir_cpu_permute_20 = ttir_cpu.permute(ttir_cpu_reshape_27, [0, 2, 3, 1])
    ttir_cpu_subtract_6 = ttir_cpu.subtract(ttir_cpu_permute_20, ttir_cpu_multiply_13)
    ttir_cpu_add_7 = ttir_cpu.add(ttnn_to_torch_35, ttir_cpu_full_0)
    ttir_cpu_sqrt_7 = ttir_cpu.sqrt(ttir_cpu_add_7)
    ttir_cpu_div_7 = ttir_cpu.div(ttnn_to_torch_36, ttir_cpu_sqrt_7)
    ttir_cpu_reshape_28 = ttir_cpu.reshape(ttir_cpu_div_7, [256, 1, 1, 1])
    ttir_cpu_multiply_14 = ttir_cpu.multiply(ttnn_to_torch_37, ttir_cpu_reshape_28)
    ttir_cpu_reshape_29 = ttir_cpu.reshape(ttnn_to_torch_38, [1, 256, 1, 1])
    ttir_cpu_reshape_30 = ttir_cpu.reshape(ttir_cpu_div_7, [1, 256, 1, 1])
    ttir_cpu_permute_21 = ttir_cpu.permute(ttir_cpu_reshape_29, [0, 2, 3, 1])
    ttir_cpu_permute_22 = ttir_cpu.permute(ttir_cpu_reshape_30, [0, 2, 3, 1])
    ttir_cpu_multiply_15 = ttir_cpu.multiply(ttir_cpu_permute_21, ttir_cpu_permute_22)
    ttir_cpu_reshape_31 = ttir_cpu.reshape(ttnn_to_torch_39, [1, 256, 1, 1])
    ttir_cpu_permute_23 = ttir_cpu.permute(ttir_cpu_reshape_31, [0, 2, 3, 1])
    ttir_cpu_subtract_7 = ttir_cpu.subtract(ttir_cpu_permute_23, ttir_cpu_multiply_15)
    ttir_cpu_add_8 = ttir_cpu.add(ttnn_to_torch_40, ttir_cpu_full_0)
    ttir_cpu_sqrt_8 = ttir_cpu.sqrt(ttir_cpu_add_8)
    ttir_cpu_div_8 = ttir_cpu.div(ttnn_to_torch_41, ttir_cpu_sqrt_8)
    ttir_cpu_reshape_32 = ttir_cpu.reshape(ttir_cpu_div_8, [64, 1, 1, 1])
    ttir_cpu_multiply_16 = ttir_cpu.multiply(ttnn_to_torch_42, ttir_cpu_reshape_32)
    ttir_cpu_reshape_33 = ttir_cpu.reshape(ttnn_to_torch_43, [1, 64, 1, 1])
    ttir_cpu_reshape_34 = ttir_cpu.reshape(ttir_cpu_div_8, [1, 64, 1, 1])
    ttir_cpu_permute_24 = ttir_cpu.permute(ttir_cpu_reshape_33, [0, 2, 3, 1])
    ttir_cpu_permute_25 = ttir_cpu.permute(ttir_cpu_reshape_34, [0, 2, 3, 1])
    ttir_cpu_multiply_17 = ttir_cpu.multiply(ttir_cpu_permute_24, ttir_cpu_permute_25)
    ttir_cpu_reshape_35 = ttir_cpu.reshape(ttnn_to_torch_44, [1, 64, 1, 1])
    ttir_cpu_permute_26 = ttir_cpu.permute(ttir_cpu_reshape_35, [0, 2, 3, 1])
    ttir_cpu_subtract_8 = ttir_cpu.subtract(ttir_cpu_permute_26, ttir_cpu_multiply_17)
    ttir_cpu_add_9 = ttir_cpu.add(ttnn_to_torch_45, ttir_cpu_full_0)
    ttir_cpu_sqrt_9 = ttir_cpu.sqrt(ttir_cpu_add_9)
    ttir_cpu_div_9 = ttir_cpu.div(ttnn_to_torch_46, ttir_cpu_sqrt_9)
    ttir_cpu_reshape_36 = ttir_cpu.reshape(ttir_cpu_div_9, [64, 1, 1, 1])
    ttir_cpu_multiply_18 = ttir_cpu.multiply(ttnn_to_torch_47, ttir_cpu_reshape_36)
    ttir_cpu_reshape_37 = ttir_cpu.reshape(ttnn_to_torch_48, [1, 64, 1, 1])
    ttir_cpu_reshape_38 = ttir_cpu.reshape(ttir_cpu_div_9, [1, 64, 1, 1])
    ttir_cpu_permute_27 = ttir_cpu.permute(ttir_cpu_reshape_37, [0, 2, 3, 1])
    ttir_cpu_permute_28 = ttir_cpu.permute(ttir_cpu_reshape_38, [0, 2, 3, 1])
    ttir_cpu_multiply_19 = ttir_cpu.multiply(ttir_cpu_permute_27, ttir_cpu_permute_28)
    ttir_cpu_reshape_39 = ttir_cpu.reshape(ttnn_to_torch_49, [1, 64, 1, 1])
    ttir_cpu_permute_29 = ttir_cpu.permute(ttir_cpu_reshape_39, [0, 2, 3, 1])
    ttir_cpu_subtract_9 = ttir_cpu.subtract(ttir_cpu_permute_29, ttir_cpu_multiply_19)
    ttir_cpu_add_10 = ttir_cpu.add(ttnn_to_torch_50, ttir_cpu_full_0)
    ttir_cpu_sqrt_10 = ttir_cpu.sqrt(ttir_cpu_add_10)
    ttir_cpu_div_10 = ttir_cpu.div(ttnn_to_torch_51, ttir_cpu_sqrt_10)
    ttir_cpu_reshape_40 = ttir_cpu.reshape(ttir_cpu_div_10, [256, 1, 1, 1])
    ttir_cpu_multiply_20 = ttir_cpu.multiply(ttnn_to_torch_52, ttir_cpu_reshape_40)
    ttir_cpu_reshape_41 = ttir_cpu.reshape(ttnn_to_torch_53, [1, 256, 1, 1])
    ttir_cpu_reshape_42 = ttir_cpu.reshape(ttir_cpu_div_10, [1, 256, 1, 1])
    ttir_cpu_permute_30 = ttir_cpu.permute(ttir_cpu_reshape_41, [0, 2, 3, 1])
    ttir_cpu_permute_31 = ttir_cpu.permute(ttir_cpu_reshape_42, [0, 2, 3, 1])
    ttir_cpu_multiply_21 = ttir_cpu.multiply(ttir_cpu_permute_30, ttir_cpu_permute_31)
    ttir_cpu_reshape_43 = ttir_cpu.reshape(ttnn_to_torch_54, [1, 256, 1, 1])
    ttir_cpu_permute_32 = ttir_cpu.permute(ttir_cpu_reshape_43, [0, 2, 3, 1])
    ttir_cpu_subtract_10 = ttir_cpu.subtract(ttir_cpu_permute_32, ttir_cpu_multiply_21)
    ttir_cpu_add_11 = ttir_cpu.add(ttnn_to_torch_55, ttir_cpu_full_0)
    ttir_cpu_sqrt_11 = ttir_cpu.sqrt(ttir_cpu_add_11)
    ttir_cpu_div_11 = ttir_cpu.div(ttnn_to_torch_56, ttir_cpu_sqrt_11)
    ttir_cpu_reshape_44 = ttir_cpu.reshape(ttir_cpu_div_11, [128, 1, 1, 1])
    ttir_cpu_multiply_22 = ttir_cpu.multiply(ttnn_to_torch_57, ttir_cpu_reshape_44)
    ttir_cpu_reshape_45 = ttir_cpu.reshape(ttnn_to_torch_58, [1, 128, 1, 1])
    ttir_cpu_reshape_46 = ttir_cpu.reshape(ttir_cpu_div_11, [1, 128, 1, 1])
    ttir_cpu_permute_33 = ttir_cpu.permute(ttir_cpu_reshape_45, [0, 2, 3, 1])
    ttir_cpu_permute_34 = ttir_cpu.permute(ttir_cpu_reshape_46, [0, 2, 3, 1])
    ttir_cpu_multiply_23 = ttir_cpu.multiply(ttir_cpu_permute_33, ttir_cpu_permute_34)
    ttir_cpu_reshape_47 = ttir_cpu.reshape(ttnn_to_torch_59, [1, 128, 1, 1])
    ttir_cpu_permute_35 = ttir_cpu.permute(ttir_cpu_reshape_47, [0, 2, 3, 1])
    ttir_cpu_subtract_11 = ttir_cpu.subtract(ttir_cpu_permute_35, ttir_cpu_multiply_23)
    ttir_cpu_add_12 = ttir_cpu.add(ttnn_to_torch_60, ttir_cpu_full_0)
    ttir_cpu_sqrt_12 = ttir_cpu.sqrt(ttir_cpu_add_12)
    ttir_cpu_div_12 = ttir_cpu.div(ttnn_to_torch_61, ttir_cpu_sqrt_12)
    ttir_cpu_reshape_48 = ttir_cpu.reshape(ttir_cpu_div_12, [128, 1, 1, 1])
    ttir_cpu_multiply_24 = ttir_cpu.multiply(ttnn_to_torch_62, ttir_cpu_reshape_48)
    ttir_cpu_reshape_49 = ttir_cpu.reshape(ttnn_to_torch_63, [1, 128, 1, 1])
    ttir_cpu_reshape_50 = ttir_cpu.reshape(ttir_cpu_div_12, [1, 128, 1, 1])
    ttir_cpu_permute_36 = ttir_cpu.permute(ttir_cpu_reshape_49, [0, 2, 3, 1])
    ttir_cpu_permute_37 = ttir_cpu.permute(ttir_cpu_reshape_50, [0, 2, 3, 1])
    ttir_cpu_multiply_25 = ttir_cpu.multiply(ttir_cpu_permute_36, ttir_cpu_permute_37)
    ttir_cpu_reshape_51 = ttir_cpu.reshape(ttnn_to_torch_64, [1, 128, 1, 1])
    ttir_cpu_permute_38 = ttir_cpu.permute(ttir_cpu_reshape_51, [0, 2, 3, 1])
    ttir_cpu_subtract_12 = ttir_cpu.subtract(ttir_cpu_permute_38, ttir_cpu_multiply_25)
    ttir_cpu_add_13 = ttir_cpu.add(ttnn_to_torch_65, ttir_cpu_full_0)
    ttir_cpu_sqrt_13 = ttir_cpu.sqrt(ttir_cpu_add_13)
    ttir_cpu_div_13 = ttir_cpu.div(ttnn_to_torch_66, ttir_cpu_sqrt_13)
    ttir_cpu_reshape_52 = ttir_cpu.reshape(ttir_cpu_div_13, [512, 1, 1, 1])
    ttir_cpu_multiply_26 = ttir_cpu.multiply(ttnn_to_torch_67, ttir_cpu_reshape_52)
    ttir_cpu_reshape_53 = ttir_cpu.reshape(ttnn_to_torch_68, [1, 512, 1, 1])
    ttir_cpu_reshape_54 = ttir_cpu.reshape(ttir_cpu_div_13, [1, 512, 1, 1])
    ttir_cpu_permute_39 = ttir_cpu.permute(ttir_cpu_reshape_53, [0, 2, 3, 1])
    ttir_cpu_permute_40 = ttir_cpu.permute(ttir_cpu_reshape_54, [0, 2, 3, 1])
    ttir_cpu_multiply_27 = ttir_cpu.multiply(ttir_cpu_permute_39, ttir_cpu_permute_40)
    ttir_cpu_reshape_55 = ttir_cpu.reshape(ttnn_to_torch_69, [1, 512, 1, 1])
    ttir_cpu_permute_41 = ttir_cpu.permute(ttir_cpu_reshape_55, [0, 2, 3, 1])
    ttir_cpu_subtract_13 = ttir_cpu.subtract(ttir_cpu_permute_41, ttir_cpu_multiply_27)
    ttir_cpu_add_14 = ttir_cpu.add(ttnn_to_torch_70, ttir_cpu_full_0)
    ttir_cpu_sqrt_14 = ttir_cpu.sqrt(ttir_cpu_add_14)
    ttir_cpu_div_14 = ttir_cpu.div(ttnn_to_torch_71, ttir_cpu_sqrt_14)
    ttir_cpu_reshape_56 = ttir_cpu.reshape(ttir_cpu_div_14, [512, 1, 1, 1])
    ttir_cpu_multiply_28 = ttir_cpu.multiply(ttnn_to_torch_72, ttir_cpu_reshape_56)
    ttir_cpu_reshape_57 = ttir_cpu.reshape(ttnn_to_torch_73, [1, 512, 1, 1])
    ttir_cpu_reshape_58 = ttir_cpu.reshape(ttir_cpu_div_14, [1, 512, 1, 1])
    ttir_cpu_permute_42 = ttir_cpu.permute(ttir_cpu_reshape_57, [0, 2, 3, 1])
    ttir_cpu_permute_43 = ttir_cpu.permute(ttir_cpu_reshape_58, [0, 2, 3, 1])
    ttir_cpu_multiply_29 = ttir_cpu.multiply(ttir_cpu_permute_42, ttir_cpu_permute_43)
    ttir_cpu_reshape_59 = ttir_cpu.reshape(ttnn_to_torch_74, [1, 512, 1, 1])
    ttir_cpu_permute_44 = ttir_cpu.permute(ttir_cpu_reshape_59, [0, 2, 3, 1])
    ttir_cpu_subtract_14 = ttir_cpu.subtract(ttir_cpu_permute_44, ttir_cpu_multiply_29)
    ttir_cpu_add_15 = ttir_cpu.add(ttnn_to_torch_75, ttir_cpu_full_0)
    ttir_cpu_sqrt_15 = ttir_cpu.sqrt(ttir_cpu_add_15)
    ttir_cpu_div_15 = ttir_cpu.div(ttnn_to_torch_76, ttir_cpu_sqrt_15)
    ttir_cpu_reshape_60 = ttir_cpu.reshape(ttir_cpu_div_15, [128, 1, 1, 1])
    ttir_cpu_multiply_30 = ttir_cpu.multiply(ttnn_to_torch_77, ttir_cpu_reshape_60)
    ttir_cpu_reshape_61 = ttir_cpu.reshape(ttnn_to_torch_78, [1, 128, 1, 1])
    ttir_cpu_reshape_62 = ttir_cpu.reshape(ttir_cpu_div_15, [1, 128, 1, 1])
    ttir_cpu_permute_45 = ttir_cpu.permute(ttir_cpu_reshape_61, [0, 2, 3, 1])
    ttir_cpu_permute_46 = ttir_cpu.permute(ttir_cpu_reshape_62, [0, 2, 3, 1])
    ttir_cpu_multiply_31 = ttir_cpu.multiply(ttir_cpu_permute_45, ttir_cpu_permute_46)
    ttir_cpu_reshape_63 = ttir_cpu.reshape(ttnn_to_torch_79, [1, 128, 1, 1])
    ttir_cpu_permute_47 = ttir_cpu.permute(ttir_cpu_reshape_63, [0, 2, 3, 1])
    ttir_cpu_subtract_15 = ttir_cpu.subtract(ttir_cpu_permute_47, ttir_cpu_multiply_31)
    ttir_cpu_add_16 = ttir_cpu.add(ttnn_to_torch_80, ttir_cpu_full_0)
    ttir_cpu_sqrt_16 = ttir_cpu.sqrt(ttir_cpu_add_16)
    ttir_cpu_div_16 = ttir_cpu.div(ttnn_to_torch_81, ttir_cpu_sqrt_16)
    ttir_cpu_reshape_64 = ttir_cpu.reshape(ttir_cpu_div_16, [128, 1, 1, 1])
    ttir_cpu_multiply_32 = ttir_cpu.multiply(ttnn_to_torch_82, ttir_cpu_reshape_64)
    ttir_cpu_reshape_65 = ttir_cpu.reshape(ttnn_to_torch_83, [1, 128, 1, 1])
    ttir_cpu_reshape_66 = ttir_cpu.reshape(ttir_cpu_div_16, [1, 128, 1, 1])
    ttir_cpu_permute_48 = ttir_cpu.permute(ttir_cpu_reshape_65, [0, 2, 3, 1])
    ttir_cpu_permute_49 = ttir_cpu.permute(ttir_cpu_reshape_66, [0, 2, 3, 1])
    ttir_cpu_multiply_33 = ttir_cpu.multiply(ttir_cpu_permute_48, ttir_cpu_permute_49)
    ttir_cpu_reshape_67 = ttir_cpu.reshape(ttnn_to_torch_84, [1, 128, 1, 1])
    ttir_cpu_permute_50 = ttir_cpu.permute(ttir_cpu_reshape_67, [0, 2, 3, 1])
    ttir_cpu_subtract_16 = ttir_cpu.subtract(ttir_cpu_permute_50, ttir_cpu_multiply_33)
    ttir_cpu_add_17 = ttir_cpu.add(ttnn_to_torch_85, ttir_cpu_full_0)
    ttir_cpu_sqrt_17 = ttir_cpu.sqrt(ttir_cpu_add_17)
    ttir_cpu_div_17 = ttir_cpu.div(ttnn_to_torch_86, ttir_cpu_sqrt_17)
    ttir_cpu_reshape_68 = ttir_cpu.reshape(ttir_cpu_div_17, [512, 1, 1, 1])
    ttir_cpu_multiply_34 = ttir_cpu.multiply(ttnn_to_torch_87, ttir_cpu_reshape_68)
    ttir_cpu_reshape_69 = ttir_cpu.reshape(ttnn_to_torch_88, [1, 512, 1, 1])
    ttir_cpu_reshape_70 = ttir_cpu.reshape(ttir_cpu_div_17, [1, 512, 1, 1])
    ttir_cpu_permute_51 = ttir_cpu.permute(ttir_cpu_reshape_69, [0, 2, 3, 1])
    ttir_cpu_permute_52 = ttir_cpu.permute(ttir_cpu_reshape_70, [0, 2, 3, 1])
    ttir_cpu_multiply_35 = ttir_cpu.multiply(ttir_cpu_permute_51, ttir_cpu_permute_52)
    ttir_cpu_reshape_71 = ttir_cpu.reshape(ttnn_to_torch_89, [1, 512, 1, 1])
    ttir_cpu_permute_53 = ttir_cpu.permute(ttir_cpu_reshape_71, [0, 2, 3, 1])
    ttir_cpu_subtract_17 = ttir_cpu.subtract(ttir_cpu_permute_53, ttir_cpu_multiply_35)
    ttir_cpu_add_18 = ttir_cpu.add(ttnn_to_torch_90, ttir_cpu_full_0)
    ttir_cpu_sqrt_18 = ttir_cpu.sqrt(ttir_cpu_add_18)
    ttir_cpu_div_18 = ttir_cpu.div(ttnn_to_torch_91, ttir_cpu_sqrt_18)
    ttir_cpu_reshape_72 = ttir_cpu.reshape(ttir_cpu_div_18, [128, 1, 1, 1])
    ttir_cpu_multiply_36 = ttir_cpu.multiply(ttnn_to_torch_92, ttir_cpu_reshape_72)
    ttir_cpu_reshape_73 = ttir_cpu.reshape(ttnn_to_torch_93, [1, 128, 1, 1])
    ttir_cpu_reshape_74 = ttir_cpu.reshape(ttir_cpu_div_18, [1, 128, 1, 1])
    ttir_cpu_permute_54 = ttir_cpu.permute(ttir_cpu_reshape_73, [0, 2, 3, 1])
    ttir_cpu_permute_55 = ttir_cpu.permute(ttir_cpu_reshape_74, [0, 2, 3, 1])
    ttir_cpu_multiply_37 = ttir_cpu.multiply(ttir_cpu_permute_54, ttir_cpu_permute_55)
    ttir_cpu_reshape_75 = ttir_cpu.reshape(ttnn_to_torch_94, [1, 128, 1, 1])
    ttir_cpu_permute_56 = ttir_cpu.permute(ttir_cpu_reshape_75, [0, 2, 3, 1])
    ttir_cpu_subtract_18 = ttir_cpu.subtract(ttir_cpu_permute_56, ttir_cpu_multiply_37)
    ttir_cpu_add_19 = ttir_cpu.add(ttnn_to_torch_95, ttir_cpu_full_0)
    ttir_cpu_sqrt_19 = ttir_cpu.sqrt(ttir_cpu_add_19)
    ttir_cpu_div_19 = ttir_cpu.div(ttnn_to_torch_96, ttir_cpu_sqrt_19)
    ttir_cpu_reshape_76 = ttir_cpu.reshape(ttir_cpu_div_19, [128, 1, 1, 1])
    ttir_cpu_multiply_38 = ttir_cpu.multiply(ttnn_to_torch_97, ttir_cpu_reshape_76)
    ttir_cpu_reshape_77 = ttir_cpu.reshape(ttnn_to_torch_98, [1, 128, 1, 1])
    ttir_cpu_reshape_78 = ttir_cpu.reshape(ttir_cpu_div_19, [1, 128, 1, 1])
    ttir_cpu_permute_57 = ttir_cpu.permute(ttir_cpu_reshape_77, [0, 2, 3, 1])
    ttir_cpu_permute_58 = ttir_cpu.permute(ttir_cpu_reshape_78, [0, 2, 3, 1])
    ttir_cpu_multiply_39 = ttir_cpu.multiply(ttir_cpu_permute_57, ttir_cpu_permute_58)
    ttir_cpu_reshape_79 = ttir_cpu.reshape(ttnn_to_torch_99, [1, 128, 1, 1])
    ttir_cpu_permute_59 = ttir_cpu.permute(ttir_cpu_reshape_79, [0, 2, 3, 1])
    ttir_cpu_subtract_19 = ttir_cpu.subtract(ttir_cpu_permute_59, ttir_cpu_multiply_39)
    ttir_cpu_add_20 = ttir_cpu.add(ttnn_to_torch_100, ttir_cpu_full_0)
    ttir_cpu_sqrt_20 = ttir_cpu.sqrt(ttir_cpu_add_20)
    ttir_cpu_div_20 = ttir_cpu.div(ttnn_to_torch_101, ttir_cpu_sqrt_20)
    ttir_cpu_reshape_80 = ttir_cpu.reshape(ttir_cpu_div_20, [512, 1, 1, 1])
    ttir_cpu_multiply_40 = ttir_cpu.multiply(ttnn_to_torch_102, ttir_cpu_reshape_80)
    ttir_cpu_reshape_81 = ttir_cpu.reshape(ttnn_to_torch_103, [1, 512, 1, 1])
    ttir_cpu_reshape_82 = ttir_cpu.reshape(ttir_cpu_div_20, [1, 512, 1, 1])
    ttir_cpu_permute_60 = ttir_cpu.permute(ttir_cpu_reshape_81, [0, 2, 3, 1])
    ttir_cpu_permute_61 = ttir_cpu.permute(ttir_cpu_reshape_82, [0, 2, 3, 1])
    ttir_cpu_multiply_41 = ttir_cpu.multiply(ttir_cpu_permute_60, ttir_cpu_permute_61)
    ttir_cpu_reshape_83 = ttir_cpu.reshape(ttnn_to_torch_104, [1, 512, 1, 1])
    ttir_cpu_permute_62 = ttir_cpu.permute(ttir_cpu_reshape_83, [0, 2, 3, 1])
    ttir_cpu_subtract_20 = ttir_cpu.subtract(ttir_cpu_permute_62, ttir_cpu_multiply_41)
    ttir_cpu_add_21 = ttir_cpu.add(ttnn_to_torch_105, ttir_cpu_full_0)
    ttir_cpu_sqrt_21 = ttir_cpu.sqrt(ttir_cpu_add_21)
    ttir_cpu_div_21 = ttir_cpu.div(ttnn_to_torch_106, ttir_cpu_sqrt_21)
    ttir_cpu_reshape_84 = ttir_cpu.reshape(ttir_cpu_div_21, [128, 1, 1, 1])
    ttir_cpu_multiply_42 = ttir_cpu.multiply(ttnn_to_torch_107, ttir_cpu_reshape_84)
    ttir_cpu_reshape_85 = ttir_cpu.reshape(ttnn_to_torch_108, [1, 128, 1, 1])
    ttir_cpu_reshape_86 = ttir_cpu.reshape(ttir_cpu_div_21, [1, 128, 1, 1])
    ttir_cpu_permute_63 = ttir_cpu.permute(ttir_cpu_reshape_85, [0, 2, 3, 1])
    ttir_cpu_permute_64 = ttir_cpu.permute(ttir_cpu_reshape_86, [0, 2, 3, 1])
    ttir_cpu_multiply_43 = ttir_cpu.multiply(ttir_cpu_permute_63, ttir_cpu_permute_64)
    ttir_cpu_reshape_87 = ttir_cpu.reshape(ttnn_to_torch_109, [1, 128, 1, 1])
    ttir_cpu_permute_65 = ttir_cpu.permute(ttir_cpu_reshape_87, [0, 2, 3, 1])
    ttir_cpu_subtract_21 = ttir_cpu.subtract(ttir_cpu_permute_65, ttir_cpu_multiply_43)
    ttir_cpu_add_22 = ttir_cpu.add(ttnn_to_torch_110, ttir_cpu_full_0)
    ttir_cpu_sqrt_22 = ttir_cpu.sqrt(ttir_cpu_add_22)
    ttir_cpu_div_22 = ttir_cpu.div(ttnn_to_torch_111, ttir_cpu_sqrt_22)
    ttir_cpu_reshape_88 = ttir_cpu.reshape(ttir_cpu_div_22, [128, 1, 1, 1])
    ttir_cpu_multiply_44 = ttir_cpu.multiply(ttnn_to_torch_112, ttir_cpu_reshape_88)
    ttir_cpu_reshape_89 = ttir_cpu.reshape(ttnn_to_torch_113, [1, 128, 1, 1])
    ttir_cpu_reshape_90 = ttir_cpu.reshape(ttir_cpu_div_22, [1, 128, 1, 1])
    ttir_cpu_permute_66 = ttir_cpu.permute(ttir_cpu_reshape_89, [0, 2, 3, 1])
    ttir_cpu_permute_67 = ttir_cpu.permute(ttir_cpu_reshape_90, [0, 2, 3, 1])
    ttir_cpu_multiply_45 = ttir_cpu.multiply(ttir_cpu_permute_66, ttir_cpu_permute_67)
    ttir_cpu_reshape_91 = ttir_cpu.reshape(ttnn_to_torch_114, [1, 128, 1, 1])
    ttir_cpu_permute_68 = ttir_cpu.permute(ttir_cpu_reshape_91, [0, 2, 3, 1])
    ttir_cpu_subtract_22 = ttir_cpu.subtract(ttir_cpu_permute_68, ttir_cpu_multiply_45)
    ttir_cpu_add_23 = ttir_cpu.add(ttnn_to_torch_115, ttir_cpu_full_0)
    ttir_cpu_sqrt_23 = ttir_cpu.sqrt(ttir_cpu_add_23)
    ttir_cpu_div_23 = ttir_cpu.div(ttnn_to_torch_116, ttir_cpu_sqrt_23)
    ttir_cpu_reshape_92 = ttir_cpu.reshape(ttir_cpu_div_23, [512, 1, 1, 1])
    ttir_cpu_multiply_46 = ttir_cpu.multiply(ttnn_to_torch_117, ttir_cpu_reshape_92)
    ttir_cpu_reshape_93 = ttir_cpu.reshape(ttnn_to_torch_118, [1, 512, 1, 1])
    ttir_cpu_reshape_94 = ttir_cpu.reshape(ttir_cpu_div_23, [1, 512, 1, 1])
    ttir_cpu_permute_69 = ttir_cpu.permute(ttir_cpu_reshape_93, [0, 2, 3, 1])
    ttir_cpu_permute_70 = ttir_cpu.permute(ttir_cpu_reshape_94, [0, 2, 3, 1])
    ttir_cpu_multiply_47 = ttir_cpu.multiply(ttir_cpu_permute_69, ttir_cpu_permute_70)
    ttir_cpu_reshape_95 = ttir_cpu.reshape(ttnn_to_torch_119, [1, 512, 1, 1])
    ttir_cpu_permute_71 = ttir_cpu.permute(ttir_cpu_reshape_95, [0, 2, 3, 1])
    ttir_cpu_subtract_23 = ttir_cpu.subtract(ttir_cpu_permute_71, ttir_cpu_multiply_47)
    ttir_cpu_add_24 = ttir_cpu.add(ttnn_to_torch_120, ttir_cpu_full_0)
    ttir_cpu_sqrt_24 = ttir_cpu.sqrt(ttir_cpu_add_24)
    ttir_cpu_div_24 = ttir_cpu.div(ttnn_to_torch_121, ttir_cpu_sqrt_24)
    ttir_cpu_reshape_96 = ttir_cpu.reshape(ttir_cpu_div_24, [256, 1, 1, 1])
    ttir_cpu_multiply_48 = ttir_cpu.multiply(ttnn_to_torch_122, ttir_cpu_reshape_96)
    ttir_cpu_reshape_97 = ttir_cpu.reshape(ttnn_to_torch_123, [1, 256, 1, 1])
    ttir_cpu_reshape_98 = ttir_cpu.reshape(ttir_cpu_div_24, [1, 256, 1, 1])
    ttir_cpu_permute_72 = ttir_cpu.permute(ttir_cpu_reshape_97, [0, 2, 3, 1])
    ttir_cpu_permute_73 = ttir_cpu.permute(ttir_cpu_reshape_98, [0, 2, 3, 1])
    ttir_cpu_multiply_49 = ttir_cpu.multiply(ttir_cpu_permute_72, ttir_cpu_permute_73)
    ttir_cpu_reshape_99 = ttir_cpu.reshape(ttnn_to_torch_124, [1, 256, 1, 1])
    ttir_cpu_permute_74 = ttir_cpu.permute(ttir_cpu_reshape_99, [0, 2, 3, 1])
    ttir_cpu_subtract_24 = ttir_cpu.subtract(ttir_cpu_permute_74, ttir_cpu_multiply_49)
    ttir_cpu_add_25 = ttir_cpu.add(ttnn_to_torch_125, ttir_cpu_full_0)
    ttir_cpu_sqrt_25 = ttir_cpu.sqrt(ttir_cpu_add_25)
    ttir_cpu_div_25 = ttir_cpu.div(ttnn_to_torch_126, ttir_cpu_sqrt_25)
    ttir_cpu_reshape_100 = ttir_cpu.reshape(ttir_cpu_div_25, [256, 1, 1, 1])
    ttir_cpu_multiply_50 = ttir_cpu.multiply(ttnn_to_torch_127, ttir_cpu_reshape_100)
    ttir_cpu_reshape_101 = ttir_cpu.reshape(ttnn_to_torch_128, [1, 256, 1, 1])
    ttir_cpu_reshape_102 = ttir_cpu.reshape(ttir_cpu_div_25, [1, 256, 1, 1])
    ttir_cpu_permute_75 = ttir_cpu.permute(ttir_cpu_reshape_101, [0, 2, 3, 1])
    ttir_cpu_permute_76 = ttir_cpu.permute(ttir_cpu_reshape_102, [0, 2, 3, 1])
    ttir_cpu_multiply_51 = ttir_cpu.multiply(ttir_cpu_permute_75, ttir_cpu_permute_76)
    ttir_cpu_reshape_103 = ttir_cpu.reshape(ttnn_to_torch_129, [1, 256, 1, 1])
    ttir_cpu_permute_77 = ttir_cpu.permute(ttir_cpu_reshape_103, [0, 2, 3, 1])
    ttir_cpu_subtract_25 = ttir_cpu.subtract(ttir_cpu_permute_77, ttir_cpu_multiply_51)
    ttir_cpu_add_26 = ttir_cpu.add(ttnn_to_torch_130, ttir_cpu_full_0)
    ttir_cpu_sqrt_26 = ttir_cpu.sqrt(ttir_cpu_add_26)
    ttir_cpu_div_26 = ttir_cpu.div(ttnn_to_torch_131, ttir_cpu_sqrt_26)
    ttir_cpu_reshape_104 = ttir_cpu.reshape(ttir_cpu_div_26, [1024, 1, 1, 1])
    ttir_cpu_multiply_52 = ttir_cpu.multiply(ttnn_to_torch_132, ttir_cpu_reshape_104)
    ttir_cpu_reshape_105 = ttir_cpu.reshape(ttnn_to_torch_133, [1, 1024, 1, 1])
    ttir_cpu_reshape_106 = ttir_cpu.reshape(ttir_cpu_div_26, [1, 1024, 1, 1])
    ttir_cpu_permute_78 = ttir_cpu.permute(ttir_cpu_reshape_105, [0, 2, 3, 1])
    ttir_cpu_permute_79 = ttir_cpu.permute(ttir_cpu_reshape_106, [0, 2, 3, 1])
    ttir_cpu_multiply_53 = ttir_cpu.multiply(ttir_cpu_permute_78, ttir_cpu_permute_79)
    ttir_cpu_reshape_107 = ttir_cpu.reshape(ttnn_to_torch_134, [1, 1024, 1, 1])
    ttir_cpu_permute_80 = ttir_cpu.permute(ttir_cpu_reshape_107, [0, 2, 3, 1])
    ttir_cpu_subtract_26 = ttir_cpu.subtract(ttir_cpu_permute_80, ttir_cpu_multiply_53)
    ttir_cpu_add_27 = ttir_cpu.add(ttnn_to_torch_135, ttir_cpu_full_0)
    ttir_cpu_sqrt_27 = ttir_cpu.sqrt(ttir_cpu_add_27)
    ttir_cpu_div_27 = ttir_cpu.div(ttnn_to_torch_136, ttir_cpu_sqrt_27)
    ttir_cpu_reshape_108 = ttir_cpu.reshape(ttir_cpu_div_27, [1024, 1, 1, 1])
    ttir_cpu_multiply_54 = ttir_cpu.multiply(ttnn_to_torch_137, ttir_cpu_reshape_108)
    ttir_cpu_reshape_109 = ttir_cpu.reshape(ttnn_to_torch_138, [1, 1024, 1, 1])
    ttir_cpu_reshape_110 = ttir_cpu.reshape(ttir_cpu_div_27, [1, 1024, 1, 1])
    ttir_cpu_permute_81 = ttir_cpu.permute(ttir_cpu_reshape_109, [0, 2, 3, 1])
    ttir_cpu_permute_82 = ttir_cpu.permute(ttir_cpu_reshape_110, [0, 2, 3, 1])
    ttir_cpu_multiply_55 = ttir_cpu.multiply(ttir_cpu_permute_81, ttir_cpu_permute_82)
    ttir_cpu_reshape_111 = ttir_cpu.reshape(ttnn_to_torch_139, [1, 1024, 1, 1])
    ttir_cpu_permute_83 = ttir_cpu.permute(ttir_cpu_reshape_111, [0, 2, 3, 1])
    ttir_cpu_subtract_27 = ttir_cpu.subtract(ttir_cpu_permute_83, ttir_cpu_multiply_55)
    ttir_cpu_add_28 = ttir_cpu.add(ttnn_to_torch_140, ttir_cpu_full_0)
    ttir_cpu_sqrt_28 = ttir_cpu.sqrt(ttir_cpu_add_28)
    ttir_cpu_div_28 = ttir_cpu.div(ttnn_to_torch_141, ttir_cpu_sqrt_28)
    ttir_cpu_reshape_112 = ttir_cpu.reshape(ttir_cpu_div_28, [256, 1, 1, 1])
    ttir_cpu_multiply_56 = ttir_cpu.multiply(ttnn_to_torch_142, ttir_cpu_reshape_112)
    ttir_cpu_reshape_113 = ttir_cpu.reshape(ttnn_to_torch_143, [1, 256, 1, 1])
    ttir_cpu_reshape_114 = ttir_cpu.reshape(ttir_cpu_div_28, [1, 256, 1, 1])
    ttir_cpu_permute_84 = ttir_cpu.permute(ttir_cpu_reshape_113, [0, 2, 3, 1])
    ttir_cpu_permute_85 = ttir_cpu.permute(ttir_cpu_reshape_114, [0, 2, 3, 1])
    ttir_cpu_multiply_57 = ttir_cpu.multiply(ttir_cpu_permute_84, ttir_cpu_permute_85)
    ttir_cpu_reshape_115 = ttir_cpu.reshape(ttnn_to_torch_144, [1, 256, 1, 1])
    ttir_cpu_permute_86 = ttir_cpu.permute(ttir_cpu_reshape_115, [0, 2, 3, 1])
    ttir_cpu_subtract_28 = ttir_cpu.subtract(ttir_cpu_permute_86, ttir_cpu_multiply_57)
    ttir_cpu_add_29 = ttir_cpu.add(ttnn_to_torch_145, ttir_cpu_full_0)
    ttir_cpu_sqrt_29 = ttir_cpu.sqrt(ttir_cpu_add_29)
    ttir_cpu_div_29 = ttir_cpu.div(ttnn_to_torch_146, ttir_cpu_sqrt_29)
    ttir_cpu_reshape_116 = ttir_cpu.reshape(ttir_cpu_div_29, [256, 1, 1, 1])
    ttir_cpu_multiply_58 = ttir_cpu.multiply(ttnn_to_torch_147, ttir_cpu_reshape_116)
    ttir_cpu_reshape_117 = ttir_cpu.reshape(ttnn_to_torch_148, [1, 256, 1, 1])
    ttir_cpu_reshape_118 = ttir_cpu.reshape(ttir_cpu_div_29, [1, 256, 1, 1])
    ttir_cpu_permute_87 = ttir_cpu.permute(ttir_cpu_reshape_117, [0, 2, 3, 1])
    ttir_cpu_permute_88 = ttir_cpu.permute(ttir_cpu_reshape_118, [0, 2, 3, 1])
    ttir_cpu_multiply_59 = ttir_cpu.multiply(ttir_cpu_permute_87, ttir_cpu_permute_88)
    ttir_cpu_reshape_119 = ttir_cpu.reshape(ttnn_to_torch_149, [1, 256, 1, 1])
    ttir_cpu_permute_89 = ttir_cpu.permute(ttir_cpu_reshape_119, [0, 2, 3, 1])
    ttir_cpu_subtract_29 = ttir_cpu.subtract(ttir_cpu_permute_89, ttir_cpu_multiply_59)
    ttir_cpu_add_30 = ttir_cpu.add(ttnn_to_torch_150, ttir_cpu_full_0)
    ttir_cpu_sqrt_30 = ttir_cpu.sqrt(ttir_cpu_add_30)
    ttir_cpu_div_30 = ttir_cpu.div(ttnn_to_torch_151, ttir_cpu_sqrt_30)
    ttir_cpu_reshape_120 = ttir_cpu.reshape(ttir_cpu_div_30, [1024, 1, 1, 1])
    ttir_cpu_multiply_60 = ttir_cpu.multiply(ttnn_to_torch_152, ttir_cpu_reshape_120)
    ttir_cpu_reshape_121 = ttir_cpu.reshape(ttnn_to_torch_153, [1, 1024, 1, 1])
    ttir_cpu_reshape_122 = ttir_cpu.reshape(ttir_cpu_div_30, [1, 1024, 1, 1])
    ttir_cpu_permute_90 = ttir_cpu.permute(ttir_cpu_reshape_121, [0, 2, 3, 1])
    ttir_cpu_permute_91 = ttir_cpu.permute(ttir_cpu_reshape_122, [0, 2, 3, 1])
    ttir_cpu_multiply_61 = ttir_cpu.multiply(ttir_cpu_permute_90, ttir_cpu_permute_91)
    ttir_cpu_reshape_123 = ttir_cpu.reshape(ttnn_to_torch_154, [1, 1024, 1, 1])
    ttir_cpu_permute_92 = ttir_cpu.permute(ttir_cpu_reshape_123, [0, 2, 3, 1])
    ttir_cpu_subtract_30 = ttir_cpu.subtract(ttir_cpu_permute_92, ttir_cpu_multiply_61)
    ttir_cpu_add_31 = ttir_cpu.add(ttnn_to_torch_155, ttir_cpu_full_0)
    ttir_cpu_sqrt_31 = ttir_cpu.sqrt(ttir_cpu_add_31)
    ttir_cpu_div_31 = ttir_cpu.div(ttnn_to_torch_156, ttir_cpu_sqrt_31)
    ttir_cpu_reshape_124 = ttir_cpu.reshape(ttir_cpu_div_31, [256, 1, 1, 1])
    ttir_cpu_multiply_62 = ttir_cpu.multiply(ttnn_to_torch_157, ttir_cpu_reshape_124)
    ttir_cpu_reshape_125 = ttir_cpu.reshape(ttnn_to_torch_158, [1, 256, 1, 1])
    ttir_cpu_reshape_126 = ttir_cpu.reshape(ttir_cpu_div_31, [1, 256, 1, 1])
    ttir_cpu_permute_93 = ttir_cpu.permute(ttir_cpu_reshape_125, [0, 2, 3, 1])
    ttir_cpu_permute_94 = ttir_cpu.permute(ttir_cpu_reshape_126, [0, 2, 3, 1])
    ttir_cpu_multiply_63 = ttir_cpu.multiply(ttir_cpu_permute_93, ttir_cpu_permute_94)
    ttir_cpu_reshape_127 = ttir_cpu.reshape(ttnn_to_torch_159, [1, 256, 1, 1])
    ttir_cpu_permute_95 = ttir_cpu.permute(ttir_cpu_reshape_127, [0, 2, 3, 1])
    ttir_cpu_subtract_31 = ttir_cpu.subtract(ttir_cpu_permute_95, ttir_cpu_multiply_63)
    ttir_cpu_add_32 = ttir_cpu.add(ttnn_to_torch_160, ttir_cpu_full_0)
    ttir_cpu_sqrt_32 = ttir_cpu.sqrt(ttir_cpu_add_32)
    ttir_cpu_div_32 = ttir_cpu.div(ttnn_to_torch_161, ttir_cpu_sqrt_32)
    ttir_cpu_reshape_128 = ttir_cpu.reshape(ttir_cpu_div_32, [256, 1, 1, 1])
    ttir_cpu_multiply_64 = ttir_cpu.multiply(ttnn_to_torch_162, ttir_cpu_reshape_128)
    ttir_cpu_reshape_129 = ttir_cpu.reshape(ttnn_to_torch_163, [1, 256, 1, 1])
    ttir_cpu_reshape_130 = ttir_cpu.reshape(ttir_cpu_div_32, [1, 256, 1, 1])
    ttir_cpu_permute_96 = ttir_cpu.permute(ttir_cpu_reshape_129, [0, 2, 3, 1])
    ttir_cpu_permute_97 = ttir_cpu.permute(ttir_cpu_reshape_130, [0, 2, 3, 1])
    ttir_cpu_multiply_65 = ttir_cpu.multiply(ttir_cpu_permute_96, ttir_cpu_permute_97)
    ttir_cpu_reshape_131 = ttir_cpu.reshape(ttnn_to_torch_164, [1, 256, 1, 1])
    ttir_cpu_permute_98 = ttir_cpu.permute(ttir_cpu_reshape_131, [0, 2, 3, 1])
    ttir_cpu_subtract_32 = ttir_cpu.subtract(ttir_cpu_permute_98, ttir_cpu_multiply_65)
    ttir_cpu_add_33 = ttir_cpu.add(ttnn_to_torch_165, ttir_cpu_full_0)
    ttir_cpu_sqrt_33 = ttir_cpu.sqrt(ttir_cpu_add_33)
    ttir_cpu_div_33 = ttir_cpu.div(ttnn_to_torch_166, ttir_cpu_sqrt_33)
    ttir_cpu_reshape_132 = ttir_cpu.reshape(ttir_cpu_div_33, [1024, 1, 1, 1])
    ttir_cpu_multiply_66 = ttir_cpu.multiply(ttnn_to_torch_167, ttir_cpu_reshape_132)
    ttir_cpu_reshape_133 = ttir_cpu.reshape(ttnn_to_torch_168, [1, 1024, 1, 1])
    ttir_cpu_reshape_134 = ttir_cpu.reshape(ttir_cpu_div_33, [1, 1024, 1, 1])
    ttir_cpu_permute_99 = ttir_cpu.permute(ttir_cpu_reshape_133, [0, 2, 3, 1])
    ttir_cpu_permute_100 = ttir_cpu.permute(ttir_cpu_reshape_134, [0, 2, 3, 1])
    ttir_cpu_multiply_67 = ttir_cpu.multiply(ttir_cpu_permute_99, ttir_cpu_permute_100)
    ttir_cpu_reshape_135 = ttir_cpu.reshape(ttnn_to_torch_169, [1, 1024, 1, 1])
    ttir_cpu_permute_101 = ttir_cpu.permute(ttir_cpu_reshape_135, [0, 2, 3, 1])
    ttir_cpu_subtract_33 = ttir_cpu.subtract(ttir_cpu_permute_101, ttir_cpu_multiply_67)
    ttir_cpu_add_34 = ttir_cpu.add(ttnn_to_torch_170, ttir_cpu_full_0)
    ttir_cpu_sqrt_34 = ttir_cpu.sqrt(ttir_cpu_add_34)
    ttir_cpu_div_34 = ttir_cpu.div(ttnn_to_torch_171, ttir_cpu_sqrt_34)
    ttir_cpu_reshape_136 = ttir_cpu.reshape(ttir_cpu_div_34, [256, 1, 1, 1])
    ttir_cpu_multiply_68 = ttir_cpu.multiply(ttnn_to_torch_172, ttir_cpu_reshape_136)
    ttir_cpu_reshape_137 = ttir_cpu.reshape(ttnn_to_torch_173, [1, 256, 1, 1])
    ttir_cpu_reshape_138 = ttir_cpu.reshape(ttir_cpu_div_34, [1, 256, 1, 1])
    ttir_cpu_permute_102 = ttir_cpu.permute(ttir_cpu_reshape_137, [0, 2, 3, 1])
    ttir_cpu_permute_103 = ttir_cpu.permute(ttir_cpu_reshape_138, [0, 2, 3, 1])
    ttir_cpu_multiply_69 = ttir_cpu.multiply(ttir_cpu_permute_102, ttir_cpu_permute_103)
    ttir_cpu_reshape_139 = ttir_cpu.reshape(ttnn_to_torch_174, [1, 256, 1, 1])
    ttir_cpu_permute_104 = ttir_cpu.permute(ttir_cpu_reshape_139, [0, 2, 3, 1])
    ttir_cpu_subtract_34 = ttir_cpu.subtract(ttir_cpu_permute_104, ttir_cpu_multiply_69)
    ttir_cpu_add_35 = ttir_cpu.add(ttnn_to_torch_175, ttir_cpu_full_0)
    ttir_cpu_sqrt_35 = ttir_cpu.sqrt(ttir_cpu_add_35)
    ttir_cpu_div_35 = ttir_cpu.div(ttnn_to_torch_176, ttir_cpu_sqrt_35)
    ttir_cpu_reshape_140 = ttir_cpu.reshape(ttir_cpu_div_35, [256, 1, 1, 1])
    ttir_cpu_multiply_70 = ttir_cpu.multiply(ttnn_to_torch_177, ttir_cpu_reshape_140)
    ttir_cpu_reshape_141 = ttir_cpu.reshape(ttnn_to_torch_178, [1, 256, 1, 1])
    ttir_cpu_reshape_142 = ttir_cpu.reshape(ttir_cpu_div_35, [1, 256, 1, 1])
    ttir_cpu_permute_105 = ttir_cpu.permute(ttir_cpu_reshape_141, [0, 2, 3, 1])
    ttir_cpu_permute_106 = ttir_cpu.permute(ttir_cpu_reshape_142, [0, 2, 3, 1])
    ttir_cpu_multiply_71 = ttir_cpu.multiply(ttir_cpu_permute_105, ttir_cpu_permute_106)
    ttir_cpu_reshape_143 = ttir_cpu.reshape(ttnn_to_torch_179, [1, 256, 1, 1])
    ttir_cpu_permute_107 = ttir_cpu.permute(ttir_cpu_reshape_143, [0, 2, 3, 1])
    ttir_cpu_subtract_35 = ttir_cpu.subtract(ttir_cpu_permute_107, ttir_cpu_multiply_71)
    ttir_cpu_add_36 = ttir_cpu.add(ttnn_to_torch_180, ttir_cpu_full_0)
    ttir_cpu_sqrt_36 = ttir_cpu.sqrt(ttir_cpu_add_36)
    ttir_cpu_div_36 = ttir_cpu.div(ttnn_to_torch_181, ttir_cpu_sqrt_36)
    ttir_cpu_reshape_144 = ttir_cpu.reshape(ttir_cpu_div_36, [1024, 1, 1, 1])
    ttir_cpu_multiply_72 = ttir_cpu.multiply(ttnn_to_torch_182, ttir_cpu_reshape_144)
    ttir_cpu_reshape_145 = ttir_cpu.reshape(ttnn_to_torch_183, [1, 1024, 1, 1])
    ttir_cpu_reshape_146 = ttir_cpu.reshape(ttir_cpu_div_36, [1, 1024, 1, 1])
    ttir_cpu_permute_108 = ttir_cpu.permute(ttir_cpu_reshape_145, [0, 2, 3, 1])
    ttir_cpu_permute_109 = ttir_cpu.permute(ttir_cpu_reshape_146, [0, 2, 3, 1])
    ttir_cpu_multiply_73 = ttir_cpu.multiply(ttir_cpu_permute_108, ttir_cpu_permute_109)
    ttir_cpu_reshape_147 = ttir_cpu.reshape(ttnn_to_torch_184, [1, 1024, 1, 1])
    ttir_cpu_permute_110 = ttir_cpu.permute(ttir_cpu_reshape_147, [0, 2, 3, 1])
    ttir_cpu_subtract_36 = ttir_cpu.subtract(ttir_cpu_permute_110, ttir_cpu_multiply_73)
    ttir_cpu_add_37 = ttir_cpu.add(ttnn_to_torch_185, ttir_cpu_full_0)
    ttir_cpu_sqrt_37 = ttir_cpu.sqrt(ttir_cpu_add_37)
    ttir_cpu_div_37 = ttir_cpu.div(ttnn_to_torch_186, ttir_cpu_sqrt_37)
    ttir_cpu_reshape_148 = ttir_cpu.reshape(ttir_cpu_div_37, [256, 1, 1, 1])
    ttir_cpu_multiply_74 = ttir_cpu.multiply(ttnn_to_torch_187, ttir_cpu_reshape_148)
    ttir_cpu_reshape_149 = ttir_cpu.reshape(ttnn_to_torch_188, [1, 256, 1, 1])
    ttir_cpu_reshape_150 = ttir_cpu.reshape(ttir_cpu_div_37, [1, 256, 1, 1])
    ttir_cpu_permute_111 = ttir_cpu.permute(ttir_cpu_reshape_149, [0, 2, 3, 1])
    ttir_cpu_permute_112 = ttir_cpu.permute(ttir_cpu_reshape_150, [0, 2, 3, 1])
    ttir_cpu_multiply_75 = ttir_cpu.multiply(ttir_cpu_permute_111, ttir_cpu_permute_112)
    ttir_cpu_reshape_151 = ttir_cpu.reshape(ttnn_to_torch_189, [1, 256, 1, 1])
    ttir_cpu_permute_113 = ttir_cpu.permute(ttir_cpu_reshape_151, [0, 2, 3, 1])
    ttir_cpu_subtract_37 = ttir_cpu.subtract(ttir_cpu_permute_113, ttir_cpu_multiply_75)
    ttir_cpu_add_38 = ttir_cpu.add(ttnn_to_torch_190, ttir_cpu_full_0)
    ttir_cpu_sqrt_38 = ttir_cpu.sqrt(ttir_cpu_add_38)
    ttir_cpu_div_38 = ttir_cpu.div(ttnn_to_torch_191, ttir_cpu_sqrt_38)
    ttir_cpu_reshape_152 = ttir_cpu.reshape(ttir_cpu_div_38, [256, 1, 1, 1])
    ttir_cpu_multiply_76 = ttir_cpu.multiply(ttnn_to_torch_192, ttir_cpu_reshape_152)
    ttir_cpu_reshape_153 = ttir_cpu.reshape(ttnn_to_torch_193, [1, 256, 1, 1])
    ttir_cpu_reshape_154 = ttir_cpu.reshape(ttir_cpu_div_38, [1, 256, 1, 1])
    ttir_cpu_permute_114 = ttir_cpu.permute(ttir_cpu_reshape_153, [0, 2, 3, 1])
    ttir_cpu_permute_115 = ttir_cpu.permute(ttir_cpu_reshape_154, [0, 2, 3, 1])
    ttir_cpu_multiply_77 = ttir_cpu.multiply(ttir_cpu_permute_114, ttir_cpu_permute_115)
    ttir_cpu_reshape_155 = ttir_cpu.reshape(ttnn_to_torch_194, [1, 256, 1, 1])
    ttir_cpu_permute_116 = ttir_cpu.permute(ttir_cpu_reshape_155, [0, 2, 3, 1])
    ttir_cpu_subtract_38 = ttir_cpu.subtract(ttir_cpu_permute_116, ttir_cpu_multiply_77)
    ttir_cpu_add_39 = ttir_cpu.add(ttnn_to_torch_195, ttir_cpu_full_0)
    ttir_cpu_sqrt_39 = ttir_cpu.sqrt(ttir_cpu_add_39)
    ttir_cpu_div_39 = ttir_cpu.div(ttnn_to_torch_196, ttir_cpu_sqrt_39)
    ttir_cpu_reshape_156 = ttir_cpu.reshape(ttir_cpu_div_39, [1024, 1, 1, 1])
    ttir_cpu_multiply_78 = ttir_cpu.multiply(ttnn_to_torch_197, ttir_cpu_reshape_156)
    ttir_cpu_reshape_157 = ttir_cpu.reshape(ttnn_to_torch_198, [1, 1024, 1, 1])
    ttir_cpu_reshape_158 = ttir_cpu.reshape(ttir_cpu_div_39, [1, 1024, 1, 1])
    ttir_cpu_permute_117 = ttir_cpu.permute(ttir_cpu_reshape_157, [0, 2, 3, 1])
    ttir_cpu_permute_118 = ttir_cpu.permute(ttir_cpu_reshape_158, [0, 2, 3, 1])
    ttir_cpu_multiply_79 = ttir_cpu.multiply(ttir_cpu_permute_117, ttir_cpu_permute_118)
    ttir_cpu_reshape_159 = ttir_cpu.reshape(ttnn_to_torch_199, [1, 1024, 1, 1])
    ttir_cpu_permute_119 = ttir_cpu.permute(ttir_cpu_reshape_159, [0, 2, 3, 1])
    ttir_cpu_subtract_39 = ttir_cpu.subtract(ttir_cpu_permute_119, ttir_cpu_multiply_79)
    ttir_cpu_add_40 = ttir_cpu.add(ttnn_to_torch_200, ttir_cpu_full_0)
    ttir_cpu_sqrt_40 = ttir_cpu.sqrt(ttir_cpu_add_40)
    ttir_cpu_div_40 = ttir_cpu.div(ttnn_to_torch_201, ttir_cpu_sqrt_40)
    ttir_cpu_reshape_160 = ttir_cpu.reshape(ttir_cpu_div_40, [256, 1, 1, 1])
    ttir_cpu_multiply_80 = ttir_cpu.multiply(ttnn_to_torch_202, ttir_cpu_reshape_160)
    ttir_cpu_reshape_161 = ttir_cpu.reshape(ttnn_to_torch_203, [1, 256, 1, 1])
    ttir_cpu_reshape_162 = ttir_cpu.reshape(ttir_cpu_div_40, [1, 256, 1, 1])
    ttir_cpu_permute_120 = ttir_cpu.permute(ttir_cpu_reshape_161, [0, 2, 3, 1])
    ttir_cpu_permute_121 = ttir_cpu.permute(ttir_cpu_reshape_162, [0, 2, 3, 1])
    ttir_cpu_multiply_81 = ttir_cpu.multiply(ttir_cpu_permute_120, ttir_cpu_permute_121)
    ttir_cpu_reshape_163 = ttir_cpu.reshape(ttnn_to_torch_204, [1, 256, 1, 1])
    ttir_cpu_permute_122 = ttir_cpu.permute(ttir_cpu_reshape_163, [0, 2, 3, 1])
    ttir_cpu_subtract_40 = ttir_cpu.subtract(ttir_cpu_permute_122, ttir_cpu_multiply_81)
    ttir_cpu_add_41 = ttir_cpu.add(ttnn_to_torch_205, ttir_cpu_full_0)
    ttir_cpu_sqrt_41 = ttir_cpu.sqrt(ttir_cpu_add_41)
    ttir_cpu_div_41 = ttir_cpu.div(ttnn_to_torch_206, ttir_cpu_sqrt_41)
    ttir_cpu_reshape_164 = ttir_cpu.reshape(ttir_cpu_div_41, [256, 1, 1, 1])
    ttir_cpu_multiply_82 = ttir_cpu.multiply(ttnn_to_torch_207, ttir_cpu_reshape_164)
    ttir_cpu_reshape_165 = ttir_cpu.reshape(ttnn_to_torch_208, [1, 256, 1, 1])
    ttir_cpu_reshape_166 = ttir_cpu.reshape(ttir_cpu_div_41, [1, 256, 1, 1])
    ttir_cpu_permute_123 = ttir_cpu.permute(ttir_cpu_reshape_165, [0, 2, 3, 1])
    ttir_cpu_permute_124 = ttir_cpu.permute(ttir_cpu_reshape_166, [0, 2, 3, 1])
    ttir_cpu_multiply_83 = ttir_cpu.multiply(ttir_cpu_permute_123, ttir_cpu_permute_124)
    ttir_cpu_reshape_167 = ttir_cpu.reshape(ttnn_to_torch_209, [1, 256, 1, 1])
    ttir_cpu_permute_125 = ttir_cpu.permute(ttir_cpu_reshape_167, [0, 2, 3, 1])
    ttir_cpu_subtract_41 = ttir_cpu.subtract(ttir_cpu_permute_125, ttir_cpu_multiply_83)
    ttir_cpu_add_42 = ttir_cpu.add(ttnn_to_torch_210, ttir_cpu_full_0)
    ttir_cpu_sqrt_42 = ttir_cpu.sqrt(ttir_cpu_add_42)
    ttir_cpu_div_42 = ttir_cpu.div(ttnn_to_torch_211, ttir_cpu_sqrt_42)
    ttir_cpu_reshape_168 = ttir_cpu.reshape(ttir_cpu_div_42, [1024, 1, 1, 1])
    ttir_cpu_multiply_84 = ttir_cpu.multiply(ttnn_to_torch_212, ttir_cpu_reshape_168)
    ttir_cpu_reshape_169 = ttir_cpu.reshape(ttnn_to_torch_213, [1, 1024, 1, 1])
    ttir_cpu_reshape_170 = ttir_cpu.reshape(ttir_cpu_div_42, [1, 1024, 1, 1])
    ttir_cpu_permute_126 = ttir_cpu.permute(ttir_cpu_reshape_169, [0, 2, 3, 1])
    ttir_cpu_permute_127 = ttir_cpu.permute(ttir_cpu_reshape_170, [0, 2, 3, 1])
    ttir_cpu_multiply_85 = ttir_cpu.multiply(ttir_cpu_permute_126, ttir_cpu_permute_127)
    ttir_cpu_reshape_171 = ttir_cpu.reshape(ttnn_to_torch_214, [1, 1024, 1, 1])
    ttir_cpu_permute_128 = ttir_cpu.permute(ttir_cpu_reshape_171, [0, 2, 3, 1])
    ttir_cpu_subtract_42 = ttir_cpu.subtract(ttir_cpu_permute_128, ttir_cpu_multiply_85)
    ttir_cpu_add_43 = ttir_cpu.add(ttnn_to_torch_215, ttir_cpu_full_0)
    ttir_cpu_sqrt_43 = ttir_cpu.sqrt(ttir_cpu_add_43)
    ttir_cpu_div_43 = ttir_cpu.div(ttnn_to_torch_216, ttir_cpu_sqrt_43)
    ttir_cpu_reshape_172 = ttir_cpu.reshape(ttir_cpu_div_43, [512, 1, 1, 1])
    ttir_cpu_multiply_86 = ttir_cpu.multiply(ttnn_to_torch_217, ttir_cpu_reshape_172)
    ttir_cpu_reshape_173 = ttir_cpu.reshape(ttnn_to_torch_218, [1, 512, 1, 1])
    ttir_cpu_reshape_174 = ttir_cpu.reshape(ttir_cpu_div_43, [1, 512, 1, 1])
    ttir_cpu_permute_129 = ttir_cpu.permute(ttir_cpu_reshape_173, [0, 2, 3, 1])
    ttir_cpu_permute_130 = ttir_cpu.permute(ttir_cpu_reshape_174, [0, 2, 3, 1])
    ttir_cpu_multiply_87 = ttir_cpu.multiply(ttir_cpu_permute_129, ttir_cpu_permute_130)
    ttir_cpu_reshape_175 = ttir_cpu.reshape(ttnn_to_torch_219, [1, 512, 1, 1])
    ttir_cpu_permute_131 = ttir_cpu.permute(ttir_cpu_reshape_175, [0, 2, 3, 1])
    ttir_cpu_subtract_43 = ttir_cpu.subtract(ttir_cpu_permute_131, ttir_cpu_multiply_87)
    ttir_cpu_add_44 = ttir_cpu.add(ttnn_to_torch_220, ttir_cpu_full_0)
    ttir_cpu_sqrt_44 = ttir_cpu.sqrt(ttir_cpu_add_44)
    ttir_cpu_div_44 = ttir_cpu.div(ttnn_to_torch_221, ttir_cpu_sqrt_44)
    ttir_cpu_reshape_176 = ttir_cpu.reshape(ttir_cpu_div_44, [512, 1, 1, 1])
    ttir_cpu_multiply_88 = ttir_cpu.multiply(ttnn_to_torch_222, ttir_cpu_reshape_176)
    ttir_cpu_reshape_177 = ttir_cpu.reshape(ttnn_to_torch_223, [1, 512, 1, 1])
    ttir_cpu_reshape_178 = ttir_cpu.reshape(ttir_cpu_div_44, [1, 512, 1, 1])
    ttir_cpu_permute_132 = ttir_cpu.permute(ttir_cpu_reshape_177, [0, 2, 3, 1])
    ttir_cpu_permute_133 = ttir_cpu.permute(ttir_cpu_reshape_178, [0, 2, 3, 1])
    ttir_cpu_multiply_89 = ttir_cpu.multiply(ttir_cpu_permute_132, ttir_cpu_permute_133)
    ttir_cpu_reshape_179 = ttir_cpu.reshape(ttnn_to_torch_224, [1, 512, 1, 1])
    ttir_cpu_permute_134 = ttir_cpu.permute(ttir_cpu_reshape_179, [0, 2, 3, 1])
    ttir_cpu_subtract_44 = ttir_cpu.subtract(ttir_cpu_permute_134, ttir_cpu_multiply_89)
    ttir_cpu_add_45 = ttir_cpu.add(ttnn_to_torch_225, ttir_cpu_full_0)
    ttir_cpu_sqrt_45 = ttir_cpu.sqrt(ttir_cpu_add_45)
    ttir_cpu_div_45 = ttir_cpu.div(ttnn_to_torch_226, ttir_cpu_sqrt_45)
    ttir_cpu_reshape_180 = ttir_cpu.reshape(ttir_cpu_div_45, [2048, 1, 1, 1])
    ttir_cpu_multiply_90 = ttir_cpu.multiply(ttnn_to_torch_227, ttir_cpu_reshape_180)
    ttir_cpu_reshape_181 = ttir_cpu.reshape(ttnn_to_torch_228, [1, 2048, 1, 1])
    ttir_cpu_reshape_182 = ttir_cpu.reshape(ttir_cpu_div_45, [1, 2048, 1, 1])
    ttir_cpu_permute_135 = ttir_cpu.permute(ttir_cpu_reshape_181, [0, 2, 3, 1])
    ttir_cpu_permute_136 = ttir_cpu.permute(ttir_cpu_reshape_182, [0, 2, 3, 1])
    ttir_cpu_multiply_91 = ttir_cpu.multiply(ttir_cpu_permute_135, ttir_cpu_permute_136)
    ttir_cpu_reshape_183 = ttir_cpu.reshape(ttnn_to_torch_229, [1, 2048, 1, 1])
    ttir_cpu_permute_137 = ttir_cpu.permute(ttir_cpu_reshape_183, [0, 2, 3, 1])
    ttir_cpu_subtract_45 = ttir_cpu.subtract(ttir_cpu_permute_137, ttir_cpu_multiply_91)
    ttir_cpu_add_46 = ttir_cpu.add(ttnn_to_torch_230, ttir_cpu_full_0)
    ttir_cpu_sqrt_46 = ttir_cpu.sqrt(ttir_cpu_add_46)
    ttir_cpu_div_46 = ttir_cpu.div(ttnn_to_torch_231, ttir_cpu_sqrt_46)
    ttir_cpu_reshape_184 = ttir_cpu.reshape(ttir_cpu_div_46, [2048, 1, 1, 1])
    ttir_cpu_multiply_92 = ttir_cpu.multiply(ttnn_to_torch_232, ttir_cpu_reshape_184)
    ttir_cpu_reshape_185 = ttir_cpu.reshape(ttnn_to_torch_233, [1, 2048, 1, 1])
    ttir_cpu_reshape_186 = ttir_cpu.reshape(ttir_cpu_div_46, [1, 2048, 1, 1])
    ttir_cpu_permute_138 = ttir_cpu.permute(ttir_cpu_reshape_185, [0, 2, 3, 1])
    ttir_cpu_permute_139 = ttir_cpu.permute(ttir_cpu_reshape_186, [0, 2, 3, 1])
    ttir_cpu_multiply_93 = ttir_cpu.multiply(ttir_cpu_permute_138, ttir_cpu_permute_139)
    ttir_cpu_reshape_187 = ttir_cpu.reshape(ttnn_to_torch_234, [1, 2048, 1, 1])
    ttir_cpu_permute_140 = ttir_cpu.permute(ttir_cpu_reshape_187, [0, 2, 3, 1])
    ttir_cpu_subtract_46 = ttir_cpu.subtract(ttir_cpu_permute_140, ttir_cpu_multiply_93)
    ttir_cpu_add_47 = ttir_cpu.add(ttnn_to_torch_235, ttir_cpu_full_0)
    ttir_cpu_sqrt_47 = ttir_cpu.sqrt(ttir_cpu_add_47)
    ttir_cpu_div_47 = ttir_cpu.div(ttnn_to_torch_236, ttir_cpu_sqrt_47)
    ttir_cpu_reshape_188 = ttir_cpu.reshape(ttir_cpu_div_47, [512, 1, 1, 1])
    ttir_cpu_multiply_94 = ttir_cpu.multiply(ttnn_to_torch_237, ttir_cpu_reshape_188)
    ttir_cpu_reshape_189 = ttir_cpu.reshape(ttnn_to_torch_238, [1, 512, 1, 1])
    ttir_cpu_reshape_190 = ttir_cpu.reshape(ttir_cpu_div_47, [1, 512, 1, 1])
    ttir_cpu_permute_141 = ttir_cpu.permute(ttir_cpu_reshape_189, [0, 2, 3, 1])
    ttir_cpu_permute_142 = ttir_cpu.permute(ttir_cpu_reshape_190, [0, 2, 3, 1])
    ttir_cpu_multiply_95 = ttir_cpu.multiply(ttir_cpu_permute_141, ttir_cpu_permute_142)
    ttir_cpu_reshape_191 = ttir_cpu.reshape(ttnn_to_torch_239, [1, 512, 1, 1])
    ttir_cpu_permute_143 = ttir_cpu.permute(ttir_cpu_reshape_191, [0, 2, 3, 1])
    ttir_cpu_subtract_47 = ttir_cpu.subtract(ttir_cpu_permute_143, ttir_cpu_multiply_95)
    ttir_cpu_add_48 = ttir_cpu.add(ttnn_to_torch_240, ttir_cpu_full_0)
    ttir_cpu_sqrt_48 = ttir_cpu.sqrt(ttir_cpu_add_48)
    ttir_cpu_div_48 = ttir_cpu.div(ttnn_to_torch_241, ttir_cpu_sqrt_48)
    ttir_cpu_reshape_192 = ttir_cpu.reshape(ttir_cpu_div_48, [512, 1, 1, 1])
    ttir_cpu_multiply_96 = ttir_cpu.multiply(ttnn_to_torch_242, ttir_cpu_reshape_192)
    ttir_cpu_reshape_193 = ttir_cpu.reshape(ttnn_to_torch_243, [1, 512, 1, 1])
    ttir_cpu_reshape_194 = ttir_cpu.reshape(ttir_cpu_div_48, [1, 512, 1, 1])
    ttir_cpu_permute_144 = ttir_cpu.permute(ttir_cpu_reshape_193, [0, 2, 3, 1])
    ttir_cpu_permute_145 = ttir_cpu.permute(ttir_cpu_reshape_194, [0, 2, 3, 1])
    ttir_cpu_multiply_97 = ttir_cpu.multiply(ttir_cpu_permute_144, ttir_cpu_permute_145)
    ttir_cpu_reshape_195 = ttir_cpu.reshape(ttnn_to_torch_244, [1, 512, 1, 1])
    ttir_cpu_permute_146 = ttir_cpu.permute(ttir_cpu_reshape_195, [0, 2, 3, 1])
    ttir_cpu_subtract_48 = ttir_cpu.subtract(ttir_cpu_permute_146, ttir_cpu_multiply_97)
    ttir_cpu_add_49 = ttir_cpu.add(ttnn_to_torch_245, ttir_cpu_full_0)
    ttir_cpu_sqrt_49 = ttir_cpu.sqrt(ttir_cpu_add_49)
    ttir_cpu_div_49 = ttir_cpu.div(ttnn_to_torch_246, ttir_cpu_sqrt_49)
    ttir_cpu_reshape_196 = ttir_cpu.reshape(ttir_cpu_div_49, [2048, 1, 1, 1])
    ttir_cpu_multiply_98 = ttir_cpu.multiply(ttnn_to_torch_247, ttir_cpu_reshape_196)
    ttir_cpu_reshape_197 = ttir_cpu.reshape(ttnn_to_torch_248, [1, 2048, 1, 1])
    ttir_cpu_reshape_198 = ttir_cpu.reshape(ttir_cpu_div_49, [1, 2048, 1, 1])
    ttir_cpu_permute_147 = ttir_cpu.permute(ttir_cpu_reshape_197, [0, 2, 3, 1])
    ttir_cpu_permute_148 = ttir_cpu.permute(ttir_cpu_reshape_198, [0, 2, 3, 1])
    ttir_cpu_multiply_99 = ttir_cpu.multiply(ttir_cpu_permute_147, ttir_cpu_permute_148)
    ttir_cpu_reshape_199 = ttir_cpu.reshape(ttnn_to_torch_249, [1, 2048, 1, 1])
    ttir_cpu_permute_149 = ttir_cpu.permute(ttir_cpu_reshape_199, [0, 2, 3, 1])
    ttir_cpu_subtract_49 = ttir_cpu.subtract(ttir_cpu_permute_149, ttir_cpu_multiply_99)
    ttir_cpu_add_50 = ttir_cpu.add(ttnn_to_torch_250, ttir_cpu_full_0)
    ttir_cpu_sqrt_50 = ttir_cpu.sqrt(ttir_cpu_add_50)
    ttir_cpu_div_50 = ttir_cpu.div(ttnn_to_torch_251, ttir_cpu_sqrt_50)
    ttir_cpu_reshape_200 = ttir_cpu.reshape(ttir_cpu_div_50, [512, 1, 1, 1])
    ttir_cpu_multiply_100 = ttir_cpu.multiply(ttnn_to_torch_252, ttir_cpu_reshape_200)
    ttir_cpu_reshape_201 = ttir_cpu.reshape(ttnn_to_torch_253, [1, 512, 1, 1])
    ttir_cpu_reshape_202 = ttir_cpu.reshape(ttir_cpu_div_50, [1, 512, 1, 1])
    ttir_cpu_permute_150 = ttir_cpu.permute(ttir_cpu_reshape_201, [0, 2, 3, 1])
    ttir_cpu_permute_151 = ttir_cpu.permute(ttir_cpu_reshape_202, [0, 2, 3, 1])
    ttir_cpu_multiply_101 = ttir_cpu.multiply(
        ttir_cpu_permute_150, ttir_cpu_permute_151
    )
    ttir_cpu_reshape_203 = ttir_cpu.reshape(ttnn_to_torch_254, [1, 512, 1, 1])
    ttir_cpu_permute_152 = ttir_cpu.permute(ttir_cpu_reshape_203, [0, 2, 3, 1])
    ttir_cpu_subtract_50 = ttir_cpu.subtract(
        ttir_cpu_permute_152, ttir_cpu_multiply_101
    )
    ttir_cpu_add_51 = ttir_cpu.add(ttnn_to_torch_255, ttir_cpu_full_0)
    ttir_cpu_sqrt_51 = ttir_cpu.sqrt(ttir_cpu_add_51)
    ttir_cpu_div_51 = ttir_cpu.div(ttnn_to_torch_256, ttir_cpu_sqrt_51)
    ttir_cpu_reshape_204 = ttir_cpu.reshape(ttir_cpu_div_51, [512, 1, 1, 1])
    ttir_cpu_multiply_102 = ttir_cpu.multiply(ttnn_to_torch_257, ttir_cpu_reshape_204)
    ttir_cpu_reshape_205 = ttir_cpu.reshape(ttnn_to_torch_258, [1, 512, 1, 1])
    ttir_cpu_reshape_206 = ttir_cpu.reshape(ttir_cpu_div_51, [1, 512, 1, 1])
    ttir_cpu_permute_153 = ttir_cpu.permute(ttir_cpu_reshape_205, [0, 2, 3, 1])
    ttir_cpu_permute_154 = ttir_cpu.permute(ttir_cpu_reshape_206, [0, 2, 3, 1])
    ttir_cpu_multiply_103 = ttir_cpu.multiply(
        ttir_cpu_permute_153, ttir_cpu_permute_154
    )
    ttir_cpu_reshape_207 = ttir_cpu.reshape(ttnn_to_torch_259, [1, 512, 1, 1])
    ttir_cpu_permute_155 = ttir_cpu.permute(ttir_cpu_reshape_207, [0, 2, 3, 1])
    ttir_cpu_subtract_51 = ttir_cpu.subtract(
        ttir_cpu_permute_155, ttir_cpu_multiply_103
    )
    ttir_cpu_add_52 = ttir_cpu.add(ttnn_to_torch_260, ttir_cpu_full_0)
    ttir_cpu_sqrt_52 = ttir_cpu.sqrt(ttir_cpu_add_52)
    ttir_cpu_div_52 = ttir_cpu.div(ttnn_to_torch_261, ttir_cpu_sqrt_52)
    ttir_cpu_reshape_208 = ttir_cpu.reshape(ttir_cpu_div_52, [2048, 1, 1, 1])
    ttir_cpu_multiply_104 = ttir_cpu.multiply(ttnn_to_torch_262, ttir_cpu_reshape_208)
    ttir_cpu_reshape_209 = ttir_cpu.reshape(ttnn_to_torch_263, [1, 2048, 1, 1])
    ttir_cpu_reshape_210 = ttir_cpu.reshape(ttir_cpu_div_52, [1, 2048, 1, 1])
    ttir_cpu_permute_156 = ttir_cpu.permute(ttir_cpu_reshape_209, [0, 2, 3, 1])
    ttir_cpu_permute_157 = ttir_cpu.permute(ttir_cpu_reshape_210, [0, 2, 3, 1])
    ttir_cpu_multiply_105 = ttir_cpu.multiply(
        ttir_cpu_permute_156, ttir_cpu_permute_157
    )
    ttir_cpu_reshape_211 = ttir_cpu.reshape(ttnn_to_torch_264, [1, 2048, 1, 1])
    ttir_cpu_permute_158 = ttir_cpu.permute(ttir_cpu_reshape_211, [0, 2, 3, 1])
    ttir_cpu_subtract_52 = ttir_cpu.subtract(
        ttir_cpu_permute_158, ttir_cpu_multiply_105
    )
    ttnn_from_torch_0 = ttnn.from_torch(ttir_cpu_multiply_0)
    ttnn_from_torch_1 = ttnn.from_torch(ttir_cpu_subtract_0)
    ttnn_from_torch_2 = ttnn.from_torch(ttir_cpu_multiply_2)
    ttnn_from_torch_3 = ttnn.from_torch(ttir_cpu_subtract_1)
    ttnn_from_torch_4 = ttnn.from_torch(ttir_cpu_multiply_4)
    ttnn_from_torch_5 = ttnn.from_torch(ttir_cpu_subtract_2)
    ttnn_from_torch_6 = ttnn.from_torch(ttir_cpu_multiply_6)
    ttnn_from_torch_7 = ttnn.from_torch(ttir_cpu_subtract_3)
    ttnn_from_torch_8 = ttnn.from_torch(ttir_cpu_multiply_8)
    ttnn_from_torch_9 = ttnn.from_torch(ttir_cpu_subtract_4)
    ttnn_from_torch_10 = ttnn.from_torch(ttir_cpu_multiply_10)
    ttnn_from_torch_11 = ttnn.from_torch(ttir_cpu_subtract_5)
    ttnn_from_torch_12 = ttnn.from_torch(ttir_cpu_multiply_12)
    ttnn_from_torch_13 = ttnn.from_torch(ttir_cpu_subtract_6)
    ttnn_from_torch_14 = ttnn.from_torch(ttir_cpu_multiply_14)
    ttnn_from_torch_15 = ttnn.from_torch(ttir_cpu_subtract_7)
    ttnn_from_torch_16 = ttnn.from_torch(ttir_cpu_multiply_16)
    ttnn_from_torch_17 = ttnn.from_torch(ttir_cpu_subtract_8)
    ttnn_from_torch_18 = ttnn.from_torch(ttir_cpu_multiply_18)
    ttnn_from_torch_19 = ttnn.from_torch(ttir_cpu_subtract_9)
    ttnn_from_torch_20 = ttnn.from_torch(ttir_cpu_multiply_20)
    ttnn_from_torch_21 = ttnn.from_torch(ttir_cpu_subtract_10)
    ttnn_from_torch_22 = ttnn.from_torch(ttir_cpu_multiply_22)
    ttnn_from_torch_23 = ttnn.from_torch(ttir_cpu_subtract_11)
    ttnn_from_torch_24 = ttnn.from_torch(ttir_cpu_multiply_24)
    ttnn_from_torch_25 = ttnn.from_torch(ttir_cpu_subtract_12)
    ttnn_from_torch_26 = ttnn.from_torch(ttir_cpu_multiply_26)
    ttnn_from_torch_27 = ttnn.from_torch(ttir_cpu_subtract_13)
    ttnn_from_torch_28 = ttnn.from_torch(ttir_cpu_multiply_28)
    ttnn_from_torch_29 = ttnn.from_torch(ttir_cpu_subtract_14)
    ttnn_from_torch_30 = ttnn.from_torch(ttir_cpu_multiply_30)
    ttnn_from_torch_31 = ttnn.from_torch(ttir_cpu_subtract_15)
    ttnn_from_torch_32 = ttnn.from_torch(ttir_cpu_multiply_32)
    ttnn_from_torch_33 = ttnn.from_torch(ttir_cpu_subtract_16)
    ttnn_from_torch_34 = ttnn.from_torch(ttir_cpu_multiply_34)
    ttnn_from_torch_35 = ttnn.from_torch(ttir_cpu_subtract_17)
    ttnn_from_torch_36 = ttnn.from_torch(ttir_cpu_multiply_36)
    ttnn_from_torch_37 = ttnn.from_torch(ttir_cpu_subtract_18)
    ttnn_from_torch_38 = ttnn.from_torch(ttir_cpu_multiply_38)
    ttnn_from_torch_39 = ttnn.from_torch(ttir_cpu_subtract_19)
    ttnn_from_torch_40 = ttnn.from_torch(ttir_cpu_multiply_40)
    ttnn_from_torch_41 = ttnn.from_torch(ttir_cpu_subtract_20)
    ttnn_from_torch_42 = ttnn.from_torch(ttir_cpu_multiply_42)
    ttnn_from_torch_43 = ttnn.from_torch(ttir_cpu_subtract_21)
    ttnn_from_torch_44 = ttnn.from_torch(ttir_cpu_multiply_44)
    ttnn_from_torch_45 = ttnn.from_torch(ttir_cpu_subtract_22)
    ttnn_from_torch_46 = ttnn.from_torch(ttir_cpu_multiply_46)
    ttnn_from_torch_47 = ttnn.from_torch(ttir_cpu_subtract_23)
    ttnn_from_torch_48 = ttnn.from_torch(ttir_cpu_multiply_48)
    ttnn_from_torch_49 = ttnn.from_torch(ttir_cpu_subtract_24)
    ttnn_from_torch_50 = ttnn.from_torch(ttir_cpu_multiply_50)
    ttnn_from_torch_51 = ttnn.from_torch(ttir_cpu_subtract_25)
    ttnn_from_torch_52 = ttnn.from_torch(ttir_cpu_multiply_52)
    ttnn_from_torch_53 = ttnn.from_torch(ttir_cpu_subtract_26)
    ttnn_from_torch_54 = ttnn.from_torch(ttir_cpu_multiply_54)
    ttnn_from_torch_55 = ttnn.from_torch(ttir_cpu_subtract_27)
    ttnn_from_torch_56 = ttnn.from_torch(ttir_cpu_multiply_56)
    ttnn_from_torch_57 = ttnn.from_torch(ttir_cpu_subtract_28)
    ttnn_from_torch_58 = ttnn.from_torch(ttir_cpu_multiply_58)
    ttnn_from_torch_59 = ttnn.from_torch(ttir_cpu_subtract_29)
    ttnn_from_torch_60 = ttnn.from_torch(ttir_cpu_multiply_60)
    ttnn_from_torch_61 = ttnn.from_torch(ttir_cpu_subtract_30)
    ttnn_from_torch_62 = ttnn.from_torch(ttir_cpu_multiply_62)
    ttnn_from_torch_63 = ttnn.from_torch(ttir_cpu_subtract_31)
    ttnn_from_torch_64 = ttnn.from_torch(ttir_cpu_multiply_64)
    ttnn_from_torch_65 = ttnn.from_torch(ttir_cpu_subtract_32)
    ttnn_from_torch_66 = ttnn.from_torch(ttir_cpu_multiply_66)
    ttnn_from_torch_67 = ttnn.from_torch(ttir_cpu_subtract_33)
    ttnn_from_torch_68 = ttnn.from_torch(ttir_cpu_multiply_68)
    ttnn_from_torch_69 = ttnn.from_torch(ttir_cpu_subtract_34)
    ttnn_from_torch_70 = ttnn.from_torch(ttir_cpu_multiply_70)
    ttnn_from_torch_71 = ttnn.from_torch(ttir_cpu_subtract_35)
    ttnn_from_torch_72 = ttnn.from_torch(ttir_cpu_multiply_72)
    ttnn_from_torch_73 = ttnn.from_torch(ttir_cpu_subtract_36)
    ttnn_from_torch_74 = ttnn.from_torch(ttir_cpu_multiply_74)
    ttnn_from_torch_75 = ttnn.from_torch(ttir_cpu_subtract_37)
    ttnn_from_torch_76 = ttnn.from_torch(ttir_cpu_multiply_76)
    ttnn_from_torch_77 = ttnn.from_torch(ttir_cpu_subtract_38)
    ttnn_from_torch_78 = ttnn.from_torch(ttir_cpu_multiply_78)
    ttnn_from_torch_79 = ttnn.from_torch(ttir_cpu_subtract_39)
    ttnn_from_torch_80 = ttnn.from_torch(ttir_cpu_multiply_80)
    ttnn_from_torch_81 = ttnn.from_torch(ttir_cpu_subtract_40)
    ttnn_from_torch_82 = ttnn.from_torch(ttir_cpu_multiply_82)
    ttnn_from_torch_83 = ttnn.from_torch(ttir_cpu_subtract_41)
    ttnn_from_torch_84 = ttnn.from_torch(ttir_cpu_multiply_84)
    ttnn_from_torch_85 = ttnn.from_torch(ttir_cpu_subtract_42)
    ttnn_from_torch_86 = ttnn.from_torch(ttir_cpu_multiply_86)
    ttnn_from_torch_87 = ttnn.from_torch(ttir_cpu_subtract_43)
    ttnn_from_torch_88 = ttnn.from_torch(ttir_cpu_multiply_88)
    ttnn_from_torch_89 = ttnn.from_torch(ttir_cpu_subtract_44)
    ttnn_from_torch_90 = ttnn.from_torch(ttir_cpu_multiply_90)
    ttnn_from_torch_91 = ttnn.from_torch(ttir_cpu_subtract_45)
    ttnn_from_torch_92 = ttnn.from_torch(ttir_cpu_multiply_92)
    ttnn_from_torch_93 = ttnn.from_torch(ttir_cpu_subtract_46)
    ttnn_from_torch_94 = ttnn.from_torch(ttir_cpu_multiply_94)
    ttnn_from_torch_95 = ttnn.from_torch(ttir_cpu_subtract_47)
    ttnn_from_torch_96 = ttnn.from_torch(ttir_cpu_multiply_96)
    ttnn_from_torch_97 = ttnn.from_torch(ttir_cpu_subtract_48)
    ttnn_from_torch_98 = ttnn.from_torch(ttir_cpu_multiply_98)
    ttnn_from_torch_99 = ttnn.from_torch(ttir_cpu_subtract_49)
    ttnn_from_torch_100 = ttnn.from_torch(ttir_cpu_multiply_100)
    ttnn_from_torch_101 = ttnn.from_torch(ttir_cpu_subtract_50)
    ttnn_from_torch_102 = ttnn.from_torch(ttir_cpu_multiply_102)
    ttnn_from_torch_103 = ttnn.from_torch(ttir_cpu_subtract_51)
    ttnn_from_torch_104 = ttnn.from_torch(ttir_cpu_multiply_104)
    ttnn_from_torch_105 = ttnn.from_torch(ttir_cpu_subtract_52)
    return (
        ttnn_from_torch_0,
        ttnn_from_torch_1,
        ttnn_from_torch_2,
        ttnn_from_torch_3,
        ttnn_from_torch_4,
        ttnn_from_torch_5,
        ttnn_from_torch_6,
        ttnn_from_torch_7,
        ttnn_from_torch_8,
        ttnn_from_torch_9,
        ttnn_from_torch_10,
        ttnn_from_torch_11,
        ttnn_from_torch_12,
        ttnn_from_torch_13,
        ttnn_from_torch_14,
        ttnn_from_torch_15,
        ttnn_from_torch_16,
        ttnn_from_torch_17,
        ttnn_from_torch_18,
        ttnn_from_torch_19,
        ttnn_from_torch_20,
        ttnn_from_torch_21,
        ttnn_from_torch_22,
        ttnn_from_torch_23,
        ttnn_from_torch_24,
        ttnn_from_torch_25,
        ttnn_from_torch_26,
        ttnn_from_torch_27,
        ttnn_from_torch_28,
        ttnn_from_torch_29,
        ttnn_from_torch_30,
        ttnn_from_torch_31,
        ttnn_from_torch_32,
        ttnn_from_torch_33,
        ttnn_from_torch_34,
        ttnn_from_torch_35,
        ttnn_from_torch_36,
        ttnn_from_torch_37,
        ttnn_from_torch_38,
        ttnn_from_torch_39,
        ttnn_from_torch_40,
        ttnn_from_torch_41,
        ttnn_from_torch_42,
        ttnn_from_torch_43,
        ttnn_from_torch_44,
        ttnn_from_torch_45,
        ttnn_from_torch_46,
        ttnn_from_torch_47,
        ttnn_from_torch_48,
        ttnn_from_torch_49,
        ttnn_from_torch_50,
        ttnn_from_torch_51,
        ttnn_from_torch_52,
        ttnn_from_torch_53,
        ttnn_from_torch_54,
        ttnn_from_torch_55,
        ttnn_from_torch_56,
        ttnn_from_torch_57,
        ttnn_from_torch_58,
        ttnn_from_torch_59,
        ttnn_from_torch_60,
        ttnn_from_torch_61,
        ttnn_from_torch_62,
        ttnn_from_torch_63,
        ttnn_from_torch_64,
        ttnn_from_torch_65,
        ttnn_from_torch_66,
        ttnn_from_torch_67,
        ttnn_from_torch_68,
        ttnn_from_torch_69,
        ttnn_from_torch_70,
        ttnn_from_torch_71,
        ttnn_from_torch_72,
        ttnn_from_torch_73,
        ttnn_from_torch_74,
        ttnn_from_torch_75,
        ttnn_from_torch_76,
        ttnn_from_torch_77,
        ttnn_from_torch_78,
        ttnn_from_torch_79,
        ttnn_from_torch_80,
        ttnn_from_torch_81,
        ttnn_from_torch_82,
        ttnn_from_torch_83,
        ttnn_from_torch_84,
        ttnn_from_torch_85,
        ttnn_from_torch_86,
        ttnn_from_torch_87,
        ttnn_from_torch_88,
        ttnn_from_torch_89,
        ttnn_from_torch_90,
        ttnn_from_torch_91,
        ttnn_from_torch_92,
        ttnn_from_torch_93,
        ttnn_from_torch_94,
        ttnn_from_torch_95,
        ttnn_from_torch_96,
        ttnn_from_torch_97,
        ttnn_from_torch_98,
        ttnn_from_torch_99,
        ttnn_from_torch_100,
        ttnn_from_torch_101,
        ttnn_from_torch_102,
        ttnn_from_torch_103,
        ttnn_from_torch_104,
        ttnn_from_torch_105,
    )


def main_const_eval_0(arg, device):
    ttnn_typecast_2 = ttnn.typecast(arg[20], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_3 = ttnn.typecast(arg[23], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_4 = ttnn.typecast(arg[24], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_5 = ttnn.typecast(arg[21], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_6 = ttnn.typecast(arg[22], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_7 = ttnn.typecast(arg[35], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_8 = ttnn.typecast(arg[38], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_9 = ttnn.typecast(arg[39], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_10 = ttnn.typecast(arg[36], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_11 = ttnn.typecast(arg[37], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_12 = ttnn.typecast(arg[30], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_13 = ttnn.typecast(arg[33], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_14 = ttnn.typecast(arg[34], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_15 = ttnn.typecast(arg[31], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_16 = ttnn.typecast(arg[32], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_17 = ttnn.typecast(arg[25], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_18 = ttnn.typecast(arg[28], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_19 = ttnn.typecast(arg[29], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_20 = ttnn.typecast(arg[26], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_21 = ttnn.typecast(arg[27], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_22 = ttnn.typecast(arg[15], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_23 = ttnn.typecast(arg[18], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_24 = ttnn.typecast(arg[19], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_25 = ttnn.typecast(arg[16], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_26 = ttnn.typecast(arg[17], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_27 = ttnn.typecast(arg[50], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_28 = ttnn.typecast(arg[53], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_29 = ttnn.typecast(arg[54], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_30 = ttnn.typecast(arg[51], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_31 = ttnn.typecast(arg[52], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_32 = ttnn.typecast(arg[45], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_33 = ttnn.typecast(arg[48], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_34 = ttnn.typecast(arg[49], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_35 = ttnn.typecast(arg[46], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_36 = ttnn.typecast(arg[47], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_37 = ttnn.typecast(arg[40], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_38 = ttnn.typecast(arg[43], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_39 = ttnn.typecast(arg[44], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_40 = ttnn.typecast(arg[41], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_41 = ttnn.typecast(arg[42], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_42 = ttnn.typecast(arg[65], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_43 = ttnn.typecast(arg[68], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_44 = ttnn.typecast(arg[69], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_45 = ttnn.typecast(arg[66], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_46 = ttnn.typecast(arg[67], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_47 = ttnn.typecast(arg[60], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_48 = ttnn.typecast(arg[63], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_49 = ttnn.typecast(arg[64], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_50 = ttnn.typecast(arg[61], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_51 = ttnn.typecast(arg[62], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_52 = ttnn.typecast(arg[55], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_53 = ttnn.typecast(arg[58], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_54 = ttnn.typecast(arg[59], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_55 = ttnn.typecast(arg[56], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_56 = ttnn.typecast(arg[57], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_57 = ttnn.typecast(arg[80], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_58 = ttnn.typecast(arg[83], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_59 = ttnn.typecast(arg[84], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_60 = ttnn.typecast(arg[81], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_61 = ttnn.typecast(arg[82], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_62 = ttnn.typecast(arg[75], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_63 = ttnn.typecast(arg[78], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_64 = ttnn.typecast(arg[79], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_65 = ttnn.typecast(arg[76], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_66 = ttnn.typecast(arg[77], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_67 = ttnn.typecast(arg[70], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_68 = ttnn.typecast(arg[73], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_69 = ttnn.typecast(arg[74], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_70 = ttnn.typecast(arg[71], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_71 = ttnn.typecast(arg[72], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_72 = ttnn.typecast(arg[10], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_73 = ttnn.typecast(arg[13], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_74 = ttnn.typecast(arg[14], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_75 = ttnn.typecast(arg[11], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_76 = ttnn.typecast(arg[12], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_77 = ttnn.typecast(arg[95], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_78 = ttnn.typecast(arg[98], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_79 = ttnn.typecast(arg[99], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_80 = ttnn.typecast(arg[96], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_81 = ttnn.typecast(arg[97], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_82 = ttnn.typecast(arg[90], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_83 = ttnn.typecast(arg[93], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_84 = ttnn.typecast(arg[94], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_85 = ttnn.typecast(arg[91], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_86 = ttnn.typecast(arg[92], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_87 = ttnn.typecast(arg[85], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_88 = ttnn.typecast(arg[88], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_89 = ttnn.typecast(arg[89], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_90 = ttnn.typecast(arg[86], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_91 = ttnn.typecast(arg[87], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_92 = ttnn.typecast(
        arg[110], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_93 = ttnn.typecast(
        arg[113], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_94 = ttnn.typecast(
        arg[114], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_95 = ttnn.typecast(
        arg[111], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_96 = ttnn.typecast(
        arg[112], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_97 = ttnn.typecast(
        arg[105], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_98 = ttnn.typecast(
        arg[108], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_99 = ttnn.typecast(
        arg[109], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_100 = ttnn.typecast(
        arg[106], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_101 = ttnn.typecast(
        arg[107], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_102 = ttnn.typecast(
        arg[100], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_103 = ttnn.typecast(
        arg[103], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_104 = ttnn.typecast(
        arg[104], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_105 = ttnn.typecast(
        arg[101], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_106 = ttnn.typecast(
        arg[102], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_107 = ttnn.typecast(
        arg[125], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_108 = ttnn.typecast(
        arg[128], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_109 = ttnn.typecast(
        arg[129], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_110 = ttnn.typecast(
        arg[126], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_111 = ttnn.typecast(
        arg[127], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_112 = ttnn.typecast(
        arg[120], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_113 = ttnn.typecast(
        arg[123], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_114 = ttnn.typecast(
        arg[124], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_115 = ttnn.typecast(
        arg[121], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_116 = ttnn.typecast(
        arg[122], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_117 = ttnn.typecast(
        arg[115], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_118 = ttnn.typecast(
        arg[118], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_119 = ttnn.typecast(
        arg[119], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_120 = ttnn.typecast(
        arg[116], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_121 = ttnn.typecast(
        arg[117], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_122 = ttnn.typecast(
        arg[140], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_123 = ttnn.typecast(
        arg[143], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_124 = ttnn.typecast(
        arg[144], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_125 = ttnn.typecast(
        arg[141], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_126 = ttnn.typecast(
        arg[142], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_127 = ttnn.typecast(
        arg[135], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_128 = ttnn.typecast(
        arg[138], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_129 = ttnn.typecast(
        arg[139], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_130 = ttnn.typecast(
        arg[136], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_131 = ttnn.typecast(
        arg[137], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_132 = ttnn.typecast(
        arg[130], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_133 = ttnn.typecast(
        arg[133], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_134 = ttnn.typecast(
        arg[134], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_135 = ttnn.typecast(
        arg[131], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_136 = ttnn.typecast(
        arg[132], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_137 = ttnn.typecast(arg[5], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_138 = ttnn.typecast(arg[8], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_139 = ttnn.typecast(arg[9], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_140 = ttnn.typecast(arg[6], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_141 = ttnn.typecast(arg[7], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_142 = ttnn.typecast(
        arg[155], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_143 = ttnn.typecast(
        arg[158], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_144 = ttnn.typecast(
        arg[159], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_145 = ttnn.typecast(
        arg[156], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_146 = ttnn.typecast(
        arg[157], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_147 = ttnn.typecast(
        arg[150], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_148 = ttnn.typecast(
        arg[153], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_149 = ttnn.typecast(
        arg[154], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_150 = ttnn.typecast(
        arg[151], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_151 = ttnn.typecast(
        arg[152], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_152 = ttnn.typecast(
        arg[145], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_153 = ttnn.typecast(
        arg[148], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_154 = ttnn.typecast(
        arg[149], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_155 = ttnn.typecast(
        arg[146], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_156 = ttnn.typecast(
        arg[147], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_157 = ttnn.typecast(
        arg[170], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_158 = ttnn.typecast(
        arg[173], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_159 = ttnn.typecast(
        arg[174], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_160 = ttnn.typecast(
        arg[171], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_161 = ttnn.typecast(
        arg[172], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_162 = ttnn.typecast(
        arg[165], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_163 = ttnn.typecast(
        arg[168], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_164 = ttnn.typecast(
        arg[169], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_165 = ttnn.typecast(
        arg[166], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_166 = ttnn.typecast(
        arg[167], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_167 = ttnn.typecast(
        arg[160], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_168 = ttnn.typecast(
        arg[163], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_169 = ttnn.typecast(
        arg[164], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_170 = ttnn.typecast(
        arg[161], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_171 = ttnn.typecast(
        arg[162], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_172 = ttnn.typecast(
        arg[185], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_173 = ttnn.typecast(
        arg[188], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_174 = ttnn.typecast(
        arg[189], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_175 = ttnn.typecast(
        arg[186], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_176 = ttnn.typecast(
        arg[187], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_177 = ttnn.typecast(
        arg[180], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_178 = ttnn.typecast(
        arg[183], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_179 = ttnn.typecast(
        arg[184], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_180 = ttnn.typecast(
        arg[181], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_181 = ttnn.typecast(
        arg[182], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_182 = ttnn.typecast(
        arg[175], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_183 = ttnn.typecast(
        arg[178], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_184 = ttnn.typecast(
        arg[179], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_185 = ttnn.typecast(
        arg[176], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_186 = ttnn.typecast(
        arg[177], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_187 = ttnn.typecast(
        arg[200], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_188 = ttnn.typecast(
        arg[203], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_189 = ttnn.typecast(
        arg[204], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_190 = ttnn.typecast(
        arg[201], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_191 = ttnn.typecast(
        arg[202], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_192 = ttnn.typecast(
        arg[195], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_193 = ttnn.typecast(
        arg[198], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_194 = ttnn.typecast(
        arg[199], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_195 = ttnn.typecast(
        arg[196], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_196 = ttnn.typecast(
        arg[197], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_197 = ttnn.typecast(
        arg[190], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_198 = ttnn.typecast(
        arg[193], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_199 = ttnn.typecast(
        arg[194], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_200 = ttnn.typecast(
        arg[191], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_201 = ttnn.typecast(
        arg[192], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_202 = ttnn.typecast(
        arg[215], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_203 = ttnn.typecast(
        arg[218], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_204 = ttnn.typecast(
        arg[219], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_205 = ttnn.typecast(
        arg[216], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_206 = ttnn.typecast(
        arg[217], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_207 = ttnn.typecast(
        arg[210], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_208 = ttnn.typecast(
        arg[213], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_209 = ttnn.typecast(
        arg[214], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_210 = ttnn.typecast(
        arg[211], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_211 = ttnn.typecast(
        arg[212], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_212 = ttnn.typecast(
        arg[205], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_213 = ttnn.typecast(
        arg[208], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_214 = ttnn.typecast(
        arg[209], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_215 = ttnn.typecast(
        arg[206], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_216 = ttnn.typecast(
        arg[207], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_217 = ttnn.typecast(
        arg[230], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_218 = ttnn.typecast(
        arg[233], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_219 = ttnn.typecast(
        arg[234], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_220 = ttnn.typecast(
        arg[231], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_221 = ttnn.typecast(
        arg[232], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_222 = ttnn.typecast(
        arg[225], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_223 = ttnn.typecast(
        arg[228], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_224 = ttnn.typecast(
        arg[229], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_225 = ttnn.typecast(
        arg[226], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_226 = ttnn.typecast(
        arg[227], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_227 = ttnn.typecast(
        arg[220], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_228 = ttnn.typecast(
        arg[223], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_229 = ttnn.typecast(
        arg[224], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_230 = ttnn.typecast(
        arg[221], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_231 = ttnn.typecast(
        arg[222], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_232 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_233 = ttnn.typecast(arg[3], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_234 = ttnn.typecast(arg[4], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_235 = ttnn.typecast(arg[1], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_236 = ttnn.typecast(arg[2], ttnn.DataType.FLOAT32, memory_config=None)
    ttnn_typecast_237 = ttnn.typecast(
        arg[245], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_238 = ttnn.typecast(
        arg[248], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_239 = ttnn.typecast(
        arg[249], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_240 = ttnn.typecast(
        arg[246], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_241 = ttnn.typecast(
        arg[247], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_242 = ttnn.typecast(
        arg[240], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_243 = ttnn.typecast(
        arg[243], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_244 = ttnn.typecast(
        arg[244], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_245 = ttnn.typecast(
        arg[241], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_246 = ttnn.typecast(
        arg[242], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_247 = ttnn.typecast(
        arg[235], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_248 = ttnn.typecast(
        arg[238], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_249 = ttnn.typecast(
        arg[239], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_250 = ttnn.typecast(
        arg[236], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_251 = ttnn.typecast(
        arg[237], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_252 = ttnn.typecast(
        arg[260], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_253 = ttnn.typecast(
        arg[263], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_254 = ttnn.typecast(
        arg[264], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_255 = ttnn.typecast(
        arg[261], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_256 = ttnn.typecast(
        arg[262], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_257 = ttnn.typecast(
        arg[255], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_258 = ttnn.typecast(
        arg[258], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_259 = ttnn.typecast(
        arg[259], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_260 = ttnn.typecast(
        arg[256], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_261 = ttnn.typecast(
        arg[257], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_262 = ttnn.typecast(
        arg[250], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_263 = ttnn.typecast(
        arg[253], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_264 = ttnn.typecast(
        arg[254], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_265 = ttnn.typecast(
        arg[251], ttnn.DataType.FLOAT32, memory_config=None
    )
    ttnn_typecast_266 = ttnn.typecast(
        arg[252], ttnn.DataType.FLOAT32, memory_config=None
    )
    (
        v_0,
        v_1,
        v_2,
        v_3,
        v_4,
        v_5,
        v_6,
        v_7,
        v_8,
        v_9,
        v_10,
        v_11,
        v_12,
        v_13,
        v_14,
        v_15,
        v_16,
        v_17,
        v_18,
        v_19,
        v_20,
        v_21,
        v_22,
        v_23,
        v_24,
        v_25,
        v_26,
        v_27,
        v_28,
        v_29,
        v_30,
        v_31,
        v_32,
        v_33,
        v_34,
        v_35,
        v_36,
        v_37,
        v_38,
        v_39,
        v_40,
        v_41,
        v_42,
        v_43,
        v_44,
        v_45,
        v_46,
        v_47,
        v_48,
        v_49,
        v_50,
        v_51,
        v_52,
        v_53,
        v_54,
        v_55,
        v_56,
        v_57,
        v_58,
        v_59,
        v_60,
        v_61,
        v_62,
        v_63,
        v_64,
        v_65,
        v_66,
        v_67,
        v_68,
        v_69,
        v_70,
        v_71,
        v_72,
        v_73,
        v_74,
        v_75,
        v_76,
        v_77,
        v_78,
        v_79,
        v_80,
        v_81,
        v_82,
        v_83,
        v_84,
        v_85,
        v_86,
        v_87,
        v_88,
        v_89,
        v_90,
        v_91,
        v_92,
        v_93,
        v_94,
        v_95,
        v_96,
        v_97,
        v_98,
        v_99,
        v_100,
        v_101,
        v_102,
        v_103,
        v_104,
        v_105,
    ) = cpu_hoisted_const_eval_3d7508d3(
        ttnn_typecast_2,
        ttnn_typecast_3,
        ttnn_typecast_4,
        ttnn_typecast_5,
        ttnn_typecast_6,
        ttnn_typecast_7,
        ttnn_typecast_8,
        ttnn_typecast_9,
        ttnn_typecast_10,
        ttnn_typecast_11,
        ttnn_typecast_12,
        ttnn_typecast_13,
        ttnn_typecast_14,
        ttnn_typecast_15,
        ttnn_typecast_16,
        ttnn_typecast_17,
        ttnn_typecast_18,
        ttnn_typecast_19,
        ttnn_typecast_20,
        ttnn_typecast_21,
        ttnn_typecast_22,
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
        ttnn_typecast_78,
        ttnn_typecast_79,
        ttnn_typecast_80,
        ttnn_typecast_81,
        ttnn_typecast_82,
        ttnn_typecast_83,
        ttnn_typecast_84,
        ttnn_typecast_85,
        ttnn_typecast_86,
        ttnn_typecast_87,
        ttnn_typecast_88,
        ttnn_typecast_89,
        ttnn_typecast_90,
        ttnn_typecast_91,
        ttnn_typecast_92,
        ttnn_typecast_93,
        ttnn_typecast_94,
        ttnn_typecast_95,
        ttnn_typecast_96,
        ttnn_typecast_97,
        ttnn_typecast_98,
        ttnn_typecast_99,
        ttnn_typecast_100,
        ttnn_typecast_101,
        ttnn_typecast_102,
        ttnn_typecast_103,
        ttnn_typecast_104,
        ttnn_typecast_105,
        ttnn_typecast_106,
        ttnn_typecast_107,
        ttnn_typecast_108,
        ttnn_typecast_109,
        ttnn_typecast_110,
        ttnn_typecast_111,
        ttnn_typecast_112,
        ttnn_typecast_113,
        ttnn_typecast_114,
        ttnn_typecast_115,
        ttnn_typecast_116,
        ttnn_typecast_117,
        ttnn_typecast_118,
        ttnn_typecast_119,
        ttnn_typecast_120,
        ttnn_typecast_121,
        ttnn_typecast_122,
        ttnn_typecast_123,
        ttnn_typecast_124,
        ttnn_typecast_125,
        ttnn_typecast_126,
        ttnn_typecast_127,
        ttnn_typecast_128,
        ttnn_typecast_129,
        ttnn_typecast_130,
        ttnn_typecast_131,
        ttnn_typecast_132,
        ttnn_typecast_133,
        ttnn_typecast_134,
        ttnn_typecast_135,
        ttnn_typecast_136,
        ttnn_typecast_137,
        ttnn_typecast_138,
        ttnn_typecast_139,
        ttnn_typecast_140,
        ttnn_typecast_141,
        ttnn_typecast_142,
        ttnn_typecast_143,
        ttnn_typecast_144,
        ttnn_typecast_145,
        ttnn_typecast_146,
        ttnn_typecast_147,
        ttnn_typecast_148,
        ttnn_typecast_149,
        ttnn_typecast_150,
        ttnn_typecast_151,
        ttnn_typecast_152,
        ttnn_typecast_153,
        ttnn_typecast_154,
        ttnn_typecast_155,
        ttnn_typecast_156,
        ttnn_typecast_157,
        ttnn_typecast_158,
        ttnn_typecast_159,
        ttnn_typecast_160,
        ttnn_typecast_161,
        ttnn_typecast_162,
        ttnn_typecast_163,
        ttnn_typecast_164,
        ttnn_typecast_165,
        ttnn_typecast_166,
        ttnn_typecast_167,
        ttnn_typecast_168,
        ttnn_typecast_169,
        ttnn_typecast_170,
        ttnn_typecast_171,
        ttnn_typecast_172,
        ttnn_typecast_173,
        ttnn_typecast_174,
        ttnn_typecast_175,
        ttnn_typecast_176,
        ttnn_typecast_177,
        ttnn_typecast_178,
        ttnn_typecast_179,
        ttnn_typecast_180,
        ttnn_typecast_181,
        ttnn_typecast_182,
        ttnn_typecast_183,
        ttnn_typecast_184,
        ttnn_typecast_185,
        ttnn_typecast_186,
        ttnn_typecast_187,
        ttnn_typecast_188,
        ttnn_typecast_189,
        ttnn_typecast_190,
        ttnn_typecast_191,
        ttnn_typecast_192,
        ttnn_typecast_193,
        ttnn_typecast_194,
        ttnn_typecast_195,
        ttnn_typecast_196,
        ttnn_typecast_197,
        ttnn_typecast_198,
        ttnn_typecast_199,
        ttnn_typecast_200,
        ttnn_typecast_201,
        ttnn_typecast_202,
        ttnn_typecast_203,
        ttnn_typecast_204,
        ttnn_typecast_205,
        ttnn_typecast_206,
        ttnn_typecast_207,
        ttnn_typecast_208,
        ttnn_typecast_209,
        ttnn_typecast_210,
        ttnn_typecast_211,
        ttnn_typecast_212,
        ttnn_typecast_213,
        ttnn_typecast_214,
        ttnn_typecast_215,
        ttnn_typecast_216,
        ttnn_typecast_217,
        ttnn_typecast_218,
        ttnn_typecast_219,
        ttnn_typecast_220,
        ttnn_typecast_221,
        ttnn_typecast_222,
        ttnn_typecast_223,
        ttnn_typecast_224,
        ttnn_typecast_225,
        ttnn_typecast_226,
        ttnn_typecast_227,
        ttnn_typecast_228,
        ttnn_typecast_229,
        ttnn_typecast_230,
        ttnn_typecast_231,
        ttnn_typecast_232,
        ttnn_typecast_233,
        ttnn_typecast_234,
        ttnn_typecast_235,
        ttnn_typecast_236,
        ttnn_typecast_237,
        ttnn_typecast_238,
        ttnn_typecast_239,
        ttnn_typecast_240,
        ttnn_typecast_241,
        ttnn_typecast_242,
        ttnn_typecast_243,
        ttnn_typecast_244,
        ttnn_typecast_245,
        ttnn_typecast_246,
        ttnn_typecast_247,
        ttnn_typecast_248,
        ttnn_typecast_249,
        ttnn_typecast_250,
        ttnn_typecast_251,
        ttnn_typecast_252,
        ttnn_typecast_253,
        ttnn_typecast_254,
        ttnn_typecast_255,
        ttnn_typecast_256,
        ttnn_typecast_257,
        ttnn_typecast_258,
        ttnn_typecast_259,
        ttnn_typecast_260,
        ttnn_typecast_261,
        ttnn_typecast_262,
        ttnn_typecast_263,
        ttnn_typecast_264,
        ttnn_typecast_265,
        ttnn_typecast_266,
    )
    ttnn.deallocate(ttnn_typecast_266, False)
    ttnn.deallocate(ttnn_typecast_265, False)
    ttnn.deallocate(ttnn_typecast_264, False)
    ttnn.deallocate(ttnn_typecast_263, False)
    ttnn.deallocate(ttnn_typecast_262, False)
    ttnn.deallocate(ttnn_typecast_261, False)
    ttnn.deallocate(ttnn_typecast_260, False)
    ttnn.deallocate(ttnn_typecast_259, False)
    ttnn.deallocate(ttnn_typecast_258, False)
    ttnn.deallocate(ttnn_typecast_257, False)
    ttnn.deallocate(ttnn_typecast_256, False)
    ttnn.deallocate(ttnn_typecast_255, False)
    ttnn.deallocate(ttnn_typecast_254, False)
    ttnn.deallocate(ttnn_typecast_253, False)
    ttnn.deallocate(ttnn_typecast_252, False)
    ttnn.deallocate(ttnn_typecast_251, False)
    ttnn.deallocate(ttnn_typecast_250, False)
    ttnn.deallocate(ttnn_typecast_249, False)
    ttnn.deallocate(ttnn_typecast_248, False)
    ttnn.deallocate(ttnn_typecast_247, False)
    ttnn.deallocate(ttnn_typecast_246, False)
    ttnn.deallocate(ttnn_typecast_245, False)
    ttnn.deallocate(ttnn_typecast_244, False)
    ttnn.deallocate(ttnn_typecast_243, False)
    ttnn.deallocate(ttnn_typecast_242, False)
    ttnn.deallocate(ttnn_typecast_241, False)
    ttnn.deallocate(ttnn_typecast_240, False)
    ttnn.deallocate(ttnn_typecast_239, False)
    ttnn.deallocate(ttnn_typecast_238, False)
    ttnn.deallocate(ttnn_typecast_237, False)
    ttnn.deallocate(ttnn_typecast_236, False)
    ttnn.deallocate(ttnn_typecast_235, False)
    ttnn.deallocate(ttnn_typecast_234, False)
    ttnn.deallocate(ttnn_typecast_233, False)
    ttnn.deallocate(ttnn_typecast_232, False)
    ttnn.deallocate(ttnn_typecast_231, False)
    ttnn.deallocate(ttnn_typecast_230, False)
    ttnn.deallocate(ttnn_typecast_229, False)
    ttnn.deallocate(ttnn_typecast_228, False)
    ttnn.deallocate(ttnn_typecast_227, False)
    ttnn.deallocate(ttnn_typecast_226, False)
    ttnn.deallocate(ttnn_typecast_225, False)
    ttnn.deallocate(ttnn_typecast_224, False)
    ttnn.deallocate(ttnn_typecast_223, False)
    ttnn.deallocate(ttnn_typecast_222, False)
    ttnn.deallocate(ttnn_typecast_221, False)
    ttnn.deallocate(ttnn_typecast_220, False)
    ttnn.deallocate(ttnn_typecast_219, False)
    ttnn.deallocate(ttnn_typecast_218, False)
    ttnn.deallocate(ttnn_typecast_217, False)
    ttnn.deallocate(ttnn_typecast_216, False)
    ttnn.deallocate(ttnn_typecast_215, False)
    ttnn.deallocate(ttnn_typecast_214, False)
    ttnn.deallocate(ttnn_typecast_213, False)
    ttnn.deallocate(ttnn_typecast_212, False)
    ttnn.deallocate(ttnn_typecast_211, False)
    ttnn.deallocate(ttnn_typecast_210, False)
    ttnn.deallocate(ttnn_typecast_209, False)
    ttnn.deallocate(ttnn_typecast_208, False)
    ttnn.deallocate(ttnn_typecast_207, False)
    ttnn.deallocate(ttnn_typecast_206, False)
    ttnn.deallocate(ttnn_typecast_205, False)
    ttnn.deallocate(ttnn_typecast_204, False)
    ttnn.deallocate(ttnn_typecast_203, False)
    ttnn.deallocate(ttnn_typecast_202, False)
    ttnn.deallocate(ttnn_typecast_201, False)
    ttnn.deallocate(ttnn_typecast_200, False)
    ttnn.deallocate(ttnn_typecast_199, False)
    ttnn.deallocate(ttnn_typecast_198, False)
    ttnn.deallocate(ttnn_typecast_197, False)
    ttnn.deallocate(ttnn_typecast_196, False)
    ttnn.deallocate(ttnn_typecast_195, False)
    ttnn.deallocate(ttnn_typecast_194, False)
    ttnn.deallocate(ttnn_typecast_193, False)
    ttnn.deallocate(ttnn_typecast_192, False)
    ttnn.deallocate(ttnn_typecast_191, False)
    ttnn.deallocate(ttnn_typecast_190, False)
    ttnn.deallocate(ttnn_typecast_189, False)
    ttnn.deallocate(ttnn_typecast_188, False)
    ttnn.deallocate(ttnn_typecast_187, False)
    ttnn.deallocate(ttnn_typecast_186, False)
    ttnn.deallocate(ttnn_typecast_185, False)
    ttnn.deallocate(ttnn_typecast_184, False)
    ttnn.deallocate(ttnn_typecast_183, False)
    ttnn.deallocate(ttnn_typecast_182, False)
    ttnn.deallocate(ttnn_typecast_181, False)
    ttnn.deallocate(ttnn_typecast_180, False)
    ttnn.deallocate(ttnn_typecast_179, False)
    ttnn.deallocate(ttnn_typecast_178, False)
    ttnn.deallocate(ttnn_typecast_177, False)
    ttnn.deallocate(ttnn_typecast_176, False)
    ttnn.deallocate(ttnn_typecast_175, False)
    ttnn.deallocate(ttnn_typecast_174, False)
    ttnn.deallocate(ttnn_typecast_173, False)
    ttnn.deallocate(ttnn_typecast_172, False)
    ttnn.deallocate(ttnn_typecast_171, False)
    ttnn.deallocate(ttnn_typecast_170, False)
    ttnn.deallocate(ttnn_typecast_169, False)
    ttnn.deallocate(ttnn_typecast_168, False)
    ttnn.deallocate(ttnn_typecast_167, False)
    ttnn.deallocate(ttnn_typecast_166, False)
    ttnn.deallocate(ttnn_typecast_165, False)
    ttnn.deallocate(ttnn_typecast_164, False)
    ttnn.deallocate(ttnn_typecast_163, False)
    ttnn.deallocate(ttnn_typecast_162, False)
    ttnn.deallocate(ttnn_typecast_161, False)
    ttnn.deallocate(ttnn_typecast_160, False)
    ttnn.deallocate(ttnn_typecast_159, False)
    ttnn.deallocate(ttnn_typecast_158, False)
    ttnn.deallocate(ttnn_typecast_157, False)
    ttnn.deallocate(ttnn_typecast_156, False)
    ttnn.deallocate(ttnn_typecast_155, False)
    ttnn.deallocate(ttnn_typecast_154, False)
    ttnn.deallocate(ttnn_typecast_153, False)
    ttnn.deallocate(ttnn_typecast_152, False)
    ttnn.deallocate(ttnn_typecast_151, False)
    ttnn.deallocate(ttnn_typecast_150, False)
    ttnn.deallocate(ttnn_typecast_149, False)
    ttnn.deallocate(ttnn_typecast_148, False)
    ttnn.deallocate(ttnn_typecast_147, False)
    ttnn.deallocate(ttnn_typecast_146, False)
    ttnn.deallocate(ttnn_typecast_145, False)
    ttnn.deallocate(ttnn_typecast_144, False)
    ttnn.deallocate(ttnn_typecast_143, False)
    ttnn.deallocate(ttnn_typecast_142, False)
    ttnn.deallocate(ttnn_typecast_141, False)
    ttnn.deallocate(ttnn_typecast_140, False)
    ttnn.deallocate(ttnn_typecast_139, False)
    ttnn.deallocate(ttnn_typecast_138, False)
    ttnn.deallocate(ttnn_typecast_137, False)
    ttnn.deallocate(ttnn_typecast_136, False)
    ttnn.deallocate(ttnn_typecast_135, False)
    ttnn.deallocate(ttnn_typecast_134, False)
    ttnn.deallocate(ttnn_typecast_133, False)
    ttnn.deallocate(ttnn_typecast_132, False)
    ttnn.deallocate(ttnn_typecast_131, False)
    ttnn.deallocate(ttnn_typecast_130, False)
    ttnn.deallocate(ttnn_typecast_129, False)
    ttnn.deallocate(ttnn_typecast_128, False)
    ttnn.deallocate(ttnn_typecast_127, False)
    ttnn.deallocate(ttnn_typecast_126, False)
    ttnn.deallocate(ttnn_typecast_125, False)
    ttnn.deallocate(ttnn_typecast_124, False)
    ttnn.deallocate(ttnn_typecast_123, False)
    ttnn.deallocate(ttnn_typecast_122, False)
    ttnn.deallocate(ttnn_typecast_121, False)
    ttnn.deallocate(ttnn_typecast_120, False)
    ttnn.deallocate(ttnn_typecast_119, False)
    ttnn.deallocate(ttnn_typecast_118, False)
    ttnn.deallocate(ttnn_typecast_117, False)
    ttnn.deallocate(ttnn_typecast_116, False)
    ttnn.deallocate(ttnn_typecast_115, False)
    ttnn.deallocate(ttnn_typecast_114, False)
    ttnn.deallocate(ttnn_typecast_113, False)
    ttnn.deallocate(ttnn_typecast_112, False)
    ttnn.deallocate(ttnn_typecast_111, False)
    ttnn.deallocate(ttnn_typecast_110, False)
    ttnn.deallocate(ttnn_typecast_109, False)
    ttnn.deallocate(ttnn_typecast_108, False)
    ttnn.deallocate(ttnn_typecast_107, False)
    ttnn.deallocate(ttnn_typecast_106, False)
    ttnn.deallocate(ttnn_typecast_105, False)
    ttnn.deallocate(ttnn_typecast_104, False)
    ttnn.deallocate(ttnn_typecast_103, False)
    ttnn.deallocate(ttnn_typecast_102, False)
    ttnn.deallocate(ttnn_typecast_101, False)
    ttnn.deallocate(ttnn_typecast_100, False)
    ttnn.deallocate(ttnn_typecast_99, False)
    ttnn.deallocate(ttnn_typecast_98, False)
    ttnn.deallocate(ttnn_typecast_97, False)
    ttnn.deallocate(ttnn_typecast_96, False)
    ttnn.deallocate(ttnn_typecast_95, False)
    ttnn.deallocate(ttnn_typecast_94, False)
    ttnn.deallocate(ttnn_typecast_93, False)
    ttnn.deallocate(ttnn_typecast_92, False)
    ttnn.deallocate(ttnn_typecast_91, False)
    ttnn.deallocate(ttnn_typecast_90, False)
    ttnn.deallocate(ttnn_typecast_89, False)
    ttnn.deallocate(ttnn_typecast_88, False)
    ttnn.deallocate(ttnn_typecast_87, False)
    ttnn.deallocate(ttnn_typecast_86, False)
    ttnn.deallocate(ttnn_typecast_85, False)
    ttnn.deallocate(ttnn_typecast_84, False)
    ttnn.deallocate(ttnn_typecast_83, False)
    ttnn.deallocate(ttnn_typecast_82, False)
    ttnn.deallocate(ttnn_typecast_81, False)
    ttnn.deallocate(ttnn_typecast_80, False)
    ttnn.deallocate(ttnn_typecast_79, False)
    ttnn.deallocate(ttnn_typecast_78, False)
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
    ttnn.deallocate(ttnn_typecast_22, False)
    ttnn.deallocate(ttnn_typecast_21, False)
    ttnn.deallocate(ttnn_typecast_20, False)
    ttnn.deallocate(ttnn_typecast_19, False)
    ttnn.deallocate(ttnn_typecast_18, False)
    ttnn.deallocate(ttnn_typecast_17, False)
    ttnn.deallocate(ttnn_typecast_16, False)
    ttnn.deallocate(ttnn_typecast_15, False)
    ttnn.deallocate(ttnn_typecast_14, False)
    ttnn.deallocate(ttnn_typecast_13, False)
    ttnn.deallocate(ttnn_typecast_12, False)
    ttnn.deallocate(ttnn_typecast_11, False)
    ttnn.deallocate(ttnn_typecast_10, False)
    ttnn.deallocate(ttnn_typecast_9, False)
    ttnn.deallocate(ttnn_typecast_8, False)
    ttnn.deallocate(ttnn_typecast_7, False)
    ttnn.deallocate(ttnn_typecast_6, False)
    ttnn.deallocate(ttnn_typecast_5, False)
    ttnn.deallocate(ttnn_typecast_4, False)
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn_typecast_267 = ttnn.typecast(v_0, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_0, False)
    ttnn_prepare_conv_weights_0 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_267,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=3,
        out_channels=64,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_267, False)
    ttnn_typecast_268 = ttnn.typecast(v_1, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_1, False)
    ttnn_prepare_conv_bias_0 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_268,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1, None
        ),
        input_layout=ttnn.Layout.TILE,
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
    )
    ttnn.deallocate(ttnn_typecast_268, False)
    ttnn_typecast_269 = ttnn.typecast(v_2, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_2, False)
    ttnn_prepare_conv_weights_1 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_269,
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
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=1,
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
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_269, False)
    ttnn_typecast_270 = ttnn.typecast(v_3, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_3, False)
    ttnn_prepare_conv_bias_1 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_270,
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
    )
    ttnn.deallocate(ttnn_typecast_270, False)
    ttnn_typecast_271 = ttnn.typecast(v_4, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_4, False)
    ttnn_prepare_conv_weights_2 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_271,
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
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_271, False)
    ttnn_typecast_272 = ttnn.typecast(v_5, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_5, False)
    ttnn_prepare_conv_bias_2 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_272,
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
    )
    ttnn.deallocate(ttnn_typecast_272, False)
    ttnn_typecast_273 = ttnn.typecast(v_6, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_6, False)
    ttnn_prepare_conv_weights_3 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_273,
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
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_273, False)
    ttnn_typecast_274 = ttnn.typecast(v_7, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_7, False)
    ttnn_prepare_conv_bias_3 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_274,
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
    )
    ttnn.deallocate(ttnn_typecast_274, False)
    ttnn_typecast_275 = ttnn.typecast(v_8, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_8, False)
    ttnn_prepare_conv_weights_4 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_275,
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
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=1,
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
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_275, False)
    ttnn_typecast_276 = ttnn.typecast(v_9, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_9, False)
    ttnn_prepare_conv_bias_4 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_276,
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
    )
    ttnn.deallocate(ttnn_typecast_276, False)
    ttnn_typecast_277 = ttnn.typecast(v_10, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_10, False)
    ttnn_prepare_conv_weights_5 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_277,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=64,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_277, False)
    ttnn_typecast_278 = ttnn.typecast(v_11, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_11, False)
    ttnn_prepare_conv_bias_5 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_278,
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
    )
    ttnn.deallocate(ttnn_typecast_278, False)
    ttnn_typecast_279 = ttnn.typecast(v_12, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_12, False)
    ttnn_prepare_conv_weights_6 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_279,
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
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_279, False)
    ttnn_typecast_280 = ttnn.typecast(v_13, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_13, False)
    ttnn_prepare_conv_bias_6 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_280,
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
    )
    ttnn.deallocate(ttnn_typecast_280, False)
    ttnn_typecast_281 = ttnn.typecast(v_14, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_14, False)
    ttnn_prepare_conv_weights_7 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_281,
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
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_281, False)
    ttnn_typecast_282 = ttnn.typecast(v_15, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_15, False)
    ttnn_prepare_conv_bias_7 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_282,
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
    )
    ttnn.deallocate(ttnn_typecast_282, False)
    ttnn_typecast_283 = ttnn.typecast(v_16, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_16, False)
    ttnn_prepare_conv_weights_8 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_283,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=64,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_283, False)
    ttnn_typecast_284 = ttnn.typecast(v_17, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_17, False)
    ttnn_prepare_conv_bias_8 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_284,
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
    )
    ttnn.deallocate(ttnn_typecast_284, False)
    ttnn_typecast_285 = ttnn.typecast(v_18, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_18, False)
    ttnn_prepare_conv_weights_9 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_285,
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
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_285, False)
    ttnn_typecast_286 = ttnn.typecast(v_19, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_19, False)
    ttnn_prepare_conv_bias_9 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_286,
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
    )
    ttnn.deallocate(ttnn_typecast_286, False)
    ttnn_typecast_287 = ttnn.typecast(v_20, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_20, False)
    ttnn_prepare_conv_weights_10 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_287,
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
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_287, False)
    ttnn_typecast_288 = ttnn.typecast(v_21, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_21, False)
    ttnn_prepare_conv_bias_10 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_288,
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
    )
    ttnn.deallocate(ttnn_typecast_288, False)
    ttnn_typecast_289 = ttnn.typecast(v_22, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_22, False)
    ttnn_prepare_conv_weights_11 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_289,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=128,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_289, False)
    ttnn_typecast_290 = ttnn.typecast(v_23, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_23, False)
    ttnn_prepare_conv_bias_11 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_290,
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
    )
    ttnn.deallocate(ttnn_typecast_290, False)
    ttnn_typecast_291 = ttnn.typecast(v_24, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_24, False)
    ttnn_prepare_conv_weights_12 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_291,
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
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_291, False)
    ttnn_typecast_292 = ttnn.typecast(v_25, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_25, False)
    ttnn_prepare_conv_bias_12 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_292,
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
    )
    ttnn.deallocate(ttnn_typecast_292, False)
    ttnn_typecast_293 = ttnn.typecast(v_26, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_26, False)
    ttnn_prepare_conv_weights_13 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_293,
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
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_293, False)
    ttnn_typecast_294 = ttnn.typecast(v_27, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_27, False)
    ttnn_prepare_conv_bias_13 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_294,
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
    )
    ttnn.deallocate(ttnn_typecast_294, False)
    ttnn_typecast_295 = ttnn.typecast(v_28, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_28, False)
    ttnn_prepare_conv_weights_14 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_295,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=512,
        batch_size=1,
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
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_295, False)
    ttnn_typecast_296 = ttnn.typecast(v_29, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_29, False)
    ttnn_prepare_conv_bias_14 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_296,
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
    )
    ttnn.deallocate(ttnn_typecast_296, False)
    ttnn_typecast_297 = ttnn.typecast(v_30, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_30, False)
    ttnn_prepare_conv_weights_15 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_297,
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
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_297, False)
    ttnn_typecast_298 = ttnn.typecast(v_31, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_31, False)
    ttnn_prepare_conv_bias_15 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_298,
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
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
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
    )
    ttnn.deallocate(ttnn_typecast_298, False)
    ttnn_typecast_299 = ttnn.typecast(v_32, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_32, False)
    ttnn_prepare_conv_weights_16 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_299,
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
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_299, False)
    ttnn_typecast_300 = ttnn.typecast(v_33, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_33, False)
    ttnn_prepare_conv_bias_16 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_300,
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
    )
    ttnn.deallocate(ttnn_typecast_300, False)
    ttnn_typecast_301 = ttnn.typecast(v_34, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_34, False)
    ttnn_prepare_conv_weights_17 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_301,
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
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_301, False)
    ttnn_typecast_302 = ttnn.typecast(v_35, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_35, False)
    ttnn_prepare_conv_bias_17 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_302,
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
    )
    ttnn.deallocate(ttnn_typecast_302, False)
    ttnn_typecast_303 = ttnn.typecast(v_36, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_36, False)
    ttnn_prepare_conv_weights_18 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_303,
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
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_303, False)
    ttnn_typecast_304 = ttnn.typecast(v_37, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_37, False)
    ttnn_prepare_conv_bias_18 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_304,
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
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
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
    )
    ttnn.deallocate(ttnn_typecast_304, False)
    ttnn_typecast_305 = ttnn.typecast(v_38, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_38, False)
    ttnn_prepare_conv_weights_19 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_305,
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
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_305, False)
    ttnn_typecast_306 = ttnn.typecast(v_39, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_39, False)
    ttnn_prepare_conv_bias_19 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_306,
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
    )
    ttnn.deallocate(ttnn_typecast_306, False)
    ttnn_typecast_307 = ttnn.typecast(v_40, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_40, False)
    ttnn_prepare_conv_weights_20 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_307,
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
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_307, False)
    ttnn_typecast_308 = ttnn.typecast(v_41, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_41, False)
    ttnn_prepare_conv_bias_20 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_308,
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
    )
    ttnn.deallocate(ttnn_typecast_308, False)
    ttnn_typecast_309 = ttnn.typecast(v_42, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_42, False)
    ttnn_prepare_conv_weights_21 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_309,
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
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_309, False)
    ttnn_typecast_310 = ttnn.typecast(v_43, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_43, False)
    ttnn_prepare_conv_bias_21 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_310,
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
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
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
    )
    ttnn.deallocate(ttnn_typecast_310, False)
    ttnn_typecast_311 = ttnn.typecast(v_44, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_44, False)
    ttnn_prepare_conv_weights_22 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_311,
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
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_311, False)
    ttnn_typecast_312 = ttnn.typecast(v_45, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_45, False)
    ttnn_prepare_conv_bias_22 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_312,
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
    )
    ttnn.deallocate(ttnn_typecast_312, False)
    ttnn_typecast_313 = ttnn.typecast(v_46, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_46, False)
    ttnn_prepare_conv_weights_23 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_313,
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
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_313, False)
    ttnn_typecast_314 = ttnn.typecast(v_47, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_47, False)
    ttnn_prepare_conv_bias_23 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_314,
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
    )
    ttnn.deallocate(ttnn_typecast_314, False)
    ttnn_typecast_315 = ttnn.typecast(v_48, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_48, False)
    ttnn_prepare_conv_weights_24 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_315,
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
        weights_format="OIHW",
        in_channels=512,
        out_channels=256,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_315, False)
    ttnn_typecast_316 = ttnn.typecast(v_49, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_49, False)
    ttnn_prepare_conv_bias_24 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_316,
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
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
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
    )
    ttnn.deallocate(ttnn_typecast_316, False)
    ttnn_typecast_317 = ttnn.typecast(v_50, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_50, False)
    ttnn_prepare_conv_weights_25 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_317,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_317, False)
    ttnn_typecast_318 = ttnn.typecast(v_51, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_51, False)
    ttnn_prepare_conv_bias_25 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_318,
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
    )
    ttnn.deallocate(ttnn_typecast_318, False)
    ttnn_typecast_319 = ttnn.typecast(v_52, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_52, False)
    ttnn_prepare_conv_weights_26 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_319,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_319, False)
    ttnn_typecast_320 = ttnn.typecast(v_53, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_53, False)
    ttnn_prepare_conv_bias_26 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_320,
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
    )
    ttnn.deallocate(ttnn_typecast_320, False)
    ttnn_typecast_321 = ttnn.typecast(v_54, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_54, False)
    ttnn_prepare_conv_weights_27 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_321,
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
        weights_format="OIHW",
        in_channels=512,
        out_channels=1024,
        batch_size=1,
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
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_321, False)
    ttnn_typecast_322 = ttnn.typecast(v_55, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_55, False)
    ttnn_prepare_conv_bias_27 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_322,
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
    )
    ttnn.deallocate(ttnn_typecast_322, False)
    ttnn_typecast_323 = ttnn.typecast(v_56, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_56, False)
    ttnn_prepare_conv_weights_28 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_323,
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
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_323, False)
    ttnn_typecast_324 = ttnn.typecast(v_57, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_57, False)
    ttnn_prepare_conv_bias_28 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_324,
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
    )
    ttnn.deallocate(ttnn_typecast_324, False)
    ttnn_typecast_325 = ttnn.typecast(v_58, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_58, False)
    ttnn_prepare_conv_weights_29 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_325,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_325, False)
    ttnn_typecast_326 = ttnn.typecast(v_59, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_59, False)
    ttnn_prepare_conv_bias_29 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_326,
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
    )
    ttnn.deallocate(ttnn_typecast_326, False)
    ttnn_typecast_327 = ttnn.typecast(v_60, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_60, False)
    ttnn_prepare_conv_weights_30 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_327,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_327, False)
    ttnn_typecast_328 = ttnn.typecast(v_61, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_61, False)
    ttnn_prepare_conv_bias_30 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_328,
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
    )
    ttnn.deallocate(ttnn_typecast_328, False)
    ttnn_typecast_329 = ttnn.typecast(v_62, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_62, False)
    ttnn_prepare_conv_weights_31 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_329,
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
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_329, False)
    ttnn_typecast_330 = ttnn.typecast(v_63, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_63, False)
    ttnn_prepare_conv_bias_31 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_330,
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
    )
    ttnn.deallocate(ttnn_typecast_330, False)
    ttnn_typecast_331 = ttnn.typecast(v_64, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_64, False)
    ttnn_prepare_conv_weights_32 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_331,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_331, False)
    ttnn_typecast_332 = ttnn.typecast(v_65, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_65, False)
    ttnn_prepare_conv_bias_32 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_332,
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
    )
    ttnn.deallocate(ttnn_typecast_332, False)
    ttnn_typecast_333 = ttnn.typecast(v_66, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_66, False)
    ttnn_prepare_conv_weights_33 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_333,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_333, False)
    ttnn_typecast_334 = ttnn.typecast(v_67, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_67, False)
    ttnn_prepare_conv_bias_33 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_334,
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
    )
    ttnn.deallocate(ttnn_typecast_334, False)
    ttnn_typecast_335 = ttnn.typecast(v_68, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_68, False)
    ttnn_prepare_conv_weights_34 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_335,
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
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_335, False)
    ttnn_typecast_336 = ttnn.typecast(v_69, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_69, False)
    ttnn_prepare_conv_bias_34 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_336,
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
    )
    ttnn.deallocate(ttnn_typecast_336, False)
    ttnn_typecast_337 = ttnn.typecast(v_70, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_70, False)
    ttnn_prepare_conv_weights_35 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_337,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_337, False)
    ttnn_typecast_338 = ttnn.typecast(v_71, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_71, False)
    ttnn_prepare_conv_bias_35 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_338,
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
    )
    ttnn.deallocate(ttnn_typecast_338, False)
    ttnn_typecast_339 = ttnn.typecast(v_72, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_72, False)
    ttnn_prepare_conv_weights_36 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_339,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_339, False)
    ttnn_typecast_340 = ttnn.typecast(v_73, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_73, False)
    ttnn_prepare_conv_bias_36 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_340,
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
    )
    ttnn.deallocate(ttnn_typecast_340, False)
    ttnn_typecast_341 = ttnn.typecast(v_74, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_74, False)
    ttnn_prepare_conv_weights_37 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_341,
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
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_341, False)
    ttnn_typecast_342 = ttnn.typecast(v_75, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_75, False)
    ttnn_prepare_conv_bias_37 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_342,
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
    )
    ttnn.deallocate(ttnn_typecast_342, False)
    ttnn_typecast_343 = ttnn.typecast(v_76, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_76, False)
    ttnn_prepare_conv_weights_38 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_343,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_343, False)
    ttnn_typecast_344 = ttnn.typecast(v_77, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_77, False)
    ttnn_prepare_conv_bias_38 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_344,
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
    )
    ttnn.deallocate(ttnn_typecast_344, False)
    ttnn_typecast_345 = ttnn.typecast(v_78, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_78, False)
    ttnn_prepare_conv_weights_39 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_345,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_345, False)
    ttnn_typecast_346 = ttnn.typecast(v_79, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_79, False)
    ttnn_prepare_conv_bias_39 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_346,
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
    )
    ttnn.deallocate(ttnn_typecast_346, False)
    ttnn_typecast_347 = ttnn.typecast(v_80, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_80, False)
    ttnn_prepare_conv_weights_40 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_347,
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
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_347, False)
    ttnn_typecast_348 = ttnn.typecast(v_81, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_81, False)
    ttnn_prepare_conv_bias_40 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_348,
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
    )
    ttnn.deallocate(ttnn_typecast_348, False)
    ttnn_typecast_349 = ttnn.typecast(v_82, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_82, False)
    ttnn_prepare_conv_weights_41 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_349,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_349, False)
    ttnn_typecast_350 = ttnn.typecast(v_83, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_83, False)
    ttnn_prepare_conv_bias_41 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_350,
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
    )
    ttnn.deallocate(ttnn_typecast_350, False)
    ttnn_typecast_351 = ttnn.typecast(v_84, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_84, False)
    ttnn_prepare_conv_weights_42 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_351,
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
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_351, False)
    ttnn_typecast_352 = ttnn.typecast(v_85, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_85, False)
    ttnn_prepare_conv_bias_42 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_352,
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
    )
    ttnn.deallocate(ttnn_typecast_352, False)
    ttnn_typecast_353 = ttnn.typecast(v_86, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_86, False)
    ttnn_prepare_conv_weights_43 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_353,
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
        weights_format="OIHW",
        in_channels=1024,
        out_channels=512,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_353, False)
    ttnn_typecast_354 = ttnn.typecast(v_87, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_87, False)
    ttnn_prepare_conv_bias_43 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_354,
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
    )
    ttnn.deallocate(ttnn_typecast_354, False)
    ttnn_typecast_355 = ttnn.typecast(v_88, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_88, False)
    ttnn_prepare_conv_weights_44 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_355,
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
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_355, False)
    ttnn_typecast_356 = ttnn.typecast(v_89, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_89, False)
    ttnn_prepare_conv_bias_44 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_356,
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
    )
    ttnn.deallocate(ttnn_typecast_356, False)
    ttnn_typecast_357 = ttnn.typecast(v_90, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_90, False)
    ttnn_prepare_conv_weights_45 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_357,
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
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_357, False)
    ttnn_typecast_358 = ttnn.typecast(v_91, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_91, False)
    ttnn_prepare_conv_bias_45 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_358,
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
    )
    ttnn.deallocate(ttnn_typecast_358, False)
    ttnn_typecast_359 = ttnn.typecast(v_92, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_92, False)
    ttnn_prepare_conv_weights_46 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_359,
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
        weights_format="OIHW",
        in_channels=1024,
        out_channels=2048,
        batch_size=1,
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
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_359, False)
    ttnn_typecast_360 = ttnn.typecast(v_93, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_93, False)
    ttnn_prepare_conv_bias_46 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_360,
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
    )
    ttnn.deallocate(ttnn_typecast_360, False)
    ttnn_typecast_361 = ttnn.typecast(v_94, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_94, False)
    ttnn_prepare_conv_weights_47 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_361,
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
        weights_format="OIHW",
        in_channels=2048,
        out_channels=512,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_361, False)
    ttnn_typecast_362 = ttnn.typecast(v_95, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_95, False)
    ttnn_prepare_conv_bias_47 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_362,
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
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
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
    )
    ttnn.deallocate(ttnn_typecast_362, False)
    ttnn_typecast_363 = ttnn.typecast(v_96, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_96, False)
    ttnn_prepare_conv_weights_48 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_363,
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
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_363, False)
    ttnn_typecast_364 = ttnn.typecast(v_97, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_97, False)
    ttnn_prepare_conv_bias_48 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_364,
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
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
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
    )
    ttnn.deallocate(ttnn_typecast_364, False)
    ttnn_typecast_365 = ttnn.typecast(v_98, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_98, False)
    ttnn_prepare_conv_weights_49 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_365,
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
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_365, False)
    ttnn_typecast_366 = ttnn.typecast(v_99, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_99, False)
    ttnn_prepare_conv_bias_49 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_366,
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
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(ttnn_typecast_366, False)
    ttnn_typecast_367 = ttnn.typecast(v_100, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_100, False)
    ttnn_prepare_conv_weights_50 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_367,
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
        weights_format="OIHW",
        in_channels=2048,
        out_channels=512,
        batch_size=1,
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
            deallocate_activation=False,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_367, False)
    ttnn_typecast_368 = ttnn.typecast(v_101, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_101, False)
    ttnn_prepare_conv_bias_50 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_368,
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
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
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
    )
    ttnn.deallocate(ttnn_typecast_368, False)
    ttnn_typecast_369 = ttnn.typecast(v_102, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_102, False)
    ttnn_prepare_conv_weights_51 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_369,
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
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_369, False)
    ttnn_typecast_370 = ttnn.typecast(v_103, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_103, False)
    ttnn_prepare_conv_bias_51 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_370,
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
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
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
    )
    ttnn.deallocate(ttnn_typecast_370, False)
    ttnn_typecast_371 = ttnn.typecast(v_104, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_104, False)
    ttnn_prepare_conv_weights_52 = ttnn.prepare_conv_weights(
        weight_tensor=ttnn_typecast_371,
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
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=1,
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
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    ttnn.deallocate(ttnn_typecast_371, False)
    ttnn_typecast_372 = ttnn.typecast(v_105, ttnn.DataType.BFLOAT16, memory_config=None)
    ttnn.deallocate(v_105, False)
    ttnn_prepare_conv_bias_52 = ttnn.prepare_conv_bias(
        bias_tensor=ttnn_typecast_372,
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
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
    )
    ttnn.deallocate(ttnn_typecast_372, False)
    return [
        ttnn_prepare_conv_weights_0,
        ttnn_prepare_conv_bias_0,
        ttnn_prepare_conv_weights_1,
        ttnn_prepare_conv_bias_1,
        ttnn_prepare_conv_weights_2,
        ttnn_prepare_conv_bias_2,
        ttnn_prepare_conv_weights_3,
        ttnn_prepare_conv_bias_3,
        ttnn_prepare_conv_weights_4,
        ttnn_prepare_conv_bias_4,
        ttnn_prepare_conv_weights_5,
        ttnn_prepare_conv_bias_5,
        ttnn_prepare_conv_weights_6,
        ttnn_prepare_conv_bias_6,
        ttnn_prepare_conv_weights_7,
        ttnn_prepare_conv_bias_7,
        ttnn_prepare_conv_weights_8,
        ttnn_prepare_conv_bias_8,
        ttnn_prepare_conv_weights_9,
        ttnn_prepare_conv_bias_9,
        ttnn_prepare_conv_weights_10,
        ttnn_prepare_conv_bias_10,
        ttnn_prepare_conv_weights_11,
        ttnn_prepare_conv_bias_11,
        ttnn_prepare_conv_weights_12,
        ttnn_prepare_conv_bias_12,
        ttnn_prepare_conv_weights_13,
        ttnn_prepare_conv_bias_13,
        ttnn_prepare_conv_weights_14,
        ttnn_prepare_conv_bias_14,
        ttnn_prepare_conv_weights_15,
        ttnn_prepare_conv_bias_15,
        ttnn_prepare_conv_weights_16,
        ttnn_prepare_conv_bias_16,
        ttnn_prepare_conv_weights_17,
        ttnn_prepare_conv_bias_17,
        ttnn_prepare_conv_weights_18,
        ttnn_prepare_conv_bias_18,
        ttnn_prepare_conv_weights_19,
        ttnn_prepare_conv_bias_19,
        ttnn_prepare_conv_weights_20,
        ttnn_prepare_conv_bias_20,
        ttnn_prepare_conv_weights_21,
        ttnn_prepare_conv_bias_21,
        ttnn_prepare_conv_weights_22,
        ttnn_prepare_conv_bias_22,
        ttnn_prepare_conv_weights_23,
        ttnn_prepare_conv_bias_23,
        ttnn_prepare_conv_weights_24,
        ttnn_prepare_conv_bias_24,
        ttnn_prepare_conv_weights_25,
        ttnn_prepare_conv_bias_25,
        ttnn_prepare_conv_weights_26,
        ttnn_prepare_conv_bias_26,
        ttnn_prepare_conv_weights_27,
        ttnn_prepare_conv_bias_27,
        ttnn_prepare_conv_weights_28,
        ttnn_prepare_conv_bias_28,
        ttnn_prepare_conv_weights_29,
        ttnn_prepare_conv_bias_29,
        ttnn_prepare_conv_weights_30,
        ttnn_prepare_conv_bias_30,
        ttnn_prepare_conv_weights_31,
        ttnn_prepare_conv_bias_31,
        ttnn_prepare_conv_weights_32,
        ttnn_prepare_conv_bias_32,
        ttnn_prepare_conv_weights_33,
        ttnn_prepare_conv_bias_33,
        ttnn_prepare_conv_weights_34,
        ttnn_prepare_conv_bias_34,
        ttnn_prepare_conv_weights_35,
        ttnn_prepare_conv_bias_35,
        ttnn_prepare_conv_weights_36,
        ttnn_prepare_conv_bias_36,
        ttnn_prepare_conv_weights_37,
        ttnn_prepare_conv_bias_37,
        ttnn_prepare_conv_weights_38,
        ttnn_prepare_conv_bias_38,
        ttnn_prepare_conv_weights_39,
        ttnn_prepare_conv_bias_39,
        ttnn_prepare_conv_weights_40,
        ttnn_prepare_conv_bias_40,
        ttnn_prepare_conv_weights_41,
        ttnn_prepare_conv_bias_41,
        ttnn_prepare_conv_weights_42,
        ttnn_prepare_conv_bias_42,
        ttnn_prepare_conv_weights_43,
        ttnn_prepare_conv_bias_43,
        ttnn_prepare_conv_weights_44,
        ttnn_prepare_conv_bias_44,
        ttnn_prepare_conv_weights_45,
        ttnn_prepare_conv_bias_45,
        ttnn_prepare_conv_weights_46,
        ttnn_prepare_conv_bias_46,
        ttnn_prepare_conv_weights_47,
        ttnn_prepare_conv_bias_47,
        ttnn_prepare_conv_weights_48,
        ttnn_prepare_conv_bias_48,
        ttnn_prepare_conv_weights_49,
        ttnn_prepare_conv_bias_49,
        ttnn_prepare_conv_weights_50,
        ttnn_prepare_conv_bias_50,
        ttnn_prepare_conv_weights_51,
        ttnn_prepare_conv_bias_51,
        ttnn_prepare_conv_weights_52,
        ttnn_prepare_conv_bias_52,
    ]


def cpu_hoisted_const_eval_382fc5dd(arg):
    ttnn_to_torch_265 = ttnn.to_torch(arg)
    ttnn_from_torch_106 = ttnn.from_torch(ttnn_to_torch_265)
    return ttnn_from_torch_106


def main_const_eval_1(arg, device):
    ttnn_typecast_373 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_382fc5dd_0 = cpu_hoisted_const_eval_382fc5dd(
        ttnn_typecast_373
    )
    ttnn.deallocate(ttnn_typecast_373, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        cpu_hoisted_const_eval_382fc5dd_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_382fc5dd_0, False)
    ttnn_to_device_0 = ttnn.to_device(
        ttnn_to_layout_1,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    return [ttnn_to_device_0]


def cpu_hoisted_const_eval_27d66922(arg):
    ttnn_to_torch_266 = ttnn.to_torch(arg)
    ttir_cpu_permute_159 = ttir_cpu.permute(ttnn_to_torch_266, [1, 0])
    ttnn_from_torch_107 = ttnn.from_torch(ttir_cpu_permute_159)
    return ttnn_from_torch_107


def main_const_eval_2(arg, device):
    ttnn_typecast_374 = ttnn.typecast(arg[0], ttnn.DataType.FLOAT32, memory_config=None)
    cpu_hoisted_const_eval_27d66922_0 = cpu_hoisted_const_eval_27d66922(
        ttnn_typecast_374
    )
    ttnn.deallocate(ttnn_typecast_374, False)
    ttnn_to_layout_2 = ttnn.to_layout(
        cpu_hoisted_const_eval_27d66922_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.deallocate(cpu_hoisted_const_eval_27d66922_0, False)
    ttnn_to_device_1 = ttnn.to_device(
        ttnn_to_layout_2,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    return [ttnn_to_device_1]


def consteval__main(ce_cache, weights, device):
    if not ce_cache:
        main_const_eval_0_0 = main_const_eval_0(
            [
                weights[
                    "resnet.encoder.stages.3.layers.0.shortcut.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.3.layers.0.shortcut.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.3.layers.0.shortcut.normalization.bias"],
                weights[
                    "resnet.encoder.stages.3.layers.0.shortcut.normalization.weight"
                ],
                weights["resnet.encoder.stages.3.layers.0.shortcut.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.0.shortcut.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.0.shortcut.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.0.shortcut.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.0.shortcut.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.0.shortcut.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.0.shortcut.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.0.shortcut.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.0.shortcut.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.0.shortcut.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.0.shortcut.convolution.weight"],
                weights[
                    "resnet.encoder.stages.0.layers.0.shortcut.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.0.layers.0.shortcut.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.0.layers.0.shortcut.normalization.bias"],
                weights[
                    "resnet.encoder.stages.0.layers.0.shortcut.normalization.weight"
                ],
                weights["resnet.encoder.stages.0.layers.0.shortcut.convolution.weight"],
                weights["resnet.embedder.embedder.normalization.running_var"],
                weights["resnet.embedder.embedder.normalization.running_mean"],
                weights["resnet.embedder.embedder.normalization.bias"],
                weights["resnet.embedder.embedder.normalization.weight"],
                weights["resnet.embedder.embedder.convolution.weight"],
                weights[
                    "resnet.encoder.stages.0.layers.0.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.0.layers.0.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.0.layers.0.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.0.layers.0.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.0.layers.0.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.0.layers.0.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.0.layers.0.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.0.layers.0.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.0.layers.0.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.0.layers.0.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.0.layers.0.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.0.layers.0.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.0.layers.0.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.0.layers.0.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.0.layers.0.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.0.layers.1.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.0.layers.1.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.0.layers.1.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.0.layers.1.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.0.layers.1.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.0.layers.1.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.0.layers.1.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.0.layers.1.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.0.layers.1.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.0.layers.1.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.0.layers.1.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.0.layers.1.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.0.layers.1.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.0.layers.1.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.0.layers.1.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.0.layers.2.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.0.layers.2.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.0.layers.2.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.0.layers.2.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.0.layers.2.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.0.layers.2.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.0.layers.2.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.0.layers.2.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.0.layers.2.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.0.layers.2.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.0.layers.2.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.0.layers.2.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.0.layers.2.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.0.layers.2.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.0.layers.2.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.0.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.0.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.0.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.0.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.0.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.0.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.0.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.0.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.0.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.0.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.0.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.0.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.0.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.0.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.0.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.1.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.1.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.1.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.1.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.1.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.1.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.1.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.1.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.1.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.1.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.1.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.1.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.1.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.1.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.1.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.2.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.2.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.2.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.2.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.2.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.2.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.2.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.2.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.2.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.2.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.2.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.2.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.2.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.2.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.2.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.3.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.3.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.3.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.3.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.3.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.3.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.3.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.3.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.3.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.3.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.1.layers.3.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.1.layers.3.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.1.layers.3.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.1.layers.3.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.1.layers.3.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.0.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.0.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.0.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.0.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.0.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.0.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.0.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.0.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.0.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.0.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.0.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.0.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.0.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.0.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.0.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.1.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.1.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.1.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.1.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.1.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.1.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.1.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.1.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.1.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.1.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.1.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.1.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.1.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.1.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.1.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.2.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.2.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.2.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.2.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.2.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.2.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.2.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.2.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.2.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.2.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.2.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.2.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.2.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.2.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.2.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.3.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.3.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.3.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.3.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.3.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.3.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.3.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.3.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.3.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.3.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.3.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.3.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.3.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.3.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.3.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.4.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.4.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.4.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.4.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.4.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.4.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.4.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.4.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.4.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.4.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.4.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.4.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.4.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.4.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.4.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.5.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.5.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.5.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.5.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.5.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.5.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.5.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.5.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.5.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.5.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.2.layers.5.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.2.layers.5.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.2.layers.5.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.2.layers.5.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.2.layers.5.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.3.layers.0.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.3.layers.0.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.3.layers.0.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.3.layers.0.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.3.layers.0.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.3.layers.0.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.3.layers.0.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.3.layers.0.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.3.layers.0.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.3.layers.0.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.3.layers.0.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.3.layers.0.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.3.layers.0.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.3.layers.0.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.3.layers.0.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.3.layers.1.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.3.layers.1.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.3.layers.1.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.3.layers.1.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.3.layers.1.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.3.layers.1.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.3.layers.1.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.3.layers.1.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.3.layers.1.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.3.layers.1.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.3.layers.1.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.3.layers.1.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.3.layers.1.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.3.layers.1.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.3.layers.1.layer.0.convolution.weight"],
                weights[
                    "resnet.encoder.stages.3.layers.2.layer.2.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.3.layers.2.layer.2.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.3.layers.2.layer.2.normalization.bias"],
                weights[
                    "resnet.encoder.stages.3.layers.2.layer.2.normalization.weight"
                ],
                weights["resnet.encoder.stages.3.layers.2.layer.2.convolution.weight"],
                weights[
                    "resnet.encoder.stages.3.layers.2.layer.1.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.3.layers.2.layer.1.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.3.layers.2.layer.1.normalization.bias"],
                weights[
                    "resnet.encoder.stages.3.layers.2.layer.1.normalization.weight"
                ],
                weights["resnet.encoder.stages.3.layers.2.layer.1.convolution.weight"],
                weights[
                    "resnet.encoder.stages.3.layers.2.layer.0.normalization.running_var"
                ],
                weights[
                    "resnet.encoder.stages.3.layers.2.layer.0.normalization.running_mean"
                ],
                weights["resnet.encoder.stages.3.layers.2.layer.0.normalization.bias"],
                weights[
                    "resnet.encoder.stages.3.layers.2.layer.0.normalization.weight"
                ],
                weights["resnet.encoder.stages.3.layers.2.layer.0.convolution.weight"],
            ],
            device,
        )
        ce_cache["main_const_eval_0"] = [
            main_const_eval_0_0[0],
            main_const_eval_0_0[1],
            main_const_eval_0_0[2],
            main_const_eval_0_0[3],
            main_const_eval_0_0[4],
            main_const_eval_0_0[5],
            main_const_eval_0_0[6],
            main_const_eval_0_0[7],
            main_const_eval_0_0[8],
            main_const_eval_0_0[9],
            main_const_eval_0_0[10],
            main_const_eval_0_0[11],
            main_const_eval_0_0[12],
            main_const_eval_0_0[13],
            main_const_eval_0_0[14],
            main_const_eval_0_0[15],
            main_const_eval_0_0[16],
            main_const_eval_0_0[17],
            main_const_eval_0_0[18],
            main_const_eval_0_0[19],
            main_const_eval_0_0[20],
            main_const_eval_0_0[21],
            main_const_eval_0_0[22],
            main_const_eval_0_0[23],
            main_const_eval_0_0[24],
            main_const_eval_0_0[25],
            main_const_eval_0_0[26],
            main_const_eval_0_0[27],
            main_const_eval_0_0[28],
            main_const_eval_0_0[29],
            main_const_eval_0_0[30],
            main_const_eval_0_0[31],
            main_const_eval_0_0[32],
            main_const_eval_0_0[33],
            main_const_eval_0_0[34],
            main_const_eval_0_0[35],
            main_const_eval_0_0[36],
            main_const_eval_0_0[37],
            main_const_eval_0_0[38],
            main_const_eval_0_0[39],
            main_const_eval_0_0[40],
            main_const_eval_0_0[41],
            main_const_eval_0_0[42],
            main_const_eval_0_0[43],
            main_const_eval_0_0[44],
            main_const_eval_0_0[45],
            main_const_eval_0_0[46],
            main_const_eval_0_0[47],
            main_const_eval_0_0[48],
            main_const_eval_0_0[49],
            main_const_eval_0_0[50],
            main_const_eval_0_0[51],
            main_const_eval_0_0[52],
            main_const_eval_0_0[53],
            main_const_eval_0_0[54],
            main_const_eval_0_0[55],
            main_const_eval_0_0[56],
            main_const_eval_0_0[57],
            main_const_eval_0_0[58],
            main_const_eval_0_0[59],
            main_const_eval_0_0[60],
            main_const_eval_0_0[61],
            main_const_eval_0_0[62],
            main_const_eval_0_0[63],
            main_const_eval_0_0[64],
            main_const_eval_0_0[65],
            main_const_eval_0_0[66],
            main_const_eval_0_0[67],
            main_const_eval_0_0[68],
            main_const_eval_0_0[69],
            main_const_eval_0_0[70],
            main_const_eval_0_0[71],
            main_const_eval_0_0[72],
            main_const_eval_0_0[73],
            main_const_eval_0_0[74],
            main_const_eval_0_0[75],
            main_const_eval_0_0[76],
            main_const_eval_0_0[77],
            main_const_eval_0_0[78],
            main_const_eval_0_0[79],
            main_const_eval_0_0[80],
            main_const_eval_0_0[81],
            main_const_eval_0_0[82],
            main_const_eval_0_0[83],
            main_const_eval_0_0[84],
            main_const_eval_0_0[85],
            main_const_eval_0_0[86],
            main_const_eval_0_0[87],
            main_const_eval_0_0[88],
            main_const_eval_0_0[89],
            main_const_eval_0_0[90],
            main_const_eval_0_0[91],
            main_const_eval_0_0[92],
            main_const_eval_0_0[93],
            main_const_eval_0_0[94],
            main_const_eval_0_0[95],
            main_const_eval_0_0[96],
            main_const_eval_0_0[97],
            main_const_eval_0_0[98],
            main_const_eval_0_0[99],
            main_const_eval_0_0[100],
            main_const_eval_0_0[101],
            main_const_eval_0_0[102],
            main_const_eval_0_0[103],
            main_const_eval_0_0[104],
            main_const_eval_0_0[105],
        ]
        main_const_eval_1_0 = main_const_eval_1([weights["classifier.1.bias"]], device)
        ce_cache["main_const_eval_1"] = main_const_eval_1_0[0]
        main_const_eval_2_0 = main_const_eval_2([weights["classifier.1.weight"]], device)
        ce_cache["main_const_eval_2"] = main_const_eval_2_0[0]
    return ce_cache
