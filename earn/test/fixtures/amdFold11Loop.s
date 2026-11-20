pearl_amd_fold_wmma11:                  ; @pearl_amd_fold_wmma11
; %bb.0:
	buffer_load_b128 v[133:136], v183, s[16:19], 0 offen
	buffer_load_b128 v[137:140], v183, s[16:19], 0 offen offset:16
	buffer_load_b128 v[141:144], v183, s[16:19], 0 offen offset:1552
	buffer_load_b128 v[153:156], v183, s[16:19], 0 offen offset:1536
	buffer_load_b128 v[145:148], v183, s[16:19], 0 offen offset:1040
	buffer_load_b128 v[157:160], v183, s[16:19], 0 offen offset:1024
	buffer_load_b128 v[149:152], v183, s[16:19], 0 offen offset:528
	buffer_load_b128 v[161:164], v183, s[16:19], 0 offen offset:512
	buffer_load_b128 v[129:132], v181, s[8:11], s2 offen
	s_branch .LBB20_3
.LBB20_2:                               ;   in Loop: Header=BB20_3 Depth=1
	s_add_i32 s2, s33, 1
	s_addk_i32 s20, 0x4000
	s_cmp_lt_u32 s2, s3
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_cbranch_scc0 .LBB20_8
.LBB20_3:                               ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s2, s33
	s_add_i32 s33, s33, 2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_u32 s33, s3
	s_cselect_b32 s22, -1, 0
	s_and_b32 s23, s22, exec_lo
	s_cselect_b32 s23, s33, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b32 s23, s23, 12
	buffer_load_b128 v[165:168], v181, s[8:11], s23 offen
	ds_load_b128 v[192:195], v191 offset:768
	ds_load_b128 v[196:199], v191 offset:1536
	ds_load_b128 v[200:203], v191 offset:2304
	ds_load_b128 v[204:207], v191
	ds_load_b128 v[208:211], v191 offset:16
	s_add_i32 s23, s20, 0xffffe000
	s_waitcnt vmcnt(8) lgkmcnt(4)
	v_wmma_i32_16x16x16_iu8 v[89:96], v[192:195], v[133:136], v[89:96] neg_lo:[1,1,0]
	s_waitcnt vmcnt(4)
	v_wmma_i32_16x16x16_iu8 v[73:80], v[192:195], v[157:160], v[73:80] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[65:72], v[192:195], v[153:156], v[65:72] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu8 v[121:128], v[204:207], v[133:136], v[121:128] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[105:112], v[204:207], v[157:160], v[105:112] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[97:104], v[204:207], v[153:156], v[97:104] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[57:64], v[196:199], v[133:136], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[41:48], v[196:199], v[157:160], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[33:40], v[196:199], v[153:156], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[25:32], v[200:203], v[133:136], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[9:16], v[200:203], v[157:160], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[1:8], v[200:203], v[153:156], v[1:8] neg_lo:[1,1,0]
	ds_load_b128 v[133:136], v191 offset:784
	ds_load_b128 v[153:156], v191 offset:1552
	ds_load_b128 v[157:160], v191 offset:2320
	s_waitcnt vmcnt(2)
	v_wmma_i32_16x16x16_iu8 v[113:120], v[204:207], v[161:164], v[113:120] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[81:88], v[192:195], v[161:164], v[81:88] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[49:56], v[196:199], v[161:164], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[17:24], v[200:203], v[161:164], v[17:24] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(3)
	v_wmma_i32_16x16x16_iu8 v[121:128], v[208:211], v[137:140], v[121:128] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[113:120], v[208:211], v[149:152], v[113:120] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[105:112], v[208:211], v[145:148], v[105:112] neg_lo:[1,1,0]
	s_waitcnt vmcnt(1)
	v_wmma_i32_16x16x16_iu8 v[97:104], v[208:211], v[141:144], v[97:104] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu8 v[89:96], v[133:136], v[137:140], v[89:96] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[81:88], v[133:136], v[149:152], v[81:88] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[73:80], v[133:136], v[145:148], v[73:80] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[65:72], v[133:136], v[141:144], v[65:72] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu8 v[57:64], v[153:156], v[137:140], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[49:56], v[153:156], v[149:152], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[41:48], v[153:156], v[145:148], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[33:40], v[153:156], v[141:144], v[33:40] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu8 v[25:32], v[157:160], v[137:140], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[17:24], v[157:160], v[149:152], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[9:16], v[157:160], v[145:148], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[1:8], v[157:160], v[141:144], v[1:8] neg_lo:[1,1,0]
	s_clause 0x7
	buffer_load_b128 v[133:136], v183, s[16:19], s23 offen
	buffer_load_b128 v[137:140], v190, s[16:19], s23 offen
	buffer_load_b128 v[141:144], v189, s[16:19], s23 offen
	buffer_load_b128 v[145:148], v188, s[16:19], s23 offen
	buffer_load_b128 v[149:152], v187, s[16:19], s23 offen
	buffer_load_b128 v[153:156], v186, s[16:19], s23 offen
	buffer_load_b128 v[157:160], v185, s[16:19], s23 offen
	buffer_load_b128 v[161:164], v184, s[16:19], s23 offen
	; sched_barrier mask(0x00000000)
	s_add_i32 s23, s2, 3
	s_waitcnt vmcnt(9)
	ds_store_b128 v182, v[129:132] offset:6144
	s_cmp_lt_u32 s23, s3
	s_waitcnt lgkmcnt(0)
	s_cselect_b32 s23, s23, s4
	s_barrier
	s_lshl_b32 s23, s23, 12
	buffer_load_b128 v[129:132], v181, s[8:11], s23 offen
	ds_load_b128 v[192:195], v191 offset:6912
	ds_load_b128 v[196:199], v191 offset:7680
	ds_load_b128 v[200:203], v191 offset:8448
	ds_load_b128 v[204:207], v191 offset:6144
	ds_load_b128 v[208:211], v191 offset:6160
	s_bitcmp0_b32 s2, 1
	s_waitcnt vmcnt(8) lgkmcnt(4)
	v_wmma_i32_16x16x16_iu8 v[89:96], v[192:195], v[133:136], v[89:96] neg_lo:[1,1,0]
	s_waitcnt vmcnt(7)
	v_wmma_i32_16x16x16_iu8 v[81:88], v[192:195], v[137:140], v[81:88] neg_lo:[1,1,0]
	s_waitcnt vmcnt(6)
	v_wmma_i32_16x16x16_iu8 v[73:80], v[192:195], v[141:144], v[73:80] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu8 v[121:128], v[204:207], v[133:136], v[121:128] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[113:120], v[204:207], v[137:140], v[113:120] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[105:112], v[204:207], v[141:144], v[105:112] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[57:64], v[196:199], v[133:136], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[49:56], v[196:199], v[137:140], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[41:48], v[196:199], v[141:144], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[25:32], v[200:203], v[133:136], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[17:24], v[200:203], v[137:140], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[9:16], v[200:203], v[141:144], v[9:16] neg_lo:[1,1,0]
	ds_load_b128 v[133:136], v191 offset:6928
	ds_load_b128 v[137:140], v191 offset:7696
	ds_load_b128 v[141:144], v191 offset:8464
	s_waitcnt vmcnt(5)
	v_wmma_i32_16x16x16_iu8 v[97:104], v[204:207], v[145:148], v[97:104] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[65:72], v[192:195], v[145:148], v[65:72] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[33:40], v[196:199], v[145:148], v[33:40] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[1:8], v[200:203], v[145:148], v[1:8] neg_lo:[1,1,0]
	s_waitcnt vmcnt(4) lgkmcnt(3)
	v_wmma_i32_16x16x16_iu8 v[121:128], v[208:211], v[149:152], v[121:128] neg_lo:[1,1,0]
	s_waitcnt vmcnt(3)
	v_wmma_i32_16x16x16_iu8 v[113:120], v[208:211], v[153:156], v[113:120] neg_lo:[1,1,0]
	s_waitcnt vmcnt(2)
	v_wmma_i32_16x16x16_iu8 v[105:112], v[208:211], v[157:160], v[105:112] neg_lo:[1,1,0]
	s_waitcnt vmcnt(1)
	v_wmma_i32_16x16x16_iu8 v[97:104], v[208:211], v[161:164], v[97:104] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(2)
	v_wmma_i32_16x16x16_iu8 v[89:96], v[133:136], v[149:152], v[89:96] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[81:88], v[133:136], v[153:156], v[81:88] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[73:80], v[133:136], v[157:160], v[73:80] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[65:72], v[133:136], v[161:164], v[65:72] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(1)
	v_wmma_i32_16x16x16_iu8 v[57:64], v[137:140], v[149:152], v[57:64] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[49:56], v[137:140], v[153:156], v[49:56] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[41:48], v[137:140], v[157:160], v[41:48] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[33:40], v[137:140], v[161:164], v[33:40] neg_lo:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_wmma_i32_16x16x16_iu8 v[25:32], v[141:144], v[149:152], v[25:32] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[17:24], v[141:144], v[153:156], v[17:24] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[9:16], v[141:144], v[157:160], v[9:16] neg_lo:[1,1,0]
	v_wmma_i32_16x16x16_iu8 v[1:8], v[141:144], v[161:164], v[1:8] neg_lo:[1,1,0]
	s_clause 0x7
	buffer_load_b128 v[133:136], v183, s[16:19], s20 offen
	buffer_load_b128 v[161:164], v190, s[16:19], s20 offen
	buffer_load_b128 v[157:160], v189, s[16:19], s20 offen
	buffer_load_b128 v[153:156], v188, s[16:19], s20 offen
	buffer_load_b128 v[137:140], v187, s[16:19], s20 offen
	buffer_load_b128 v[149:152], v186, s[16:19], s20 offen
	buffer_load_b128 v[145:148], v185, s[16:19], s20 offen
	buffer_load_b128 v[141:144], v184, s[16:19], s20 offen
	s_cbranch_scc1 .LBB20_5
; %bb.4:                                ;   in Loop: Header=BB20_3 Depth=1
	v_xor_b32_e32 v192, v122, v121
	v_xor_b32_e32 v193, v124, v123
	v_xor_b32_e32 v194, v58, v57
	v_xor_b32_e32 v195, v60, v59
.LBB20_5:                               ;   in Loop: Header=BB20_3 Depth=1
	; sched_barrier mask(0x00000000)
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_cbranch_vccnz .LBB20_2
; %bb.6:                                ;   in Loop: Header=BB20_3 Depth=1
	ds_store_b128 v182, v[165:168]
	s_branch .LBB20_2
.LBB20_8:
	s_load_b512 s[16:31], s[0:1], 0x28
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pearl_amd_fold_wmma11
		.amdhsa_next_free_vgpr 217
	.end_amdhsa_kernel
	.text
.Lfunc_end20:
