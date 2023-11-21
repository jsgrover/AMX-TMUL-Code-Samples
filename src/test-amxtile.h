//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <iostream>
#include <iomanip>
#include "omp.h"
#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64
#define STRIDE64 64
#define STRIDE32 32
#define STRIDE16 16
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

//Define tile config data structure 
typedef struct __tile_config
{
  uint8_t palette_id;         //1  bytes
  uint8_t start_row;          //1  bytes
  uint8_t reserved_0[14];     //14 bytes
  uint16_t colsb[8];          //16 bytes
  uint8_t reserved_1[16];     //16 bytes
  uint8_t rows[8];            //8 bytes
  uint8_t reserved_3[8];      //8 bytes
                              //Total 64 bytes
} __tilecfg;


 void init_tile_config (__tilecfg *);
 void init_buffer (int8_t *, int8_t );
 void init_buffer_fp16( _Float16* , _Float16 );
 void init_buffer_fP32 (_Float32* buf, _Float32 value);
 void init_buffer32 (int32_t *, int32_t );
 bool set_tiledata_use();
 void print_buffer(int8_t* , int32_t , int32_t ); 
 void print_buffer_fp16(_Float16* , int32_t , int32_t ); 
 void print_buffer_fp32(_Float32* , int32_t , int32_t );
 void print_buffer32(int32_t* , int32_t , int32_t );

 void init_buffer_bf16(__bfloat16 [], __bfloat16 );
 void init_buffer_bf32 (_Float32 [], _Float32 );
 void print_buffer_bf16(__bfloat16 [], int32_t , int32_t ); 



 void DotMatrixInt8();
 void DotMatrix_bf16();
 void DotMatrix_fp16();
