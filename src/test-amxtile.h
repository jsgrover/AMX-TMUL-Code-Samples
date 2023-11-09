//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64
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



static void init_tile_config (__tilecfg *);
static void init_buffer (int8_t *, int8_t );
static void init_buffer32 (int32_t *, int32_t );
static bool set_tiledata_use();
static void print_buffer(int8_t* , int32_t , int32_t ); 
static void print_buffer32(int32_t* , int32_t , int32_t );

static void DotMatrixInt8();
static void DoMatrix_bf16();
static void DoMatrix_fp16();
