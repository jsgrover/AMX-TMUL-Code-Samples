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
#include <exception>
#include "test-amxtile.h"




int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <i,b,f> for int8 bf16, fp16. \n",argv[0]);
        return 1;
    }

    char *datatype = argv[1];

    switch (*datatype) {
        case 'i':
            DotMatrixInt8();
            break;
        case 'b':
            DoMatrix_bf16();
            break;
        case 'f':
            DoMatrix_fp16();
            break;
        default:
            printf("Invalid datatype %c\n",*datatype);
            printf("Usage: %s <i,b,f> for int8 bf16, fp16. \n",argv[0]);
            return 1;
    }

    return 0;
}

static void DoMatrix_bf16()
{

  printf("Dot Matrix BF16\n");
}

static void DoMatrix_fp16()
{
  printf("Dot Matrix FP16\n");
}

static void DotMatrixInt8()
{
    __tilecfg tile_data = {0};
    int8_t src1[MAX];
    int8_t src2[MAX];
    int32_t res[MAX/4];
    int32_t res2[MAX/4];
    int32_t res3[MAX/4];
    int rows  = MAX_ROWS;
    int colsb = MAX_COLS;

    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
      exit(-1);

    // Load tile configuration 
    init_tile_config (&tile_data);

    // Init src matrix buffers with data
    init_buffer (src1, 2);
    print_buffer(src1, rows, colsb);

    init_buffer (src2, 2);
    print_buffer(src2, colsb, rows);

    // Init dst matrix buffers with data
    init_buffer32 (res, 0);



    //Tile 0,1,2
    // Load tile rows from memory
    _tile_loadd (1, src1, MAX_COLS);
    _tile_loadd (2, src2, MAX_ROWS);
    _tile_loadd (0, res, MAX_COLS);


    // Compute dot-product of bytes in tiles 
    _tile_dpbssd (0, 1, 2);
    // Store the tile data to memory
    _tile_stored (0, res, STRIDE);
    print_buffer32(res, rows, colsb/4);



    //Tile 3,4,5
    // Init dst matrix buffers with data
    init_buffer32 (res, 0);
    // Load tile rows from memory
    _tile_loadd (4, src1, MAX_COLS);
    _tile_loadd (5, src2, MAX_ROWS);
    _tile_loadd (3, res, MAX_COLS);


    // Compute dot-product of bytes in tiles 
    _tile_dpbssd (3, 4, 5);
    // Store the tile data to memory
    _tile_stored (3, res, STRIDE);
    print_buffer32(res, rows, colsb/4);

    //Tile 5,6,7
    // Init dst matrix buffers with data
    init_buffer32 (res, 0);
    // Load tile rows from memory
    _tile_loadd (6, src1, MAX_COLS);
    _tile_loadd (7, src2, MAX_ROWS);
    _tile_loadd (5, res, MAX_COLS);


    // Compute dot-product of bytes in tiles 
    _tile_dpbssd (5, 6, 7);
    // Store the tile data to memory
    _tile_stored (5, res, STRIDE);
    print_buffer32(res, rows, colsb/4);

    // Release the tile configuration to return to the init state, 
    // which releases all storage it currently holds
    _tile_release();

}
/* Initialize tile config */
static void init_tile_config (__tilecfg *tileinfo)
{
  int i;
  tileinfo->palette_id = 1;
  tileinfo->start_row = 0;


  for (i = 0; i < 8; ++i)
  {
    printf("Load Config Index: %d\n",i);
    tileinfo->colsb[i] = MAX_COLS;
    tileinfo->rows[i] =  MAX_ROWS;
  }
  
  

  _tile_loadconfig (tileinfo);
  
}

/* Initialize int8_t buffer */
static void init_buffer (int8_t *buf, int8_t value)
{
  int rows, colsb, i, j;
  rows  = MAX_ROWS;
  colsb = MAX_COLS;

  for (i = 0; i < rows; i++)
    for (j = 0; j < colsb; j++)
    {
        buf[i * colsb + j] = value;
    }
}

/* Initialize int32_t buffer */
static void init_buffer32 (int32_t *buf, int32_t value)
{
  int rows, colsb, i, j;
  rows  = MAX_ROWS;
  colsb = MAX_COLS;
  int colsb2=colsb/4;

  for (i = 0; i < rows; i++)
    for (j = 0; j < (colsb2); j++)
    {
        buf[i * colsb2 + j] = value;
    }
}

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use()
{
   if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) 
   {
      printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
      return false;
   }
   else
   {
      printf("\n TILE DATA USE SET - OK \n\n");
      return true;
   }

   return true;
}

/* Print int8_t buffer */
static void print_buffer(int8_t* buf, int32_t rows, int32_t colsb) 
{
   for (int i = 0; i < rows; i++) {
     for (int j = 0; j < (colsb); j++)
     {
         printf("%d ", buf[i * colsb + j]);
     }
     printf("\n");
   }
   printf("\n");
}

/* Print int32_t buffer */
static void print_buffer32(int32_t* buf, int32_t rows, int32_t colsb)
{
   for (int i = 0; i < rows; i++) {
     for (int j = 0; j < (colsb); j++)
     {
         printf("%d ", buf[i * colsb + j]);
     }
     printf("\n");
   }
   printf("\n");
}