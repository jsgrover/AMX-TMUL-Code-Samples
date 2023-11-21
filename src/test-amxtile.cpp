//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include "test-amxtile.h"


bool PRINTIT=true;



int main(int argc, char* argv[]) {
    char *datatype = argv[1];
    char ch = *datatype;
    if (argc != 2) 
    {
        printf("Usage: %s <i,b,f> for int8 bf16, fp16. \n",argv[0]);
        return 1;
    }
    else if (ch != 'i' && ch != 'b' && ch != 'f')
    {
        printf("Invalid datatype %c\n",*datatype);
        printf("Usage: %s <i,b,f> for int8 bf16, fp16. \n",argv[0]);
        return 1;
    }

    if (PRINTIT)
    {
        switch (*datatype) 
        {
            case 'i':
                    DotMatrixInt8();
                break;
            case 'b':
                    DotMatrix_bf16();
                break;
            case 'f':
                DotMatrix_fp16();
                break;

        }
    }
    else
    { 
      for (int x=0;x<10000000;++x)
      {
        #pragma omp parallel
        {
          switch (*datatype) 
          {
              case 'i':
                      DotMatrixInt8();
                  break;
              case 'b':
                      DotMatrix_bf16();
                  break;
              case 'f':
                      DotMatrix_fp16();
                  break;
          }
        }
      }
    }
    return 0;
}

 void DotMatrix_bf16()
{
  if (PRINTIT) printf("Dot Matrix BF16\n");
  __tilecfg tile_data = {0};
  __bfloat16 src1[MAX];
  __bfloat16 src2[MAX];
  _Float32  res[MAX/4];

    int rows  = MAX_ROWS;
    int colsb = MAX_COLS;

    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
       exit(-1);

    // Load tile configuration 
    init_tile_config (&tile_data);

        // Init src matrix buffers with data
    init_buffer_bf16(src1, (__bfloat16)(2.0));

    if (PRINTIT) print_buffer_bf16(src1, rows, colsb);

    init_buffer_bf16(src2, (__bfloat16)(2.0));
    if (PRINTIT) print_buffer_bf16(src2, colsb, rows);

    // Init dst matrix buffers with data
    init_buffer_bf32(res, 0);



    // //Tile 0,1,2
    // // Load tile rows from memory
    _tile_loadd (1, src1, STRIDE64);
    _tile_loadd (2, src2, STRIDE16);
    _tile_loadd (0, res, STRIDE64);

    if (PRINTIT) printf("Compute dot-product of data in tiles.\n");
    // Compute dot-product of data in tiles 
    try
    {
      _tile_dpbf16ps (0, 1, 2);
    }
    catch(const std::exception& e)
    {
      std::cerr << e.what() << '\n';
    }
    
    
    //Store the tile data to memory
    _tile_stored (0, res, STRIDE64);
    if (PRINTIT) print_buffer_fp32(res, rows, colsb/4);



    _tile_release();

    if (PRINTIT) printf("BF16 done\n");
  
}

 void DotMatrix_fp16()
{
  if (PRINTIT) printf("Dot Matrix FP16\n");
    __tilecfg tile_data = {0};
    _Float16 src1[MAX];
    _Float16 src2[MAX];
    _Float32 res[MAX/4];

    int rows  = MAX_ROWS;
    int colsb = MAX_COLS;

    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
       exit(-1);

    // Load tile configuration 
    init_tile_config (&tile_data);

    // Init src matrix buffers with data
    init_buffer_fp16(src1, (_Float16)(2.0));
    if (PRINTIT)print_buffer_fp16(src1, rows, colsb);

    init_buffer_fp16(src2, (_Float16)(2.0));
    if (PRINTIT) print_buffer_fp16(src2, colsb, rows);

    // Init dst matrix buffers with data
    init_buffer_fP32(res, 0);



    // //Tile 0,1,2
    // // Load tile rows from memory
    _tile_loadd (1, src1, STRIDE64);
    _tile_loadd (2, src2, STRIDE16);
    _tile_loadd (0, res, STRIDE64);

    if (PRINTIT) printf("Compute dot-product of data in tiles.\n");
    // Compute dot-product of data in tiles 
    try
    {
      _tile_dpfp16ps (0, 1, 2);
    }
    catch(const std::exception& e)
    {
      std::cerr << e.what() << '\n';
    }
    
    
    //Store the tile data to memory
    _tile_stored (0, res, STRIDE64);
    if (PRINTIT) print_buffer_fp32(res, rows, colsb/4);


    // Release the tile configuration to return to the init state, 
    // which releases all storage it currently holds
    _tile_release();

    if (PRINTIT) printf("Float16 done\n");

}

 void DotMatrixInt8()
{
    __tilecfg tile_data = {0};
    int8_t src1[MAX];
    int8_t src2[MAX];
    int32_t res[MAX/4];

    int colsb=MAX_COLS;
    int rows=MAX_ROWS;


    // Request permission to linux kernel to run AMX 
    if (!set_tiledata_use())
      exit(-1);

    // Load tile configuration 
    init_tile_config (&tile_data);

    // Init src matrix buffers with data
    init_buffer (src1, 2);
    if (PRINTIT) print_buffer(src1, rows, colsb);

    init_buffer (src2, 2);
    if (PRINTIT) print_buffer(src2, colsb, rows);

    // Init dst matrix buffers with data
    init_buffer32 (res, 0);



    //Tile 0,1,2
    // Load tile rows from memory
    _tile_loadd (1, src1, STRIDE64);
    _tile_loadd (2, src2, STRIDE16);
    _tile_loadd (0, res, STRIDE64);


    // Compute dot-product of bytes in tiles 
    _tile_dpbssd (0, 1, 2);

    // Store the tile data to memory
    _tile_stored (0, res, STRIDE64);
    if (PRINTIT) print_buffer32(res, rows, colsb/4);

    // Release the tile configuration to return to the init state, 
    // which releases all storage it currently holds
    _tile_release();
  
    if (PRINTIT) printf("INT8 Done\n");


}
/* Initialize tile config */
 void init_tile_config(__tilecfg *tileinfo)
{

  _tile_release();
  tileinfo->palette_id = 1;
  tileinfo->start_row = 0;

  int cols=MAX_COLS;
  int rows=MAX_ROWS;


  for (int i=0; i < 8; ++i)
  {
    if (PRINTIT) printf("Load Config Index: %d\n",i);
    tileinfo->rows[i] =  rows;
    tileinfo->colsb[i] = cols;
   
    if (PRINTIT) printf("Rows : %d\n", tileinfo->rows[i]);
    if (PRINTIT) printf("Colsb: %d\n", tileinfo->colsb[i]);

  }
  
  if (PRINTIT) printf("Load final config.\n");

  _tile_loadconfig (tileinfo);
  
}

/* Initialize int8_t buffer */
 void init_buffer(int8_t *buf, int8_t value)
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

 void init_buffer_fp16(_Float16 * buf, _Float16 value)
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

 void init_buffer_bf16(__bfloat16 buf[], __bfloat16 value)
{
  for (int i = 0; i < MAX; i++)
  {
        buf[i] = value;
  }
}



/* Initialize int32_t buffer */
 void init_buffer32 (int32_t *buf, int32_t value)
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
/* Initialize int32_t buffer */
 void init_buffer_fP32 (_Float32* buf, _Float32 value)
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


/* Initialize int32_t buffer */
 void init_buffer_bf32 (_Float32 buf [], _Float32 value)
{

  for (int i = 0; i < MAX/4; i++)
  {
       buf[i ] = value;
  }

}

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
 bool set_tiledata_use()
{
   if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) 
   {
      if (PRINTIT) printf("\n Fail to do XFEATURE_XTILEDATA \n\n");
      return false;
   }
   else
   {
    if (PRINTIT) printf("\n TILE DATA USE SET - OK \n\n");
      return true;
   }

   return true;
}

/* Print int8_t buffer */
 void print_buffer(int8_t* buf, int32_t rows, int32_t colsb) 
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


/* Print fp16 buffer */
 void print_buffer_fp16(_Float16* buf, int32_t rows, int32_t colsb) 
{
   for (int i = 0; i < rows; i++) {
     for (int j = 0; j < (colsb); j++)
     {
         printf("%.2f ", (float) buf[i * colsb + j]);
     }
     printf("\n");
   }
   printf("\n");
}

/* Print bf16 buffer */
 void print_buffer_bf16(__bfloat16 buf[], int32_t rows, int32_t colsb) 
{
   for (int i = 0; i < rows; i++) 
   {
     for (int j = 0; j < (colsb); j++)
     {
         float v =buf[i * colsb + j];
         printf("%.0f ", v);
     }
     printf("\n");
   }
   printf("\n");
}



/* Print fp32 buffer */
 void print_buffer_fp32(_Float32* buf, int32_t rows, int32_t colsb) 
{
   for (int i = 0; i < rows; i++) {
     for (int j = 0; j < (colsb); j++)
     {
         printf("%.2f ", (float)(buf[i * colsb + j]));
     }
     printf("\n");
   }
   printf("\n");
}





/* Print int32_t buffer */
 void print_buffer32(int32_t* buf, int32_t rows, int32_t colsb)
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