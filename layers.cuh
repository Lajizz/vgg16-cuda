#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

#define R 244
#define C 244
#define S 244 * 244


/* Declaration */
__global__ void Conv(float*, float*, float*, int, int, int, int, int, int, int);

__global__ void MaxPool(float*, float*, int, int, int, int, int);

__global__ void Gemm1(float*, float*, float*, int, int, int);

__global__ void Gemm2(float*, float*, float*, int, int, int);


/* Implementation */

/* In default, kernel_shape = 3, padding = 1, strides_x = 1, strides_y = 1
 * the blockId is the serial number of kernel
 */ 
__global__ void Conv(float* input, float* output, float* kernel, float* bias, const int depth, const int row, const int col, int kernel_shape = 3, int padding = 1, int strides_x = 1, int strides_y = 1){
    // printf("Convolution Layers\n");
    // basic information
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int blockSize = blockDim.x;
    // 3*3*3 int the first convolution layer;
    extern __shared__ float single_kernel[];
    for(int i = threadId; i < depth * kernel_shape * kernel_shape; i += blockSize){
        single_kernel[i] = kernel[blockId * depth * kernel_shape * kernel_shape + i];
    }
    __syncthreads();
    
    // if(threadId == 0){
    //     printf("single kernel:%f\n",single_kernel[0]);
    // }
    for(int id = threadId; id < row * col; id += blockSize){
        //target position , is also the start place
        int r = id / col;
        int c = id % col;
        float temp = 0;
        //convolution kernel size default 3
        //dot conv
        for(int j = 0; j < depth; j++){
            for(int m = 0; m < kernel_shape; m++){
                for(int n = 0; n < kernel_shape; n++){
                    int posx = r + m - 1;
                    int posy = c + n - 1;
                    if(posx >= 0 && posx < row && posy >= 0 && posy < col)
                        temp += input[j * col * row + posx * col + posy] * single_kernel[j * kernel_shape * kernel_shape + m * kernel_shape + n];
                    // printf(" single_kernel:%f,temp:%f\n", single_kernel[m * 3 + n],temp);
                }
            } 
            
        }
        temp += bias[blockId];
        // Relu
        output[blockId * row * col + id] = MAX(temp,0);

        // for test
        // break;
    }

}

/* In default, padding = 0, strides_x = 2, strides_y = 2
 * the blockId is the serial number of feature maps
 */
__global__ void MaxPool(float* input, float* output, int input_row, int input_col, int row, int col, int padding = 0, int strides_x = 2, int strides_y = 2){
    // printf("MaxPool Layer\n");
    // basic information
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int blockSize = blockDim.x;

    for(int id = threadId;id < row * col;id += blockSize){
        int x = id / col * 2; 
        int y = id % col * 2;
        float val1 = input[blockId * input_row * input_col + x * input_col + y]; 
        float val2 = input[blockId * input_row * input_col + x * input_col + (y + 1)];
        float val3 = input[blockId * input_row * input_col + (x + 1) * input_col + y];
        float val4 = input[blockId * input_row * input_col + (x + 1) * input_col + (y + 1)];
        output[blockId * row * col + id] = MAX(MAX(MAX(val1,val2),val3),val4); 
    }
}

__global__ void Gemm1(float* left, float *right, float *bias, float* output, int X, int Y,int Z)
{
    // printf("Gemm Layer\n");
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int blockSize = blockDim.x;
    int id = blockId * blockSize + threadId;
    if(id < X * Z){
        int x = id / Z;//row 
        int y = id % Z;//col
        float res = 0;
        for(int i = 0;i < Y;i++){
            res += left[x * Y + i] * right[i * Z + y] ;// (x,i) * (i,y)
        }
        res += bias[id];

        output[id]=res;
    }
}
__global__ void Gemm2(float* left, float *right, float *bias, float* output, int X, int Y,int Z)
{
    // printf("Gemm Layer\n");
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int blockSize = blockDim.x;
    int id = blockId * blockSize + threadId;
    if(id < X * Z){
        int x = id / Z;//row 
        int y = id % Z;//col
        float res = 0;
        for(int i = 0;i < Y;i++){
            res += left[x * Y + i] * right[i * Z + y] ;// (x,i) * (i,y)
        }
        res += bias[id];
        
        output[id]=MAX(res,0);
    }
}