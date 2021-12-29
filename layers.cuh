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
/*
 * @brief convolution kernel function for cuda
 * @param input            input matrix
 * @param output           convolution destination output matrix 
 * @param kernel           convolution kernel matrix
 * @param bias             bias matrix
 * @param channels         the channels of input matrix
 * @param rows             the rows of input matrix
 * @param cols             the cols of input matrix
 * @param kernel_shape     kernel shape of convolution kernel, default kernel_shape = 3
 * @param padding          padding, default padding = 1
 * @param strides_x        strides_x, default strides_x = 1 
 * @param strides_y        strides_y, default strides_y = 1
 */
__global__ void Conv(float* input, float* output, float* kernel, float* bias, const int channels, const int rows, const int cols, int kernel_shape = 3, int padding = 1, int strides_x = 1, int strides_y = 1){
    // printf("Convolution Layers\n");
    // basic information
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    int blockSize = blockDim.x;
    // 3*3*3 int the first convolution layer;
    extern __shared__ float single_kernel[];
    for(int i = threadId; i < channels * kernel_shape * kernel_shape; i += blockSize){
        single_kernel[i] = kernel[blockId * channels * kernel_shape * kernel_shape + i];
    }
    __syncthreads();
    

    for(int id = threadId; id < rows * cols; id += blockSize){
        //target position , is also the central of destination block
        int r = id / cols;
        int c = id % cols;
        float temp = 0;
        //convolution kernel size default 3
        for(int j = 0; j < channels; j++){
            for(int m = 0; m < kernel_shape; m++){
                for(int n = 0; n < kernel_shape; n++){
                    int posx = r + m - 1;
                    int posy = c + n - 1;
                    if(posx >= 0 && posx < rows && posy >= 0 && posy < cols)
                        temp += input[j * rows * cols + posx * cols + posy] * single_kernel[j * kernel_shape * kernel_shape + m * kernel_shape + n];
                }
            } 
            
        }
        temp += bias[blockId];
        // Relu
        output[blockId * rows * cols + id] = MAX(temp,0);
    }

}

/*
 * @brief maxpool kernel function for cuda
 * @param input            input matrix
 * @param output           maxpool destination output matrix 
 * @param input_row        the rows of input matrix
 * @param input_col        the cols of input matrix
 * @param row              the rows of output matrix
 * @param col              the cols of output matrix
 * @param padding          padding, default padding = 0
 * @param strides_x        strides_x, default strides_x = 2 
 * @param strides_y        strides_y, default strides_y = 2
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



/*
 * @brief Gemm kernel funtion without Relu
 * @param left              left matrix
 * @param right             right matrix 
 * @param bias              bias
 * @param output            output matrix
 * @param X                 the first demonsion of the left matrix
 * @param Y                 the second demonsion of the left matrix, the first demonsion of the right matrix
 * @param Z                 the second demonsion of the right matrix
 */
__global__ void Gemm1(float* left, float *right, float *bias, float* output, int X, int Y,int Z)
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int blockSize = blockDim.x;
    int id = blockId * blockSize + threadId;
    if(id < X * Z){
        int x = id / Z;//row 
        int y = id % Z;//col
        float res = 0;
        for(int i = 0;i < Y;i++){
            res += left[x * Y + i] * right[i * Z + y] ;
        }
        res += bias[id];

        output[id]=res;
    }
}

/*
 * @brief Gemm kernel funtion with Relu
 * @param left              left matrix
 * @param right             right matrix 
 * @param bias              bias
 * @param output            output matrix
 * @param X                 the first demonsion of the left matrix
 * @param Y                 the second demonsion of the left matrix, the first demonsion of the right matrix
 * @param Z                 the second demonsion of the right matrix
 */
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