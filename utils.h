/*
 * @Author: group3
 * @Date: 2021-12-25 17:05:31
 * @LastEditTime: 2021-12-26 17:24:03
 */
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "helper_functions.h"


#include "json.hpp"
#include "layers.cuh"

using json = nlohmann::json;

float* copytogpu(float* src,float* des,int size){
    cudaError_t cudaStatus = cudaMalloc((void **)&des, size * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!%d\n", cudaStatus);
        exit(EXIT_FAILURE);
    }
    cudaStatus = cudaMemcpy(des, src, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!\n");
        exit(EXIT_FAILURE);
    }
    return des;
}
cudaError_t copyfromgpu(float* src,float* des,int size){
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaError_t cudaStatus = cudaMemcpy(des, src, size * sizeof(float) , cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(cudaStatus));
        exit(EXIT_FAILURE);
    }
    return cudaStatus;
}

float* gpumalloc(float* ptr,int size){
    cudaError_t cudaStatus = cudaMalloc((void **)&ptr, size * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!%d\n", cudaStatus);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

int vector_multi(std::vector<int> v){
    int sum = 1;
    for(auto i:v){
        sum *= i;
    }
    return sum;
    
}

void vector_printf(std::vector<int> v){
    printf("[");
    for(auto i:v){
        printf("%d,",(int)i);
    }
    printf("]");
}

