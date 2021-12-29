#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "model.hpp"

#define INPUTSHAPE 3 * 244 * 244
#define OUTPUTSHAPE 1000
#define TESTNUM 10
#define ITERNUM 500
float inputArr[TESTNUM][INPUTSHAPE];
float benchOutArr[TESTNUM][OUTPUTSHAPE];

void readInput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < INPUTSHAPE; j++)
            fscanf(fp, "%f", &inputArr[i][j]);
}

void readOutput(char *filename)
{
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    for (int i = 0; i < TESTNUM; i++)
        for (int j = 0; j < OUTPUTSHAPE; j++)
            fscanf(fp, "%f", &benchOutArr[i][j]);
}

void checkOutput(float *out1, float *out2)
{
    float maxDiff = 0;
    for (int i = 0; i < OUTPUTSHAPE; i++)
    {
        maxDiff = (fabs(out1[i] - out2[i]) > maxDiff) ? fabs(out1[i] - out2[i]) : maxDiff;
    }
    if (maxDiff > 1e-5)
    {
        printf("Output dismatch. MaxDiff is %.7f\n", maxDiff);
        exit(-1);
    }
}

// TODO: 读取权重
VGG16* model;
void initModel(){
    model = new VGG16("./paras.json","../model/parasbin/");
    // model->print_message();
};

// TODO: 实现自己的inference
void inference(float *input, float *output){
    model->inference(input,output);
};


int main()
{

    initModel(); // 读取网络权重

    readInput("../../models/vgg16Input.txt");   // 读取输入
    readOutput("../../models/vgg16Output.txt"); // 读取标准输出
    float sumTime = 0;
    float inferOut[1000];
    // printf("inputArr[0]:%f\n",inputArr[0][0]);
    // model->inference(inputArr[0],inferOut);
    // for(int i = 0;i < 1000;i++)
    //     printf("%f ",inferOut[i]);
    for (int i = 0; i < TESTNUM; i++)
    {
        float inferOut[1000];
        for (int j = 0; j < ITERNUM; j++)
        {
            float Onetime;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);

            // 执行Inference
            inference(inputArr[i], inferOut);

            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&Onetime, start, stop);
            // 累加单次推理消耗时间
            sumTime += Onetime;
        }
        checkOutput(benchOutArr[i], inferOut);
    }
    
    printf("Average Time is: %f\n", (sumTime / TESTNUM / ITERNUM));
    delete model;
}