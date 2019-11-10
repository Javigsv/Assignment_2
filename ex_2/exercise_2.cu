#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TPB 256
#define ARRAY_SIZE 1000000000
#define MARGIN 1e-6

__global__ void saxpyGPU(float* x, float* y, float a)
{
    int myId = blockIdx.x * blockDim.x + threadIdx.x;
    if(myId < ARRAY_SIZE) 
    {
        y[myId] += a*x[myId];
    }
}

void saxpyCPU(float* x, float* y, float a)
{
    for(int i = 0; i < ARRAY_SIZE; i++) 
    {
        y[i] += x[i]*a;
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
 }

int compare(float* x, float* y)
{
    int res = 1;
    for(int i = 0; i < ARRAY_SIZE && res; i++)
    {
        if(abs(x[i] - y[i]) > MARGIN)
        {
            res = 0;
        }
    }
    return res;
}

void initArray(float* p)
{
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        p[i] = (float) rand() / RAND_MAX;
    }
}

int main()
{
    double iStart, iElapsCPU, iElapsGPU;

    //Initialization of arrays
    float* xpg = NULL;
    float* ypg = NULL;
    cudaMalloc(&ypg, ARRAY_SIZE*sizeof(float));
    cudaMalloc(&xpg, ARRAY_SIZE*sizeof(float));
    float* xpc = (float*) malloc(ARRAY_SIZE*sizeof(float));
    float* ypc = (float*) malloc(ARRAY_SIZE*sizeof(float));
    initArray(xpc);
    initArray(ypc);
    const float a = 2.2;
    float* yForeing = (float*) malloc(ARRAY_SIZE*sizeof(float));

    //Moving data to the device    
    cudaMemcpy(ypg, ypc, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(xpg, xpc, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
    
    //Computing by CPU
    printf("Computing by CPU... ");
    iStart = cpuSecond();
    saxpyCPU(xpc, ypc, a);
    iElapsCPU = cpuSecond() - iStart;
    printf("Done\n");
    
    //Computing by GPU
    printf("Computing by GPU... ");
    iStart = cpuSecond();
    saxpyGPU<<<(ARRAY_SIZE + TPB - 1)/TPB, TPB>>>(xpg, ypg, a);
    cudaDeviceSynchronize();
    iElapsGPU = cpuSecond() - iStart;
    printf("Done\n");

    //Sum up
    printf("Size of the array: %d\n", ARRAY_SIZE);
    printf("CPU time: %2f\nGPU time: %2f\n", iElapsCPU, iElapsGPU);

    cudaMemcpy(yForeing, ypg, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    int comp = compare(yForeing, ypc);

    if (comp)
    {
        printf("Both arrays are equal\n");
    }
    else 
    {
        printf("Differences between arrays\n");
    }
    return 0;
}
