#include <stdio.h>

#define TPB 256
#define BPG 1

__global__ void printing()
{
    int myID = blockIdx.x *blockDim.x + threadIdx.x;
    printf("Hello world! My thread ID is %d", myID);
}

int main()
{
    printing<<<BPG, TPB>>>();
    return 0;
}