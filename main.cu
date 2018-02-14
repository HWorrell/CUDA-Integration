#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <curand.h>
#include <curand_kernel.h>
#include "support.h"
#include "kernel.cu"


int main(int argc, char* argv[])
{
    Timer timer;

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);
	
	//device arrays
	unsigned long long *counters_d;
	//host arrays
	unsigned long long *counters;
	//information
	float a, b;
	int top, trials;
    cudaError_t cuda_ret;	
    dim3 dim_grid, dim_block;

	//number of blocks
	int numblocks = 20;
	//lower bound
	a = 0;
	//upper bound
	b = 1;
	//"top" (value above the maximum of the function on that range
	top = 1;
	//how many times each thread should run
	trials = 100000;

	//need 1 counter per block (host)
	counters = (unsigned long long*)malloc(numblocks * sizeof(unsigned long long));
    if(counters == NULL) FATAL("Unable to allocate host");
	
	stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	
	printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);
	
	//need 1 counter per block (device)
	cuda_ret = cudaMalloc((void**)&counters_d, numblocks * sizeof(unsigned long long));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

	//sync
	cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

	//set all the device data to 0
    cuda_ret = cudaMemset(counters_d, 0, numblocks * sizeof(long long));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    dim_block.x = BLOCK_SIZE; dim_block.y = dim_block.z = 1;
    dim_grid.x = numblocks; dim_grid.y = dim_grid.z = 1;

	//use kernel to randomly test intervals
    integrate<<<dim_grid, dim_block>>>(counters_d, a, b, top, trials);
    cuda_ret = cudaDeviceSynchronize();
	
	
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel: %d" , cuda_ret);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

	//get the data from the device
    cuda_ret = cudaMemcpy(counters, counters_d, numblocks * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));	

	//sum the total "successes" (under the curve)
	
	unsigned long long total = 0;

	printf("Total of: %llu Successes out of %d Trials\n", *counters * numblocks, BLOCK_SIZE * numblocks * trials);
		
	//diabling multi block until I work out the bugs	
	for(int i = 0; i < numblocks; i++){
		
		total += counters[i];
		
	}
	//using doubles here because they will not be used on the device, so we can get additional accuracy without
	//the preformance issues of using doubles on the device
	double areaofbox = (b - a) * top;
	
	printf("Area used: %f \n", areaofbox);
	
	double percentofarea = ((double)*counters * numblocks) / ((double)BLOCK_SIZE * (double)trials * numblocks);
	
	printf("Percent of area used: %f \n", percentofarea);
	
	double integraltotal = areaofbox * percentofarea;
	
	printf("\n The integral total is: %f \n", integraltotal);
	
	free(counters);
	
	cudaFree(counters_d);

}	

