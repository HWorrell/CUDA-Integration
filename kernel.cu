#include <stdio.h>
#define BLOCK_SIZE 1024

__device__ float f(float x){
	//these are integrals chosen to be very difficult to solve;
	//I was not able to solve them myself, but I verified the value
	//against several sources online
		//return sqrt(1 - (x * x * x * x));
	
		return pow(x,x);

	
	//these are very simple integrals, chosen to be "simple" and allow
	//for testing of accuracy with a known value.
		//return x;
	
		//return x * x;

}


__global__ void integrate(unsigned long long *counters_d, float a, float b, int top, int trials){

	//implement shared to allow for reduction tree approach
	__shared__ unsigned long long successes_shared[BLOCK_SIZE + 1];	
	
	unsigned long long successes = 0;
	//only need the block ID and the thread id.  Each thread will work independantly of the others, and 
	//so thread id is only important in the context of the block for the reduction tree
	int bidx = blockIdx.x;
	int tidx = threadIdx.x;
		
	//the following is based in part on an example I found in a presentation online from 
	//a tech conference on GPU methods, in addition to the api  
	curandState state;
	/* Each thread gets same seed, a different sequence number, no offset. */
	curand_init ( 0, blockIdx.x + threadIdx.x, 0, &state );

	
	//implement cuda rng
	float domain = b - a;
	float x;
	float y;
	float fx;
	for(int i = 0; i < trials; i++){
	
		//get random number for x
		x = a + (domain * curand_uniform(&state));
		//get random number for y
		y = (float)top * curand_uniform(&state);		
		//check if f(x) <= y
		fx = f(x);
		//if(f(x) <= Y), successes++			
		if(y < fx){
			
			successes++;
			
		}


	}
	//sync
	__syncthreads();

	
	//atomicAdd(&counters_d[bidx], successes);


	//__syncthreads();

	
	//the following is an attempt to use a reduction tree to parallelize the addition of all the 
	//counters in a block.  I was unable to get it functional, I kept getting an error with
	//and ran out of time to debug it.
					//add successes to local memory
	successes_shared[tidx] = successes;
					
					//atomicAdd(counters_d[bidx], (float)successes);
					
					
					//use a basic reduction implementation to add all the shared variables together
					
	
	
	for(int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
	__syncthreads();
		if(threadIdx.x % stride == 0){
			successes_shared[threadIdx.x] += successes_shared[threadIdx.x + stride];
		//write block sum of successes to the global array

			
		
		
		}

	}
	if(threadIdx.x == 0)
		counters_d[bidx] = successes_shared[0];	

}
