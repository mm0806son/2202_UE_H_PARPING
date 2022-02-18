#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

static inline void check(cudaError_t err, const char* context) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << context << ": "
			<< cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

#define CHECK(x) check(x, #x)

__host__ __device__ inline int p5 (int i)
{
	return i*i*i*i*i;
}

__host__ __device__ inline int value(int x)
{
	int a = 0;
	for (int i = 0; i < 30; ++i)
	{
		if (x & (1 << i))
		{
			a += p5(i+1);
		}
		else
		{
			a -= p5(i+1);
		}
	}
	return abs(a);
}


void display_final_result(int x)
{
	int a = 0;
	int b = 0;
	for (int i = 0; i < 30; ++i)
	{
		if (x & (1 << i))
		{
			a += p5(i+1);
		}
		else
		{
			b += p5(i+1);
		}
	}
	std::cout << "a: " << a << std::endl;
	std::cout << "b: " << b << std::endl;
	std::cout << "x: " << std::hex << x << std::endl;
}


__global__ void mykernel(int* r) {
	// constexpr int iterations = ??;

	// compute best x for given block and thread
}

int main() {
	// compute number of blocks and threads
	constexpr int blocks = 1 << 10;
	constexpr int threads = 1 << 6;

	// create pointer and allocate GPU memory

	// launch kernel

	// allocate memory on CPU side

	// transfer values from GPU to CPU

	// free allocated GPU memory

	int best_x = 0;
	int best_v = value(best_x);

	// final step : determine best x amongs the ones that have been returned by
	// the GPU

	for (int i = 0; i < blocks * threads; ++i)
	{
	}
	display_final_result(best_x);
}
