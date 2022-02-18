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

static inline int divup(int a, int b) { // The quotient of a divided by b, rounded up
	return (a + b - 1)/b; // (a-1)/b + 1
}

#define CHECK(x) check(x, #x)

// MAGIC
__global__ void mykernel(float* c, const float* a, const float* b, int n) {
	int i = blockIdx.x * threadIdx.x;
	if (i > n) // if nothing left to do (num of threads > n)
		return;
	c[i] = a[i] + b[i];
}

void step(float* c, const float* a, const float* b, int n) {
	// allocate memory in gpu
	float* aGPU = NULL;
	CHECK(cudaMalloc((void**)&aGPU, n * sizeof(float)));
	float* bGPU = NULL;
	CHECK(cudaMalloc((void**)&bGPU, n * sizeof(float)));
	float* cGPU = NULL;
	CHECK(cudaMalloc((void**)&cGPU, n * sizeof(float)));

	// transfer a & b to GPU memory
	CHECK(cudaMemcpy(aGPU, a, n * sizeof(float), cudaMemcpyHostToDevice)); 
	CHECK(cudaMemcpy(bGPU, b, n * sizeof(float), cudaMemcpyHostToDevice)); 

	// do the magic in GPU
	const int threads = 32;
	int blocks = divup(n, 32);
	// const int blocks = 1562500;

	// Run kernel
	mykernel<<<blocks, threads>>>(cGPU, aGPU, bGPU, n);
	CHECK(cudaGetLastError());

	// get c into CPU memory
	CHECK(cudaMemcpy(c, cGPU, n * sizeof(float), cudaMemcpyDeviceToHost)); 

	CHECK(cudaFree(aGPU));
	CHECK(cudaFree(bGPU));
	CHECK(cudaFree(cGPU));
}

int main() {
	int n = 50000000;
	int seed = 0;

	std::mt19937 gen(seed);
	std::uniform_real_distribution<> dis(1.0, 8.0);

	std::vector<float> a(n);
	std::vector<float> b(n);
	std::vector<float> c(n);

	for (int i = 0; i < a.size(); i++)
	{
		a[i] = dis(gen);
		b[i] = dis(gen);
	}

	auto t1 = std::chrono::high_resolution_clock::now();

	step(c.data(), a.data(),b.data(), n);

	auto t2 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<float,std::milli> duration_gpu = t2 - t1;
	std::cout << "Duration GPU: " << duration_gpu.count() << std::endl;

	auto t3 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < n; i++)
		c[i] = a[i] + b[i];

	auto t4 = std::chrono::high_resolution_clock::now();

	std::chrono::duration<float,std::milli> duration_cpu = t4 - t3;
	std::cout << "Duration CPU: " << duration_cpu.count() << std::endl;

	// what's the fastest ?
}