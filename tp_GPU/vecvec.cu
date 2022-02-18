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
__global__ void mykernel() {

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

	// allocate memory in gpu

	auto t1 = std::chrono::high_resolution_clock::now();

	// transfer a & b to GPU memory
	// do the magic in GPU
	// get c into CPU memory

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