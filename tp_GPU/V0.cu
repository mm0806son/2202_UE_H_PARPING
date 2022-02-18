#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>

// Throw out an error if there is a prob
static inline void check(cudaError_t err, const char* context) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << context << ": "
			<< cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

#define CHECK(x) check(x, #x)

// Find the minimum value v from node i to j
__global__ void mykernel(float* r, const float* d, int n) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= n || j >= n) // if nothing left to do (num of threads > num of nodes)
		return;
	float v = HUGE_VALF; // v=infinity
	//from node i to node j, traverse all possible intermediate node, find the shortest path
	for (int k = 0; k < n; ++k) {
		float x = d[n*i + k]; // distance from i to k
		float y = d[n*k + j]; // distance from k to j
		float z = x + y;
		v = min(v, z); // update the minimum distance
	}
	r[n*i + j] = v; // save the shortest path form i to j 
}

static inline int divup(int a, int b) { // The quotient of a divided by b, rounded up
	return (a + b - 1)/b; // (a-1)/b + 1
}

// !? Why we defined this but never used ?
static inline int roundup(int a, int b) { // The smallest number greater than a and is divisible by b
	return divup(a, b) * b;
}

void step(float* r, const float* d, int n) {
	// Allocate memory & copy data to GPU
	float* dGPU = NULL;
	CHECK(cudaMalloc((void**)&dGPU, n * n * sizeof(float)));
	float* rGPU = NULL;
	CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
	CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice)); // copy d to gpu

	// Run kernel
	dim3 dimBlock(16, 16); // define the size of each block
	dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y)); // define the number of blocks
	mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, n);
	CHECK(cudaGetLastError());

	// Copy data back to CPU & release memory
	CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost)); // copy r to cpu
	CHECK(cudaFree(dGPU));
	CHECK(cudaFree(rGPU));
}

int main() {
	int n = 5000; // number of nodes
	int seed = 0; // for random

	std::mt19937 gen(seed);
	std::uniform_real_distribution<> dis(1.0, 8.0);

	// generate the distance of network
	std::vector<float> d(n*n);
	for (int i = 0; i < d.size(); i++)
		d[i] = dis(gen);
	for (int i = 0; i < n; i++)
		d[i*n + i] = 0;

	std::vector<float> r(n*n);
	auto t1 = std::chrono::high_resolution_clock::now(); // to mesure time

	step(r.data(), d.data(), n); // execute

	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration_s = t2 - t1;
	std::cout << "Duration: " << duration_s.count() << std::endl;

	// for (int i = 0; i < n; ++i) {
	//     for (int j = 0; j < n; ++j) {
	//         std::cout << d[i*n + j] << " ";
	//     }
	//     std::cout << "\n";
	// }

	// for (int i = 0; i < n; ++i) {
	//     for (int j = 0; j < n; ++j) {
	//         std::cout << r[i*n + j] << " ";
	//     }
	//     std::cout << "\n";
	// }
}