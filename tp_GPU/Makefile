all: V0CPP V0 V1 V1_transpose V2 V3 splitting vecvec

V0CPP: V0.cpp
	g++ -o V0CPP V0.cpp

V0: V0.cu
	nvcc -o V0 V0.cu

V1: V1.cu
	nvcc -o V1 V1.cu

V1_transpose: V1_transpose.cu
	nvcc -o V1_transpose V1_transpose.cu

V2: V2.cu
	nvcc -o V2 V2.cu

V3: V3.cu
	nvcc -o V3 V3.cu

splitting: splitting.cu
	nvcc -o splitting splitting.cu

vecvec: vecvec.cu
	nvcc -o vecvec vecvec.cu