#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/time.h>
#include "common.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
using namespace std;
void cufft1d(double *signal, long int row, Complex *r_signal, int batch){
	long int bytes=batch*row*sizeof(cufftDoubleComplex);
	cufftDoubleComplex* h_signal=new cufftDoubleComplex[row*batch];
	for(long int i=0; i<row*batch; i++){
		h_signal[i].x=signal[i];
		h_signal[i].y=0.;
	}
	cufftDoubleComplex* d_signal, *o_signal;
	cudaMalloc((void**)&d_signal, bytes);
	cudaMalloc((void**)&o_signal, bytes);
	cudaMemcpy(d_signal, h_signal, bytes, cudaMemcpyHostToDevice);

	cufftHandle plan;
	int dim_arry[3]={row, 1, 1};
	//cufftPlan1d(&plan, row, CUFFT_C2C, 1);
	cufftPlanMany(&plan, 1, dim_arry, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, batch);
	//warm up
	cufftExecZ2Z(plan, d_signal, o_signal, CUFFT_FORWARD);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for(int i=0; i<COUNT; i++)
		cufftExecZ2Z(plan, d_signal, d_signal, CUFFT_FORWARD);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime=0.0f;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	elapsedTime *=1.e-3;
	float gflops=COUNT*batch*row*5*log(row)/log(2)/elapsedTime*1.e-9;
	cout<<row<<", "<<elapsedTime/COUNT<<" s/cufft1d "<<gflops<<" gflops"<<endl;
	cudaMemcpy(r_signal,(Complex*)d_signal, bytes, cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	delete[] h_signal;
	cudaFree(d_signal);
}
void cufft2d(double *signal, int row, int col, Complex *r_signal, int batch){
	int bytes=batch*row*col*sizeof(cufftDoubleComplex);
	cufftDoubleComplex* h_signal=new cufftDoubleComplex[row*col*batch];
	for(unsigned int i=0; i<row*col*batch; i++){
		h_signal[i].x=signal[i];
		h_signal[i].y=0.;
	}
	cufftDoubleComplex* d_signal, *o_signal;
	cudaMalloc((void**)&d_signal, bytes);
	cudaMalloc((void**)&o_signal, bytes);
	cudaMemcpy(d_signal, h_signal, bytes, cudaMemcpyHostToDevice);

	cufftHandle plan;
	int dim_arry[3]={row, col, 1};
	cufftPlanMany(&plan, 2, dim_arry, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, batch);
	//cufftPlan2d(&plan, row, col, CUFFT_C2C);
	//warm up
	//cufftExecC2C(plan, d_signal, o_signal, CUFFT_FORWARD);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for(int i=0; i<COUNT; i++)
		cufftExecZ2Z(plan, d_signal, d_signal, CUFFT_FORWARD);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime=0.0f;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	elapsedTime *=1.e-3;
	float gflops = batch*COUNT*5*row*col*log(row*col)/elapsedTime*1.e-9;
	cout<<"row="<<row<<" col="<<col<<", "<<elapsedTime/COUNT<<" s/cufft2d "<<gflops<<" gflops"<<endl;
	cudaMemcpy(r_signal,(Complex*)d_signal, bytes, cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	delete[] h_signal;
	cudaFree(d_signal);
}
void cufft3d(double *signal, int row, int col, int nz, Complex *r_signal, int batch){
	int bytes=batch*row*col*nz*sizeof(cufftDoubleComplex);
	cufftDoubleComplex* h_signal=new cufftDoubleComplex[row*col*nz*batch];
	for(unsigned int i=0; i<row*col*nz*batch; i++){
		h_signal[i].x=signal[i];
		h_signal[i].y=0.;
	}
	cufftDoubleComplex* d_signal, *o_signal;
	cudaMalloc((void**)&d_signal, bytes);
	cudaMalloc((void**)&o_signal, bytes);
	cudaMemcpy(d_signal, h_signal, bytes, cudaMemcpyHostToDevice);

	cufftHandle plan;
	int dim_arry[3]={row, col, nz};
	cufftPlanMany(&plan, 3, dim_arry, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, batch);
	//cufftPlan3d(&plan, row, col, nz, CUFFT_C2C);
	//warm up
	//cufftExecC2C(plan, d_signal, o_signal, CUFFT_FORWARD);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for(int i=0; i<COUNT; i++)
		cufftExecZ2Z(plan, d_signal, d_signal, CUFFT_FORWARD);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float elapsedTime=0.0f;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	elapsedTime *=1.e-3;
	float gflops = batch*COUNT*5*row*col*nz*log(row*col*nz)/elapsedTime*1.e-9;
	cout<<"row="<<row<<" col="<<col<<" nz="<<nz<<", "<<elapsedTime/COUNT<<" s/cufft3d "<<gflops<<" gflops"<<endl;
	cudaMemcpy(r_signal,(Complex*)d_signal, bytes, cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	delete[] h_signal;
	cudaFree(d_signal);
}
