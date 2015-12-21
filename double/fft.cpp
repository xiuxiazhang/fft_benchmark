#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mkl.h"
#include "mkl_dfti.h"
#include "common.h"
double sec (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
using namespace std;
void fft1d(double *signal, long int row, Complex *r_signal, int batch){
//	struct _MKL_Complex16 *x;
	DFTI_DESCRIPTOR_HANDLE my_desc_handle;
	MKL_LONG status, l[1];
//	x=(struct _MKL_Complex16*)malloc(sizeof(struct _MKL_Complex16)*row);
	for(long int i=0;i<row*batch;i++)
	{
		r_signal[i].x=signal[i];
		r_signal[i].y=0.0;
	}
	l[0] = row;// l[1] = N_f_Column;
	status = DftiCreateDescriptor( &my_desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 1, l[0]);
	status = DftiCommitDescriptor( my_desc_handle);
//warm up
	status = DftiComputeForward( my_desc_handle, (struct _MKL_Complex16*)r_signal);

	double start=sec();
	for(int i=0; i<COUNT; i++)
		for(int j=0; j<BATCH; j++)
		status = DftiComputeForward( my_desc_handle, (struct _MKL_Complex16*)&r_signal[j*row]);
	double stop=sec();
	float gflops=batch*COUNT*row*5*log(row)/log(2)/(stop-start)*1.e-9;
	cout<<row<<", "<<(stop-start)/COUNT<<" s/fft1d "<<gflops<<" gflops"<<endl;
	status = DftiFreeDescriptor(&my_desc_handle);
}
void fft2d(double *signal, int row, int col, Complex *r_signal, int batch){
//	struct _MKL_Complex16 *x;
	DFTI_DESCRIPTOR_HANDLE my_desc_handle;
	MKL_LONG status, l[2];
//	x=(struct _MKL_Complex16*)malloc(sizeof(struct _MKL_Complex16)*row);
	for(int i=0;i<row*col*batch;i++)
	{
		r_signal[i].x=signal[i];
		r_signal[i].y=0.0;
	}
	l[0] = row; l[1] = col;
	status = DftiCreateDescriptor( &my_desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 2, l);
	status = DftiCommitDescriptor( my_desc_handle);
//warm up
	//status = DftiComputeForward( my_desc_handle, (struct _MKL_Complex16*)r_signal);

	double start=sec();
	for(int i=0; i<COUNT; i++)
		for(int j=0; j<batch;j++)
		status = DftiComputeForward( my_desc_handle, (struct _MKL_Complex16*)&r_signal[j*row*col]);
	double stop=sec();

	float gflops=batch*COUNT*row*col*5*log(row*col)/(stop-start)*1.e-9;
	cout<<"row="<<row<<" col="<<col<<", "<<(stop-start)/COUNT<<" s/fft2d "<<gflops<<" gflops"<<endl;
	status = DftiFreeDescriptor(&my_desc_handle);
}

void fft3d(double *signal, int row, int col, int nz, Complex *r_signal, int batch){
//	struct _MKL_Complex16 *x;
	DFTI_DESCRIPTOR_HANDLE my_desc_handle;
	MKL_LONG status, l[3];
//	x=(struct _MKL_Complex16*)malloc(sizeof(struct _MKL_Complex16)*row);
	for(int i=0;i<row*col*nz*batch;i++)
	{
		r_signal[i].x=signal[i];
		r_signal[i].y=0.0;
	}
	l[0] = row; l[1] = col; l[2]=nz;
	status = DftiCreateDescriptor( &my_desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, 3, l);
	status = DftiCommitDescriptor( my_desc_handle);
//warm up
	//status = DftiComputeForward( my_desc_handle, (struct _MKL_Complex16*)r_signal);

	double start=sec();
	for(int i=0; i<COUNT; i++)
		for(int j=0; j<batch; j++)
		status = DftiComputeForward( my_desc_handle, (struct _MKL_Complex16*)&r_signal[j*row*col*nz]);
	double stop=sec();

	float gflops=batch*COUNT*row*col*nz*5*log(row*col*nz)/(stop-start)*1.e-9;
	cout<<"row="<<row<<" col="<<col<<" nz="<<nz<<", "<<(stop-start)/COUNT<<" s/fft3d "<<gflops<<" gflops"<<endl;
	status = DftiFreeDescriptor(&my_desc_handle);
}

