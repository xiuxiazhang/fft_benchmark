#include <iostream>
#include <math.h>
#include <stdlib.h>
#include "common.h"
void cufft1d(float *, long int, Complex *, int);
void cufft2d(float *, int, int, Complex *, int);
void cufft3d(float *, int, int, int, Complex *, int);
void fft1d(float *, long int, Complex *, int);
void fft2d(float *, int, int, Complex *, int);
void fft3d(float *, int, int, int, Complex *, int);

using namespace std;
int main(){
	cout<<"Batch="<<BATCH<<endl;
	cout<<"COUNT="<<COUNT<<endl;
	float *signal=(float*)malloc(sizeof(float)*ROW*COL*NZ*BATCH);
	Complex *cufft=(Complex*)malloc(sizeof(Complex)*ROW*COL*NZ*BATCH);
	Complex *fft=(Complex*)malloc(sizeof(Complex)*ROW*COL*NZ*BATCH);

	for(long long int i=0; i<ROW*COL*NZ*BATCH; i++)
		signal[i]=i%100;

	cufft1d(signal, ROW, cufft, BATCH);
	fft1d(signal, ROW, fft, BATCH);

	//cufft2d(signal, ROW, COL, cufft, BATCH);
	//fft2d(signal, ROW, COL, fft, BATCH);

	//cufft3d(signal, ROW, COL, NZ, cufft, BATCH);
	//fft3d(signal, ROW, COL, NZ, fft, BATCH);

	cout<<"cufft:("<<cufft[100].x<<" ,"<<cufft[100].y<<")"<<endl;
	cout<<"fft:("<<fft[100].x<<" ,"<<fft[100].y<<")"<<endl;

	for(long long int i=0; i<ROW*COL*NZ*BATCH; i++)
		if((fabs(cufft[i].x-fft[i].x)>1.e-5)||(fabs(cufft[i].y-fft[i].y)>1.0e-5)){
			cout<<"not pass!\n";
			return -1;
		}
	cout<<"pass!\n";
	return 0;
}
