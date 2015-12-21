
DFT :=/opt/intel/composer_xe_2015.0.090/mkl/lib/intel64/libmkl_intel_lp64.a -Wl,--start-group /opt/intel/composer_xe_2015.0.090/mkl/lib/intel64/libmkl_intel_thread.a  /opt/intel/composer_xe_2015.0.090/mkl/lib/intel64/libmkl_core.a  -Wl,--end-group -L/opt/intel/composer_xe_2015.0.090/mkl/lib/intel64/ -liomp5 -lpthread

NVCCFLAGES=-O2 #-arch sm_35
LDFLAGES = -L/usr/local/cuda-6.5/lib64 -lcufft -lcudart
NVCCINCLUDE=/usr/local/cuda-6.5/include
MKLINCLUDE=/opt/intel/mkl/include
benchmark: cufft.cu_o fft.o benchmark.o
	g++ -O2 -o $@ $^ $(DFT) $(LDFLAGES)
#benchmark: cufft.cu_o benchmark.o
#	g++ -O2 -o $@ $^ $(LDFLAGES)

cufft.cu_o:cufft.cu
	nvcc -c -o $@ $^ -I$(NVCCINCLUDE) $(NVCCFLAGES)
benchmark.o:benchmark.cpp
	g++ -c -o $@ $^
fft.o:fft.cpp
	g++ -c -o $@ $^  -I$(MKLINCLUDE)

clean:
	rm *.o *.cu_o benchmark
