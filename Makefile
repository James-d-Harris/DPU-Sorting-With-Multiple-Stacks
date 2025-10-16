cc=gcc

flags= -O3 -Wall -fopenmp
dpucflags= `dpu-pkg-config --cflags dpu`
dpulibs=   `dpu-pkg-config --libs dpu`
tasklets = 16
dpus = 2546
stack_size = 256

NUM_SETS = 8
TOTAL_RANKS = 40
MAX_ELEMS_PER_DPU = 16384
NUM_THREADS = 8

objs = host_main.o bucketing.o dpu_exec.o

all: main quicksort_dpu

quicksort_dpu: quicksort_dpu.c
	dpu-upmem-dpurte-clang -O2 \
		-DNR_TASKLETS=$(tasklets) \
		-DNR_DPUS=$(dpus) \
		-DSTACK_SIZE_DEFAULT=$(stack_size) \
		-DMAX_ELEMS_PER_DPU=$(MAX_ELEMS_PER_DPU) \
		quicksort_dpu.c -o quicksort_dpu

main: $(objs)
	$(cc) $(flags) $(objs) -lm $(dpulibs) -o main

host_main.o: host_main.c bucketing.h dpu_exec.h
	$(cc) -c host_main.c $(flags) $(dpucflags) \
		-DDPU_EXE="\"./quicksort_dpu\"" \
		-DNUM_SETS=$(NUM_SETS) -DTOTAL_RANKS=$(TOTAL_RANKS) -DMAX_ELEMS_PER_DPU=$(MAX_ELEMS_PER_DPU) \
		-o host_main.o

bucketing.o: bucketing.c bucketing.h
	$(cc) -c bucketing.c $(flags) -DNUM_THREADS=$(NUM_THREADS) -o bucketing.o

dpu_exec.o: dpu_exec.c dpu_exec.h
	$(cc) -c dpu_exec.c $(flags) $(dpucflags) \
		-DNR_TASKLETS=$(tasklets) -DNR_DPUS=$(dpus) -DSTACK_SIZE_DEFAULT=$(stack_size) \
		-DMAX_ELEMS_PER_DPU=$(MAX_ELEMS_PER_DPU) \
		-o dpu_exec.o

clean:
	rm -rf main quicksort_dpu *.o *.out
