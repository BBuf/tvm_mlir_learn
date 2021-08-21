#define _GNU_SOURCE
#include "cpufp_kernel_x86.h"
#include "smtl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time(struct timespec *start,
	struct timespec *end)
{
	return end->tv_sec - start->tv_sec +
		(end->tv_nsec - start->tv_nsec) * 1e-9;
}

#ifdef _AVX512F_
#define AVX512F_FP32_COMP (0x20000000L * 320)
#define AVX512F_FP64_COMP (0x20000000L * 160)

static void thread_func_avx512_fp32(void *params)
{
	cpufp_kernel_x86_avx512f_fp32();
}

static void thread_func_avx512_fp64(void *params)
{
	cpufp_kernel_x86_avx512f_fp64();
}

#ifdef _AVX512_VNNI_
#define AVX512_VNNI_INT8_COMP (0x20000000L * 1280)
static void thread_func_avx512_8b(void *params)
{
	cpufp_kernel_x86_avx512_vnni_8b();
}
#endif

void cpufp_x86_avx512(int num_threads)
{
	int i;
	struct timespec start, end;
	double time_used, perf;

	smtl_handle sh;
	smtl_init(&sh, num_threads);

#ifdef _AVX512_VNNI_
	// warm up
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_avx512_8b, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_avx512_8b, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
	perf = AVX512_VNNI_INT8_COMP * num_threads / time_used * 1e-9;
	printf("avx512_vnni int8 perf: %.4lf gops.\n", perf);
#endif

	// warm up
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_avx512_fp32, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_avx512_fp32, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
	perf = AVX512F_FP32_COMP * num_threads / time_used * 1e-9;
	printf("avx512f fp32 perf: %.4lf gflops.\n", perf);

	// warm up
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_avx512_fp64, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_avx512_fp64, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
	perf = AVX512F_FP64_COMP * num_threads / time_used * 1e-9;
	printf("avx512f fp64 perf: %.4lf gflops.\n", perf);

	smtl_fini(sh);
}
#endif

#ifdef _FMA_
#define FMA_FP32_COMP (0x40000000L * 160)
#define FMA_FP64_COMP (0x40000000L * 80)

static void thread_func_fma_fp32(void *params)
{
	cpufp_kernel_x86_fma_fp32();
}

static void thread_func_fma_fp64(void *params)
{
	cpufp_kernel_x86_fma_fp64();
}

void cpufp_x86_fma(int num_threads)
{
	int i;
	struct timespec start, end;
	double time_used, perf;
    
	smtl_handle sh;
	smtl_init(&sh, num_threads);

	// warm up
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_fma_fp32, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_fma_fp32, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
	perf = FMA_FP32_COMP * num_threads / time_used * 1e-9;
	printf("fma fp32 perf: %.4lf gflops.\n", perf);

	// warm up
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_fma_fp64, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_fma_fp64, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
	perf = FMA_FP64_COMP * num_threads / time_used * 1e-9;
	printf("fma fp64 perf: %.4lf gflops.\n", perf);

	smtl_fini(sh);
}
#endif

#ifdef _AVX_
#define AVX_FP32_COMP (0x40000000L * 96)
#define AVX_FP64_COMP (0x40000000L * 48)

static void thread_func_avx_fp32(void *params)
{
	cpufp_kernel_x86_avx_fp32();
}

static void thread_func_avx_fp64(void *params)
{
	cpufp_kernel_x86_avx_fp64();
}

void cpufp_x86_avx(int num_threads)
{
	int i;
	struct timespec start, end;
	double time_used, perf;

    smtl_handle sh;
	smtl_init(&sh, num_threads);

	// warm up
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_avx_fp32, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_avx_fp32, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
	perf = AVX_FP32_COMP * num_threads / time_used * 1e-9;
	printf("avx fp32 perf: %.4lf gflops.\n", perf);

	// warm up
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_avx_fp64, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_avx_fp64, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
	perf = AVX_FP64_COMP * num_threads / time_used * 1e-9;
	printf("avx fp64 perf: %.4lf gflops.\n", perf);

	smtl_fini(sh);
}
#endif

#ifdef _SSE_
#define SSE_FP32_COMP (0x30000000L * 64)
#define SSE_FP64_COMP (0x30000000L * 32)

static void thread_func_sse_fp32(void *params)
{
	cpufp_kernel_x86_sse_fp32();
}

static void thread_func_sse_fp64(void *params)
{
	cpufp_kernel_x86_sse_fp64();
}

void cpufp_x86_sse(int num_threads)
{
	int i;
	struct timespec start, end;
	double time_used, perf;

    smtl_handle sh;
	smtl_init(&sh, num_threads);

	// warm up
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_sse_fp32, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_sse_fp32, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
	perf = SSE_FP32_COMP * num_threads / time_used * 1e-9;
	printf("sse fp32 perf: %.4lf gflops.\n", perf);

	// warm up
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_sse_fp64, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (i = 0; i < num_threads; i++)
	{
		smtl_add_task(sh, thread_func_sse_fp64, NULL);
	}
	smtl_begin_tasks(sh);
	smtl_wait_tasks_finished(sh);
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);

	time_used = get_time(&start, &end);
	perf = SSE_FP64_COMP * num_threads / time_used * 1e-9;
	printf("sse fp64 perf: %.4lf gflops.\n", perf);

	smtl_fini(sh);
}
#endif

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s num_threads.\n", argv[0]);
		exit(0);
	}

	int num_threads = atoi(argv[1]);
	printf("Thread(s): %d\n", num_threads);

#ifdef _AVX512F_
	cpufp_x86_avx512(num_threads);
#endif
#ifdef _FMA_
	cpufp_x86_fma(num_threads);
#endif
#ifdef _AVX_
	cpufp_x86_avx(num_threads);
#endif
#ifdef _SSE_
	cpufp_x86_sse(num_threads);
#endif

	return 0;
}

