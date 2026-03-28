#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <cuda_runtime.h>
#include <stdio.h>

/**
 * Defines the necessary variables for the timer.
 * Creates two cudaEvent_t objects (start and stop) and a float to store duration.
 */
#define CUDA_TIMER_DEF(name) \
    cudaEvent_t start_##name, stop_##name; \
    float elapsed_##name = 0.0f;

/**
 * Initializes the CUDA events.
 * Must be called before starting the timer.
 */
#define CUDA_TIMER_INIT(name) \
    cudaEventCreate(&start_##name); \
    cudaEventCreate(&stop_##name);

/**
 * Records the start event.
 * Places a marker in the GPU command stream to capture the start time.
 */
#define CUDA_TIMER_START(name) \
    cudaEventRecord(start_##name);

/**
 * Stops the timer and calculates duration.
 * 1. Records the stop event.
 * 2. Synchronizes the CPU with the stop event (waits for GPU to finish).
 * 3. Computes the time difference between start and stop in milliseconds.
 */
#define CUDA_TIMER_STOP(name) \
    cudaEventRecord(stop_##name); \
    cudaEventSynchronize(stop_##name); \
    cudaEventElapsedTime(&elapsed_##name, start_##name, stop_##name);

/**
 * Returns the elapsed time in milliseconds.
 */
#define CUDA_TIMER_ELAPSED(name) (elapsed_##name)

/**
 * Destroys the CUDA events to free up GPU resources.
 * Should be called when the timer is no longer needed.
 */
#define CUDA_TIMER_CLEAN(name) \
    cudaEventDestroy(start_##name); \
    cudaEventDestroy(stop_##name);

#endif