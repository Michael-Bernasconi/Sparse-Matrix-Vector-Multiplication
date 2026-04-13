/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <stdio.h>

/**
 * Helper function to convert CUDA Compute Capability version to the number of CUDA Cores.
 * The ratio of cores per Streaming Multiprocessor (SM) changes with each architecture.
 */
int _ConvertSMVer2Cores_Local(int major, int minor) {
    switch (major) {
        case 2: return (minor == 1) ? 48 : 32; // Fermi
        case 3: return 192;                    // Kepler
        case 5: return 128;                    // Maxwell
        case 6: return (minor == 0) ? 64 : 128;// Pascal
        case 7: return 64;                     // Volta/Turing
        case 8: return (minor == 0) ? 64 : 128;// Ampere
        case 9: return 128;                    // Hopper/Ada Lovelace
        default: return 128;
    }
}

int main(int argc, char **argv) {
    printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    // Get the total number of CUDA-capable devices available on the system
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    // Error handling if the CUDA driver or runtime fails to initialize
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
               static_cast<int>(error_id), cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    // Iterate through every detected GPU
    for (dev = 0; dev < deviceCount; ++dev) {
        // Set the current device to perform operations on
        cudaSetDevice(dev);
        
        // Structure to hold hardware properties (name, memory, architecture, etc.)
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Retrieve the installed Driver and Runtime versions
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10,
               runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        
        // Print Compute Capability (e.g., 8.6)
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               deviceProp.major, deviceProp.minor);

        // Calculate and display Global Memory in MBytes
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                 static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                 (unsigned long long)deviceProp.totalGlobalMem);
        printf("%s", msg);

        // Calculate Total CUDA Cores: Cores/MP * Number of Multiprocessors
        printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores_Local(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores_Local(deviceProp.major, deviceProp.minor) *
                   deviceProp.multiProcessorCount);

        // Local variables to store hardware attributes retrieved via API
        int clockRate, memoryClockRate, asyncEngineCount, execTimeout, coopLaunch, computeMode;
        
        // Query specific hardware attributes using the cudaDeviceGetAttribute API
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev);
        cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, dev);
        cudaDeviceGetAttribute(&asyncEngineCount, cudaDevAttrAsyncEngineCount, dev);
        cudaDeviceGetAttribute(&execTimeout, cudaDevAttrKernelExecTimeout, dev);
        cudaDeviceGetAttribute(&coopLaunch, cudaDevAttrCooperativeLaunch, dev);
        cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, dev);

        printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n",
               clockRate * 1e-3f, clockRate * 1e-6f);
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",
               deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n",
                   deviceProp.l2CacheSize);
        }

        // Display memory limits per block and warp sizes
        printf("  Total amount of constant memory:                %zu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        
        // Check if GPU can transfer data and execute kernels at the same time
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n",
               (asyncEngineCount > 0 ? "Yes" : "No"), asyncEngineCount);
        
        // Check if there is a time limit for kernel execution (common on Windows/WDDM)
        printf("  Run time limit on kernels:                     %s\n",
               execTimeout ? "Yes" : "No");
        
        // Check if the GPU supports cooperative groups (grid synchronization)
        printf("  Supports Cooperative Kernel Launch:            %s\n",
               coopLaunch ? "Yes" : "No");

        // Human-readable labels for different GPU Compute Modes
        const char *sComputeMode[] = {
            "Default (multiple host threads can use ::cudaSetDevice())",
            "Exclusive (only one host thread can use)",
            "Prohibited (no host thread can use)",
            "Exclusive Process (many threads in one process can use)",
            "Unknown", NULL};
        
        // Index protection for the Compute Mode array
        if (computeMode < 0 || computeMode > 3) computeMode = 4;
        printf("  Compute Mode:\n     < %s >\n", sComputeMode[computeMode]);
    }

    printf("\nResult = PASS\n");
    return 0;
}