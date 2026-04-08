/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <stdio.h>

//Capability -> Cores
int _ConvertSMVer2Cores_Local(int major, int minor) {
    switch (major) {
        case 2: return (minor == 1) ? 48 : 32;
        case 3: return 192;
        case 5: return 128;
        case 6: return (minor == 0) ? 64 : 128;
        case 7: return 64;
        case 8: return (minor == 0) ? 64 : 128;
        case 9: return 128;
        default: return 128;
    }
}

int main(int argc, char **argv) {
    printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

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

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10,
               runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               deviceProp.major, deviceProp.minor);

        char msg[256];
        snprintf(msg, sizeof(msg),
                 "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                 static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
                 (unsigned long long)deviceProp.totalGlobalMem);
        printf("%s", msg);

        printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores_Local(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores_Local(deviceProp.major, deviceProp.minor) *
                   deviceProp.multiProcessorCount);

        // Recovery attribute with API
        int clockRate, memoryClockRate, asyncEngineCount, execTimeout, coopLaunch, computeMode;
        
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

        printf("  Total amount of constant memory:                %zu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n",
               (asyncEngineCount > 0 ? "Yes" : "No"), asyncEngineCount);
        
        printf("  Run time limit on kernels:                     %s\n",
               execTimeout ? "Yes" : "No");
        
        printf("  Supports Cooperative Kernel Launch:            %s\n",
               coopLaunch ? "Yes" : "No");

        const char *sComputeMode[] = {
            "Default (multiple host threads can use ::cudaSetDevice())",
            "Exclusive (only one host thread can use)",
            "Prohibited (no host thread can use)",
            "Exclusive Process (many threads in one process can use)",
            "Unknown", NULL};
        
        // Protection index out of bound
        if (computeMode < 0 || computeMode > 3) computeMode = 4;
        printf("  Compute Mode:\n     < %s >\n", sComputeMode[computeMode]);
    }

    printf("\nResult = PASS\n");
    return 0;
}