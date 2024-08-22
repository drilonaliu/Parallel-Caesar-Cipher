
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring>
#include <stdio.h>
#include <iostream>

__global__ void cudaEncryptMessage(char* message, int key, int textLength);
__global__ void cudaDecryptMessage(char* encrypted, int key, int textLength);