#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstring>
#include <stdio.h>
#include <iostream>
#include "ParallelCaesar.cuh"
#include "KernelParallelCaesar.cuh"

using namespace std;


/*
* Encrypts message using ceaser cipher in parallel
* 
* @param message - source message.
* @param key - any number from 0 to 26.
* 
* @return encrypted message
*/
string parallelCaesarEncrypt(string message, int key) {

	//Convert message to char array
	char* plainText = new char[message.length() + 1];
	strcpy(plainText, message.c_str());

	//Text Size
	int size = (message.length() + 1) * sizeof(char);

	//Device pointers
	char* d_plainText = nullptr;

	//Memory Allocation
	cudaMalloc((void**)&d_plainText, size);

	//Memory copy
	cudaMemcpy(d_plainText, plainText, size, cudaMemcpyHostToDevice);

	//Launch Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (int)ceil(message.length()*1.0 / threadsPerBlock);
	int totalThreadsLaunched = threadsPerBlock * blocksPerGrid;
	cudaEncryptMessage << <blocksPerGrid, threadsPerBlock >> > (d_plainText, key, message.length());

	//Wait for Cuda
	cudaDeviceSynchronize();

	//Retrieving the results 
	cudaMemcpy(plainText, d_plainText, size, cudaMemcpyDeviceToHost);


	string encrypted = plainText;
	cudaFree(d_plainText);
	delete[] plainText;

	return encrypted;
}

/*
* Decrypts an ecnrypted message with the Caesar cipher.
* 
* @param message - encrypted message.
* @param key - key used to encrypt the original message.
* 
* @return decrypted message.
*/
string parallelCaesarDecrypt(string message, int key){

	//Convert message to char array
	char* encryptedText = new char[message.length() + 1];
	strcpy(encryptedText, message.c_str());
	char* d_encryptedText = nullptr;

	//Text size
	int size = (message.length() + 1) * sizeof(char);

	//Memory Allocation
	cudaMalloc((void**)&d_encryptedText, size);

	//Memory copy
	cudaMemcpy(d_encryptedText, encryptedText, size, cudaMemcpyHostToDevice);

	//Launch Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (int)ceil(message.length() * 1.0 / threadsPerBlock);
	int totalThreadsLaunched = threadsPerBlock * blocksPerGrid;
	cudaDecryptMessage << <blocksPerGrid, threadsPerBlock >> > (d_encryptedText, key,message.length());

	//Wait for cuda
	cudaDeviceSynchronize();

	//Retrieve the results the results.
	cudaMemcpy(encryptedText, d_encryptedText, size, cudaMemcpyDeviceToHost);
	string decrypted = encryptedText;

	//Free memory
	cudaFree(d_encryptedText);
	delete[] encryptedText;

	return decrypted;
}

