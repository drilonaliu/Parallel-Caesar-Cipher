#include "KernelParallelCaesar.cuh"


/*
* Kernel method for encrypting message with Caeser encryption.
* Encrypts the char array 'message'.
* Result is written in 'message' array.
*
* @param message - the charr array from the message.
* @param key - any number from 0 to 26.
* @param textLength - length of the message.
*
*/
__global__ void cudaEncryptMessage(char* message, int key, int textLength) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < textLength) {
		char letter = message[i];
		int letterNum = int(letter);

		int beginIntervalIndex = 0;
		bool isUpperCase = (letterNum >= 65 && letterNum <= 90);
		bool isLowerCase = (letterNum >= 97 && letterNum <= 122);

		//Encrypt only lowercase or uppercase characters
		if (isLowerCase || isUpperCase) {
			if (isLowerCase) {
				beginIntervalIndex = 97;
			}
			else {
				beginIntervalIndex = 65;
			}
			//We shift back the index of the character to the interval [0,26]. Example: character A with code 65 shifts back to 0.
			int shiftedBack = letterNum - beginIntervalIndex;
			//Encrypt index with the ceaser cipher.
			int encrypted = (shiftedBack + key) % 26;
			//Shift back the index to the uppercase or lowercase interval.
			letterNum = encrypted + beginIntervalIndex;
		}
		letter = letterNum;

		message[i] = letter;
	}
}


/*
* Kernel method for decrypting message, encrypted with Caeser.
* Decrypts the char array 'encrypted'.
* Result is written in 'encrypted' array.
*
* 
* @param message - the charr array from the message.
* @param key - the key with which the orignal message was encrypted.
* @param textLength - length of the encrypted text.
*/
__global__ void cudaDecryptMessage(char* encrypted, int key, int textLength) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < textLength) {
		char letter = encrypted[i];
		int letterNum = int(letter);

		bool isUpperCase = (letterNum >= 65 && letterNum <= 90);
		bool isLowerCase = (letterNum >= 97 && letterNum <= 122);

		int beginIntervalIndex;
		//Decrypt only the lowercase or uppercase letters
		if (isLowerCase || isUpperCase) {
			if (isLowerCase) {
				beginIntervalIndex = 97;
			}
			else {
				beginIntervalIndex = 65;
			}
			//We shift back the index of the character to the interval [0,26]. Example: character A with index 65 shifts back to 0.
			int shiftedBack = letterNum - beginIntervalIndex;
			int x = shiftedBack - key;
			// x can be negative. Example: encrypted character A with index 65 is sent back to -2 if the key is 2
			while (x < 0) {
				x += 26;
			}
			//Decrypt index with the ceaser cipher. 
			int decrypted = x % 26;
			//Shift back the index to the uppercase or lowercase interval.
			letterNum = decrypted + beginIntervalIndex;
		}

		letter = letterNum;
		encrypted[i] = letter;
	}
}

