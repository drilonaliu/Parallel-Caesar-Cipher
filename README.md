
In the encryption/decryption method we encrypt/decrypt all characters that are letters. 
The range for uppercase letters is [65, 90], and for lowercase letters, it is [97, 122].
In the sequential version, we iterate through each character, whereas in the parallel version, the i-th thread corresponds to the i-th character of the text.
For each character, we first check if it is a letter. If it is, we encrypt/decrypt it; if not, we simply copy that character to the encrypted text.

The encryption is performed by shifting the index of the letter backward within the range [0, 26]. 
Then, we encrypt this index, which results in another number within the [0, 26] range. This index is then mapped back to the letter range: [65, 90] if we have encrypted an uppercase letter, and [97, 122] if we have encrypted a lowercase letter.

```
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
```

