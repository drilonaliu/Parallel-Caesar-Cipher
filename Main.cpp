#include <iostream>
#include "ParallelCaesar.cuh"
#include "TextReader.h"
#include <chrono>

using namespace chrono;
using namespace std;


int main() {

	//Encryption and decryption of a string
	string plainText = "Parallel Programming";
	int key = 23;
	string encrypted = parallelCaesarEncrypt(plainText, key);
	string decrypted = parallelCaesarDecrypt(encrypted, key);

	cout << "\nOriginal Message: " + plainText
		<< "\nEncrypted Message: " + encrypted
		<< "\nDecrypted Messasge: " + decrypted;

	// Encryption and decryption on a text file
	string plainTxt = readTextFile("Texts/PlainText_1.txt"); 
	string encryptedTxt = parallelCaesarEncrypt(plainTxt, key);

	outputTextFile("Texts/Encrypted.txt", encryptedTxt);
	encryptedTxt = readTextFile("Texts/Encrypted.txt");
	string decryptedTxt = parallelCaesarDecrypt(encryptedTxt, key);
	outputTextFile("Texts/Decrypted.txt", decryptedTxt);
}
