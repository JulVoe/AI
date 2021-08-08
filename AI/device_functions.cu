#pragma once
#include "OperationGraph.cpp"

__global__ void fillTensorConstant(void* mem, uint32_t size[5], uint32_t stride[5], void* value, uint32_t typeSize) {

}

template<typename T>
__global__ void fillTensorRandom(void* mem, uint32_t size[5], uint32_t stride[5], double stddev, double mean) {

}

void writeTensorToFile(TensorDesc* tensor, FILE* file) {

}
//Can pass an already existing TensorDesc*. If pointer is nullptr, a new descriptor will be returned
void readTensorFromFile(TensorDesc*& tensor, FILE* file) {

}

void writeConstantToFile(ConstantDesc* constant, FILE* file) {
	TYPE constType = constant->getTypeId();
	fwrite(&constType, sizeof(TYPE), 1, file);
	
	bool useIndirection;
	constant->getUseIndirecion();
	fwrite(&useIndirection, sizeof(bool), 1, file);

	void* mem;
	constant->getValue(&mem);
	fwrite(mem, sizeOfType(constType), 1, file);
}
//Can pass an already existing ConstantDesc*. If pointer is nullptr, a new descriptor will be returned
void readConstantFromFile(ConstantDesc*& constant, FILE* file) {

}