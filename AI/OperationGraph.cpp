#pragma once
#include <vector>
#include <array>
#include <inttypes.h>
#include <stdexcept>
#include <set>
#include <utility>

#include <cuda.h>

#include "util.cpp"
#include "device_functions.cu"

//TODO: Intermediate types

//============================================================
//==================|Declare new classes|=====================
//============================================================
struct TensorDesc;
struct ConstantDesc;
class  Operation;
class  Operation_Add;
class  Operation_GEMM;
class  Operation_Conv;
class  Operation_Reduce;
class  Operation_Pointwise;
class  Operation_Host;
struct OperationNode;
struct Dependendencies;
class  OperationGraph;

//====================================================
//==================|Descriptors|=====================
//====================================================
/*
	A descriptor holds all the relevant information on a variable like its size and so on. It also holds the dependencies guard for the memory
*/

struct TensorDesc {
private:					          
	uint32_t alignment;               //The alignment requirement of the memory of this tensor in bytes

	std::array<uint32_t, 5> size;     //Sizes of the dimensions in the order NCDHW
	std::array<uint32_t, 5> strides;  //Strides for the dimensions (this shows the true memory layout and padding)
	  
	TYPE typeId;                      //This is the id of the type of the data of this tensor
	  
	bool isNeeded;                    //If this is true, the tensor will be backed by memory (isVirtual will be false). One might set this if the information is needed for back propagation
	bool useIndirection;              //True if indirection pointer should be used. Only relevant when "isNeeded" is true

protected:
	bool isVirtual;                   //Specifies whether tensor is backed by memory. This is not set by user but application with the users request in mind. However, not needed tensors might also be not virtual.
	void** mem;                       //Pointer on host to pointer on device to memory on device of this tensor. Set by application, not user

	Dependencies dep;                 //Dependencies guard for the memory of this tensor. Used in the graph compilation to generate the dependencies for the OperationNodes. Only used and set by application

public:
	//Constructors
	/*
		Default constructor. All information must be set with the setter methods
	*/
	TensorDesc() :
		alignment((uint32_t)(-1)), size(), strides(), typeId((TYPE)(-1)), isNeeded(false), useIndirection(false), isVirtual(true), dep()
	{
		cudaMalloc((void**)&mem, sizeof(void*));
		__setMem(nullptr);
	}
	TensorDesc(TensorDesc* other) :
		alignment(other->alignment), size(other->size), strides(other->strides), typeId(other->typeId), isNeeded(false), useIndirection(false), isVirtual(true), dep()
	{
		cudaMalloc((void**)&mem, sizeof(void*));
		__setMem(nullptr);
	}
	TensorDesc(TensorDesc&  other) = delete;    //No copy constructor as objects are identified by their pointer!
	TensorDesc(TensorDesc&& other) = delete;    //No move constructor as objects are identified by their pointer!

	//Setter methods. They all return the "this" pointer so they are chainable
	inline TensorDesc* setAlignment(uint32_t alignment_) {
		alignment = alignment_;
		return this;
	}
	inline TensorDesc* setStrides(std::array<uint32_t, 5> strides_) {
		strides = strides_;
		return this;
	}
	inline TensorDesc* setSize(std::array<uint32_t, 5> size_) {
		size = size_;
		return this;
	}
	inline TensorDesc* setTypeId(TYPE typeId_) {
		typeId = typeId_;
		return this;
	}
	inline TensorDesc* setIsNeeded(bool isNeeded_) {
		isNeeded = isNeeded_;
		return this;
	}
	inline TensorDesc* setUseIndirection(bool useIndirection_) {
		useIndirection_ = useIndirection_;
		return this;
	}

	//Getter Methods
	inline uint32_t                getAlignment() const {
		return alignment;
	}
	inline std::array<uint32_t, 5> getStrides() const {
		return strides;
	}
	inline std::array<uint32_t, 5> getSize() const {
		return size;
	}
	inline TYPE                    getTypeId() const {
		return typeId;
	}
	inline bool                    getIsNeeded() const {
		return isNeeded;
	}
	inline bool                    getUseIndirection() const {
		return useIndirection; 
	}

	//Methods that should only be called by the application
	inline void  __setIsVirtual(bool isVirtual_) {
		isVirtual = isVirtual_;
	}
	inline void  __setMem(void* mem_) {
		cudaMemcpy((void*)mem, (void*)&mem_, sizeof(void*), cudaMemcpyHostToDevice);
	}
	inline bool  __getIsVirtual() const {
		return isVirtual;
	}
	inline void* __getMem() const {
		void* ret;
		cudaMemcpy((void*)&ret, (void*)mem, sizeof(void*), cudaMemcpyDeviceToHost);

		return ret; 
	}

	template<typename T>
	inline void __fillConstant(T constant) {
		void* constantConv = malloc(sizeOfType(typeId));
		convertType(constant, constantConv, typeId);

		fillTensorConstant<<<1,1>>>(*mem, size.data(), stride.data(), constantConv, sizeOfType(typeId));
		free(constantConv);
	}
	inline void __fillRandom(TensorDesc* tensor, double stddev, double mean) {
		switch (typeId) {
		case TYPE::TYPE_UINT8 :
			fillTensorRandom<uint8_t ><<<1, 1>>>(*mem, size.data(), stride.data(), stddev, mean);
			break;
		case TYPE::TYPE_INT8  :
			fillTensorRandom<int8_t  ><<<1, 1>>>(*mem, size.data(), stride.data(), stddev, mean);
			break;
		case TYPE::TYPE_UINT16:
			fillTensorRandom<uint16_t><<<1, 1>>>(*mem, size.data(), stride.data(), stddev, mean);
			break;
		case TYPE::TYPE_INT16 :
			fillTensorRandom<int16_t ><<<1, 1>>>(*mem, size.data(), stride.data(), stddev, mean);
			break;
		case TYPE::TYPE_UINT32:
			fillTensorRandom<uint32_t><<<1, 1>>>(*mem, size.data(), stride.data(), stddev, mean);
			break;
		case TYPE::TYPE_INT32 :
			fillTensorRandom<int32_t ><<<1, 1>>>(*mem, size.data(), stride.data(), stddev, mean);
			break;
		case TYPE::TYPE_HALF  :
			fillTensorRandom<half    ><<<1, 1>>>(*mem, size.data(), stride.data(), stddev, mean);
			break;
		case TYPE::TYPE_FLOAT :
			fillTensorRandom<float   ><<<1, 1>>>(*mem, size.data(), stride.data(), stddev, mean);
			break;
		case TYPE::TYPE_DOUBLE:
			fillTensorRandom<double  ><<<1, 1>>>(*mem, size.data(), stride.data(), stddev, mean);
			break;
		}
	}

	void __resetDependencies() {
		dep.clear();
	}
	template<bool write, bool atomic = false>
	inline void __registerOperation(OperationNode* node) {
		dep.apply<write, atomic>(node;)
	}

	//Serialization TODO: Leaks memory when not packed
	/*
		Serializes type, size, strides and type of tensor to file.

		@param file: The file to serialize to. Has to already be open
	*/
	inline void serialize(FILE* file) const {
		fwrite(&typeId       , sizeof(TYPE)    , 1, file); 
		fwrite(size.data()   , sizeof(uint32_t), 5, file);
		fwrite(strides.data(), sizeof(uint32_t), 5, file);

		uint32_t combinedSize = size[0] * strides[0] + size[1] * strides[1] + size[2] * strides[2] + size[3] * strides[3] + size[4] * strides[4];
		void* hostBuf = malloc(combinedSize * sizeOfType(typeId));
		cudaMemcpy(hostBuf, *mem, combinedSize * sizeOfType(typeId), cudaMemcpyDeviceToHost);

		fwrite(hostBuf, sizeOfType(typeId), combinedSize, file);

		free(hostBuf);
	}

	//Can pass an already existing TensorDesc*. If pointer is nullptr, a new descriptor will be returned
	inline static void deserialize(TensorDesc*& tensor, FILE* file) {
		TYPE typeId_;
		std::array<uint32_t, 5> size_;
		std::array<uint32_t, 5> strides_;
		void* hostBuf;

		fread(&typeId_       , sizeof(TYPE)    , 1, file);
		fread(size_.data()   , sizeof(uint32_t), 5, file);
		fread(strides_.data(), sizeof(uint32_t), 5, file);

		if (tensor == nullptr) {
			tensor = new TensorDesc()
				->setSize(size_)
				->setStrides(strides_)
				->setTypeId(typeId_);
		}
		else {
			if (typeId_  != tensor->getTypeId())  throw new std::runtime_error("[ERROR] Trying to deserialize tensor from file into tensor of different variable type");
			if (size_    != tensor->getSize())    throw new std::runtime_error("[ERROR] Trying to deserialize tensor from file into tensor of different size");
			if (strides_ != tensor->getStrides()) throw new std::runtime_error("[ERROR] Trying to deserialize tensor from file into tensor of different strides");
		}

		uint32_t combinedSize = size_[0] * strides_[0] + size_[1] * strides_[1] + size_[2] * strides_[2] + size_[3] * strides_[3] + size_[4] * strides_[4];
		hostBuf = malloc(combinedSize * sizeOfType(typeId_));
		void* newMem;
		cudaMalloc(&newMem, combinedSize * sizeOfType(typeId_));
		fread(hostBuf, sizeOfType(typeId_), combinedSize, file);

		if (tensor->__getMem()) {
			cudaFree(tensor->__getMem());
		}
		tensor->__setMem(newMem);
		cudaMemcpy(newMem, hostBuf, combinedSize * sizeOfType(typeId_), cudaMemcpyHostToDevice);

		free(hostBuf);
	}
};

struct ConstantDesc {
private:
	bool useIndirection;   //If true, use indirection pointers for this value
	
	TYPE typeId;           //The id of the data type of this constant

	void* value;           //Pointer to value on device. Even when user provides value, allocate space on gpu for this
public:
	//Constructors
	/*
		Default constructor. All information must be set with setter methods.
	*/
	ConstantDesc(TYPE typeId) :
		useIndirection(false), typeId(typeId)
	{
		cudaMalloc(&value, sizeOfType(typeId));
	}
	ConstantDesc(ConstantDesc&  other) = delete;  //No copy constructor as objects are identified by their pointer
	ConstantDesc(ConstantDesc&& other) = delete;  //No move constructor as objects are identified by their pointer

	//Setter Methods. They all return the "this" pointer so they are chainable
	inline ConstantDesc* setUseIndirection(bool useIndirection_) {
		useIndirection = useIndirection_;
	
		return this;
	}
	template<typename T> 
	inline ConstantDesc* setValue(T value_) {
		void* valueConv = malloc(sizeofType(typeId));
		convertType<T>(value_, valueConv, typeId);
		cudaMemcpy(value, valueConv, sizeofType(typeId), cudaMemcpyHostToDevice);
		free(valueConv);

		return this;
	}

	//Getter methods
	inline bool  getUseIndirecion() const {
		return useIndirection; 
	}
	inline TYPE  getTypeId() const {
		return typeId; 
	}
	inline void  getValue(void** out) const {
		cudaMemcpy((void*)out, value, sizeOfType(typeId), cudaMemcpyDeviceToHost);
	}
	inline void* getValueHandle() const {
		return value;
	}
};

//========================================================
//==================|Operation Graph|=====================
//========================================================
/*
	An Operation Graph consists of multiple OperationNodes which all specify one operation and all the operations before it it depends on. The operation graph will optimize the specified operations and compile them to an executable cuda graph.
*/

enum OPERATION_TYPE   : uint32_t { OPERATION_FILL = 0, OPERATION_COPY = 1, OPERATION_BIAS = 2, OPERATION_BINARY = 3, OPERATION_GEMM = 4, OPERATION_CONV = 5, OPERATION_REDUCE = 6, OPERATION_POINTWISE = 7, OPERATION_ACTIVATION_FORWARD = 8, OPERATION_ACTIVATION_BACKWARD = 9, OPERATION_HOST = 10 };
enum POINTWISE_TYPE   : uint32_t { POINTWISE_ADD = 0, POINTWISE_SUB = 1, POINTWISE_MUL = 2, POINTWISE_DIV = 3, POINTWISE_SQUARE = 4, POINTWISE_ROOT = 5, POINTWISE_REC = 6, POINTWISE_ROOT_REC = 7 , POINTWISE_POW = 8, POINTWISE_SIGN = 9, POINTWISE_CLIP = 10};
enum BINARY_TYPE      : uint32_t { BINARY_ADD = 0, BINARY_SUB = 1, BINARY_MUL = 2, BINARY_DIV = 3 };
enum ACTIVATION_TYPE  : uint32_t { ACTIVATION_RELU = 0, ACTIVATION_SIGMOID = 1, ACTIVATION_TANH = 2, ACTIVATION_SOFTPLUS = 3, ACTIVATION_SOFTMAX = 4, ACTIVATION_SOFTMAX_TEMP = 5 };
enum CONVOLUTION_TYPE : uint32_t { CONVOLUTION = 0, CROSS_CORRELATION = 1 };
enum REDUCE_TYPE      : uint32_t { SUM = 0, MEAN = 1, VARIANCE = 2 };

//TODO: Check input and output parameters for consistency, heuristics, differentiation. Computation type

class Operation {
protected:
	std::vector<TensorDesc*>   in;     //Tensor descriptors of tensors that are read during the operation
	std::vector<TensorDesc*>   out;    //Tensor descriptor  of tensor  where the result of the operation is stored
	std::vector<ConstantDesc*> consts; //Constants used in the operation

public:
	//Constructors
	/*
		Default constructors. All information has to be provided using setter methods
	*/
	Operation() :
		in(), out()
	{}

	//Setter methods
	//	profided by each subclass differently

	//Getter tensors
	inline  std::vector<TensorDesc*> getInTensors()     const  {
		return in;
	}
	inline  std::vector<TensorDesc*> getOutTensors()    const {
		return out;
	}
	inline std::vector<ConstantDesc*> getConstants()    const {
		return consts;
	}
	virtual OPERATION_TYPE           getOperationType() const {
		throw new std::runtime_error("[ERROR] Trying to get type of object of Operation base class");
	}
};
class Operation_Fill : public Operation {
	/*
		Fill whole tensor with a constant
	*/
public:
	//Constructor
	Operation_Fill() {
		this->in = std::vector<TensorDesc*>();
		this->in.push_back(nullptr);
		this->out = std::vector<TensorDesc*>();
		this->out.resize(1);
		this->consts = std::vector<ConstantDesc*>(); //Blending constants + fill value
		this->consts.resize(3);
		this->consts[2] = nullptr;
	}

	//Setter methods
	inline Operation_Fill* setConstant (ConstantDesc* in ) {
		//0.: Check arguments
		if (!in) {
			fprintf(stderr, "[ERROR] ConstantDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}
		if (this->in[0] != nullptr) {
			fprintf(stderr, "[ERROR] Fill constant can oly be from Tensor or Constant but not both! (File %s, Lins %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->consts[2] = in;

		return this;
	}
	inline Operation_Fill* setConstant (TensorDesc* in) {
		//0.: Check arguments
		if (!in) {
			fprintf(stderr, "[ERROR] ConstantDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}
		std::array<uint32_t, 5> sizes = in->getSize();
		if (sizes[1] * sizes[2] * sizes[3] * sizes[4] != 1) {
			fprintf(stderr, "[ERROR] If fill constant comes from tensor, it must only contain one element per batch! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}
		if (this->consts[2] != nullptr) {
			fprintf(stderr, "[ERROR] Fill constant can oly be from Tensor or Constant but not both! (File %s, Lins %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}


		//1.: Set value
		this->in[0] = in;

		return this;
	}
	inline Operation_Fill* setOutTensor(TensorDesc* out) {
		//0.: Check arguments
		if (!out) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->out[0] = out;

		return this;
	}
	inline Operation_Fill* setBlendConstants(std::array<ConstantDesc*, 2> blendingConstants) {
		this->consts[0] = blendingConstants[0];
		this->consts[1] = blendingConstants[1];

		return this;
	}

	//Getter methods
	inline  std::array<ConstantDesc*, 2> getBlendConstants() const {
		return std::array<ConstantDesc*, 2>({ this->consts[0], this->consts[1] });
	}
	virtual OPERATION_TYPE               getOperationType()  const override {
		return OPERATION_TYPE::OPERATION_FILL;
	}
};
class Operation_Copy : public Operation {
	/*
		Copies the content of a tensor element per element to a different one using blending coefficients
	*/

public:
	//Constructor
	Operation_Copy()
	{
		this->in = std::vector<TensorDesc*>();
		this->in.resize(1);
		this->out = std::vector<TensorDesc*>();
		this->out.resize(1);
		this->consts = std::vector<ConstantDesc*>();
		this->consts.resize(2); //Blend constants ={alpha, beta}. alpha scales output of operation, beta scales previous value
	}

	//Setter methods
	inline Operation_Copy* setInTensor (TensorDesc* in) {
		//0.: Check arguments
		if (!in) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__ );
			std::exit(-1);
		}

		//1.: Set value
		this->in[0] = in;

		return this;
	}
	inline Operation_Copy* setOutTensor(TensorDesc* out) {
		//0.: Check arguments
		if (!out) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->out[0] = out;

		return this;
	}
	inline Operation_Copy* setBlendConstants(std::array<ConstantDesc*, 2> blendConstants) {
		this->consts[0] = blendConstants[0];
		this->consts[1] = blendConstants[1];

		return this;
	}

	//Getter methods
	inline std::array<ConstantDesc*, 2> getBlendConstants() const {
		return std::array<ConstantDesc*, 2>({this->consts[0], this->consts[1]});
	}
	virtual OPERATION_TYPE getOperationType() const override {
		return OPERATION_TYPE::OPERATION_COPY;
	}
};
class Operation_Bias                  : public Operation {
	/*
		Performs a addition of two tensors and writes the result to another. For this, both tensors need to share the same size except that when the size of one dimension of the bias is 1, the value is broadcasted to every value of mem
	*/

public:
	//Constructor
	Operation_Bias()
	{
		this->in  = std::vector<TensorDesc*>();
		this->in.resize(2);
		this->out = std::vector<TensorDesc*>();
		this->out.resize(1);
		this->consts = std::vector<ConstantDesc*>();
		this->consts.resize(2); //Blend constants ={alpha, beta}. alpha scales output of operation, beta scales previous value
	}

	//Setter methods
	inline Operation_Bias* setInTensors(TensorDesc* mem, TensorDesc* bias) {
		//0.: Check arguments
		if (!mem || !bias) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[0] = mem;
		this->in[1] = bias;

		return this;
	}
	inline Operation_Bias* setOutTensor(TensorDesc* out) {
		//0.: Check arguments
		if (!out) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->out[0] = out;

		return this;
	}
	inline Operation_Bias* setBlendConstants(std::array<ConstantDesc*, 2> blendConstants) {
		this->consts[0] = blendConstants[0];
		this->consts[1] = blendConstants[1];

		return this;
	}
	
	//Getter methods
	inline std::array<ConstantDesc*, 2> getBlendConstants() const {
		return std::array<ConstantDesc*, 2>({ this->consts[0], this->consts[1] });;
	}
	virtual OPERATION_TYPE getOperationType() const override {
		return OPERATION_TYPE::OPERATION_BIAS;
	}
};
class Operation_Pointwise : public Operation {
	/*
		Performs an operation on each element of a tensor
	*/

private:
	POINTWISE_TYPE pointwiseType;                //Which pointwise operation to perform

public:
	//Constructor
	Operation_Pointwise() :
		pointwiseType((POINTWISE_TYPE)-1)
	{
		this->in = std::vector<TensorDesc*>();
		this->in.resize(1);
		this->out = std::vector<TensorDesc*>();
		this->out.resize(1);
		this->consts = std::vector<ConstantDesc*>(); //Blend constatns + arguments ([2:])
		this->consts.resize(2); //Blend constants ={alpha, beta}. alpha scales output of operation, beta scales previous value
	}

	//Setter methods
	inline Operation_Pointwise* setInTensor(TensorDesc* in) {
		//0.: Check arguments
		if (!in) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[0] = in;

		return this;
	}
	inline Operation_Pointwise* setOutTensor(TensorDesc* out) {
		//0.: Check arguments
		if (!out) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->out[0] = out;

		return this;
	}
	inline Operation_Pointwise* setPointwiseType(POINTWISE_TYPE type) {
		pointwiseType = type;

		return this;
	}
	inline Operation_Pointwise* setBlendConstants(std::array<ConstantDesc*, 2> blendConstants) {
		this->consts[0] = blendConstants[0];
		this->consts[1] = blendConstants[1];

		return this;
	}
	
	/*
		Only needed when operation needs argument. Otherwise, this can be ignored
	*/
	inline Operation_Pointwise* setArgument(std::vector<ConstantDesc*> arguments) {
		this->consts.insert(this->consts.begin() + 2, arguments.begin(), arguments.end());
	}

	//Getter methods
	inline POINTWISE_TYPE               getPointwiseType()  const {
		return pointwiseType;
	}
	inline std::array<ConstantDesc*, 2> getBlendConstants() const {
		return std::array<ConstantDesc*, 2>({ this->consts[0], this->consts[1] });
	}
	inline std::vector<ConstantDesc*>   getArguments()      const {
		return std::vector(this->consts.begin()+2, this->consts.end());
	}
	virtual OPERATION_TYPE              getOperationType()  const override {
		return OPERATION_TYPE::OPERATION_POINTWISE;
	}
};
class Operation_Binary : public Operation {
	/*
		Performs a binary operation on two tensor (elementwise; out=op(in1, in2))
	*/

private:
	BINARY_TYPE binaryType;

public:
	//Constructor
	Operation_Binary()
	{
		this->in = std::vector<TensorDesc*>();
		this->in.resize(2);
		this->out = std::vector<TensorDesc*>();
		this->out.resize(1);
		this->consts = std::vector<ConstantDesc*>();
		this->consts.resize(2);
	}

	//Setter methods
	inline Operation_Binary* setInTensors(TensorDesc* in1, TensorDesc* in2) {
		//0.: Check arguments
		if (!in1 || !in2) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[0] = in1;
		this->in[1] = in2;

		return this;
	}
	inline Operation_Binary* setOutTensor(TensorDesc* out) {
		//0.: Check arguments
		if (!out) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->out[0] = out;

		return this;
	}
	inline Operation_Binary* setBlendConstants(std::array<ConstantDesc*, 2> blendConstants) {
		this->consts[0] = blendConstants[0];
		this->consts[1] = blendConstants[1];

		return this;
	}
	inline Operation_Binary* setBinaryType(BINARY_TYPE type) {
		binaryType = type;
	}

	//Getter methods
	inline std::array<ConstantDesc*, 2> getBlendConstants() const {
		return std::array<ConstantDesc*, 2>({ this->consts[0], this->consts[1] });
	}
	inline BINARY_TYPE                  getBinaryType()     const {
		return binaryType;
	}
	virtual OPERATION_TYPE              getOperationType()  const override {
		return OPERATION_TYPE::OPERATION_BINARY;
	}
};
class Operation_GEMM                 : public Operation {
	/*
		Matrix multiplication. (out = in[0] x in[1])
	*/
private:
	std::array<bool, 2> transpositions_;

public:
	//Constructor
	Operation_GEMM() :
		transpositions_()
	{
		this->in  = std::vector<TensorDesc*>();
		this->in.resize(2);
		this->out = std::vector<TensorDesc*>();
		this->out.resize(1);
		this->consts = std::vector<ConstantDesc*>();
		this->consts.resize(2);
	}

	//Setter methods
	inline Operation_GEMM* setInTensors(TensorDesc* in1, TensorDesc* in2) {
		//0.: Check arguments
		if (!in1 || !in2) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[0] = in1;
		this->in[1] = in2;

		return this;
	}
	inline Operation_GEMM* setOutTensor(TensorDesc* out) {
		//0.: Check arguments
		if (!out) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->out[0] = out;

		return this;
	}
	inline Operation_GEMM* setBlendConstants(std::array<ConstantDesc*, 2> blendConstants) {
		this->consts[0] = blendConstants[0];
		this->consts[1] = blendConstants[1];

		return this;
	}
	inline Operation_GEMM* setTranspositions(std::array<bool, 2> transpositions) {
		transpositions_ = transpositions;

		return this;
	}

	//Getter methods
	inline std::array<ConstantDesc*, 2> getBlendConstants() const {
		return std::array<ConstantDesc*, 2>({ this->consts[0], this->consts[1] });
	}
	inline std::array<bool, 2>          getTranspositions() const {
		return transpositions_;
	}
	virtual OPERATION_TYPE getOperationType() const override {
		return OPERATION_TYPE::OPERATION_GEMM;
	}
};
class Operation_Conv         : public Operation {
	/*
		Convolution operation (in[0] is feature map, in[1] is kernel)

		4D - 3 spatial dimensions and one for number of feature maps / number of kernels
	*/
private:
	CONVOLUTION_TYPE convType;
	std::array<uint32_t, 3> prePadding_;          //Number of 0's appended to the beginning of the feature map per dimension
	std::array<uint32_t, 3> postPadding_;         //Number of 0's appended to the end       of the feature map per dimension
	std::array<uint32_t, 3> dilation_;            //Kernel dilation per dimension
	std::array<uint32_t, 3> stride_;              //Stride of kernel motion of feature map per dimension
	
	//Application only attributes
	std::array<uint32_t, 3> zeroInterleave_;      //"Fractional stride"; interleaves feature map with zeros. Only needed for backpropagation and should thus only be set by application. 
	bool rotatedFeatureMap_;                      //True, if the feature map of the convolution is first rotated 180° degrees. Only needed for backpropagation and should thus only be set by application. Notice that because of padding, dilation and stride, this does not simply change the convType.

public:
	//Constructor
	Operation_Conv() :
		convType(), prePadding_(), postPadding_(), dilation_({ 1u, 1u, 1u, 1u }), stride_({ 1u, 1u, 1u, 1u }), zeroInterleave_({ 1u, 1u, 1u, 1u }), rotatedFeatureMap_(false)
	{
		this->in = std::vector<TensorDesc*>();
		this->in.resize(2);
		this->out = std::vector<TensorDesc*>();
		this->out.resize(1);
		this->consts = std::vector<ConstantDesc*>();
		this->consts.resize(2); //Blending constants
	}

	//Setter methods
	inline Operation_Conv* setInFeatureMap  (TensorDesc* inFeatureMap) {
		//0.: Check arguments
		if (!inFeatureMap) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[0] = inFeatureMap;
	
		return this;
	}
	inline Operation_Conv* setKernel        (TensorDesc* kernel) {
		//0.: Check arguments
		if (!kernel) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[1] = kernel;

		return this;
	}
	inline Operation_Conv* setOutTensor     (TensorDesc* out) {
		//0.: Check arguments
		if (!out) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->out[0] = out;

		return this;
	}
	inline Operation_Conv* setBlendConstants(std::array<ConstantDesc*, 2> blendConstants) {
		this->consts[0] = blendConstants[0];
		this->consts[1] = blendConstants[1];

		return this;
	}
	inline Operation_Conv* setConvType      (CONVOLUTION_TYPE             type) {
		convType = type;
	}
	inline Operation_Conv* setPrePadding    (std::array<uint32_t, 3>      prePadding) {
		prePadding_ = prePadding;
	}
	inline Operation_Conv* setPostPadding   (std::array<uint32_t, 3>      postPadding) {
		postPadding_ = postPadding;
	}
	inline Operation_Conv* setDilation      (std::array<uint32_t, 3>      dilation) {
		dilation_ = dilation;
	}
	inline Operation_Conv* setStride        (std::array<uint32_t, 3>      stride) {
		stride_ = stride;
	}

	//Getter methods
	inline std::array<ConstantDesc*, 2> getBlendConstants() const {
		return std::array<ConstantDesc*, 2>({ this->consts[0], this->consts[1] });
	}
	inline CONVOLUTION_TYPE             getConvType()       const {
		return convType;
	}
	inline std::array<uint32_t     , 3> getPrePadding()     const {
		return prePadding_;
	}
	inline std::array<uint32_t     , 3> getPostPadding()    const {
		return postPadding_;
	}
	inline std::array<uint32_t     , 3> getDilation()       const {
		return dilation_;
	}
	inline std::array<uint32_t     , 3> getStride()         const {
		return stride_;
	}
	virtual OPERATION_TYPE getOperationType() const override {
		return OPERATION_TYPE::OPERATION_CONV;
	}

	//Heuristic methods
	inline void checkConsistency() const {
		// "getExpectedOutputShape" internally checks stride,dilation,size consistency
		if (this->getOutTensors()[0]->getSize() != getExpectedOutputShape()) throw new std::runtime_error("[ERROR] Convolution consistency check failed: The output shape is inconsistent with the input shapes!");
	
		if(this->getOutTensors()[0] && this->getOutTensors()[0]->getTypeId() != this->getInTensors()[0]->getTypeId()) throw new std::runtime_error("[ERROR] Convolutional layer requires input and output feature map to have the same type");
	}
	inline std::array<uint32_t, 5> getExpectedOutputShape() const{
		std::array<uint32_t, 5>  inputSizes = this->getInTensors()[0]->getSize();
		std::array<uint32_t, 5> kernelSizes = this->getInTensors()[1]->getSize();

		//Checks
		uint32_t dim1Numerator   = (inputSizes[2] + prePadding_[0] + postPadding_[0] - 1u) - (kernelSizes[2] - 1u) * dilation_[0];
		uint32_t dim2Numerator   = (inputSizes[3] + prePadding_[1] + postPadding_[1] - 1u) - (kernelSizes[3] - 1u) * dilation_[1];
		uint32_t dim3Numerator   = (inputSizes[4] + prePadding_[2] + postPadding_[2] - 1u) - (kernelSizes[4] - 1u) * dilation_[2];
		if(dim1Numerator % stride_[0] != 0) throw new std::runtime_error("[ERROR] Convolution consistency check failed: Because of your choice of dilation and stride, the last row of the first dimension of the input feature map will not be read! (decrease it)");
		if(dim2Numerator % stride_[1] != 0) throw new std::runtime_error("[ERROR] Convolution consistency check failed: Because of your choice of dilation and stride, the last row of the second dimension of the input feature map will not be read! (decrease it)");
		if(dim3Numerator % stride_[2] != 0) throw new std::runtime_error("[ERROR] Convolution consistency check failed: Because of your choice of dilation and stride, the last row of the third dimension of the input feature map will not be read! (decrease it)");

		if (gcd(stride_[0], dilation_[0]) != 1) throw new std::runtime_error("[ERROR] Convolution layer has stride and dilation with an gcd that is not 1 in dimension 1, thus the input feature map will only be used sparsely");
		if (gcd(stride_[1], dilation_[1]) != 1) throw new std::runtime_error("[ERROR] Convolution layer has stride and dilation with an gcd that is not 1 in dimension 2, thus the input feature map will only be used sparsely");
		if (gcd(stride_[3], dilation_[2]) != 1) throw new std::runtime_error("[ERROR] Convolution layer has stride and dilation with an gcd that is not 1 in dimension 3, thus the input feature map will only be used sparsely");

		if (this->getInTensors()[0]->getTypeId() != this->getInTensors()[1]->getTypeId()) throw new std::runtime_error("[ERROR] Convolutional layer requires kernel and input feature map to have the same type");

		return {inputSizes[0], kernelSizes[0], dim1Numerator / stride_[0], dim2Numerator / stride_[1], dim3Numerator / stride_[2]};
	}

	//Differentiate
	Operation_Conv* differentiateByFeatureMap(TensorDesc* gradientsIn, TensorDesc* gradientsOut) {
		checkConsistency();
		
		return new Operation_Conv()
			->setInFeatureMap(gradientsIn)
			->setKernel(this->getInTensors()[1])
			->setOutTensor(gradientsOut)
			->setConvType((CONVOLUTION_TYPE)!getConvType())
			->setBlendConstants({
					new ConstantDesc(gradientsOut->getTypeId())->setUseIndirection(false)->setValue(1.),
					new ConstantDesc(gradientsOut->getTypeId())->setUseIndirection(false)->setValue(0.)
				})
			->setPrePadding({ 
					(this->getInTensors()[1]->getSize()[2] - 1u) * getDilation()[0] - getPrePadding()[0],
					(this->getInTensors()[1]->getSize()[3] - 1u) * getDilation()[1] - getPrePadding()[1],
					(this->getInTensors()[1]->getSize()[4] - 1u) * getDilation()[2] - getPrePadding()[2]
				})
			->setPostPadding({
					(this->getInTensors()[1]->getSize()[2] - 1u) * getDilation()[0] - getPostPadding()[0],
					(this->getInTensors()[1]->getSize()[3] - 1u) * getDilation()[1] - getPostPadding()[1],
					(this->getInTensors()[1]->getSize()[4] - 1u) * getDilation()[2] - getPostPadding()[2]
				})
			->setDilation(getDilation())
			->setStride({ 1u, 1u, 1u })
			->setZeroInterleave(getStride())
			->setRotatedFeatureMap(false);
	}
	Operation_Conv* differentiateByKernel    (TensorDesc* gradientsIn, TensorDesc* gradientsOut) {
		checkConsistency();

		return new Operation_Conv()
			->setInFeatureMap(this->getInTensors()[0])
			->setKernel(gradientsIn)
			->setOutTensor(gradientsOut)
			->setConvType(getConvType())
			->setBlendConstants({
					new ConstantDesc(gradientsOut->getTypeId())->setUseIndirection(false)->setValue(1.),
					new ConstantDesc(gradientsOut->getTypeId())->setUseIndirection(false)->setValue(0.)
				})
			->setPrePadding ((getConvType() == CONVOLUTION_TYPE::CROSS_CORRELATION) ? getPrePadding()  : getPostPadding())
			->setPostPadding((getConvType() == CONVOLUTION_TYPE::CROSS_CORRELATION) ? getPostPadding() : getPrePadding() )
			->setDilation(getStride())
			->setStride(getDilation())
			->setZeroInterleave({ 1u, 1u, 1u })
			->setRotatedFeatureMap(getConvType()==CONVOLUTION_TYPE::CONVOLUTION);
	}

protected:
	//Application only methods

	//Setter methods
	inline Operation_Conv* setZeroInterleave(std::array<uint32_t, 3> zeroInterleave) {
		zeroInterleave_ = zeroInterleave;
	}
	inline Operation_Conv* setRotatedFeatureMap(bool rotatedFeatureMap) {
		rotatedFeatureMap_ = rotatedFeatureMap;
	}
	
	//Getter methods
	inline std::array<uint32_t, 3> getZeroInterleave   () const {
		return zeroInterleave_;
	}
	inline bool                    getRotatedFeatureMap() const {
		return rotatedFeatureMap_;
	}
};
class Operation_Reduce : public Operation{
	/*
		Performs a reduction of one tensors along one of its dimensions.

		The reduce dimensions are determined by the input and output
	*/
private:
	std::vector<uint32_t> reduceDimensions; //The dimension to reduce
	REDUCE_TYPE reduce_type_;

public:
	//Constructor
	Operation_Reduce() :
		reduceDimensions(), reduce_type_((REDUCE_TYPE)-1)
	{
		this->in = std::vector<TensorDesc*>();
		this->in.resize(1);
		this->out = std::vector<TensorDesc*>();
		this->out.resize(1);
		this->consts = std::vector<ConstantDesc*>();
		this->consts.resize(2);
	}

	//Setter methods
	inline Operation_Reduce* setInTensor (TensorDesc* input) {
		//0.: Check arguments
		if (!input) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[0] = input;

		return this;
	}
	inline Operation_Reduce* setOutTensor(TensorDesc* out) {
		//0.: Check arguments
		if (!out) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->out[0] = out;

		return this;
	}
	inline Operation_Reduce* setReduceDimensions(std::vector<uint32_t> dimensions) {
		for (uint32_t d : dimensions) {
			if (d > 4) {
				fprintf(stderr, "[ERROR] Reduce dimension has to be in [0,4]!\n");
				std::exit(-1);
			}
		}

		reduceDimensions = dimensions;
	}
	inline Operation_Reduce* setReduceType(REDUCE_TYPE reduce_type) {
		reduce_type_ = reduce_type;

		return this;
	}
	inline Operation_Reduce* setBlendConstants(std::array<ConstantDesc*, 2> blendConstants) {
		this->consts[0] = blendConstants[0];
		this->consts[1] = blendConstants[1];

		return this;
	}

	//Getter methods
	inline std::array<ConstantDesc*, 2> getBlendConstants()   const {
		return std::array<ConstantDesc*, 2>({ this->consts[0], this->consts[1] });
	}
	inline std::vector<uint32_t>        getReduceDimensions() const {
		return reduceDimensions;
	}
	inline REDUCE_TYPE                  getRduceType()        const {
		return reduce_type_;
	}
	virtual OPERATION_TYPE              getOperationType()    const override {
		return OPERATION_TYPE::OPERATION_REDUCE;
	}
};
class Operation_Activation_Forward   : public Operation {
	/*
		Applys an activation function to a tensor
	*/
private:
	ACTIVATION_TYPE activationType;               //Which activation to apply

public:
	//Constructor
	Operation_Activation_Forward() :
		activationType((ACTIVATION_TYPE)-1)
	{
		this->in = std::vector<TensorDesc*>();
		this->in.resize(2);
		this->in[1] = nullptr;
		this->out = std::vector<TensorDesc*>();
		this->out.resize(1);
		this->consts = std::vector<ConstantDesc*>();
		this->consts.resize(2);
	}

	//Setter methods
	inline Operation_Activation_Forward* setInTensor      (TensorDesc*                  in            ) {
		//0.: Check arguments
		if (!in) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[0] = in;

		return this;
	}
	/*
		Only needed when operation needs argument. Otherwise, this can be ignored
	*/
	inline Operation_Activation_Forward* setArgument      (TensorDesc*                  argument      ) {
		//0.: Check arguments
		if (!argument) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[1] = argument;
		
		return this;
	}
	inline Operation_Activation_Forward* setOutTensor     (TensorDesc*                  out           ) {
		this->out[0] = out;

		return this;
	}
	inline Operation_Activation_Forward* setActivationType(ACTIVATION_TYPE              type          ) {
		activationType = type;

		return this;
	}
	inline Operation_Activation_Forward* setBlendConstants(std::array<ConstantDesc*, 2> blendConstants) {
		this->consts[0] = blendConstants[0];
		this->consts[1] = blendConstants[1];

		return this;
	}

	//Getter methods
	inline TensorDesc*                  getArgument()        const {
		return this->in[1];
	}
	inline ACTIVATION_TYPE              getActivationType()  const {
		return activationType;
	}
	inline std::array<ConstantDesc*, 2> getBlendConstants()  const {
		return std::array<ConstantDesc*, 2>({ this->consts[0], this->consts[1] });
	}
	virtual OPERATION_TYPE              getOperationType()   const override {
		return OPERATION_TYPE::OPERATION_ACTIVATION_FORWARD;
	}
};
class Operation_Activation_Backward  : public Operation {
	/*
		Computes derivative of activation function
	*/
private:
	ACTIVATION_TYPE activationType;               //Which activation to apply

public:
	//Constructor
	Operation_Activation_Backward() :
		activationType((ACTIVATION_TYPE)-1)
	{
		this->in = std::vector<TensorDesc*>();
		this->in.resize(3);
		this->in[2] = nullptr;
		this->out = std::vector<TensorDesc*>();
		this->out.resize(1);
		this->consts = std::vector<ConstantDesc*>();
		this->consts.resize(2);
	}

	//Setter methods
	inline Operation_Activation_Backward* setInTensor   (TensorDesc* in) {
		//0.: Check arguments
		if (!in) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[0] = in;

		return this;
	}
	inline Operation_Activation_Backward* setDeltaTensor(TensorDesc* deltas) {
		//0.: Check arguments
		if (!deltas) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[1] = deltas;

		return this;
	}
	/*
		Only needed when operation needs argument. Otherwise, this can be ignored. Has to be the same that was used for forward activation.
	*/
	inline Operation_Activation_Backward* setArgument(TensorDesc* argument) {
		//0.: Check arguments
		if (!argument) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->in[2] = argument;
		
		return this;
	}
	inline Operation_Activation_Backward* setOutTensor(TensorDesc* out) {
		//0.: Check arguments
		if (!out) {
			fprintf(stderr, "[ERROR] TensorDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		this->out[0] = out;

		return this;
	}
	inline Operation_Activation_Backward* setActivationType(ACTIVATION_TYPE type) {
		activationType = type;

		return this;
	}
	inline Operation_Activation_Backward* setBlendConstants(std::array<ConstantDesc*, 2> blendConstants) {
		this->consts[0] = blendConstants[0];
		this->consts[1] = blendConstants[1];

		return this;
	}

	//Getter methods
	inline TensorDesc*                  getArgument()        const {
		return this->in[2];
	}
	inline ACTIVATION_TYPE              getActivationType()  const {
		return activationType;
	}
	inline std::array<ConstantDesc*, 2> getBlendConstants() const {
		return blendConstants_;
	}
	virtual OPERATION_TYPE              getOperationType()  const override {
		return OPERATION_TYPE::OPERATION_ACTIVATION_BACKWARD;
	}
};
class Operation_If : public Operation {
	/*
		If a constant is 1, performs one step of operations, otherwise another.
	*/
private:
	std::vector<Operation*> true_operations;
	std::vector<Operation*> false_operations;

public:
	//Constructor
	Operation_If() :
		true_operations(), false_operations()
	{
		this->in     = std::vector<TensorDesc*>();
		this->out    = std::vector<TensorDesc*>();
		this->consts = std::vector<ConstantDesc*>();
		this->consts.resize(1);
	}

	//Setter methods
	inline Operation_If* setConditionConstant(ConstantDesc* in) {
		//0.: Check arguments
		if (!in) {
			fprintf(stderr, "[ERROR] ConstantDesc* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set condition constant
		this->consts[0] = in;

		return this;
	}
	inline Operation_If* addOperationTrue(Operation* op) {
		//0.: Check arguments
		if (!op) {
			fprintf(stderr, "[ERROR] Operation* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		true_operations.push_back(op);

		return this;
	}
	inline Operation_If* addOperationFalse(Operation* op) {
		//0.: Check arguments
		if (!op) {
			fprintf(stderr, "[ERROR] Operation* cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		false_operations.push_back(op);

		return this;
	}

	//Getter methods
	inline std::vector<Operation*> getTrueOperations() const {
		return true_operations;
	}
	inline std::vector<Operation*> getFalseOperations() const {
		return false_operations;
	}
	virtual OPERATION_TYPE         getOperationType()  const override {
		return OPERATION_TYPE::OPERATION_FILL;
	}
};
class Operation_Host                 : public Operation {
	/*
		Calls an c++-only function (no calls to the cuda api, kernel launches, etc. are allowed)
	*/
	
private:
	void* functionPointer_;                //Pointer of host function to call
	std::vector<ConstantDesc*> arguments_; //Descriptor of the call parameters in the right order

public:
	//Constructor
	Operation_Host() :
		functionPointer_(nullptr), arguments_()
	{
		this->in = std::vector<TensorDesc*>();
		this->out = std::vector<TensorDesc*>();
		this->consts = std::vector<ConstantDesc*>();
	}

	//Setter methods
	inline Operation_Host* setFunctionPointer(void* functionPointer) {
		//0.: Check arguments
		if (!in) {
			fprintf(stderr, "[ERROR] Function pointer cannot be nullptr! (File %s, Line %d)\n", __FILE__, __LINE__);
			std::exit(-1);
		}

		//1.: Set value
		functionPointer_ = functionPointer;

		return this;
	}
	inline Operation_Host* setArguments(std::vector<ConstantDesc*> arguments) {
		arguments_ = arguments;

		return this;
	}

	//Getter methods
	inline  void*                      getFunctionPointer() const {
		return functionPointer_;
	}
	inline  std::vector<ConstantDesc*> getArguments()       const {
		return arguments_;
	}
	virtual OPERATION_TYPE             getOperationType()   const override {
		return OPERATION_TYPE::OPERATION_HOST;
	}
};

//Fused operations
//Fuse move/scale/fill/add/mult after any operation using blending coefficients, if tensor is not needed and result is only used once. Copy into itself=mult. Eliminate moves. Fill = pointwise add/sub with indirection ->transform. Divsion = multiplication with invers
struct Operation_Fused_GEMM_Add_Pointwise : public Operation {
	/*
		Fused (matrix multiplication), (bias addition), (pointwise operation): out = poinwise(in[0] * in[1] + in[2])
	*/
private:
	std::array<ConstantDesc*, 2> blendConstants; //={alpha, beta}. alpha scales output of operation, beta scales previous value

public:
	//Setter method
	inline Operation* setInTensors(TensorDesc* mul1, TensorDesc* mul2, TensorDesc* add) {
		this->in = std::vector<TensorDesc*>({ mul1, mul2, add});

		return this;
	}
	inline Operation* setBlendConstants(std::array<ConstantDesc*, 2> blendConstants_) {
		blendConstants = blendConstants_;

		return this;
	}

	//Getter method
	inline std::array<ConstantDesc*, 2> getBlendConstants() const {
		return blendConstants;
	}
};
struct Operation_Fused_Convolution_Add_Pointwise : public Operation {
	/*
		Fused (convolution), (bias addition), (poitwise operation): out = poinwise(convolution(features=in[0], kernel=in[1]) + in[2])
	*/
private:
	std::array<ConstantDesc*, 2> blendConstants; //={alpha, beta}. alpha scales output of operation, beta scales previous value

public:
	//Setter methods
	inline Operation* setInTensors(TensorDesc* featureMap, TensorDesc* kernel, TensorDesc* add) {
		this->in = std::vector<TensorDesc*>({ featureMap, kernel, add });

		return this;
	}
	inline Operation* setBlendConstants(std::array<ConstantDesc*, 2> blendConstants_) {
		blendConstants = blendConstants_;

		return this;
	}
	
	//Getter methods
	inline std::array<ConstantDesc*, 2> getBlendConstants() const {
		return blendConstants;
	}
};

struct OperationNode {
private:
	Operation* operation;                    //The operation of this node

	std::set<OperationNode*> dependencies;   //Operations this operation depends on. Will be set during operation graph compilation by the application. Not used by user
	cudaGraphNode_t graphNode;               //The graph node this operation is compiled to. Set by application during graph compilation, not used by used

public:
	//Constructors
	/*
		Default constructor
	*/
	OperationNode() :
		dependencies(), operation(), graphNode()
	{}
	/*
		Constructs node from operation. Default initialises all other variables
	*/
	OperationNode(Operation* op) :
		operation(op)
	{}

	//Setter methods
	inline void setOperation(Operation* operation_) {
		operation = operation_;
	}

	//Getter methods
	inline Operation* getOperation() const {
		return operation;
	}
	
	//Methods that should only be called by the application
	inline void __addDependencies(const std::vector<OperationNode*>& new_dependencies) {
		dependencies.insert(new_dependencies.begin(), new_dependencies.end());
	}
	inline std::vector<OperationNode*>&& __getDependencies() const {
		std::vector<OperationNode*> ret(dependencies.begin(), dependencies.end());
		return std::move(ret);
	}
	inline void __setGraphNode(const cudaGraphNode_t& graphNode_) {
		graphNode = graphNode_;
	}
	inline cudaGraphNode_t __getGraphNode() const {
		return graphNode;
	}
};


//Takes ownership of all not yet allocated tensor desriptors
class OperationGraph {
	/*
		All operation to be performed are added to an operation graph, which optimizes them and compiles them to an executable graph. It also allocates the memory of all non-virtual pointers
	
		The order the operations are added matters, as the reads of later nodes depend on the writes to the same location of previous nodes.
	*/
private:
	std::vector<OperationNode> nodes; //All nodes in this graph order by dependencies

public:
	//Constructors
	/*
		Default constructor
	*/
	OperationGraph() :
		nodes()
	{}

	//Methods
	/*
		This appends a new node of the given operation to the graph.

		@param operation: The operation of the new node. Does not take ownership
	*/
	inline void addNode(Operation* operation) {
		nodes.emplace_back(operation);
	}
	/*
		Compiles the graph: Generates an executable cudaGraph_t and allocates the needed memory

		@param outGraph            : The generated cudaGraph_t
		@param workspaceRequirement: The memory requirement of the cublas/cudnn workspaces needed for the execution of the graph
	*/
	inline void compile(cudaGraph_t& outGraph, MemoryRequirement& workspaceRequirement, uint32_t minimalAlignment = 16) { //TODO
		//1.: Walk the graph and fuse operations and delete unused operations
		//2.: Decide, which tensors are virtual (set this in tensor descriptors) and which are tmp (mem can be reused)
		//3.: Allocate and distribute memory (set in tensor descriptors). Because of the different alignment criteria, pick the memory layout that minimizes memory consumption
		//4.: Generate the dependencies (call Dependency guards of tensors which will generate dependencies of nodes)
		//5.: Build the cuda graph

		//JIT?!

		//Reuse temporary memory
	}
};

//================================================================
//==================|Dependencies Management|=====================
//================================================================

struct Dependencies {
	/*
		An object of this class guards a memory region. Each operation that is perfomed on this memory region has to register how it uses the memory region and this class in turn computes the dependencies of this operation
	*/

	//If both vectors contain elements, the last operation was a read and the writes are only stored because each new read depends on them
private:
	std::vector<OperationNode*> unblocked_reads;
	std::vector<OperationNode*> unblocked_write;           //Could be multiple if using atomics. If there are also read dependencies, all write dependencies are blocked but need to be stored as each new read needs to depend on them

public:
	Dependencies() :
		unblocked_reads(), unblocked_write()
	{}

	/*
		Applies dependecies to a node that performs a operation on the memory region guarded by this

		@param write: True, if "node" writes to the memory segment guarded by this. False, if it just reads it
		@param node : The node that either reads or writes to the memory segment guarded by this
	*/
	template<bool write, bool atomic = false>
	void apply(OperationNode* node) {
		static_assert(!atomic, "[ERROR] Dependencie::apply error: Atomic operations not implemented yet!");
		static_assert(write || !atomic, "[ERROR] Dependencies::apply error: A read cannot be atomic (or rather is always atomic... Anyhow, just leave second template parameter empty, ok?)!");

		if constexpr (write) { //node depends on read and write dependencies of "dep"    
			if (unblocked_reads.size()) {                      //There are unblocked reads. Thus, all writes are blocked and we only need to depend on every read
				//1.: Apply dependencies
				node->__addDependencies(unblocked_reads);

				//2.: Update dependencies
				unblocked_reads.clear();
				unblocked_write.clear();
				unblocked_write.push_back(node);
			}
			else {                                             //There are no unblocked reads. Thus, the node is only dependend on the last writes
				//1.: Apply dependencies
				node->__addDependencies(unblocked_write);

				//2.: Update dependencies
				unblocked_write.clear();
				unblocked_write.push_back(node);
			}
		}
		else {       //node only interferes with writes
			//1.: Applies dependencies
			node->__addDependencies(unblocked_write);

			//2.: Update dependencies
			unblocked_reads.push_back(node);
		}
	}

	/*
		Resets this dependency guard to its starting state
	*/
	void clear() {
		unblocked_reads.clear();
		unblocked_write.clear();
	}
};