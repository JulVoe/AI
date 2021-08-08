#pragma once
#include "OperationGraph.cpp"
#include "Optimizer.cpp"
#include <vector>

//============================================================
//==================|Declare new classes|=====================
//============================================================
enum ACTIVATION_TYPE : uint32_t { ACTIVATION_IDENTITY = 0, ACTIVATION_RELU = 1, ACTIVATION_SOFTMAX = 2, ACTIVATION_SOFTMAX_TEMP = 3, ACTIVATION_SIGMOID = 4, ACTIVATION_TANH = 5, ACTIVATION_SOFTPLUS = 6 };  //DON'T CHANGE THESE VALUES AS IT WILL BREAK OLD CHECKPOINT FILES!
class Activation;
class Activation_Identity;
class Activation_Relu;
class Activation_Softmax;
class Activation_SoftmaxTemp;
class Activation_Sigmoid;
class Activation_Tanh;
class Activation_Softplus;

//===================================================
//==================|Activations|====================
//===================================================

class Activation {
public:
	TYPE data_type;
	TYPE computation_type;

protected:
	TensorDesc* memory;                           //The memory this activates

	std::vector<ConstantDesc*> internalDesc;      //Descriptors of constants needed for computations (e.g.  temperature)
	std::vector<TensorDesc*>   tmpTensors;        //Temporary tensors of computation

public:
	//Constructor
	Activation() {
		throw new std::runtime_error("[ERROR] Trying to create a base class activation");
	}

	//Setters
	inline void setMemory(TensorDesc* memory_) {
		if (memory_->getTypeId() != data_type) throw new std::runtime_error("[ERROR] Trying to set activation memory to a type that is not the data type of the activation");

		memory = memory_;
	}

	//Getters
	inline TensorDesc* getMemory() const {
		return memory;
	}

	//Methods
	virtual void forward(OperationGraph* graph) {
		throw new std::runtime_error("[ERROR] Trying to forward propagate through base class activation");
	}

	virtual void backward(OperationGraph* graph, Optimizer* optimizer) {
		throw new std::runtime_error("[ERROR] Trying to backward propagate through base class activation");
	}

	//Serialization
	/*
		Return the derived class type of the object
	*/
	virtual ACTIVATION_TYPE getType() const {
		throw new std::runtime_error("[ERROR] Trying to call \"getType\" on base class optimizer");
	}
	/*
		Returns a pointer to a newly created default initialized object of the specified derived class

		@param type: The derived class of the returned object
	*/
	inline static Activation* getActivationOfType(ACTIVATION_TYPE type) {
		switch (type) {
		case ACTIVATION_TYPE::ACTIVATION_IDENTITY:
			return new Activation_Identity();
			break;
		case ACTIVATION_TYPE::ACTIVATION_RELU:
			return new Activation_Relu();
			break;
		case ACTIVATION_TYPE::ACTIVATION_SOFTMAX:
			return new Activation_Softmax();
			break;
		case ACTIVATION_TYPE::ACTIVATION_SOFTMAX_TEMP:
			return new Activation_SoftmaxTemp();
			break;
		case ACTIVATION_TYPE::ACTIVATION_SIGMOID:
			return new Activation_Sigmoid();
			break;
		case ACTIVATION_TYPE::ACTIVATION_TANH:
			return new Activation_Tanh();
			break;
		case ACTIVATION_TYPE::ACTIVATION_SOFTPLUS:
			return new Activation_Softplus();
			break;
		default:
			throw new std::runtime_error("[ERROR] Trying to create an activation of unsupported type");
		}
	}
	/*
		Serializes the type of this activation as well as the internal memory to the specified file. The information can be deserialized using the "deserialize" method

		@param file: FILE* to an already opened file where the information is written to
	*/
	inline void serialize(FILE* file) const {
		ACTIVATION_TYPE type = getType();
		fwrite(&type, sizeof(type), 1, file);

		uint32_t num_intern_bufs = internalDesc.size();
		fwrite(&num_intern_bufs, sizeof(uint32_t), 1, file);
		for (ConstantDesc* constant : internalDesc) {
			writeConstantToFile(constant);
		}
	}
	/*
		Deserializes the type of this activation from the specified file. Return an object of this type and reads in all the internal buffers. The file has to have been created using the "serialize" method

		@param file: FILE* to an already opened file where the information is read to
	*/
	inline static Activation* deserialize(FILE* file) {
		ACTIVATION_TYPE type;
		fread(&type, sizeof(type), 1, file);

		Activation* ret = getActivationOfType(type);

		uint32_t num_intern_bufs;
		fread(&num_intern_bufs, sizeof(uint32_t), 1, file);
		for (uint32_t i = 0; i != num_intern_bufs; i++) {
			ConstantDesc* constant = nullptr;
			readConstantFromFile(constant, file);
			ret->internalDesc.push_back(constant);
		}

		return ret;
	}
};

class Activation_Identity : public Activation {
	//internalDesc = {}
	//tmpTensors   = {}

public:
	//Constructor
	Activation_Identity() {
		this->internalDesc     = std::vector<ConstantDesc*>();
		this->tmpTensors       = std::vector<TensorDesc*>();
		this->data_type        = (TYPE)-1;
		this->computation_type = (TYPE)-1;
	}

	//Methods
	virtual void forward(OperationGraph* graph) override {	/*Do nothing*/ }
	virtual void backward(OperationGraph* graph, Optimizer* optimizer) override {	/*Do nothing*/ }

	//Serialization
	virtual ACTIVATION_TYPE getType() const override {
		return ACTIVATION_TYPE::ACTIVATION_IDENTITY;
	}
};

class Activation_Relu : public Activation {
	//internalDesc = {}
	//tmpTensors   = {}

public:
	//Constructor
	Activation_Relu() {
		this->internalDesc     = std::vector<ConstantDesc*>();
		this->tmpTensors       = std::vector<TensorDesc*>();
		this->data_type        = (TYPE)-1;
		this->computation_type = (TYPE)-1;
	}

	//Methods
	virtual void forward(OperationGraph* graph) override {	
		Operation_Pointwise* op = new Operation_Pointwise()
			->setInTensor(this->getMemory())
			->setOutTensor(this->getMemory())
			->setPointwiseType(POINTWISE_TYPE::POINTWISE_RELU)
			->setBlendConstants({
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.),
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.)
				});

		graph->addNode(op);
	}
	virtual void backward(OperationGraph* graph, Optimizer* optimizer) override {
		Operation_Pointwise* op = new Operation_Pointwise()
			->setInTensor (this->getMemory())
			->setOutTensor(this->getMemory())
			->setPointwiseType(POINTWISE_TYPE::POINTWISE_RELU)
			->setBlendConstants({
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(1.),
					new ConstantDesc(this->data_type)->setUseIndirection(false)->setValue(0.)
				});

		graph->addNode(op);
	}

	//Serialization
	virtual ACTIVATION_TYPE getType() const override {
		return ACTIVATION_TYPE::ACTIVATION_RELU;
	}
};