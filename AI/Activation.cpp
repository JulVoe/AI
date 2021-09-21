#pragma once
#include "OperationGraph.cpp"
#include "Optimizer.cpp"
#include <vector>

//============================================================
//==================|Declare new classes|=====================
//============================================================
//The following is already declared in OperationGraph.cpp:
//enum ACTIVATION_TYPE : uint32_t { ACTIVATION_IDENTITY = 0, ACTIVATION_RELU = 1, ACTIVATION_SOFTMAX = 2, ACTIVATION_SOFTMAX_TEMP = 3, ACTIVATION_SIGMOID = 4, ACTIVATION_TANH = 5, ACTIVATION_ELU = 6, ACTIVATION_SWISH = 7, ACTIVATION_SOFTPLUS = 8 };  //DON'T CHANGE THESE VALUES AS IT WILL BREAK OLD CHECKPOINT FILES!
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
	TYPE data_type;                   //Should be set in constructor
	TYPE computation_type;            //Should be set in constructor

protected:
	TensorDesc* memory_;                                   //The memory this activates

	std::unordered_map<std::string, ConstantDesc*> constants; //Constants
	std::unordered_map<std::string, TensorDesc*  > variables; //Variables

public:
	//Constructor
	Activation() {
		throw std::runtime_error("[ERROR] Trying to create a base class activation");
	}

	//Setters
	inline void setMemory(TensorDesc* memory) {
		if (memory->getTypeId() != data_type) throw std::runtime_error("[ERROR] Trying to set activation memory to a type that is not the data type of the activation");

		memory_ = memory;
	}

	//Getters
	inline TensorDesc* getMemory() const {
		return memory_;
	}

	//Methods
	virtual void forward(OperationGraph* graph) {
		throw std::runtime_error("[ERROR] Trying to forward propagate through base class activation");
	}

	virtual void backward(OperationGraph* graph, Optimizer* optimizer) {
		throw std::runtime_error("[ERROR] Trying to backward propagate through base class activation");
	}

	//Serialization    TODO!!!!
	/*
		Return the derived class type of the object
	*/
	virtual ACTIVATION_TYPE getType() const {
		throw std::runtime_error("[ERROR] Trying to call \"getType\" on base class optimizer");
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
public:
	//Constructor
	Activation_Identity(TYPE data_type = TYPE::TYPE_FLOAT, TYPE computation_type = TYPE::TYPE_FLOAT) {
		this->constants        = std::unordered_map<std::string, ConstantDesc*>();
		this->variables        = std::unordered_map<std::string, TensorDesc*  >();
		this->data_type        = data_type;
		this->computation_type = computation_type;
	}

	//Methods
	virtual void forward(OperationGraph* graph) override {	/*Do nothing*/ }
	virtual void backward(OperationGraph* graph, Optimizer* optimizer) override {	/*Do nothing*/ }

	//Serialization
	virtual ACTIVATION_TYPE getType() const override {
		return ACTIVATION_TYPE::ACTIVATION_IDENTITY;
	}							
};

#define SIMPLE_ACTIVATION(class_name, activation_type)                                                               \
class class_name : public Activation {																				 \
public:																												 \
	/*Constructor*/																									 \
	class_name(TYPE data_type = TYPE::TYPE_FLOAT, TYPE computation_type = TYPE::TYPE_FLOAT) {						 \
		this->constants = std::unordered_map<std::string, ConstantDesc*>();											 \
		this->variables = std::unordered_map<std::string, TensorDesc*  >();											 \
		this->data_type = data_type;																				 \
		this->computation_type = computation_type;																	 \
	}																												 \
																													 \
	/*Methods*/																										 \
	virtual void forward(OperationGraph* graph) override {															 \
		graph->addNode((new Operation_Activation_Forward())															 \
			->setActivationType(activation_type)													                 \
			->setInTensor(this->memory_)																			 \
			->setOutTensor(this->memory_)																			 \
			->setBlendConstants({																					 \
				new ConstantDesc(this->computation_type)->setUseIndirection(false)->setValue(1.),					 \
				new ConstantDesc(this->computation_type)->setUseIndirection(false)->setValue(0.)					 \
			})																										 \
		);																											 \
	}																												 \
	virtual void backward(OperationGraph* graph, TensorDesc*& deltas, Optimizer* optimizer) override {    			 \
		graph->addNode((new Operation_Activation_Backward())												    	 \
			->setActivationType(activation_type)													                 \
			->setInTensor(this->memory_)																			 \
			->setOutTensor(this->memory_)																			 \
			->setDeltaTensor(deltas)																				 \
			->setBlendConstants({																					 \
				new ConstantDesc(this->computation_type)->setUseIndirection(false)->setValue(1.),					 \
				new ConstantDesc(this->computation_type)->setUseIndirection(false)->setValue(0.)					 \
			})   																									 \
		);																											 \
	}																												 \
																													 \
	/*Serialization*/																							     \
	virtual ACTIVATION_TYPE getType() const override {																 \
		return activation_type;																	                     \
	}																												 \
};

#define ONE_VAR_ACTIVATION(class_name, param_name, activation_type)                                                  \
class class_name : public Activation {																				 \
public:																												 \
	/*Constructor*/																									 \
	class_name(TYPE data_type = TYPE::TYPE_FLOAT, TYPE computation_type = TYPE::TYPE_FLOAT) {						 \
		this->constants = std::unordered_map<std::string, ConstantDesc*>();											 \
		this->variables = std::unordered_map<std::string, TensorDesc*  >();											 \
		this->data_type = data_type;																				 \
		this->computation_type = computation_type;																	 \
																													 \
		this->variables.emplace(param_name, (new TensorDesc())														 \
			->setTypeId(this->data_type)																			 \
			->setSize({1u, 1u, 1u, 1u, 1u})																			 \
			->setStrides({1u, 1u, 1u, 1u, 1u})																		 \
			->setAlignment(MIN_ALIGN)																				 \
			->setIsNeeded(true)																						 \
			->setUseIndirection(false)																				 \
		);														                                                     \
	}																												 \
																													 \
	/*Methods*/																										 \
	virtual void forward(OperationGraph* graph) override {															 \
		graph->addNode((new Operation_Activation_Forward())															 \
			->setActivationType(activation_type)													                 \
			->setInTensor(this->memory_)																			 \
			->setArgument(this->variables[param_name])											     				 \
			->setOutTensor(this->memory_)																			 \
			->setBlendConstants({																					 \
				new ConstantDesc(this->computation_type)->setUseIndirection(false)->setValue(1.),					 \
				new ConstantDesc(this->computation_type)->setUseIndirection(false)->setValue(0.)					 \
			})																										 \
		);																											 \
	}																												 \
	virtual void backward(OperationGraph* graph, Optimizer* optimizer) override {									 \
		graph->addNode((new Operation_Activation_Backward())														 \
			->setActivationType(activation_type)													                 \
			->setInTensor(this->memory_)																			 \
			->setArgument(this->variables[param_name])											     				 \
			->setOutTensor(this->memory_)																			 \
			->setBlendConstants({																					 \
				new ConstantDesc(this->computation_type)->setUseIndirection(false)->setValue(1.),					 \
				new ConstantDesc(this->computation_type)->setUseIndirection(false)->setValue(0.)					 \
				})																									 \
		);																											 \
		
	}																												 \
																													 \
	/*Serialization*/																							     \
	virtual ACTIVATION_TYPE getType() const override {																 \
		return activation_type;																	                     \
	}																												 \
};


SIMPLE_ACTIVATION(Activation_Relu    , ACTIVATION_TYPE::ACTIVATION_RELU    );
SIMPLE_ACTIVATION(Activation_Softmax , ACTIVATION_TYPE::ACTIVATION_SOFTMAX );
SIMPLE_ACTIVATION(Activation_Sigmoid , ACTIVATION_TYPE::ACTIVATION_SIGMOID );
SIMPLE_ACTIVATION(Activation_Tanh    , ACTIVATION_TYPE::ACTIVATION_TANH    );
SIMPLE_ACTIVATION(Activation_Swish   , ACTIVATION_TYPE::ACTIVATION_SWISH   );
SIMPLE_ACTIVATION(Activation_Softplus, ACTIVATION_TYPE::ACTIVATION_SOFTPLUS);




