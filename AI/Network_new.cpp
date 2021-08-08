#include "OperationGraph.cpp"
#include "Activation.cpp"
#include "Optimizer.cpp"

class Layer {
public:
	uint32_t data_type;
	uint32_t computation_type;

private:
	uint32_t* batch_size;

	ACTIVATION_TYPE activation;
	
	std::vector<TensorDesc>* internalDesc;      //Descriptors of memory that needs to be saved (weights, biases, kernels). These need to have "isNeeded=true"
	std::vector<TensorDesc*> intermediateDesc;  //Descriptors of memory that does not need to be saved (intermediate computation steps. These could be needed for computation and thus do not neccessarily have "isNeeded=false", e.g. output)
	TensorDesc* outputDesc;

	Layer* prevLayer;

public:
	//Change alignment requirement of precious layers output
	virtual void setPreviousLayer(Layer* prevLayer_) {
		prevLayer = prevLayer_;
	}

	virtual void forward(OperationGraph* graph) {

	}

	virtual void backward(OperationGraph* graph, Optimizer* optimizer) {

	}

	//Has to be called after operation graph has been compiled
	virtual void initMem(std::vector<std::pair<uint32_t, void*>> uid_memory_map) {

	}
};


class Architecture {
private:
	uint32_t data_type;
	uint32_t compute_type;

	std::vector<Layer> layers;
	uint32_t batch_size;
	uint32_t min_alignment;

public:
	void saveArchitecture(std::string path) {}
	void saveArchitectureAndData(std::string path) {}
	void loadArchitecture(std::string path) {}

	//Connects layers
	void connect() {

	}

	//Adds forward propagation to graph
	void addForwardPass(OperationGraph* graph) {}

	//Adds backward propagation to graph
	void addBackwardPass(OperationGraph* graph) {}

	void initialize() {

	}
	void initialize(std::string path) {

	}



	void getFirstAndLastLayer(Layer*& firstLayer, Layer*& lastLayer) {

	}
};