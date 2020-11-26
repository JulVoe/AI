//#define THE_VERSION_JULIAN_DID_NOT_SCREW_WITH
#ifdef __NVCC__
#pragma warning( disable : 4514)
#pragma warning( disable : 4711)
#pragma warning( disable : 4710)
#pragma warning( disable : 5039)
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
//#include <mma.h>
//#include <cublasXt.h>

#include <inttypes.h>
#include <stdio.h>

#include "Dataset.cpp"

/*
Notation conventions:
 - A matrix has size height*width.
 - layer[0] is the input layer
 - weight[i] are the weights between layer[i] and layer[i+1]
*/

//TODO: Check launch parameters
//TODO: Const, restrict
//TODO: Check comments, move files
//TODO: Put methods in classes
//TODO: CASH COHERENCE
//=========================================================
//==================|HELPER FUNCTIONS|=====================
//=========================================================

#define LAUNCH_PARAM(N) (int)(1. / ((10. / ((double)(1 << 13)) + 32. / ((double)(N))))), 32
#define GRID_SIZE(N) (int)(1. / ((10. / ((double)(1 << 13)) + 32. / ((double)(N))))), 1, 1

//================================================
//==================|GLOBALS|=====================
//================================================

cublasHandle_t cublas_handle;

//========================================================================
//==================|Move to classes and kernell.cpp|=====================
//========================================================================

/* 
    Returns a graph that captures the following operation:
    Computes either the matrix multiplication C=trans_A(A)*trans_B(B) or C+=trans_A(A)*trans_B(B).
    All matrices (A,B,C) have to be stored column major.

    @param T: The computation type.
    @param A: Left  factor of matrix product that will be computed.
    @param B: Right factor of matrix product that will be computed
    @param C: Where to store the result of the multiplication
    @param trans_A: Whether to transpose A before multiplication (swap height and width)
    @param trans_B: Whether to transpose B before multiplication (swap height and width)
    @param overwrite: If this is true, we overwrite C with the result of the matrix multiplication. If it is false, we add the result to the data in C
    @param y1, x1, x2: trans_A(A) has size y1*x1. trans_B(B) has size x1*x2. C has size y1*y2.
    @param cublasConst: ={(T)0, (T)1}. Used to determine the alpha and beta of cublas kernel depending on "overwrite". Has to persist in memory during the invocation of cuda graph!
    @param captureStream: The stream which is used to stream capture the cublas kernel.
*/
template<typename T, bool trans_A, bool trans_B, bool overwrite> //TODO:cublasConst
inline cudaGraph_t getMatmulGraph(T* A, T* B, T* C, uint32_t y1, uint32_t x1, uint32_t x2, cudaStream_t captureStream) {
        //0.: Make sure T is a recognized type
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, half>::value, "[Error] Matrix multiplication is not supported with this type!");
        
        //1.: Start stream capture
        cudaGraph_t capGraph;
        cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeThreadLocal);
        
        //2.: Enqueue cublas kernel
        if constexpr (std::is_same<T, float>::value)
            cublasSgemm( cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, (float*) &cublasConst[1], (float*) A, trans_A?x1:y1, (float*) B, trans_B?x2:x1, (float*)&cublasConst[!overwrite], (float*)C, y1);
        if constexpr (std::is_same<T, double>::value)
            cublasDgemm( cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, (double*)&cublasConst[1], (double*)A, trans_A?x1:y1, (double*)B, trans_B?x2:x1, (double*)&cublasConst[!overwrite], (double*)C, y1);
        if constexpr (std::is_same<T, half>::value)
            cublasGemmEx(cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, (half*)  &cublasConst[1], (half*)  A, CUDA_R_16F, trans_A?x1:y1, (half*)B, CUDA_R_16F, trans_B?x2:x1, (half*)&cublasConst[!overwrite], (half*)C, CUDA_R_16F, y1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
        //3.: Stop capture and return graph
        cudaStreamEndCapture(captureStream, capGraph);
        return capGraph;
}

template<typename T>
inline void addBiasNode(cudaGraph_t graph, T* out, T* in, uint32_t num_out, uint32_t num_in, cudaGraphNode_t* node){
    void* biasArgs[] = {
        (void*)& out,
        (void*)& in, 
        (void*)& num_out,
        (void*)& num_in
    };
    cudaKernelNodeParams biasParam {
        (void*)set_repeating<T>,              //Function pointer
        dim3(GRID_SIZE(num_out)),             //Grid dimensions
        dim3(32, 1, 1),                       //Block dimensions
        0u,                                   //Dyn. shared-mem per block in bytes
        (void**)&biasArgs,                    //Array of pointers to individual kernel arguments
        nullptr                               //Pointer to kernel arguments in the "extra" format
    };
    cudaGraphAddKernelNode(node, graph, nullptr, 0, &biasParam);
}

template<typename T>
inline void addActivationNode(cudaGraph_t graph, T* mem, uint32_t outStateSize, uint32_t batch_size, Activation<T> act, cudaGraphNode_t* node){
    if (act == Activation::RELU){
        constexpr auto ldb = []__device__(T in) { return in > (T)0 ? in : (T)0; };
        uint32_t outStateSizeBatched = outStateSize * batch_size;
        
        void* reluArgs[] = {
            (void*)& mem,
            (void*)& outStateSizeBatched, 
            (void*)& ldb
        };
        cudaKernelNodeParams reluParam {
            (void*)transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&reluArgs,                    //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(node, graph, nullptr, 0, &reluParam);
    }
    else if(act == Activation::SOFTMAX){
        const T temp = (T)1;
        void* softmaxArgs[] = {
            (void*)& mem,
            (void*)& outStateSize, 
            (void*)& batch_size,
            (void*)& temp
        };
        cudaKernelNodeParams softmaxParam {
            (void*)softmaxTemperature<T>,         //Function pointer
            dim3(GRID_SIZE(outStateSize)),        //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            sizeof(T),                            //Dyn. shared-mem per block in bytes
            (void**)&softmaxArgs,                 //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(node, graph, nullptr, 0, &softmaxParam);
    }
    else if(act == Activation::SIGMOID) {
        constexpr auto ldb = []__device__(T in) { return (T)1 / ((T)1 + exponential<T>(in)); };
        uint32_t outStateSizeBatched = outStateSize * batch_size;
        
        void* sigmoidArgs[] = {
            (void*)& mem,
            (void*)& outStateSizeBatched, 
            (void*)& ldb
        };
        cudaKernelNodeParams sigmoidParam {
            (void*)transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes */
            (void**)&sigmoidArgs,                 //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(node, graph, nullptr, 0, &sigmoidParam);
    }
    else if(act == Activation::SOFTPLUS) {
        constexpr auto ldb = []__device__(T in) { return logarithm<T>((T)1 + exponential<T>(in)); };
        uint32_t outStateSizeBatched = outStateSize * batch_size;
        
        void* softplusdArgs[] = {
            (void*)& mem,
            (void*)& outStateSizeBatched, 
            (void*)& ldb
        };
        cudaKernelNodeParams softPlusdParam {
            (void*)transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes */
            (void**)&softplusdArgs,               //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(node, graph, nullptr, 0, &softplusParam);
    } else {
        fprintf(stderr, "[ERROR] Unable to add activation function to cuda graph as it is unkown!\n");
    }
}

template<typename T>
inline void addElementwiseMultNode(cudaGraph_t graph, T* A, T* B, uint32_t len, cudaGraphNode_t* node){
    constexpr auto ldb = []__device__(T a, T b) { return a * b; };
    
    void* elemMultArgs[] = {
        (void*)& A,
        (void*)& B, 
        (void*)& len,
        (void*)& ldb
    };
    cudaKernelNodeParams elemMultParam {
        (void*)transform<T, decltype(ldb)>,   //Function pointer
        dim3(GRID_SIZE(len)),                 //Grid dimensions
        dim3(32, 1, 1),                       //Block dimensions
        0u,                                   //Dyn. shared-mem per block in bytes
        (void**)&elemMultArgs,                //Array of pointers to individual kernel arguments
        nullptr                               //Pointer to kernel arguments in the "extra" format
    };
    cudaGraphAddKernelNode(node, graph, nullptr, 0, &elemMultParam);
}
//==============

enum OPTIMIZER_TYPE: uint32_t {SGD=0};   //DON'T CHANGE THESE VALUES AS IT WILL BREAK OLD CHECKPOINT FILES!
template<typename T>
class Optimizer {
protected:
    T* optBuf;
    double* learningRates;

public:
    void setLearningRates(double* learningRates_) { learningRates = learningRates_; }
    virtual uint64_t setMem(T*& mem); //Uses amount specified in "getMemoryRequirements" for internal memory. Increments mem by this amount and returns offset to prevoous value
    virtual void getMemoryRequirements(uint64_t& num_values, uint64_t optimizables);
    virtual void initMem();
    virtual void addNode(cudaGraph_t graph, T* mem, T* cur, T* delta, T* input, uint32_t y, uint32_t x, uint32_t batch_size, cudaGraphNode_t* node);
    virtual void addNode(cudaGraph_t graph, T* mem, T* cur, T* delta, uint32_t y, uint32_t batch_size, cudaGraphNode_t* node);
    virtual OPTIMIZER_TYPE getOptimizerType();

    /*
        Serializes an object to a FILE*. It can be deserialized using the "deserialize" method.
        The serialize method should be called from the derived class. It is required to first write "getOptimizerType".
        Deserialization is invoked from this base class. It reads the layer type first and afterwards invokes the "deserialize"
        method of the corresponding derived class.
    */
    virtual void serialize(FILE* file);
    /*
        Deserializes an object from a FILE* that was serialized using the "serialize" method.
        The deserialize method should be called from the base class. It reads the optimizer type first and afterwards invokes the "deserialize"
        method of the corresponding derived class.
    */
    virtual static Optimizer<T> deserialize(FILE* file) {
        OPTIMIZER_TYPE opt_type;
        fread(&opt_type, sizeof(OPTIMIZER_TYPE), 1, file);
        switch(opt_type){
        case OPTIMIZER_TYPE::SGD:
            return SGD_Optimizer<T>::deserialize(file);
            break;
        default:
            fprintf(stderr, "[ERROR] The optimizer has an unknown OPTIMIZER_TYPE!");
            exit(-1);
        }
    }
};


enum ACTIVATION_TYPE: uint32_t {RELU=0};  //DON'T CHANGE THESE VALUES AS IT WILL BREAK OLD CHECKPOINT FILES!
template<typename T>
class Activation {
protected:
    T* params;      //TODO: At the moment, they are constant and can't be learned.

public:
    virtual void addActivationNode(cudaGraph_t graph, T* mem, uint32_t outStateSize, uint32_t batch_size, cudaGraphNode_t* node)
    virtual void addActivationDerivNode(cudaGraph_t graph, T* mem, uint32_t outStateSize, uint32_t batch_size, cudaGraphNode_t* node)
    virtual ACTIVATION_TYPE getActivationType();

    /*
        Serializes an object to a FILE*. It can be deserialized using the "deserialize" method.
        The serialize method should be called from the derived class. It is required to first write "getActivationType".
        Deserialization is invoked from this base class. It reads the layer type first and afterwards invokes the "deserialize"
        method of the corresponding derived class.
    */
    virtual void serialize(FILE* file);
    /*
        Deserializes an object from a FILE* that was serialized using the "serialize" method.
        The deserialize method should be called from the base class. It reads the activation type first and afterwards invokes the "deserialize"
        method of the corresponding derived class.
    */
    virtual static Activation<T> deserialize(FILE* file) {
        ACTIVATION_TYPE act_type;
        fread(&act_type, sizeof(ACTIVATION_TYPE), 1, file);
        switch(act_type){
        case ACTIVATION_TYPE::RELU:
            return RELU_Activation<T>::deserialize(file);
            break;
        default:
            fprintf(stderr, "[ERROR] The activation has an unknown ACTIVATION_TYPE!");
            exit(-1);
        }
    }
};

enum LAYER_TYPE: uint32_t {FULLY_CONNECTED = 0};  //DON'T CHANGE THESE VALUES AS IT WILL BREAK OLD CHECKPOINT FILES!
template<typename T>
class Layer {
    /*
        A layer is a helper class that does not actually own any memory. It is only used by the NetworkBuilder to construct
        the execution graph.

        This is only the base class that should never be used itself. Each actual layer type should be implemented as a
        new derived class that overwrites the respective methods.
    */
protected:
    Layer* layerBefore;
    uint32_t batch_size;

    Activation<T> act;

    Image_Shape outStateShape;                    //Shape of T's in state that correspond to output !for batch size 1!
    T* state;                                     //State of this layer (everything dependend on batch size). Has to have size returned in getMemoryRequirements. First outStateSizeBatched values correspond to output state, rest can correspond to internal state.
    T* other;                                     //Other memory (independent of batch size, e.g. weights and biases). This includes (but is not limited to) all parameters that are optimized.

public:
    /*
        Call methods in the order they are defined:
        1.: setLayerBefore
        2.: setBatchSize
        3.: getMemoryRequirement
        4.: setMem
        5.: initMem
        6.: forwardProp
        7.: backwardProp

        The following methods are only helper methods:
         - getLayerType
         - serialize
         - deserialize
    */
    
    /*
        Sets the variable holding a pointer to the layer right in front of this one in the network.
    */
    void setLayerBefore(Layer* l) { layerBefore = l; }
    /*
        Sets the batch size used by this layer. Note, that this does not reallocate state memory.
        Thus, this called should be followed up by a call to "setMem"

        @param batch_size_: The new batch size.
    */
    void setBatchSize(uint32_t batch_size_) { batch_size = batch_size_; }
    /*
        Returns the memory requirements of this layer to the given variables.

        @param state_nums  : Out parameter. The number of T's on the gpu needed to store the state of the layer
        @param other_nums  : Out parameter. The number of T's on the gpu needed to store other data of the layer (weights, biases, ...)
        @param optimizables: Out parameter. The number of T's on the gpu that need to be optimized (weights, biases,...)
        @param tmp         : Out parameter. The number of T's on the gpu needed as temporary storage (shared between all Layers, non persistent)
        @param num_node    : Out parameter. The number of cudaGraphNode_t needed for forward and backwards pass
    */
    virtual void getMemoryRequirement(uint64_t &state_nums, uint64_t &other_nums, uint64_t &optimizables, uint64_t &tmp, uint64_t &num_nodes);
    /*
        Sets the internal memory pointers of the layer. The passed pointers will be incremented by the memory needed by this layer as returned by getMemoryRequirements.
        Returns the offset to the values before as two uint64_t's. ret>>64 is Δstate_mem, ret&(((0b1)<<64)-1) is Δother_mem
        The passed pointer need to be 32bit aligned.

        @param state_mem: Uses the in getMemoryRequirement() returned number of bytes to store its own state. Advances pointer by same amount.
        @param other_mem: Uses the in getMemoryRequirement() returned number of bytes to store other data. Advances pointer by same amount.
    */
    virtual __int128 setMem(uint8_t* &state_mem, uint8_t* &other_mem);
    /* 
        Initializes "state" and "other" memory of the size returned in getMemoryRequirements.
    */
    virtual void initMem();
    /*
        Adds forward propagation through this layer to an execution graph. Takes in the state of the layer before and computes own state.
     
        We can not simply return a graph, as this would make it impossible to express dependencies correctly (bias depends on nothing while matmul does).

        @param graph: The execution graph to construct
        @param depNode: The node that finishes compution the state of the previous layer, as this forward propagation depends on it. At the end, the pointer will be cahnged to the own node that was used to finalise the state.
        @param output: This is where all new graph nodes are stored. The pointer is advanced accordingly.
        @param captureStream: The function can use stream capture to generate the call graph. This is the stream it will use for this.
        @param tmp: Temporary storage of at least the size requested in "getMemoryRequirements".
    */
    virtual void forwardProp(cudaGraph_t graph, cudaGraphNode_t*& depNode, cudaGraphNode_t* &output, cudaStream_t captureStream, T* tmp);
    /*
        Adds backpropagation through this layer to an execution graph.
        Assumes, that the state of this layer was already set to dL/do where L is Loss and o the output after forwardProp.
        This updates the own weights and biases of this layer using optimizer opt and writes dL/do of the layer before in his state.
    
        @param graph: The execution graph to construct
        @param depNode: The node that finishes writing dL/do into this layer's state as this function depends on this data. At the end, the pointer will be cahnged to the own node that was used to finalise the state.
        @param opt: The optimizer used to change the internal weights.
        @param cur_opt: The pointer to the correspondent element in the optimization buffer.
        @param output: This is where all new graph nodes are stored. The pointer is advanced accordingly.
        @param captureStream: The function can use stream capture to generate the call graph. This is the stream it will use for this.
        @param tmp: Temporary storage of at least the size requested in "getMemoryRequirements".
    */
    virtual void backwardProp(cudaGraph_t graph, cudaGraphNode_t*& depNode, Optimizer<T>* opt, T* &cur_opt, cudaGraphNode_t* &output, cudaStream_t captureStream, T* tmp);
    
    /*
        Return the type of the layer.
    */
    virtual LAYER_TYPE getLayerType();
    /*
        Serializes an object to a FILE*. It can be deserialized using the "deserialize" method.
        The serialize method should be called from the derived class. It is required to first write "getLayerType".
        Deserialization is invoked from this base class. It reads the layer type first and afterwards invokes the "deserialize"
        method of the corresponding derived class.
    */
    virtual void serialize(FILE* file);
    /*
        Deserializes an object from a FILE* that was serialized using the "serialize" method.
        The deserialize method should be called from the base class. It reads the layer type first and afterwards invokes the "deserialize"
        method of the corresponding derived class.
    */
    virtual static Layer<T> deserialize(FILE* file) {
        LAYER_TYPE layer_type;
        fread(&layer_type, sizeof(LAYER_TYPE), 1, file);
        switch(layer_type){
        case LAYER_TYPE::FULLY_CONNECTED:
            return FullyConnected_Layer<T>::deserialize(file);
            break;
        default:
            fprintf(stderr, "[ERROR] The layer has an unknown LAYER_TYPE!");
            exit(-1);
        }
    }
};

template<typename T>
class FullyConnected_Layer : public Layer<T> {
public:
    FullyConnected_Layer(Activation<T>& act_, uint32_t num_neurons) :
        layerBefore(nullptr), batch_size(0u), act(act_), outStateShape(Image_Shape(num_neurons, 1u, 1u)), state(nullptr), other(nullptr)
    {}

    void getMemoryRequirement(uint64_t &state_nums, uint64_t &other_nums, uint64_t &optimizables, uint64_t &tmp, uint64_t &num_nodes) overwrite {
        state_nums = outStateShape.prod() * batch_size;

        other_nums = (uint64_t)outStateShape.prod() * (1ull + (uint64_t)layerBefore->outStateShape.prod()); //Bias + Weights

        optimizables = other_nums;                                                                          //Bias + Weights

        tmp = layerBefore->outStateShape.prod() * batch_size;                                               //For backprop
    
        num_nodes = 3 + 5;                                                               //3 for forwardProp, 5 for backProp
    }

    __int128 setMem(T* &state_mem, T* &other_mem) overwrite {
        //1.: Compute return value
        __int128 upper = state_mem - state;
        __int128 lower = other_mem - other;
        __int128 ret   = (upper << 64) ^ lower;

        //2.: Update internal values
        state = state_mem;
        other = other_mem;

        //3.: Increment parameters
        state_mem += outStateShape.prod() * batch_size;
        other_mem += (uint64_t)outStateShape.prod() * (1ull + (uint64_t)layerBefore->outStateShape.prod());
    
        //4.: Return
        return ret;
    }

    //TODO/FIXIT: MAKE WORK AND DEPENDENT ON ACTIVATION FUNCTION
    void initMem() overwrite {
        set_random<T, true>(other_bias   , outStateShape.prod());
        set_random<T, true>(other_weights, layerBefore->outStateShape.prod(), outStateShape.prod());
    }

    void forwardProp(cudaGraph_t graph, cudaGraphNode_t*& depNode, cudaGraphNode_t* &output, cudaStream_t captureStream, T* tmp) overwrite {
        //output={biasNode, multiplyNode, activationNode}
        
        //0.: Usefuls variables
        uint32_t outStateSize = outStateShape.prod();
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        T* bias    = other;
        T* weights = bias + outStateSize;

        //1.: Bias
        addBiasNode<T>(graph, state, bias, outStateSizeBatched, outStateSize, output);

        //2.: Weight multiplication        
        cudaGraph_t multGraph = getMatmulGraph<T, false, false, false>(weights, layerBefore->state, state, outStateShape.prod(), layerBefore->outStateShape.prod(), batch_size, captureStream);
        cudaGraphAddChildGraphNode(output+1, graph, depNode, 1, multGraph);
        //cudaGraphAddDependencies(graph, output/*biasNode*/, output+1/*multiplyNode*/, 1); //TODO:DETERMINE WHETHER THIS IS NECCESARRY. Probably, the gemm will only interact with output using atomicAdd. Thus, it can run in parallel with bias operation.
    
        //3.: Activation
        act.addActivationNode(graph, state, outStateSize, batch_size, output+2);
        cudaGraphAddDependencies(graph, output  , output+2, 1);  //Activation depends on bias
        cudaGraphAddDependencies(graph, output+1, output+2, 1);  //Activation depends on weight multiplication

        //4.: Update pointers
        depNode = output+2; //activationNode;
        output += 3;        //biasNode, multiplyNode, activationNode
    }

    void backwardProp(cudaGraph_t graph, cudaGraphNode_t*& depNode, Optimizer<T>* opt, T* &cur_opt, cudaGraphNode_t* &output, cudaStream_t captureStream, T* tmp) overwrite {
        //output={multNode, optWeightsNode, optBiasNode, activationNode, elemMultNode}

        //0.: Usefull variables
        uint32_t outStateSize = outStateShape.prod();
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        T* bias    = other;
        T* weights = bias + outStateSize;

        //1.: Multiply deltas backwards (deltas_{before} = w^T * deltas)
        cudaGraph_t multGraph = getMatmulGraph<T, true, false, true>(weights, state, tmp, layerBefore->outStateShape.prod(), outStateShape.prod(), batch_size, captureStream);
        cudaGraphAddChildGraphNode(output, graph, depNode, 1, multGraph);       //Dependent on depNode

        //2.: Apply optimizers to weights
        opt->addNode(graph, weights, cur_opt, state, layerBefore->state, outStateSize, layerBefore->outStateShape.prod(), batch_size, output+1);
        cudaGraphAddDependencies(graph, output, output+1, 1); //Optimizer depends on multGraph as it will change weights
        cur_opt += (uint64_t)outStateShape.prod() * (uint64_t)layerBefore->outStateShape.prod();

        //3.: Apply optimizer to bias
        opt->addNode(graph, bias, state, outStateSize, batch_size, output+2);
        cudaGraphAddDependencies(graph, output, output+2, 1); //Optimizer depends on multGraph as it will change bias
        cur_opt += outStateShape.prod();

        //4.: Get derivative of activation in layer before
        layerBefore->act.addActivationDerivNode(graph, layerBefore->state, layerBefore->outStateShape.prod(), batch_size, output+3);
        cudaGraphAddDependencies(graph, output+1, output+3, 1); //Depends on weights optimization as it needs original value

        //5.: Multiply buffer elementwise with derivatives in previous layer
        addElementwiseMultNode(graph, layerBefore->state, tmp, layerBefore->outStateShape.prod() * batch_size, output+4);
        cudaGraphAddDependencies(graph, output  , output+4, 1); //Dependend on multNode as it generates tmp
        cudaGraphAddDependencies(graph, output+3, output+4, 1); //Dependend on activationNode as it generates layerBefore->state
    
        //6.: Update pointers
        depNode = output + 4;  //elemMultNode
        output += 5;
    }

    LAYER_TYPE getLayerType() overwrite {
        return LAYER_TYPE::FULLY_CONNECTED;
    }

    void serialize(FILE* file) overwrite {
        LAYER_TYPE layer_type = getLayerType();
        fwrite(&layer_type, sizeof(layer_type), 1, file);
        
        fwrite(&layer_before, sizeof(layer_before), 1, file);
        fwrite(&batch_size  , sizeof(batch_size)  , 1, file);
        act.serialize(file);
        outStateShape.serialize(file);
        fwrite(&state, sizeof(state), 1, file);
        fwrite(&other, sizeof(other), 1, file);
    }

    static Layer<T> deserialize(FILE* file) overwrite {
        FullyConnected_Layer<T> ret{};
        
        fread(&ret.layer_before, sizeof(ret.layer_before), 1, file);
        fread(&ret.batch_size  , sizeof(ret.batch_size)  , 1, file);
        ret.act           = Activation<T>::deserialize(file);
        ret.outStateShape = Image_Shape::deserialize(file);
        fread(&ret.state, sizeof(ret.state), 1, file);
        fread(&ret.other, sizeof(ret.other), 1, file);

        return ret;
    }
};

template<typename T>
class NetworkBuilder {
    /*
        Checkpoint file structure:
         - 7 bytes signature ("JVCHECK")
         - 2 bytes library version
         - 4 bytes the used type (T)
         - Dump of NetworkBuilder (except "tmp" and "nodes")
         - Optimizer type identifier
         - For each layer, type identifier of activation followed by type identifier of layer
         - Dump of all layers (except "state" and "other")
         - Lenght of other_mem in bytes
         - Length of state_mem in bytes
         - Dump of other_mem
         - Dump of state_mem
    */
private:
    Layer* layers;
    uint32_t num_layers;

    uint32_t batch_size;

    Optimizer<T> opt;

    T* state_mem;           //Has to be realloced when batch size changes.
    T* other_mem;           //Fixed size.
    T* tmp_mem;             //Temporary buffer, has to be realloced when batch size changes.
    cudaGraphNode_t* nodes; //Graph nodes

public:
    NetworkBuilder(Layer* layers_, uint32_t num_layers_, uint32_t batch_size_, Optimizer<T> opt_) :
        layers(layers_),
        num_layers(num_layers_),
        batch_size(batch_size_),
        opt(opt_),
        state_mem(nullptr),
        other_tmp_mem(nullptr),
        tmp_mem(nullptr),
        nodes(nullptr)
    {
        //1.: Connect the layers
        for(uint32_t ind = 1; ind != num_layers; ind++)
            l[ind].setLayerBefore(&l[ind-1])
    }
    
    /*
        Loads from a checkpoint file.

        @param save_file: Path to the Checkpoint.
    */
    NetworkBuilder(char* save_file) {
        //1.: Open file
        FILE* file = fopen(save_file, "rb");

        //2.: Check signature
        printf("[INFO] Parsing checkpoint file...\n");
        
        char sig[7];
        fread(+sig, sizeof(char), 7, file);
        if(strcmp(sig, "JVCHECK", 7) == 0)
            printf("[INFO] \t - Signature matches\n");
        else {
            printf("[Error] \t - Signature mismatch!");
            exit(-1);
        }

        //3.: Version
        uint16_t ver;
        fread(&ver, sizeof(uint16_t), 1, file);
        if (ver < AI_VERSION)
            printf("[INFO] \t - File version: %u. This is an old version, since this is library version %u\n", (uint32_t)ver, AI_VERSION);
        else if (ver == AI_VERSION)
            printf("[INFO] \t - File version: %u. This is the recent version\n", (uint32_t)ver);
        else 
            fprintf(stderr, "[WARNING] \t - The checkpoint file is of a newer version (%u) than the library (%u)! Good chance, this will not work!\n", (uint32_t)ver, AI_VERSION);

        //4.: Type
        uint32_t type;
        fread(&type, sizeof(uint32_t), 1, file);
        if(type == type_hash<T>())
            printf("[INFO] \t - Type matches.\n");
        else{
            fprintf(stderr, "[ERROR] \t - Type mismatch!");
            exit(-1);
        }
            
        //5.: Read in this object
        //The stored pointers point to the old memory location. They are still safed, as they provide relative information.
        Layer* layers_;
        T *state_mem_old, *other_mem_old;

        fread(&layers_   , sizeof(layers_)   , 1, file);
        fread(&num_layers, sizeof(num_layers), 1, file);
        fread(&batch_size, sizeof(batch_size), 1, file);
        opt = Optimizer<T>::deserialize(file):
        fread(&state_mem_old, sizeof(state_mem_old), 1, file);
        fread(&other_mem_old, sizeof(other_mem_old), 1, file);
        
        //6.: Read in layers
        layers = (Layer*)malloc(sizeof(Layer) * num_layers);

        for(uint32_t ind = 0; ind != num_layers; ind++)
            layers[ind] = Layer<T>::deserialize(file);

        //7.: Get memory requirements of layers
        uint64_t state_nums=0, other_nums=0, optimizables=0, tmp=0, num_nodes=0; //Accumulator
        uint64_t s,ot,op,t,n;                                                    //Per layer

        for(uint32_t ind = 0; ind != num_layers; ind++){
            layers[ind].getMemoryRequirement(s,ot,op,t,n);

            state_nums   += s;
            other_nums   += ot;
            optimizables += op;
            tmp          += t;
            num_nodes    += n;
        }

        opt.getMemoryRequirements(ot, optimizables);
        other_nums += ot;

        uint64_t other_nums_, state_nums_;
        fread(&other_nums_, sizeof(other_nums_), 1, file);
        fread(&state_nums_, sizeof(state_nums_), 1, file);
        assert(other_nums == other_nums_):
        assert(state_nums == state_nums_);

        //8.: Allocate memory
        printf("[INFO] Trying to allocate %llumb on gpu for the network...", (bytes_state + bytes_other + bytes_tmp) / (1024ull * 1024ull));

        gpuErrchk(cudaMalloc(&other_mem, bytes_other));
        gpuErrchk(cudaMalloc(&state_mem, bytes_state));
        gpuErrchk(cudaMalloc(&  tmp_mem, bytes_tmp));
        if (other_tmp_mem != nullptr && state_mem != nullptr && tmp_mem != nullptr)
            printf("Success!\n");
        else {
            printf("Failure!\n");
            std::exit(-1);
        }
        nodes = (cudaGraphNode_t*)malloc(sizeof(cudaGraphNode_t) * num_nodes);

        //9.: Set memory of layers and optimizer
        __int128 expected_off = (((__int128)(state_mem - state_mem_old)) << 64) ^ ((__int128)(other_mem - other_mem_old));

        T* state_mem_ = state_mem;
        T* other_mem_ = other_mem;
        for(uint32_t ind = 0; ind != num_layers; ind++) {
            if(ind != 0) layer[ind].setLayerBefore(&layer[ind-1]);
            __int128 off = layers[ind].setMem(state_mem_, other_mem_);

            if (off != expected_off) {
                fprintf(stderr, "[ERROR] The memory requirements of layer %u changed!", ind-1);
                std::exit(-1);
            }
        }

        uint64_t off = opt.setMem(other_mem_);                                    //Should be original pointer + state_mem - state_mem_
        if (off != other_mem - other_mem_old) {
            fprintf(stderr, "[ERROR] The memory requirements of the last layer changed!");
            std::exit(-1);
        }

        //10.: Read in memory from file
        fread(&other_mem, sizeof(T), other_nums, file);
        fread(&state_mem, sizeof(T), state_nums, file);
    }

    /*
        Allocates memory for optimizer and layers.
    */
    void allocate(){
        //0.: Initialize variables
        uint64_t state_nums=0, other_nums=0, optimizables=0, tmp=0, num_nodes=0; //Accumulator
        uint64_t s,ot,op,t,n;                                                    //Per layer

        //1.: Accumulate memory requirements of the layers
        for(uint32_t ind = 0; ind != num_layers; ind++){
            layers[ind].getMemoryRequirement(s,ot,op,t,n);

            state_nums   += s;
            other_nums   += ot;
            optimizables += op;
            tmp          += t;
            num_nodes    += n;
        }

        //2.: Get requirements of optimizer
        opt.getMemoryRequirement(ot, optimizables);
        other_nums += ot;

        //3.: Allocation
        uint64_t bytes_state = sizeof(T) * state_nums;
        uint64_t bytes_other = sizeof(T) * other_nums;
        uint64_t bytes_tmp   = sizeof(T) * tmp;
        printf("[INFO] Trying to allocate %llumb on gpu for the network...", (bytes_state + bytes_other + bytes_tmp) / (1024ull * 1024ull));

        gpuErrchk(cudaMalloc(&other_mem, bytes_other));
        gpuErrchk(cudaMalloc(&state_mem, bytes_state));
        gpuErrchk(cudaMalloc(&  tmp_mem, bytes_tmp  ));
        if(other_tmp_mem != nullptr && state_mem != nullptr && tmp_mem != nullptr)
            printf("Success!\n");
        else {
            printf("Failure!\n");
            std::exit(-1);
        }

        //4.: Set pointers of layers
        T* state_mem_ = state_mem;
        T* other_mem_ = other_mem;
        for(uint32_t ind = 0; ind != num_layers; ind++)
            layers[ind].setMem(state_mem_, other_mem_);

        //5.: Set pointer of optimizer
        opt.setMem(other_mem_);

        //6.: Allocate memory for the graph nodes
        nodes = (cudaGraphNode_t*)malloc(sizeof(cudaGraphNode_t) * num_nodes);
    }

    /*
        Initialize the allocated memory
    */
    void initialize(){
        //1.: Initialize layers
        for(uint32_t ind = 0; ind != num_layers; ind++)
            layers[ind].initMem();

        //2.: Initialize optimizer
        opt.initMem();
    }

    /*
        Resets the batch size and handels all reallocations.
        Note: This clears the state of each layer.
    */
    void resetBatchSize(uint32_t new_batchSize){
        //1.: Reset variables
        batch_size = new_batchSize;
        for(uint32_t ind = 0, ind != num_layers; ind++)
            layers[ind].setBatchSize(new_batchSize);

        //2.: Reallocate
        cudaFree(state_mem);
        cudaFree(tmp_mem);

        uint64_t state_nums=0, tmp=0; //Accumulator
        uint64_t s,t;                 //Per layer

        for(uint32_t ind = 0; ind != num_layers; ind++){
            layers[ind].getMemoryRequirement(s,ot,op,t,n);

            state_nums   += s;
            tmp          += t;
        }

        gpuErrchk(cudaMalloc(&state_mem, sizeof(T) * state_nums));
        gpuErrchk(cudaMalloc(&  tmp_mem, sizeof(T) * tmp));
        
        //3.: Set new memory
        T* state_mem_ = state_mem;
        T* other_mem_ = other_mem;
        for(uint32_t ind = 0; ind != num_layers; ind++)
            layers[ind].setMem(state_mem_, other_mem_);
    }

    /*
        Safes a checkpoint file.
    */
    void serialize(char* out){
        //1.: Open output file
        FILE* out_file = fopen(out, "wb");

        //2.: Write header
        char     signature[] = "JVCHECK";
        uint8_t  version     = AI_VERSION;
        uint32_t type        = type_hash<T>();
        fwrite(+signature, sizeof(signature) - 1, 1, out_file); //Signature
        fwrite(&version  , sizeof(version)      , 1, out_file); //Library version
        fwrite(&type     , sizeof(type)         , 1, out_file); //The underlying type
        
        //3.: Dump this class to file
        fwrite(&layers    , sizeof(layers)    , 1, out_file);
        fwrite(&num_layers, sizeof(num_layers), 1, out_file);
        fwrite(&batch_size, sizeof(batch_size), 1, out_file);
        opt.serialize(out_file);
        fwrite(&state_mem , sizeof(state_mem) , 1, out_file);
        fwrite(&other_mem , sizeof(other_mem) , 1, out_file);
        //tmp will not be changed. It can be computed using "other_nums". Data is not stored as it is temporary and lenght could easily change during implementations
        //Nodes will not be safed, as it is perfectly valid to change the execution graph, e.g. to be more efficient.
            
        //3.: Layer dump
        for(uint32_t ind = 0; ind != num_layer; ind++)
            layers[ind].serialize(out_file);

        //4.: Gpu Memory dump
        uint64_t state_nums=0, other_nums=0, optimizables=0, tmp=0, num_nodes=0; //Accumulator
        uint64_t s,ot,op,t,n;                                                    //Per layer

        for(uint32_t ind = 0; ind != num_layers; ind++){                         //Add up requirements of layers
            layers[ind].getMemoryRequirement(s,ot,op,t,n);

            state_nums   += s;
            other_nums   += ot;
            optimizables += op;
            tmp          += t;
            num_nodes    += n;
        }

        opt.getMemoryRequirement(ot, optimizables);                              //Requirement of optimizer
        other_nums += ot;

        fwrite(&other_nums, sizeof(other_nums), 1         , out_file);
        fwrite(&state_nums, sizeof(state_nums), 1         , out_file);
        fwrite( other_mem , sizeof(T)         , other_nums, out_file);
        fwrite( state_mem , sizeof(T)         , state_nums, out_file);
    }

    /*
        Builds cuda execution graph
    */
    cudaGraph_t getForwardGraph(){

    }
    
    /*
        Builds cuda execution graph
    */
    cudaGraph_t getBackwardsGraph(){

    }
};


template<typename T>
class Scheduler{
    /*
        Schedules learning rates, the learning proces, ...
        Also builds the executable graph using input node and loss node
        Can be used to show loss of model, graph it, early stopping, etc.
    */
}


//TODOS (BOTH FILES!!)
//TODO: Block all kernel calls
//TODO: CONVOLUTIONS AND TILING DESIGN
//TODO: Unified allocator
//TODO: TYPE HANDLING
//TODO: REGULARIZATION AND NORMALIZATION

//======================================================
//==================|DEEP LEARNING|=====================
//======================================================

enum Activation { RELU = 0, SIGMOID = 1, SOFTMAX = 2, SOFTPLUS = 3 };
enum Optimizer_Type { SGD, ADAM, DEMON_ADAM };

struct OptVariables {
    Optimizer_Type opt_t;
    float learningRate;
    float beta1;
    float beta2;
    float betaInit;

    uint32_t t;
    uint32_t T;
};

template<typename T>
class MultiLayerPerceptron {
private:
    uint32_t  num_layer;     //Number of layers
    uint32_t* layer_size;    //Number of neurons per layer
    Activation* activations; //The activation function for every layer
    uint32_t  batch_size;    //Number of samples per batch

    uint64_t num_neurons;  //Number of neurons !not in the input layer!
    uint64_t num_weights;  //Number of weights

    T** weights;  //Pointer to 2D-Array                                 | weights[i] has dimension layer_size[i+1] * layer_size[i], column major
    T** bias;     //Pointer to Array                                    | bias[i]    has dimension layer_size[i+1] * 1            , column major 
    T** output;   //Pointer to Array, after activation                  | output[i]  has dimension layer_size[i+1] * batch_size   , column major

    OptVariables<T> opt_var; //Holds information on what optimizer to use
    T*  opt_buf;             //Size of biggest output[] buffer. Temporary storage for optimizer
    T** opt_momentum_buf[2]; //Stores the momentum of all weights, if opt\in\{ADAM, DEMON_ADAM\}. Otherwise should be set to nullptr
    
    DatasetHandler<T> dataset;     //Dataset to use.
    T* cur_in;                     //Pointer to input data to use (testing, validation or custom)
    T* cur_out;                    //Pointer to output data to use (testing, validation or custom)

    T* cublasConst;               //={0, 1}. In GPU memory. Used to tell cublas whether to add or overwrite (x in A = x*A + y*(B*C))

    //=========================================================
    //========================|UTILITY|========================
    //=========================================================
    /*
        Allocates memory for weights and biases. Furthermore, it computes the correct pointers for each layer. Sets globals num_weights and num_neurons.

        Memory is allocated in a contingous chunk.
    */
    inline void allocate() {
        //1.: Count neurons and weights
        num_weights = 0;
        num_neurons = 0;
        for (uint32_t layer = 0; layer != num_layer - 1; layer++) {
            num_weights += layer_size[layer + 1] * layer_size[layer];
            num_neurons += layer_size[layer + 1];
        }

        //2.: Allocate gpu memory
        T* raw_mem;
        cudaMalloc(raw_mem, sizeof(T) * (num_weights + num_neurons));

        //3.: Set weights pointers
        weights = (T**)malloc(sizeof(T*) * (num_layer - 1)); //First layer needs no weights
        for (uint32_t layer = 0; layer != num_layer - 1; layer++) {
            weights[layer] = raw_mem;
            raw_mem += layer_size[layer + 1] * layer_size[layer];
        }

        //4.: Set bias pointers
        bias = (T**)malloc(sizeof(T*) * (num_layer - 1)); //First layer needs no bias
        for (uint32_t layer = 0; layer != num_layer - 1; layer++) {
            bias[layer] = raw_mem;
            raw_mem += layer_size[layer + 1];
        }
    }

    /* 
        Computes either the matrix multiplication C=trans_A(A)*trans_B(B) or C+=trans_A(A)*trans_B(B).
        All matrices (A,B,C) have to be stored column major.

        @param A: Left  factor of matrix product that will be computed.
        @param B: Right factor of matrix product that will be computed
        @param C: Where to store the result of the multiplication
        @param trans_A: Whether to transpose A before multiplication (swap height and width)
        @param trans_B: Whether to transpose B before multiplication (swap height and width)
        @param overwrite: If this is true, we overwrite C with the result of the matrix multiplication. If it is false, we add the result to the data in C
        @param y1, x1, x2: trans_A(A) has size y1*x1. trans_B(B) has size x1*x2. C has size y1*y2.
    */
    template<bool trans_A, bool trans_B, bool overwrite>
    void matmul(T* A, T* B, T* C, uint32_t y1, uint32_t x1, uint32_t x2) {
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, half>::value, "[Error] Matrix multiplication is not supported with this type!");
        
        if constexpr (std::is_same<T, float>::value)
            cublasSgemm( cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, (float*) &cublasConst[1], (float*) A, trans_A?x1:y1, (float*) B, trans_B?x2:x1, (float*)&cublasConst[!overwrite], (float*)C, y1);
        if constexpr (std::is_same<T, double>::value)
            cublasDgemm( cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, (double*)&cublasConst[1], (double*)A, trans_A?x1:y1, (double*)B, trans_B?x2:x1, (double*)&cublasConst[!overwrite], (double*)C, y1);
        if constexpr (std::is_same<T, half>::value)
            cublasGemmEx(cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, (half*)  &cublasConst[1], (half*)  A, CUDA_R_16F, trans_A?x1:y1, (half*)B, CUDA_R_16F, trans_B?x2:x1, (half*)&cublasConst[!overwrite], (half*)C, CUDA_R_16F, y1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    
    inline void forward_weight_mul(uint32_t ind, T* in) {//output[ind] += weights[ind] * in.
        matmul<false, false, false>(weights[ind], in, output[ind], layer_size[ind + 1], layer_size[ind], batch_size);
    }
    inline void backward_delta_mul(uint32_t ind, T* out) {//out = weights[ind+1]^T * output[ind+1]
        matmul<true, false, true>(weights[ind + 1, output[ind + 1], out, layer_size[ind + 1], layer_size[ind + 2], batch_size);
    }
    inline void set_bias(uint32_t ind) {//Copy bias[i] to output[i] for all samples in batch
        uint32_t size_out = layer_size[ind + 1] * batch_size;
        set_repeating<T><<<LAUNCH_PARAM(size_out)>>>(output[ind], bias[ind], size_out, layer_size[ind + 1]);
    }

    /*
        Applies an activation function to a layer.

        @param a: Specifies, which activation function to use
        @param i: Specifies the layer (output[i], that means layer[i+1])
    */
    template<Activation a> void activate(uint32_t i) {
        static_assert(a == Activation::RELU || a == Activation::SIGMOID || a == Activation::SOFTPLUS || a == Activation::SOFTMAX, "[Error] Unknown activation");

        uint32_t s = layer_size[i + 1] * batch_size;
        if constexpr (a == Activation::RELU)     relu<T>(output[i], s, LAUNCH_PARAM(s));
        if constexpr (a == Activation::SIGMOID)  sigmoid<T>(output[i], s, LAUNCH_PARAM(s));
        if constexpr (a == Activation::SOFTPLUS) softplus<T>(output[i], s, LAUNCH_PARAM(s));
        if constexpr (a == Activation::SOFTMAX) { //Because the output of softmax for every neuron depends on all neurons in the same batch, we need to apply softmax for every batch seperatly
            T* sta = output[i];
            for (int b = 0; b != batch_size; b++, sta += layer_size[i + 1])
                softmax<T>(sta, layer_size[i + 1], LAUNCH_PARAM(layer_size[i + 1]));
        }
    }
    /*
        When f is the activation function, it replaces f(x) with f'(x).

        @param i: Specifies the layer (output[i], that means layer[i+1])
    */
    inline void activate_derivative(uint32_t i) {//Replaces f(x) with f'(x)
        static_assert(ACT == Activation::RELU || ACT == Activation::SIGMOID || ACT == Activation::SOFTPLUS, "[Error] Unknown activation for hidden layer");
        
        uint32_t s = layer_size[i + 1] * batch_size;
        if constexpr (ACT == Activation::RELU)     relu_deriv<T>(output[i], s, LAUNCH_PARAM(s));
        if constexpr (ACT == Activation::SIGMOID)  sigmoid_deriv<T>(output[i], s, LAUNCH_PARAM(s));
        if constexpr (ACT == Activation::SOFTPLUS) softplus_deriv<T>(output[i], s, LAUNCH_PARAM(s));
    }
    /*
        When f is the activation function, it computes f'(x) from f(x) and multiplies the result elementwise to an array.

        @param in: Input pointer which stores f(x)
        @param o:  Output array which to elementwise multiply f'(x) to
        @param s: Number of elements in "in" and "o"
        @param negate: If this is ture, multiply -f'(x) to o instead of f'(x)
    */
    template<bool negate> void activate_derivative_mul(T* in, T* o, uint32_t s) {//*o *= f'(in)
        static_assert(ACT == Activation::RELU || ACT == Activation::SIGMOID || ACT == Activation::SOFTPLUS, "Unknown activation for hidden layer");

        if constexpr (ACT == Activation::RELU)     relu_deriv_mul<T, negate>(in, o, s, LAUNCH_PARAM(s));
        if constexpr (ACT == Activation::SIGMOID)  sigmoid_deriv_mul<T, negate>(in, o, s, LAUNCH_PARAM(s));
        if constexpr (ACT == Activation::SOFTPLUS) softplus_deriv_mul<T, negate>(in, o, s, LAUNCH_PARAM(s));
    }

    void optimizer_apply(uint32_t l) {//Changes weights[l] according to lambdas in output[l] and data in output[l - 1]. Uses optimizer specified in opt_var
        //https://github.com/JRC1995/DemonRangerOptimizer/blob/master/optimizers.py
        switch (opt_var.opt_t) {
        case Optimizer_Type::SGD:
            sgd<T>(weights[l], output[l], output[l - 1], learning_factor, layer_size[l + 1], layer_size[l], batch_size);
            break;
        case DEMON_ADAM:
            T temp = (T)1 - (opt_var.t / opt_var.T);
            opt_var.beta1 = opt_var.betaInit * temp / (((T)1 - opt_var.betaInit) + opt_var.betaInit * temp);
        case ADAM:
            adam<T>(weights[l], output[l], output[l - 1], opt_momentum_buf[0][l], opt_momentum_buf[1][l], -learning_factor, opt_var.beta1, opt_var.beta2, (T)0.00000001, layer_size[l + 1], layer_size[l], batch_size, opt_var.t);
        }
    }

public:
    /*
        Usage: 
        1.: Constructor
        2.: set_dataset
        3.: set_optimizer
        4.: train
    */

    MultiLayerPerceptron(uint32_t n, uint32_t* l, bool smart_weights = true)
        : num_layer(n), layer_size(l)
    {
        //1.: Allocate memory
        allocate();

        //2.: Initialize memory
        if (smart_weights) {
            for (uint32_t layer = 0; layer != num_layer - 1; layer++) {
                set_random<T, true>(weights[layer], layer_size[layer], layer_size[layer + 1]);
            }
        }
        else {
            for (uint32_t layer = 0; layer != num_layer - 1; layer++) {
                set_random<T, false>(weights[layer], layer_size[layer], layer_size[layer + 1]);
            }
        }
        
        cudaMemsetAsync(bias[0], 0, num_neurons * sizeof(T));
    }
    /*
        Initializes with the values from a file.
        Layout of file:
        |typeid(T)|num_layer|----layer_size-----|------weights--------|-----bias------|
    */
    MultiLayerPerceptron(char* file) {
        //1.: Open file
        int fd = open(file, O_RDONLY);

        //2.: Check, that computation types of file and object match up
        std::type_info* type;
        read(fd, type, sizeof(std::type_info));
        assert(*type == typeid(T));

        //3.: Get num_layer and layer_size
        read(fd, &num_layer, sizeof(uint32_t));
        layer_size = (uint32_t*)malloc(sizeof(uint32_t) * num_layer);
        read(fd, layer_size, sizeof(uint32_t) * num_layer);

        //4.: Allocate memory
        allocate(num_neurons, num_weights);

        //5.: Initialize memory
        read(fd, weights[0], (num_weights + num_neurons) * sizeof(T)); //Exploites, that the weights and bias lay packed in this order in memory

        //6.: Close file
        close(fd);
    }
    void store_parameters(char* file) {//Stores the object, so that it can be re-initialized by MultiLayerPerceptron
        //0.:File stuff
        int fd = open(file, O_WRONLY);

        write(fd, &(typeid(T)), sizeof(std::type_info));     //1.: Type
        write(fd, &(num_layers), sizeof(uint32_t));          //2.: num_layers
        write(fd, layer_size, sizeof(uint32_t) * num_layer); //3.: layer_size
        write(fd, weights[0], sizeof(T) * num_weights);      //4.: weights
        write(fd, bias[0]   , sizeof(T) * num_neurons);      //5.: bias
    }

    uint32_t get_batchSize() {
        return batch_size;
    }
    void set_batchSize(uint32_t batch_size_) {//Sets the new batch size and allocates the output memory and opt_buf   TODO
        //1.: Set batch_size in this and dataset
        if (batch_size_ != dataset.batch_size) {      //Called not for first time?
            if (!dataset.set_batch_size(batch_size_)) //Is resize possible?
                return;

            //Free old output
            free(output[0]);                          
            free(opt_buf);
        }
        batch_size = batch_size_;

        //2.: Allocate output and get max_output
        uint32_t max_output = 0;
        cudaMalloc(&output[0], sizeof(T) * num_neurons * batch_size);
        for (uint32_t layer = 1; layer != num_layer - 1; layer++) {
            uint32_t size = layer_size[layer] * batch_size;
            output[layer] = output[layer - 1] + size;

            if (size > max_output)
                max_output = size;
        }

        //3.: Allocate opt_buf
        cudaMalloc(&opt_buf, sizeof(T) * max_output);
    }
    OptVariables<T> get_optimizer() {
        return opt_var;
    }
    void set_optimizer(OptVariables<T> opt_var_) {
        if (opt_var.opt_t == Optimizer_Type::SGD && opt_var_.opt_t != Optimizer_Type::SGD) { //Allocate opt_momentum_buf
            //Allocate memory
            T* raw_mem;
            cudaMalloc(raw_mem, sizeof(T) * num_weights * 2);
            
            cudaMallocHost(&opt_momentum_buf[0], sizeof(T*) * (num_layer - 1)); //First layer needs no weights
            cudaMallocHost(&opt_momentum_buf[1], sizeof(T*) * (num_layer - 1)); //First layer needs no weights

            opt_momentum_buf[0][0] = raw_mem;
            opt_momentum_buf[1][0] = raw_mem + sizeof(T) * num_weights;

            for (uint32_t layer = 1; layer != num_layer - 1; layer++) {
                uint32_t off = layer_size[layer - 1] * layer_size[layer];
                opt_momentum_buf[0][layer] = opt_momentum_buf[0][layer - 1] + off;
                opt_momentum_buf[1][layer] = opt_momentum_buf[1][layer - 1] + off;
            }
        }
        if (opt_var.opt_t != Optimizer_Type::SGD && opt_var_.opt_t == Optimizer_Type::SGD) {  //Free opt_momentum_buf
            cudaFree(opt_momentum_buf[0][0]);
            cudaFreeHost(opt_momentum_buf[0]);
            cudaFreeHost(opt_momentum_buf[1]);
        }
        opt_var = opt_var_;
    }
    DatasetHandler<T> get_dataset() {
        return dataset;
    }
    void set_dataset(DatasetHandler<T> dataset_) {//Sets dataset and batch_size
        dataset = dataset_;
        set_batchSize(dataset.batch_size);
    }
   
    T* get_cur_in() {
        return cur_in;
    }
    T* get_cur_out() {
        return cur_out;
    }
    void set_cur_in(T* p) { //Shoud only be used for manual changes and not for iteration over training or validation samples
        cur_in = p;
    }
    void set_cur_out(T* p) {//Shoud only be used for manual changes and not for iteration over training or validation samples
        cur_out = p;
    }
    template<bool training> bool nextBatch() { //Sets cur_in and cur_out to new batch
        dataset.advance<training>(cur_in, cur_out);              //This loads new data to gpu, thus this has overhead. Synchronize after, not before this!!!
    }
    
    /*
        Call-pipeline for training:
        1.: Initialize this class.
        2.: set_batchSize, set_optimizer, set_dataset
        
        3.1.: train

        3.2.: set_cur_in and set_cur_out (for example using nextBatch)
        3.3.: forward_propagate
        3.4.: compute_ll_delta or set_ll_delta
        3.5.: backpropagate
        3.6.: Optionally: opt_var.t++
    */
    void train(uint32_t num_batches) {
        for (int batch = 0; batch != num_batches; batch++) {
            nextBatch<true>(); //Insert gpu synchronise after, not before!

            forward_propagate();
            compute_ll_delta();
            backward_propagate();

            opt_var.t++;
        }
    }
    /*
        Does forward propagation of data in in_data
    */
    void forward_propagate() {
        set_bias(0);
        forward_weight_mul(0, cur_in);
        activate<ACT>(0);
        for (int l = 1; l < num_layer - 2; l++) {
            set_bias(l);
            forward_weight_mul(l, output[l - 1]);
            activate<ACT>(l);
        }
        set_bias(num_layer - 2);
        forward_weight_mul(num_layer - 2, output[num_layer - 3]);
        activate<Activation::SOFTMAX>(num_layer - 2);
    }
    /*
        Computes the deltas for the last layer based on cur_out.
        Has to be called right after forward_propagate
    */
    void compute_ll_delta() {
        //Last layer
        softmax_cross_entropy_deriv(output[num_layer - 2], cur_out, layer_size[num_layer - 2], LAUNCH_PARAM(layer_size[num_layer - 2]));
    }
    /*
        Set the output for the last layer to delta. 
        @param delta can be located either on host or device. Size is layer_size[num_layer - 1] * batch_size
        Has to be called right after forward_propagate
    */
    void set_ll_delta(T* delta) {
        cudaMemcpy(output[num_layer - 2], delta, layer_size[num_layer - 1] * batch_size * sizeof(T), cudaMemcpyDefault)
    }
    /*
        Does Backpropagation from layer before output layer to first hidden layer. 
        Has to be called right after compute_ll_delta or set_ll_delta
    */
    void backward_propagate() {
        //Hidden Layers
        for (int l = num_layer - 2; l > 0; l--) {
            backward_delta_mul(l, opt_buf);                                   //Compute delta for output[l-1]
            optimizer_apply(l);                                               //Apply weight changes to weights[l] between output[l-1] and output[l]
            activate_derivative(l - 1);                                       //Get the derivative of output[l-1]
            uint32_t s = layer_size[l] * batch_size;                          //Size of output[l - 1]
            multiplyElementwise(output[l - 1], opt_buf, s, LAUNCH_PARAM(s));  //Multiply these derivatives with the result of backward_delta_mul in opt_buf
        }

        //Input Layer
        optimizer_apply(0);
    }
    /*
        Same as before, but returns delta for input. Usefull for concatenating networks
        @param out: Where deltas of input layer will be stored
        @param negate: if true, will return negative deltas
        @param apply: if false, will not update weights
    */
    template<bool negate, bool apply>
    void backward_propagate(T* out) {
        set_cublasMode(CublasMode::OVERWRITE);

        //Hidden Layers
        for (int l = num_layer - 2; l > 0; l--) {
            backward_delta_mul(l, opt_buf);                                   //Compute delta for output[l-1]
            if constexpr (apply) optimizer_apply(l);                          //Apply weight changes to weights[l] between output[l-1] and output[l]
            activate_derivative(l - 1);                                       //Get the derivative of output[l-1]
            uint32_t s = layer_size[l] * batch_size;                          //Size of output[l - 1]
            multiplyElementwise(output[l - 1], opt_buf, s, LAUNCH_PARAM(s));  //Multiply these derivatives with the result of backward_delta_mul in opt_buf
        }

        //Input Layer
        backward_delta_mul(0, out);
        if constexpr (apply) optimizer_apply(0);
        activate_derivative_mul<negate>(cur_in, out, layer_size[0] * batch_size);
    }

    /*
        Computed the loss average. 
        Can be called anytime during training (just not between forward_propagate, compute_ll_delta or set_ll_delta, backward_propagate and get_delta_input)
    */
    T get_validation_loss() {
        T loss_acc = 0;

        uint32_t size_out = layer_size[num_layer - 2];

        uint32_t i = 0;
        while (nextBatch<false>()) {
            i++;
            forward_propagate();
            loss_acc += cross_entropy_loss(output[num_layer - 2], cur_out, size_out, LAUNCH_PARAM(size_out));
        }
        return loss_acc / i;
    }
    void get_output(T* in, T* out) {
        uint32_t batch_size_tmp = batch_size;
        batch_size = 1;                       //Do not call set_batchSize to avoid unnecessary reallocations
        cur_in     = in;
        cur_out    = out;

        forward_propagate();

        batch_size = batch_size_tmp;          //Do not call set_batchSize to avoid unnecessary reallocations
        cudaMemcpy(out, output[num_layer - 2], layer_size[num_layer - 1], cudaMemcpyDefault);
    }
};

//=============================================
//==================|MAIN|=====================
//=============================================

int main()
{
    CUBLAS_ERROR(cublasCreate(&cublas_handle));
    //Logging
    cublasSetAtomicsMode(cublas_handle, CUBLAS_ATOMICS_ALLOWED);
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    //Block dimension

    //Set cuda stream
    //set vector, matrixes (async), pointer mode

    //Initialize random
    Random::init_rand();


    //Maybe have to change indexing mode to start with 0 instead of 1 using #define IDX2C(i,j,ld) (((j)*(ld))+(i))



    CHECK_CUDA_ERROR();

    CUBLAS_ERROR(cublasDestroy(cublas_handle));

    return 0;
}

//OLD BENCHMARKS
#if 0
#define MAX_N (1<<18)
float* mem;
cudaMalloc(&mem, sizeof(float) * 2 * MAX_N);

float time_sums[4] = { 0.f, 0.f, 0.f, 0.f };
int num_best[4] = { 0, 0, 0, 0 };
for (int N = 32; N != MAX_N; N++) {
    num_best[BENCHMARK(mem, mem + N, N, LAUNCH_PARAM(N), time_sums)]++;

    if (N % 512 == 0)
        printf("%d %d %d %d %f %f %f %f\n", num_best[0], num_best[1], num_best[2], num_best[3], time_sums[0], time_sums[1], time_sums[2], time_sums[3]);
}







//Memory
float* mem;
cudaMalloc(&mem, 2ull * sizeof(float) * MAX_N);

float* global_timing = (float*)malloc(sizeof(float) * (MAX_N + 1)); //Fastest Time
int* global_blocks = (int*)malloc(sizeof(int) * (MAX_N + 1)); //Block size for fastest time
int* global_grids = (int*)malloc(sizeof(int) * (MAX_N + 1)); //Grid size for fastest time
int* global_algos = (int*)malloc(sizeof(int) * (MAX_N + 1)); //Algorithm for fastest time
for (int i = 0; i != 32; i++) {
    global_timing[i] = 0.03f;                          //Not right
    global_blocks[i] = 0;
    global_grids[i] = 0;
    global_algos[i] = -1;
}

//Timing stuff
float time;
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0);

//Debugging stuff
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
std::chrono::high_resolution_clock::time_point sta[20], sto;
#pragma clang diagnostic pop

int count_algs[9] = { 0,0,0,0,0,0,0,0,0 };
for (int32_t N_ = 8192; N_ <= MAX_N; N_++) {
    uint32_t N = (uint32_t)N_;
    global_timing[N] = 999999999999999999999999999999999.f;
    for (uint32_t g = 1; g <= min(2560u, 2 * N / 32); g++) {
        DO_BENCH(N, g, global_timing, global_blocks, global_grids, global_algos, mem);
    }

    printf("Size: %d \t|Red_Algo: %d \t|Mul_Algo: %d \t|Atomics: %d \t|Block_Size: %d \t|Grid_Size: %d \t|Time: %f\n",
        N, UNHASH_ALGO(global_algos[N]), global_blocks[N], global_grids[N], global_timing[N]);

    count_algs[UNHASH_RED(global_algos[N]) * 3 + UNHASH_MUL(global_algos[N])]++;
    if (N % 512 == 0) {
        for (int i = 0; i != 9; i++)
            printf("%d ", count_algs[i]);
        printf("\n");
    }
}
#endif
