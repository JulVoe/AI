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

#include <cmath>

#include <inttypes.h>
#include <stdio.h>
#include <utility>
#include <vector>

#include "util.cpp"
#include "Dataset.cpp"

/*
Notation conventions:
 - A matrix has size height*width.
 - layer[0] is the input layer
 - weight[i] are the weights between layer[i] and layer[i+1]

 Memory handling:
 Each class is responsible for its parameters and should set its memory itself recursivly. "state" memory needs to be reallocated when the batch size changes,
 "other" memory does not need to be reallocated. The number of nodes needed for the execution graph also shall not change when the batch size changes. Every
 class has to supply a "getMemoryRequirements", "setMem", "initMem" method. Before those, there should be one method that sets the internal parameters. Internal
 parameters create redundancies but it keeps the code clean as they do not need to be passed with each call. This way, "getMemoryRequirements" has only out 
 parameters. These functions are used as they allow the parent class to batch allocations. This is faster, saves memory and reduces fragementation (despite
 for the obvious reason, also because cudaMalloc return 256bit aligned memory (see "Memory management" in "util.cpp")). It also enables the future capability
 of serialization by just writing the complete memory to the file by the parent in one call.

Indirection pointers:
The input pointer changes every batch. Thus, it should not be captured in the forward propagation graph. As a consequence, one has to use an indirection
pointer: the graph stores a pointer which is dereferenced for every batch. This results in the true pointer pointing to the data. This works, because the
memory of the indirection pointer can be cahnged between runs and when it is dereferenced each batch, it always gives back a different pointer pointing to
the current data. However, either all or none of the arguments have to be indirection pointers. Thus, the layer after the input layer has to use indirection
pointers for its own weights and state too. The correct graph methods have names ending on "Indirection".

Serialization / Deserialization rules:
 - Each class that has subclasses must provide a method that constructs a specific sublclass at a specific memory location.
 - Each serialization method just takes a FILE* and writes the contents of the class to the file, so it can be deserialized and all its content can be restored.
 - Each deserialization method has to be a static method that has a FILE* and a memory pointer. It constructs a object using the contents of the file at
   the specified location. This way, derived classes can be handled correctly (see next point)
 - Serialization and Deserialization methods are always implemented in the base class. When dealing with derived classes, one always first serializes a
   unique subclass id and after that the contents of the class. When deserializing, one reads in this identifier first and then uses method desribed in
   point 1 to create an object of the correct type. After that, one reads in the serialized information from the file and uses this to correctly set the
   member variables.

Cuda Graphs:
There is no real documentation of how cuda graphs work. This is what I found out through some tests:
 - After a node was added to a graph, a change of the parameters used for it does not change the operation. The storage for the parameters can thus be freed 
   after the node was generated.                 (from experiment)
 - There is no need to store the cudaGraphNode_t of a graph. The generated objects can all be freed. The information on the nodes is not stored in them.
                                                 (from experiment)
 - A graph does not store pointers to its nodes. Instead, cudaNode_t probably store a pointer to the internal structure of the graph. Thus, copying a node is
   not a problem and nodes can be moved without a problem. Thus, the nodes can bes tores in a vector as moving doesn't break anything. They can still be used for
   "cudaGraphAddDependencies" without a problem. (from experiment)
 - As "cudaGraphAdd*Node" takes graph by value, a cudaGraph_t object does not contain any information on nodes itself. Together with the points before, I would
   guess that there is a private object on the heap created by "cudaCreateGraph" that contains all the information on the graph while "cudaGraph_t" merely holds
   a pointer to this object an "cudaGraphNode_t" holds a pointer to information on itself saved in an array of the private object. Nonetheless, a graph can surely
   be moved without any nasty sideeffects.
*/

//TODO: Softmax kernel (needed as a combination of transform/reduce operation would lead to a dynamic amount of calls)
//TODO: Loss
//TODO: Destructors
//TODO: Adam. Layer types
//TODO: Learn activation parameters
//TODO: Implement regularization and decay in optimizer (seperate for weights and biases as according to "Bag of tricks for Image classification using CNNs", biases should not be regularized)
//TODO: Dataset must load validation samples sequencially
//TODO: Measure training loss to detect overfitting
//TODO: Implement serialization and other derived class methods in LRScheduler and Loss class
//TODO: constexpr the "getMemoryRequirements" functions. Also __restrict__ and const in other functions
//TODO: Expand Scheduler debugging with: Show lr, loss, episode, timings, in-&output, model, occupiancy ...
//TODO: Change batchSize in Scheduler, get Input_Layer from every layer, NetworkBuilder takes argument whether first layer is the dataset or not
//TODO: Check launch parameters and smem
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

/*
    Same as above, with one distinction: It does not safe C itself, but a pointer to it which is only dereferenced at runtime.
    Thus, the *C can change and point to different locations between different runs without changing anything
*/
template<typename T, bool trans_A, bool trans_B, bool overwrite> //TODO:cublasConst
inline cudaGraph_t getMatmulGraphIndirection(T** A, T** B, T** C, uint32_t y1, uint32_t x1, uint32_t x2, cudaStream_t captureStream) {
    //0.: Make sure T is a recognized type
    static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, half>::value, "[Error] Matrix multiplication is not supported with this type!");

    //1.: Start stream capture
    cudaGraph_t capGraph;
    cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeThreadLocal);

    //2.: Enqueue cublas kernel
    if constexpr (std::is_same<T, float>::value)
        cublasSgemmBatched(cublas_handle, trans_A ? CUBLAS_OP_T : CUBLAS_OP_N, trans_B ? CUBLAS_OP_T : CUBLAS_OP_N, y1, x2, x1, (float*)&cublasConst[1], (float**)A, trans_A ? x1 : y1, (float**)B, trans_B ? x2 : x1, (float*)&cublasConst[!overwrite], (float**)C, y1, 1);
    if constexpr (std::is_same<T, double>::value)
        cublasDgemmBatched(cublas_handle, trans_A ? CUBLAS_OP_T : CUBLAS_OP_N, trans_B ? CUBLAS_OP_T : CUBLAS_OP_N, y1, x2, x1, (double*)&cublasConst[1], (double**)A, trans_A ? x1 : y1, (double**)B, trans_B ? x2 : x1, (double*)&cublasConst[!overwrite], (double**)C, y1, 1);
    if constexpr (std::is_same<T, half>::value)
        cublasGemmBatchedEx(cublas_handle, trans_A ? CUBLAS_OP_T : CUBLAS_OP_N, trans_B ? CUBLAS_OP_T : CUBLAS_OP_N, y1, x2, x1, (half*)&cublasConst[1], (half**)A, CUDA_R_16F, trans_A ? x1 : y1, (half**)B, CUDA_R_16F, trans_B ? x2 : x1, (half*)&cublasConst[!overwrite], (half**)C, CUDA_R_16F, y1, 1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

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
inline void addElementwiseMultNode(cudaGraph_t graph, T* A, T* B, uint32_t len, cudaGraphNode_t& node){
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
    cudaGraphAddKernelNode(&node, graph, nullptr, 0, &elemMultParam);
}

//===================================================
//==================|kernell.cu|=====================
//===================================================

template<typename T, bool safe>
__global__ void softmax(T* mem, uint32_t outStateSize, uint32_t batch_size) {

}

//temp is device pointer
template<typename T, bool safe>
__global__ void softmaxTemp(T* mem, uint32_t outStateSize, uint32_t batch_size, T* temp) {

}

template<typename T, bool safe>
__global__ void softmax_deriv(T* mem, T* out, uint32_t outStateSize, uint32_t batch_size) {

}

//temp is device pointer 
template<typename T, bool safe>
__global__ void softmaxTemp_deriv(T* mem, T* out, uint32_t outStateSize, uint32_t batch_size, T* temp) {

}

template<typename T, typename F>
__global__ void transform_indirection(T* in1, T** in2_, uint32_t n, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T* in2 = *in2_;

    for (int i = idx; i < n / 4; i += blockDim.x * gridDim.x) {
        var4<T> val = reinterpret_cast<var4<T>*>(in1)[i];
        var4<T> val = reinterpret_cast<var4<T>*>(in2)[i];
        val1.a = f(val1.a, val2.a);
        val1.b = f(val1.b, val2.b);
        val1.c = f(val1.c, val2.c);
        val1.d = f(val1.d, val2.d);
        reinterpret_cast<var4<T>*>(in1)[i] = val1;
    }
    int i = idx + n / 4 * 4;
    if (i < n)
        in1[i] = f(in1[i], in2[i]);
}

template<typename T, typename F, DIVISIBILITY N_divisible_blocksize, DIVISIBILITY N_divisible_32, bool write>
__global__ void transform_reduce_indirection(T* in1, T** in2_, T* out, uint32_t n, F f) {
    //0.: Compute unknown divisibilities
    bool N_divisible_blocksize_, N_divisible_32_;
    if constexpr (N_divisible_blocksize == DIVISIBILITY::UNKNOWN)
        N_divisible_blocksize_ = ((n % blockDim.x) == 0);
    if constexpr (N_divisible_32 == DIVISIBILITY::UNKNOWN)
        N_divisible_32_ = ((n % 32) == 0);

    T* in2 = *in2_;

    //1.: Initialize variables
    T sum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //2.: Reduce multiple elements per thread
    for (int i = idx; i < n / 4; i += blockDim.x * gridDim.x) {
        var4<T> val1 = reinterpret_cast<var4<T>*>(in1)[i];
        var4<T> val2 = reinterpret_cast<var4<T>*>(in2)[i];
        val1.a = f(val1.a, val2.a);
        val1.b = f(val1.b, val2.b);
        val1.c = f(val1.c, val2.c);
        val1.d = f(val1.d, val2.d);
        sum += (val1.a + val1.b) + (val1.c + val1.d);
        if constexpr (write) { reinterpret_cast<var4<T>*>(in1)[i] = val1; }
    }
    int i = idx + n / 4 * 4;
    if (i < n) {
        T tmp = f(in1[i], in2[i]);
        sum += tmp;
        if constexpr (write) in1[i] = tmp;
    }

    //3.: Store results
    if constexpr (N_divisible_32 == DIVISIBILITY::DIVISIBLE) {
        for (int offset = 32 / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

        if ((threadIdx.x & 31) == 0) atomicAdd(out, sum);
    }
    if constexpr (N_divisible_32 == DIVISIBILITY::NOT_DIVISIBLE) {
        if (((threadIdx.x + blockIdx.x * blockDim.x) & ~(0b11111)) + 32 < n) {
            for (int offset = 32 / 2; offset > 0; offset /= 2)
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

            if ((threadIdx.x & 31) == 0) atomicAdd(out, sum);
        }
        else {
            atomicAdd(out, sum);
        }
    }
    if constexpr (N_divisible_32 == DIVISIBILITY::UNKNOWN) {
        if (N_divisible_32_ || ((threadIdx.x + blockIdx.x * blockDim.x) & ~(0b11111)) + 32 < n) {
            for (int offset = 32 / 2; offset > 0; offset /= 2)
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

            if ((threadIdx.x & 31) == 0) atomicAdd(out, sum);
        }
        else {
            atomicAdd(out, sum);
        }
    }
}

template<typename T, typename L>
__global__ void sgdMul(T* weigths, T* delta, T* in, L* lrate, uint32_t w_y, uint32_t w_x, uint32_t batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < w_y * w_x * batch_size) {
        int b = idx % batch_size;
        idx /= batch_size;
        int y = idx % w_y;
        int x = idx / w_y;

        atomicAdd(&weigths[y + x * w_y], -*lrate * in[b * w_x + x] * delta[b * w_y + y]]));
    } //index of "weigths" is "idx", compiler will optimize it away
}

//Assumes that "in" only stores "1" (which is the case for the input of biases). Thus, the parameter can be ommited. Therefore only one size paramter is needed as well an "w_x" can be omitted. "w_y" is now th esize of "weights".
template<typename T, typename L>
__global__ void sgdSimple(T* weigths, T* delta, L* lrate, uint32_t w_y, uint32_t batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < w_y * batch_size) {
        int b = idx % batch_size;
        idx /= batch_size;

        atomicAdd(&weigths[idx], -*lrate * delta[b * w_y + idx]]));
    } //index of "weigths" is "idx", compiler will optimize it away
}

//===================================================
//==================|Optimizers|=====================
//===================================================

enum OPTIMIZER_TYPE: uint32_t {SGD=0, ADAM=1};   //DON'T CHANGE THESE VALUES AS IT WILL BREAK OLD CHECKPOINT FILES!
template<typename T, typename L = T>              //T is type of data, L is type of learning rates
class Optimizer {
protected:
    uint64_t optimizables;                        //Number of T's this optimizer optimizes
    
    T* optBuf;                                    //Buffer that stores information on every variable it optimizes (e.g. momentum). Optimizer has no authority over layout.
    L* learningRates;                             //Learning rates (e.g. momentum decay, ...).    Variable names: alpha=learning rate; beta1,beta2=first and second order momentum decay.

    /*
        Returns the memory requirements of every buffer to the given variables.

        @param optBuf_req: Out parameter. The number of bytes on the gpu needed to store the optimization buffer (momentum, ...)
        @param lRates_req: Out parameter. The number of bytes on the gpu needed to store the learning rates (also momentum decay, ...)
    */
    virtual void getOwnMemoryRequirements(MemoryRequirement& optBuf_req, MemoryRequirement lRates_req);

public:
    /*
        Call methods in the order they are definded:
        1.: setNumOptimizables
        2.: getMemoryRequirements
        3.: setMem
        4.: initMem
        5.: addNodeWeights and addNodeBias
    */

    /*
        Updates the member variable "optimizables" 
    */
    void setNumOptimizables(uint64_t num_optimizables) { optimizables = num_optimizables; }

    /*
        Returns the aggregated memory requirements of this optimizer to the given variables.

        @param other_requirement: Out parameter. The number of "other" bytes on the gpu needed for this optimizer
    */
    void getMemoryRequirements(MemoryRequirement& other_requirement) {
        MemoryRequirement optBuf_req, lRates_req;
        getOwnMemoryRequirements(optBuf_req, lRates_req);

        other_requirement = optBuf_req + lRates_req;
    } 
    /*
        Sets the internal memory pointers of the optimizer. The passed pointer will be set to the first byte after the by this optimizer used memory 
        region. The pointer does not need to be aligned as first they will be padded so they satisfy the alignment requirement and then incrementen by the 
        space requirement of this optimizer, as specified in "getMemoryRequirement".

        @param mem: Pointer to enough free memory that this optimzier will use to store "optBuf" and "learningRates". Will be set after the region used for this.
    */
    void setMem(uint8_t*& mem) {
        //1.: Get memory requirements
        MemoryRequirement other_requirement, lrates_requirement;
        getOwnMemoryRequirements(buffer_requirement, lrates_requirement);

        //2.: Set "optBuf"
        mem = align_pointer_unsafe(mem, buffer_requirement.alignment);
        optBuf = (T*)mem;
        mem += buffer_requirement.num_bytes;

        //3.: Set "learningRates"
        mem = align_pointer_unsafe(mem, lrates_requirement.alignment);
        learningRates = (L*)learningRates;
        mem += lrates_requirement.num_bytes;
    }
    /*
        Initializes the optimization buffer
    */
    virtual void initMem();
    
    /*
        Adds a node that optimizes a weights matrix multiplication to a graph.
        (Optimizes matrix "mem" of shape y*x. Gradient is product of respective elements from input, a vector of lenght x, with delta, a vector of lenght y.)
        ("mem" correspondes with weights, "input" with layerBefore->state and "delta" with the derivative of the loss with respect to state before activation)

        @param mem       : The memory that should be optimized (of the weights in this case). Matrix of shape y*x
        @param index     : The first element of "mem" is the index-ed value that this optimizer optimizes (used as a index to "optBuf"). Will be updated
        @param delta     : The delta of the output of this layer before acitivation to the loss. Vector of length y.              =Δo_n
        @param input     : The output of the layer before. Vector of length x.                                                    =layerBefore->state
        @param y         : The y-dimension of "mem"                                                                               =height weights
        @param x         : The x-dimension of "mem"                                                                               =width weights
        @param batch_size: The used batch size
        @param graph     : The graph the optimization node should be added to
        @param depMem    : The dependencies on "mem". Will be updated
        @param depDelta  : The dependencies on "delta". Will be updated
        @param depInput  : The dependencies on "input". Will be updated
    */
    virtual void addNodeWeights(T* mem, uint64_t& index, T* delta, T* input, uint32_t y, uint32_t x, uint32_t batch_size, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput);
    /*
        Adds a node that optimizes a bias addition to a graph.
        (Optimizes vector "mem" of lenght y. Gradient is repective element of "delta", another vector of lenght "y".)
        ("mem" corresponds to bias and "delta" to the derivative of the loss with respect to state before activation.)

        @param mem       : The memory that should be optimized (of the bias in this case). Vector of length y
        @param index     : The first element of "mem" is the index-ed value that this optimizer optimizes (used as a index to "optBuf"). Will be updated
        @param delta     : The delta of the output of this layer before acitivation to the loss. Vector of length y               =Δo_n
        @param y         : The length of "mem"                                                                                    =number of biases (in b_n)
        @param batch_size: The used batch size
        @param graph     : The graph the optimization node should be added to
        @param depMem    : The dependencies on "mem". Will be updated
        @param depDelta  : The dependencies on "delta". Will be updated
    */
    virtual void addNodeBias(T* mem, uint64_t& index, T* delta, uint32_t y, uint32_t batch_size, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta);
    
    /*
        Sets learning rates. If some of the learning rates aren't actually used, the parameter is just ignored.

        @param alpha : The normal learning rate to use. Host pointer created by "cudaMallocHost"
        @param beta1 : The first order momentum decay to use. Host pointer created by "cudaMallocHost"
        @param beta2 : The second order momentum decay to use. Host pointer created by "cudaMallocHost"
        @param stream: The stream to use when copying the previous parameters to the gpu
    */
    virtual void setLR(L* alpha, L* beta1, L* beta2, cudaStream_t stream);

    /*
        Returns the sublcass of the optimizer.
    */
    virtual static OPTIMIZER_TYPE getOptimizerType();
    /*
        Constructs an optimizer of a specific subclass at a specific memory location

        @param ot : The derived class of the optimizer to create
        @param out: The memory location where the optimizer should be created
    */
    static void getOptimizerOfType(OPTIMIZER_TYPE ot, Optimizer<T>* out) {
        switch (ot) {
        case OPTIMIZER_TYPE::SGD:
            new (out) SGD_Optimizer<T>();
            break;
        case OPTIMIZER_TYPE::ADAM:
            new (out) Adam_Optimizer<T>();
            break;
        default:
            fprintf(stderr, "[ERROR] %llu is not a known optimizer type!", (uint64_t)ot);
            exit(-1);
        }
    }

    // /+=============+\
    // ||SERIALIZATION||
    // \+=============+/

    /*
        Serialization according to the serialization rules.
    */
    void serialize(FILE* file) {
        //1.: Write layer type
        OPTIMIZER_TYPE optimizer_type = getOptimizerType();
        fwrite(&optimizer_type, sizeof(optimizer_type), 1, file);

        //2.: Write variables
        fwrite(&optimizables, sizeof(optimizables), 1, file);

        //3.: Write memory
        MemoryRequirement buffer_requirement, lrates_requirement;
        getOwnMemoryRequirement(buffer_requirement, lrates_requirement);

        fwrite(&buffer_requirement.num_bytes, sizeof(buffer_requirement.num_bytes), 1, file);
        fwrite(optBuf, 1, buffer_requirement.num_bytes, file);

        fwrite(&lrates_requirement.num_bytes, sizeof(lrates_requirement.num_bytes), 1, file);
        fwrite(learningRates, 1, lrates_requirement.num_bytes, file);
    }
    /*
        Deserialization according to deserialization rules
    */
    static void deserialize(FILE* file, Optimizer<T, L>* out) {
        //1.: Create correct derived class
        OPTIMIZER_TYPE optimizer_type;
        fread(&optimizer_type, sizeof(OPTIMIZER_TYPE), 1, file);
        Optimizer<T, L>::getOptimizerOfType(optimizer_type, out);

        //2.: Read in variables
        fread(out->optimizables, sizeof(out->optimizables), 1, file);

        //3.: Get memory requirements
        MemoryRequirement buffer_requirement, lrates_requirement;
        out->getOwnMemoryRequirement(buffer_requirement, lrates_requirement);

        //4.: Read in memory
        uint64_t buffer_bytes, lrates_bytes;

        fread(&buffer_bytes, sizeof(buffer_bytes), 1, file);
        cudaMallocAligned(&out->optBuf, buffer_requirement);
        fread(out->optBuf, 1, buffer_bytes, file);

        fread(&lrates_bytes, sizeof(lrates_bytes), 1, file);
        cudaMallocAligned(&out->learningRates, lrates_requirement);
        fread(out->learningRates, 1, lrates_bytes, file);

        //5.: Check consistency
        if (buffer_requirement.num_bytes != buffer_bytes) {
            fprintf(stderr, "[ERROR] Trying to create a optimizer of type %llu with %llu state bytes, even though it requires %llu", (uint64_t)optimizer_type, (uint64_t)buffer_bytes, (uint64_t)buffer_requirement.num_bytes);
            exit(-1);
        }
        if (lrates_requirement.num_bytes != lrates_bytes) {
            fprintf(stderr, "[ERROR] Trying to create a optimizer of type %llu with %llu other bytes, even though it requires %llu", (uint64_t)optimizer_type, (uint64_t)lrates_bytes, (uint64_t)lrates_requirement.num_bytes);
            exit(-1);
        }
    }

    /*
        Intended for checkpoint files. Only writes information needed to recreate this optimizer.
        All alignment is ignored and padding is removed to save space. Dynamically allocates ram (as intermidiate for gpu->file transfer).

        If "data==false", inverse is "getOptimizerFromCompression". Otherwise, inverse is "initMemFromCompression".

        @param data: If false, only writes specifiers. If true, only writes data.
        @param file: The file to write the compressed data to.
    */
    template<bool data>
    void compress(FILE* file) {
        if constexpr (data) {
            //1.: Write "optBuf"
            MemoryRequirement optBuf_req, lRates_req;
            getOwnMemoryRequirements(optBuf_req, lRates_req);

            void* ram_buffer;
            cudaMallocHost(&ram_buffer, optBuf_req.num_bytes);
            cudaMemcpy(ram_buffer, optBuf, optBuf_req.num_bytes, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(NULL);
            fwrite(ram_buffer, 1, optBuf_req.num_bytes, file);
            cudaFreeHost(ram_buffer);
        }
        else {
            //1.: Write specifiers
            OPTIMIZER_TYPE ot = getOptimizerType();
            fwrite(&ot, sizeof(OPTIMIZER_TYPE), 1, file);
        }
    }
    /*
        Inverse of "compress<false>"
    */
    static void getOptimizerFromCompression(FILE* file, Layer<T>* out) {
        OPTIMIZER_TYPE ot;
        fread(&ot, sizeof(OPTIMIZER_TYPE), 1, file);
        getOptimizerOfType(ot, out);
    }
    /*
        Inverse of "compress<true>"
    */
    void initMemFromCompression(FILE* file) {
        MemoryRequirement optBuf_req, lRates_req;
        getOwnMemoryRequirements(optBuf_req, lRates_req);

        void* ram_buffer;
        cudaMallocHost(&ram_buffer, optBuf_req.num_bytes);
        fread(ram_buffer, 1, optBuf_req.num_bytes, file);
        cudaMemcpy(optBuf, ram_buffer, optBuf_req.num_bytes, cudaMemcpyHostToDevice);
        cudaStreamSynchronize(NULL);
        cudaFreeHost(ram_buffer);
    }
};

template<typename T, typename L = T>
class SGD_Optimizer : public Optimizer<T, L> {
    //learningRates = {alpha}
protected:
    virtual void getOwnMemoryRequirements(MemoryRequirement& optBuf_req, MemoryRequirement lRates_req) override {
        optBuf_Req = MemoryRequirement(0ull, 1u);                //No memory needed
        lRates_req = MemoryRequirement(sizeof(L), alignof(L));   //alpha
    }

public:
    SGD_Optimizer() = default;
    
    virtual void initMem() override {}     //No memory to initializes

    virtual static OPTIMIZER_TYPE getOptimizerType() override {
        return OPTIMIZER_TYPE::SGD;
    }
    
    virtual void setLR(L* alpha, L* beta1, L* beta2, cudaStream_t stream) override {
        cudaMemcpyAsync(learningRates, alpha, sizeof(L), cudaMemcpyHostToDevice, stream);  //SGD only uses alpha
    }

    virtual void addNodeWeights(T* mem, uint64_t& index, T* delta, T* input, uint32_t y, uint32_t x, uint32_t batch_size, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput) override {
        //1.: Variables used
        uint32_t outStateSizeBatched = x * y * batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        void* sgdArgs[] = {
            (void*)&mem,
            (void*)&delta,
            (void*)&input,
            (void*)&learningRates,
            (void*)y,
            (void*)x,
            (void*)batch_size
        };
        cudaKernelNodeParams sgdParam{
            (void*)sgdMul<T, L>,                  //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&sgdArgs,                     //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam);
        depsMem.apply<true>(graph, node);
        depsDelta.apply<false>(graph, node);
        depInput.apply<false>(graph, node);
    }

    virtual void addNodeBias(T* mem, uint64_t& index, T* delta, uint32_t y, uint32_t batch_size, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta) override {
        //1.: Variables used
        uint32_t outStateSizeBatched = x * y * batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        void* sgdArgs[] = {
            (void*)&mem,
            (void*)&delta,
            (void*)&learningRates,
            (void*)y,
            (void*)batch_size
        };
        cudaKernelNodeParams sgdParam{
            (void*)sgdSimple<T, L>,               //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&sgdArgs,                     //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam);
        depsMem.apply<true>(graph, node);
        depsDelta.apply<false>(graph, node);
    }
};

//=============================================================
//==================|Activation Functions|=====================
//=============================================================

enum ACTIVATION_TYPE: uint32_t {IDENTITY=0, RELU=1, SOFTMAX=2, SOFTMAX_TEMP=3, SIGMOID=4, TANH=5, SOFTPLUS=6};  //DON'T CHANGE THESE VALUES AS IT WILL BREAK OLD CHECKPOINT FILES!
template<typename T>
class Activation {
protected:
    uint32_t outStateSize;
    uint32_t batch_size;

    T* params;      //TODO: At the moment, they are constant and can't be learned.

public:
    /*
        Call sequence: (the order of definition)
        1.: setSizes
        2.: getMemoryRequirements
        3.: setMem
        4.: initMem
        5.: addActivation[Deriv]Node
    */

    /*
        Sets member variables.

        @param outStateSize_: The number of values to activate !!!per batch!!!. (sample_size)
        @param batch_size   : The batch size
    */
    void setSizes(uint32_t outStateSize_, uint32_t batch_size_) {
        outStateSize = outStateSize_;
        batch_size   = batch_size_;
    }
    /*
        Returns the memory requirements of this activation to "other_requirement".

        @param other_requirement: Out parameter. The number of bytes on the gpu needed to store "params" (softmax temperature, ...)
        @param tmp_requirement  : Out parameter. The number of bytes on the gpu needed to for temporary storage (accumulator for softmax, ...)
    */
    virtual void getMemoryRequirements(MemoryRequirement& other_requirement, MemoryRequirement& tmp_requirement);
    /*
        Sets the internal memory pointers of the actiavtion. The passed pointer will be set to the first byte after the by this activation used memory
        region. The pointers do not need to be aligned as first they will be padded so they satisfy the alignment requirement and then incrementen by the 
        space requirement of this activation, as specified in "getMemoryRequirement".

        Does not include memory for activation!

        @param param_memory: Pointer to enough free memory that this activation will use to store "params". Will be set after the region used for this.
    */
    void setMem(uint8_t*& param_memory) {
        //1.: Get memory requirements
        MemoryRequirement other_requirement;
        getMemoryRequirement(other_requirement);

        //2.: Add padding to create alignment
        param_memory = align_pointer_unsafe(param_memory, other_requirement.alignment);

        //3.: Set internal variables
        params = param_memory;

        //4.: Increment parameters
        param_memory += other_requirement.num_bytes;
    }
    /*
        Initializes the memory pointed to by "params"
    */
    virtual void initMem();

    /*
        Adds a graph node to a graph that performs forward propagation through this activation.

        @param mem         : The input memory that the activation should be applied to
        @param tmp         : Temporary memory satisfying the requirements returned in "getMemoryRequirements"
        @param graph       : The graph generated nodes will be added to
        @param depsMem     : The dependencies on "mem". Will be updated
        @param captureStream: The stream used to capture cublas operations
    */
    virtual void addActivationNode     (T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream);
    /*
        Adds nodes to a graph that compute the derivative of the loss with respect to the input of the activation (backpropagation)

        @param mem          : The input memory that the backpropagation should be applied to
        @param deltas       : The derivative of the loss with respect to the output of the activation
        @param outStateSize : The number of T's the backpropagation should be applied to per sample
        @param batch_size   : The number of samples per batch
        @param tmp          : Temporary memory satisfying the requirements returned in "getMemoryRequirements"
        @param graph        : The graph the generated nodes will be added to
        @param depsMem      : The dependencies on "mem". Will be updated
        @param depsDeltas   : The dependencies on "deltas". Wil be updated
        @param captureStream: The stream used to capture cublas operations
    */
    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream);
    
    /*
        Returns the type of this activation
    */
    virtual ACTIVATION_TYPE getActivationType();
    /*
        Creates an activation of the given derived class at a specific memory location.

        @param at:  The derived class to create
        @param out: The memory location the object should be created at
    */
    static void getActivationOfType(ACTIVATION_TYPE at, Activation<T>* out) { //TODO
        switch (at) {
        case ACTIVATION_TYPE::IDENTITY:
            new (out) IDENTITY_Activation<T>();
            break;
        case ACTIVATION_TYPE::RELU:
            new (out) RELU_Activation<T>();
            break;
        case ACTIVATION_TYPE::SOFTMAX:
            new (out) Softmax_Activation<T>();
            break;
        case ACTIVATION_TYPE::SOFTMAX_TEMP:
            new (out) SoftmaxTemp_Activation<T>();
            break;
        case ACTIVATION_TYPE::SIGMOID:
            new (out) Sigmoid_Activation<T>();
            break;
        case ACTIVATION_TYPE::TANH:
            new (out) Tanh_Activation<T>();
            break;
        case ACTIVATION_TYPE::SOFTPLUS:
            new (out) Softplus_Activation<T>();
            break;
        default:
            fprintf(stderr, "[ERROR] %llu is not a known activation type!", (uint64_t)at);
            exit(-1);
        }
    }

    // /+=============+\
    // ||SERIALIZATION||
    // \+=============+/

    /*
        Serializes an object to a FILE*. It can be deserialized using the "deserialize" method.
        The serialize method should be called from the derived class. It is required to first write "getActivationType".
        Deserialization is invoked from this base class. It reads the layer type first and afterwards invokes the "deserialize"
        method of the corresponding derived class.
    */
    void serialize(FILE* file) {
        //1.: Write activation type
        ACTIVATION_TYPE activation_type = getActivationType();
        fwrite(&activation_type, sizeof(activation_type), 1, file);

        //2.: Write variables
        fwrite(&outStateSize, sizeof(outStateSize), 1, file);
        fwrite(&batch_size  , sizeof(batch_size  ), 1, file);

        //3.: Write memory
        MemoryRequirement other_requirement;
        getMemoryRequirement(other_requirement);

        fwrite(&other_requirement.num_bytes, sizeof(other_requirement.num_bytes), 1, file);
        fwrite(params, 1, other_requirement.num_bytes, file);
    }
    /*
        Deserializes an object from a FILE* that was serialized using the "serialize" method.
        The deserialize method should be called from the base class. It reads the activation type first and afterwards invokes the "deserialize"
        method of the corresponding derived class.
    */
    static void deserialize(FILE* file, Activation<T>* out) {
        //1.: Create correct derived class
        ACTIVATION_TYPE activation_type;
        fread(&activation_type, sizeof(ACTIVATION_TYPE), 1, file);
        Activation>T>::getActivationOfType(activation_type, out);

        //2.: Get memory requirements
        MemoryRequirement other_requirement;
        out->getMemoryRequirement(other_requirement);

        //3.: Read in variables
        fread(&outStateSize, sizeof(outStateSize), 1, file);
        fread(&batch_size, sizeof(batch_size), 1, file);

        //4.: Read in memory
        uint64_t other_bytes;

        fread(&other_bytes, sizeof(other_bytes), 1, file);
        cudaMallocAligned(&out->params, other_requirement);
        fread(out->params, 1, other_bytes, file);

        //5.: Check consistency
        if (other_requirement.num_bytes != other_bytes) {
            fprintf(stderr, "[ERROR] Trying to create a activation of type %llu with %llu other bytes, even though it requires %llu", (uint64_t)activation_type, (uint64_t)other_bytes, (uint64_t)other_requirement.num_bytes);
            exit(-1);
        }
    }

    /*
        Intended for checkpoint files. Only writes information needed to recreate this activation.
        All alignment is ignored and padding is removed to save space. Dynamically allocates ram (as intermidiate for gpu->file transfer).

        If "data==false", inverse is "getActivationFromCompression". Otherwise, inverse is "initMemFromCompression".

        @param data: If false, only writes specifiers. If true, only writes data.
        @param file: The file to write the compressed data to.
    */
    template<bool data>
    void compress(FILE* file) {
        if constexpr (data) {
            //1.: Write "params"
            MemoryRequirement other_req, tmp_req;
            uint64_t nodes;
            getMemoryRequirements(other_req, tmp_req, nodes);

            void* ram_buffer;
            cudaMallocHost(&ram_buffer, other_req.num_bytes);
            cudaMemcpy(ram_buffer, params, other_req.num_bytes, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(NULL);
            fwrite(ram_buffer, 1, other_req.num_bytes, file);
            cudaFreeHost(ram_buffer);
        }
        else {
            //1.: Write specifiers
            ACTIVATION_TYPE at = getActivationType();
            fwrite(&at, sizeof(ACTIVATION_TYPE), 1, file);
        }
    }
    /*
        Inverse of "compress<false>"
    */
    static void getActivationFromCompression(FILE* file, Layer<T>* out) {
        ACTIVATION_TYPE at;
        fread(&at, sizeof(ACTIVATION_TYPE), 1, file);
        getActivationOfType(at, out);
    }
    /*
        Inverse of "compress<true>"
    */
    void initMemFromCompression(FILE* file) {
        MemoryRequirement other_req, tmp_req;
        uint64_t nodes;
        getMemoryRequirements(other_req, tmp_req, nodes);

        void* ram_buffer;
        cudaMallocHost(&ram_buffer, other_req.num_bytes);
        fread(ram_buffer, 1, other_req.num_bytes, file);
        cudaMemcpy(params, ram_buffer, other_req.num_bytes, cudaMemcpyHostToDevice);
        cudaStreamSynchronize(NULL);
        cudaFreeHost(ram_buffer);
    }
};

template<typename T>
class IDENTITY_Activation : public Activation<T> {
public:
    //1.: Constructors
    IDENTITY_Activation() = default;   //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual ACTIVATION_TYPE getActivationOfType() override {
        return ACTIVATION_TYPE::IDENTITY;
    }

    virtual void getMemoryRequirements(MemoryRequirement& other_requirement, MemoryRequirement& tmp_requirement) override {
        other_requirement = MemoryRequirement(0ull, 1u);  //No memory needed
        tmp_requirement   = MemoryRequirement(0ull, 1u);  //No memory needed
    }

    virtual void initMem() override {} //No memory to initialize

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) override {}

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) override {}
};

template<typename T>
class RELU_Activation : public Activation<T> {
public:
    //1.: Constructors
    RELU_Activation() = default;   //For "getActivationOfType" and user
    
    //2.: Overloaded functions
    virtual ACTIVATION_TYPE getActivationOfType() override {
        return ACTIVATION_TYPE::RELU;
    }

    virtual void getMemoryRequirements(MemoryRequirement& other_requirement, MemoryRequirement& tmp_requirement) override {
        other_requirement =  MemoryRequirement(0ull, 1u);  //No memory needed
        tmp_requirement   =  MemoryRequirement(0ull, 1u);  //No memory needed
    }

    virtual void initMem() override {} //No memory to initialize

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) override {
        //1.: Variables used
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        constexpr auto ldb = []__device__(T in) { return in > (T)0 ? in : (T)0; };
        void* reluArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams reluParam{
            (void*)transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&reluArgs,                    //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &reluParam);
        depsMem.apply<true>(graph, node);
    }

    //TODO: Can be fused into one kernel
    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) override {
        //0: Variables
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        cudaGraphNode_t node;
        
        //1.: Calculate derivatives of output of activation with respect to input of activation
        constexpr auto ldb = []__device__(T in) { return in > (T)0 ? (T)1 : (T)0; };

        void* reluArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams reluParam{
            (void*)transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&reluArgs,                    //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &reluParam);
        depsMem.apply<true>(graph, node);
        
        //2.: Calculate derivatives of loss with respect to the input of the activation
        addElementwiseMultNode(graph, mem, deltas, outStateSizeBatched, node);
        depsMem.apply<true>(graph, node);
        depsDeltas.apply<false>(graph, node);
    }
};

template<typename T>
class Softmax_Activation : public Activation<T> {
    //1.: Constructors
    Softmax_Activation() = default; //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual ACTIVATION_TYPE getActivationOfType() override {
        return ACTIVATION_TYPE::SOFTMAX;
    }

    virtual void getMemoryRequirements(MemoryRequirement& other_requirement, MemoryRequirement& tmp_requirement) override {
        other_requirement = MemoryRequirement(sizeof(T), alignof(T));                         //Need to store one T for temperature
        tmp_requirement   = MemoryRequirement(batch_size * outStateSize * outStateSize, 16u); //For backprop
    }

    virtual void initMem() override {
        //Initialize temperature to 1
        T temp = (T)1;
        cudaMemcpy(params, &temp, sizeof(T), cudaMemcpyHostToDevice);
    }

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) override {
        //1.: Variables
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        cudaGraphNode_t node;
        
        //2.: Kernell
        void* softmaxArgs[] = {
            (void*)&mem,
            (void*)&outStateSize,
            (void*)&batch_size
        };
        cudaKernelNodeParams softmaxArgs{
            (void*)softmax<T, true>,                 //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),    //Grid dimensions
            dim3(32, 1, 1),                          //Block dimensions
            0u,                                      //Dyn. shared-mem per block in bytes
            (void**)softmaxArgs,                     //Array of pointers to individual kernel arguments
            nullptr                                  //Extra
        };        
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &softmaxArgs);
        depsMem.apply<true>(graph, node);
    }

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) override {
        //1.: Variables
        uint32_t outStateSizeBatched = outStateSize * batch_size;
        Dependencies depsTmp;

        cudaGraphNode_t node;

        //2.: Generate derivatives of softmax. tmp=derivs[batch_size][outStateSize][outStateSize].
        //derivs[b][x][y] (yep, column major) stores derivative of output of x with respect to input x in batch b.

        void* softmaxArgs[] = {
            (void*)&mem,
            (void*)&tmp,
            (void*)&outStateSize,
            (void*)&batch_size
        };
        cudaKernelNodeParams softmaxArgs{
            (void*)softmax_deriv<T, true>,           //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),    //Grid dimensions
            dim3(32, 1, 1),                          //Block dimensions
            0u,                                      //Dyn. shared-mem per block in bytes
            (void**)softmaxArgs,                     //Array of pointers to individual kernel arguments
            nullptr                                  //Extra
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &softmaxArgs);
        depsMem.apply<false>(graph, node);
        depsTmp.apply<true> (graph, node);

        //3.: Multiply generated derivative matrix with deltas
        cudaGraph_t mulGraph = getMatmulGraph<T, false, false, true>(tmp, deltas, mem, outStateSize, outStateSize, 1u, captureStream);
        cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, mulGraph);

        depsMem.apply<true> (graph, node);
        depsTmp.apply<false>(graph, node);
    }
};

template<typename T>
class SoftmaxTemp_Activation : public Activation<T> {
    //1.: Constructors
    SoftmaxTemp_Activation() = default; //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual ACTIVATION_TYPE getActivationOfType() override {
        return ACTIVATION_TYPE::SOFTMAX_TEMP;
    }

    virtual void getMemoryRequirements(MemoryRequirement& other_requirement, MemoryRequirement& tmp_requirement) override {
        other_requirement = MemoryRequirement(sizeof(T), alignof(T));                         //Need to store one T for temperature
        tmp_requirement   = MemoryRequirement(batch_size * outStateSize * outStateSize, 16u); //For backprop
    }

    virtual void initMem() override {
        //Initialize temperature to 1
        T temp = (T)1;
        cudaMemcpy(params, &temp, sizeof(T), cudaMemcpyHostToDevice);
    }

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) override {
        //1.: Variables
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        cudaGraphNode_t node;

        //2.: Kernell
        void* softmaxArgs[] = {
            (void*)&mem,
            (void*)&outStateSize,
            (void*)&batch_size
            (void*)&params
        };
        cudaKernelNodeParams softmaxArgs{
            (void*)softmaxTemp<T, true>,             //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),    //Grid dimensions
            dim3(32, 1, 1),                          //Block dimensions
            0u,                                      //Dyn. shared-mem per block in bytes
            (void**)softmaxArgs,                     //Array of pointers to individual kernel arguments
            nullptr                                  //Extra
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &softmaxArgs);
        depsMem.apply<true>(graph, node);
    }

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) override {
        //1.: Variables
        uint32_t outStateSizeBatched = outStateSize * batch_size;
        Dependencies depsTmp;

        cudaGraphNode_t node;

        //2.: Generate derivatives of softmax. tmp=derivs[batch_size][outStateSize][outStateSize].
        //derivs[b][x][y] (yep, column major) stores derivative of output of x with respect to input x in batch b.

        void* softmaxArgs[] = {
            (void*)&mem,
            (void*)&tmp,
            (void*)&outStateSize,
            (void*)&batch_size,
            (void*)&params
        };
        cudaKernelNodeParams softmaxArgs{
            (void*)softmaxTemp_deriv<T, true>,       //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),    //Grid dimensions
            dim3(32, 1, 1),                          //Block dimensions
            0u,                                      //Dyn. shared-mem per block in bytes
            (void**)softmaxArgs,                     //Array of pointers to individual kernel arguments
            nullptr                                  //Extra
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &softmaxArgs);
        depsMem.apply<false>(graph, node);
        depsTmp.apply<true>(graph, node);

        //3.: Multiply generated derivative matrix with deltas
        cudaGraph_t mulGraph = getMatmulGraph<T, false, false, true>(tmp, deltas, mem, outStateSize, outStateSize, 1u, captureStream);
        cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, mulGraph);
        depsMem.apply<true>(graph, node);
        depsTmp.apply<false>(graph, node);
    }
};

template<typename T>
class Sigmoid_Activation : public Activation<T> {
    //1.: Constructors
    Sigmoid_Activation() = default; //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual ACTIVATION_TYPE getActivationOfType() override {
        return ACTIVATION_TYPE::SIGMOID;
    }

    virtual void getMemoryRequirements(MemoryRequirement& other_requirement, MemoryRequirement& tmp_requirement) override {
        other_requirement = MemoryRequirement(0ull, 1u); //No memory needed
        tmp_requirement   = MemoryRequirement(0ull, 1u); //For backprop
    }

    virtual void initMem() override {} //No memory to initialize

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) override {
        //1.: Variables used
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        constexpr auto ldb = []__device__(T in) { return (T)1 / ((T)1 + exponential<T>(-in)); };
        void* sigArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams sigParam{
            (void*)transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&sigArgs,                     //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sigParam);
        depsMem.apply<true>(graph, node);
    }

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) override {
        //0: Variables
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        cudaGraphNode_t node;

        //1.: Calculate derivatives of output of activation with respect to input of activation
        constexpr auto ldb = []__device__(T in) { return in * ((T)1 - in); };

        void* sigArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams sigParam{
            (void*)transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&sigArgs,                     //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sigParam);
        depsMem.apply<true>(graph, node);

        //2.: Calculate derivatives of loss with respect to the input of the activation
        addElementwiseMultNode(graph, mem, deltas, outStateSizeBatched, node);
        depsMem.apply<true>(graph, node);
        depsDeltas.apply<false>(graph, node);
    }
};

template<typename T>
class Tanh_Activation : public Activation<T> {
    //1.: Constructors
    Tanh_Activation() = default; //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual ACTIVATION_TYPE getActivationOfType() override {
        return ACTIVATION_TYPE::TANH;
    }

    virtual void getMemoryRequirements(MemoryRequirement& other_requirement, MemoryRequirement& tmp_requirement) override {
        other_requirement = MemoryRequirement(0ull, 1u); //No memory needed
        tmp_requirement   = MemoryRequirement(0ull, 1u); //For backprop
    }

    virtual void initMem() override {} //No memory to initialize

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) override {
        //1.: Variables used
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        constexpr auto ldb = []__device__(T in) { return (exponential<T>(x) - exponential<T>(-x)) / (exponential<T>(x) + exponential<T>(-x)); };
        void* tanhArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams tanhParam{
            (void*)transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&tanhArgs,                    //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &tanhParam);
        depsMem.apply<true>(graph, node);
    }

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) override {
        //0: Variables
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        cudaGraphNode_t node;

        //1.: Calculate derivatives of output of activation with respect to input of activation
        constexpr auto ldb = []__device__(T in) { return (T)1 - in * in; };

        void* tanhArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams tanhParam{
            (void*)transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&tanhArgs,                    //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &tanhParam);
        depsMem.apply<true>(graph, node);

        //2.: Calculate derivatives of loss with respect to the input of the activation
        addElementwiseMultNode(graph, mem, deltas, outStateSizeBatched, node);
        depsMem.apply<true>(graph, node);
        depsDeltas.apply<false>(graph, node);
    }
};

template<typename T>
class Softplus_Activation : public Activation<T> {
    //1.: Constructors
    Softplus_Activation() = default; //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual ACTIVATION_TYPE getActivationOfType() override {
        return ACTIVATION_TYPE::SOFTPLUS;
    }

    virtual void getMemoryRequirements(MemoryRequirement& other_requirement, MemoryRequirement& tmp_requirement) override {
        other_requirement = MemoryRequirement(0ull, 1u); //No memory needed
        tmp_requirement   = MemoryRequirement(0ull, 1u); //For backprop
    }

    virtual void initMem() override {} //No memory to initialize

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) override {
        //1.: Variables used
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        //2.: Add node to graph
        constexpr auto ldb = []__device__(T in) { return logarithm<T>((T)1 + exponential<T>(in)); };
        void* softplusArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams softplusParam{
            (void*)transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&softplusArgs,                //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(out_node, graph, nullptr, 0, &softplusParam);
        depsMem.apply<true>(graph, out_node);

        out_node++;

        //3.: Set out parameters     
        //depsMem reamains depsMem and thus remains the same. out_node was also already incremented
    }

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) override {
        //0: Variables
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        cudaGraphNode_t node;

        //1.: Calculate derivatives of output of activation with respect to input of activation
        constexpr auto ldb = []__device__(T in) { return (T)1 - exponential<T>(-in); };

        void* softplusArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams softplusParam{
            (void*)transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&softplusArgs,                //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &softplusParam);
        depsMem.apply<true>(graph, node);

        //2.: Calculate derivatives of loss with respect to the input of the activation
        addElementwiseMultNode(graph, mem, deltas, outStateSizeBatched, node);
        depsMem.apply<true>(graph, node);
        depsDeltas.apply<false>(graph, node);
    }
};

//============================================
//==================|Layers|==================
//============================================

enum LAYER_TYPE: uint32_t {INPUT=0, FULLY_CONNECTED=1, CNN=2, RNN=3, LSTM=4, TRANSFORMER=5, POOL=6};  //DON'T CHANGE THESE VALUES AS IT WILL BREAK OLD CHECKPOINT FILES!
template<typename T, typename L = T>
class Layer {
    friend Scheduler<T, L>;
    friend Loss<T, L>;

    /*
        A layer is a helper class that does not actually own any memory. It is only used by the NetworkBuilder to construct
        the execution graph.

        This is only the base class that should never be used itself except for static methods. Each actual layer type should be implemented as a
        new derived class that overwrites the respective methods. Each virtual method except "getMemoryRequirements" (has a default implementation) 
        needs to be overwritten.

        Each derived class is only allowed to have "trivial" constructor and shall suppy a consdtructor without arguments.
    */
protected:
    Layer<T, L>* layerBefore;        //Pointer to the layer before this one. If this is the first layer, this has to be "nullptr"
    uint32_t  batch_size;         //Number of smaples in a batch

    Activation<T> act;            //The activation of this layer

    Image_Shape outStateShape;    //Shape of one output sample
    void* state;                  //State of this layer (everything dependend on batch size). Starts with output state consisting of outStateShape.prod()*batch_size T's, rest can correspond to internal state.
    void* other;                  //Other memory (independent of batch size, e.g. weights and biases). This includes (but is not limited to) all parameters that are optimized.

    /*
        Returns the memory requirements of this layer excluding the activation to the given variables.

        @param state_requirement : Out parameter. The number of bytes on the gpu needed to store the state of the layer(output, ...)
        @param other_requirement : Out parameter. The number of bytes on the gpu needed to store other data of the layer (weights, biases, activation params, ...)
        @param optimizables      : Out parameter. The number of T's   on the gpu that need to be optimized (weights, biases, ...)
        @param tmp_requirement   : Out parameter. The number of bytes on the gpu needed as temporary storage (shared between all layers and activation, non persistent)
    */ 
    virtual constexpr static void getOwnMemoryRequirement(MemoryRequirement& state_requirement, MemoryRequirement& other_requirement, uint64_t& optimizables, MemoryRequirement& tmp_requirement)

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
    */
    
    /*
        Sets the variable holding a pointer to the layer right in front of this one in the network.
    */
    void setLayerBefore(Layer<T>* l) { layerBefore = l; }
    /*
        Sets the batch size used by this layer. Note, that this does not reallocate state memory.
        This also passes the information on to "act".

        @param batch_size_: The new batch size.
    */
    void setBatchSize(uint32_t batch_size_) { batch_size = batch_size_; act.setSizes(outStateShape.prod(), batch_size); }
    /*
        Returns the memory requirements of this layer including the activation to the given variables.

        @param state_requirement : Out parameter. The number of bytes on the gpu needed to store the state of the layer(output, ...)
        @param other_requirement : Out parameter. The number of bytes on the gpu needed to store other data of the layer (weights, biases, activation params, ...)
        @param optimizables      : Out parameter. The number of T's   on the gpu that need to be optimized (weights, biases, ...)
        @param tmp_requirement   : Out parameter. The number of bytes on the gpu needed as temporary storage (shared between all layers and activation, non persistent)
    */
    virtual constexpr static void getMemoryRequirement(MemoryRequirement& state_requirement, MemoryRequirement& other_requirement, uint64_t& optimizables, MemoryRequirement& tmp_requirement) {
        //1.: Own requirements
        getOwnMemoryRequirement(state_requirement, other_requirement, optimizables, tmp_requirement);


        //2.: Requirements of activation
        MemoryRequirement other_req_activ, tmp_req_activ;
        uint32_t nodes_activ;
        act.getMemoryRequirements(other_req_activ, tmp_req_activ, nodes_activ);

        other_nums      += other_req_activ;
        tmp_requirements = max(tmp_requirements, tmp_req_activ);
    }
    /*
        Sets the internal memory pointers of the layer and activation. The passed pointer will be set to the first byte after the used memory region.
        The pointers do not need to be aligned as first they will be padded so they satisfy the alignment requirement and then incrementen by the space
        required, as specified in "getMemoryRequirement".

        @param state_mem: Pointer to enough free memory to store "state". Has to be reallocated whenever batch size changes
        @param other_mem: Pointer to enough free memory to store "other" and "act.params". Does not need to be reallocated
    */
    void setMem(uint8_t*& state_mem, uint8_t*& other_mem) {
        //1.: This layer
        //1.1: Get memory requirements
        MemoryRequirement state_requirement, other_requirement, tmp_requirement;
        uint64_t optimizables, num_nodes;
        getMemoryRequirement(state_requirement, other_requirement, optimizables, tmp_requirement, num_nodes);
        
        //1.2.: Add padding to create alignment
        state_mem = align_pointer_unsafe(state_mem, state_requirement.alignment);
        other_mem = align_pointer_unsafe(other_mem, other_requirement.alignment);

        //1.3.: Set internal variables
        state = state_mem;
        other = other_mem;

        //1.4.: Increment parameters
        state += state_requirement.num_bytes;
        other += other_requirement.num_bytes;

        //2.: Activation
        act.setMem(other);
    }

    /* 
        Initializes "state" and "other" memory of the size returned in getMemoryRequirements.
        Also has to intialize memory of "act".
    */
    virtual void initMem();
    /*
        Adds forward propagation through this layer to an execution graph. Takes in the state of the layer before and computes own state.
        Returns indirection pointers (the layer after the input cannot use the input layer's state directly, as it changes) and the number of them.

        We can not simply return a graph, as this would make it impossible to express dependencies correctly (bias depends on nothing while matmul does).

        @param after_input  : True if this is the first layer after the input layer
        @param graph        : The execution graph to construct
        @param depPrevState : The dependencies on previous output state. Will be updated
        @param captureStream: The function can use stream capture to generate the call graph. This is the stream it will use for this.
        @param tmp          : Temporary storage of at least the size and alignment requested in "getMemoryRequirements".
        @param after_dataset: True, when this layer is the first layer after the input layer that refers to the dataset
    */
    virtual std::vector<T**> forwardProp(cudaGraph_t graph, Dependencies& depPrevState, cudaStream_t captureStream, T* tmp, bool after_dataset);
    /*
        Adds backpropagation through this layer to an execution graph.
        Assumes, that the state of this layer was already set to dL/do where L is Loss and o the output state after forwardProp.
        This updates the own weights and biases of this layer using optimizer opt and writes dL/do of the layer before in his state.
    
        @param graph            : The execution graph to construct
        @param depState         : The dependencies on own state Will be updated
        @param opt              : The optimizer used to change the internal weights.
        @param optimizable_index: The index of the first element to the optimization buffer. Will be advanced (according to "getMemoryRequirements")
        @param captureStream    : The function can use stream capture to generate the call graph. This is the stream it will use for this.
        @param tmp              : Temporary storage of at least the size and alignment requested in "getMemoryRequirements".
        @param after_dataset    : True, when this layer is the first layer after the input layer that refers to the dataset
    */
    virtual std::vector<T**> backwardProp(cudaGraph_t graph, Dependencies& depState, Optimizer<T, L>* opt, uint64_t& optimizable_index, cudaStream_t captureStream, T* tmp, bool after_dataset);
    
    /*
        Return the type of the layer.
    */
    virtual static LAYER_TYPE getLayerType();
    static void getLayerOfType(LAYER_TYPE lt, Layer<T>* out) {
        switch (lt) {
        case LAYER_TYPE::INPUT:
            new (out) Input_Layer<T>();
            break;
        case LAYER_TYPE::FULLY_CONNECTED:
            new (out) FullyConnected_Layer<T>();
            break;
        default:
            fprintf(stderr, "[ERROR] %llu is not a known layer type!", (uint64_t)lt);
            exit(-1);
        }
    }
    /*
        This function can be called by user to create a wanted layer. It is also used for Networks deserialization
    */
    static void getLayerFromSpecifiers(LAYER_TYPE lt, Image_Size shape, ACTIVATION_TYPE at, Layer<T>* out) {
        getLayerOfType(lt, out);
        out->outStateShape = shape;
        Activation<T>::getActivationOfType(at, &out->act);
    }
    
    // /+=============+\
    // ||SERIALIZATION||
    // \+=============+/

    /*
        Serialization according to the serialization rules.
    */
    void serialize(FILE* file) {
        //1.: Write layer type
        LAYER_TYPE layer_type = getLayerType();
        fwrite(&layer_type, sizeof(layer_type), 1, file);

        //2.: Write variables
        fwrite(&layer_before, sizeof(layer_before), 1, file);
        fwrite(&batch_size  , sizeof(batch_size), 1, file);
        act.serialize(file);
        outStateShape.serialize(file);

        //3.: Write memory
        MemoryRequirement state_requirement, other_requirement, tmp_requirement;
        uint64_t optimizables, num_nodes;
        getMemoryRequirement(state_requirement, other_requirement, optimizables, tmp_requirement, num_nodes);

        fwrite(&state_requirement.num_bytes, sizeof(state_requirement.num_bytes), 1, file);
        fwrite(state, 1, state_requirement.num_bytes, file);

        fwrite(&other_requirement.num_bytes, sizeof(other_requirement.num_bytes), 1, file);
        fwrite(other, 1, other_requirement.num_bytes, file);
    }
    /*
        Deserialization according to deserialization rules
    */
    static void deserialize(FILE* file, Layer<T>* out) {
        //1.: Create correct derived class
        LAYER_TYPE layer_type;
        fread(&layer_type, sizeof(LAYER_TYPE), 1, file);
        Layer<T>::getLayerOfType(layer_type, out);

        //2.: Read in variables
        fread(out->layer_before, sizeof(out->layer_before), 1, file);
        fread(out->batch_size, sizeof(out->batch_size), 1, file);
        Activation<T>::deserialize(file, &out->act);
        Image_Shape::deserialize(file, &out->outStateShape);

        //3.: Get memory requirements
        MemoryRequirement state_requirement, other_requirement, tmp_requirement;
        uint64_t optimizables, num_nodes;
        out->getMemoryRequirement(state_requirement, other_requirement, optimizables, tmp_requirement, num_nodes);

        //4.: Read in memory
        uint64_t state_bytes, other_bytes;

        fread(&state_bytes, sizeof(state_bytes), 1, file);
        cudaMallocAligned(&out->state, state_requirement);
        fread(out->state, 1, state_bytes, file);

        fread(&other_bytes, sizeof(other_bytes), 1, file);
        cudaMallocAligned(&out->other, other_requirement);
        fread(out->other, 1, other_bytes, file);

        //5.: Check consistency
        if (state_requirement.num_bytes != state_bytes) {
            fprintf(stderr, "[ERROR] Trying to create a layer of type %llu with %llu state bytes, even though it requires %llu", (uint64_t)layer_type, (uint64_t)state_bytes, (uint64_t)state_requirement.num_bytes);
            exit(-1);
        }
        if (other_requirement.num_bytes != other_bytes) {
            fprintf(stderr, "[ERROR] Trying to create a layer of type %llu with %llu other bytes, even though it requires %llu", (uint64_t)layer_type, (uint64_t)other_bytes, (uint64_t)other_requirement.num_bytes);
            exit(-1);
        }
    }

    /*
        Intended for checkpoint files. Only writes information needed to recreate this layer. For example, "state" is not needed for a checkpoint.
        Also writes "act". All alignment is ignored and padding is removed to save space. Dynamically allocates ram (as intermidiate for gpu->file transfer).
    
        If "data==false", inverse is "getLayerFromCompression". Otherwise, inverse is "initMemFromCompression".

        @param data: If false, only writes specifiers. If true, only writes data.
        @param file: The file to write the compressed data to.
    */
    template<bool data>
    void compress(FILE* file) {
        if constexpr (data) {
            //1.: Write "other"
            MemoryRequirement state_req, other_req, tmp_req;
            uint64_t optim, nodes;
            getOwnMemoryRequirement(state_req, other_req, optim, tmp_req, nodes);

            void* ram_buffer;
            cudaMallocHost(&ram_buffer, other_req.num_bytes);
            cudaMemcpy(ram_buffer, other, other_req.num_bytes, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(NULL);
            fwrite(ram_buffer, 1, other_req.num_bytes, file);
            cudaFreeHost(ram_buffer);

            //2.: Compress activation
            act.compress<true>(file);
        }
        else {
            //1.: Write specifiers
            LAYER_TYPE lt = getLayerType();
            fwrite(&lt           , sizeof(LAYER_TYPE ), 1, file);
            fwrite(&outStateShape, sizeof(Image_Shape), 1, file);

            //2.: Write acitvation specifiers
            act.compress<false>(file);
        }
    }
    /*
        Inverse of "compress<false>"
    */
    static void getLayerFromCompression(FILE* file, Layer<T>* out) {
        //1.: Layer
        LAYER_TYPE lt;
        fread(&lt           , sizeof(LAYER_TYPE ), 1, file);
        getLayerOfType(lt, out);

        fread(&out->outStateShape, sizeof(Image_Shape), 1, file);

        //2.: Activation
        Activation<T>::getActivationFromCompression(file, &out->act);
    }
    /*
        Inverse of "compress<true>"
    */
    void initMemFromCompression(FILE* file) {
        //1.: Layer
        MemoryRequirement state_req, other_req, tmp_req;
        uint64_t optim, nodes;
        getOwnMemoryRequirement(state_req, other_req, optim, tmp_req, nodes);

        void* ram_buffer;
        cudaMallocHost(&ram_buffer, other_req.num_bytes);
        fread(ram_buffer, 1, other_req.num_bytes, file);
        cudaMemcpy(other, ram_buffer, other_req.num_bytes, cudaMemcpyHostToDevice);
        cudaStreamSynchronize(NULL);
        cudaFreeHost(ram_buffer);

        //2.: Activation
        act.initMemFromCompression(file);
    }
};

template<typename T, typename L = T>
class Input_Layer : public Layer<T, L> {
    //*state={pointer to current gpu tile}
    //*other={}
public:
    //1.: Constructors
    //Used in "getLayerOfType"
    Input_Layer() :
        layerBefore(nullptr),
        batch_size(0),
        outStateShape(Image_Shape(0, 0, 0)),
        state(nullptr),
        other(nullptr)
    {
        getActivationOfType(ACTIVATION_TYPE::IDENTITY, &act);
    }

    //Used by user
    Input_Layer(uint32_t num_neurons):
        layerBefore(nullptr),
        batch_size(0),
        outStateShape(Image_Shape(num_neurons, 1, 1)),
        state(nullptr),
        other(nullptr)
    {
        getActivationOfType(ACTIVATION_TYPE::IDENTITY, &act);
    }

protected:
    virtual static void getOwnMemoryRequirement(MemoryRequirement& state_requirement, MemoryRequirement& other_requirement, uint64_t& optimizables, MemoryRequirement& tmp_requirement, uint64_t& num_nodes) override {
        state_requirement = MemoryRequirement(sizeof(T*), 1u);   //Points to current gpu tile
        other_requirement = MemoryRequirement(0ull      , 1u);   //No memory needed
        optimizables      = 0;                                   //No variables to optimize
        tmp_requirement   = MemoryRequirement(0ull      , 1u);   //No memory needed
        num_nodes         = 0;                                   //No node needed
    }

public:
    virtual void initMem() override { T* cur_gpu_tile = nullptr;  cudaMemcpy(state, &cur_gpu_tile, sizeof(T*), cudaMemcpyHostToDevice); } //Write "nullptr" to state

    /*
        Set the pointer to the gpu tile to a specified pointer.

        @param host_indirectionPointer: Must be a pointer allocated by "cudaMallocHost" that points to a pointer on the host that points to the correct gpu tile
        @param stream: The stream used for the memory transfer
    */
    void setInputPointer(T** host_indirectionPointer, cudaStream_t stream) {
        cudaMemcpyAsync(state, host_indirectionPointer, sizeof(T*), cudaMemcpyHostToDevice, stream);
    }

    virtual std::vector<T**> forwardProp(const cudaGraph_t& graph, Dependencies& depPrevState, cudaStream_t captureStream, T* tmp, bool after_dataset) override {
        fprintf(stderr, "[ERROR] You are trying to compute forward pass through input layer!");
        std::exit(-1);
    }

    virtual std::vector<T**> backwardProp(const cudaGraph_t& graph, Dependencies& depState, Optimizer<T, L>* opt, uint64_t& optimizable_index, cudaStream_t captureStream, T* tmp, bool after_dataset) override {
        fprintf(stderr, "[ERROR] You are trying to compute backwards pass through input layer!");
        std::exit(-1);
    }

    virtual static LAYER_TYPE getLayerType() override {
        return LAYER_TYPE::INPUT;
    }
};

template<typename T, typename L = T>
class FullyConnected_Layer : public Layer<T, L> {
    //other={bias|weights}
    //state={output}
public:
    // /+============+\
    // ||Constructors||
    // \+============+/
    //Used in "getLayerOfType":
    FullyConnected_Layer() = default;
    //Used by user:
    FullyConnected_Layer(Activation<T>& act_, uint32_t num_neurons) :    
        layerBefore(nullptr), batch_size(0u), act(act_), outStateShape(Image_Shape(num_neurons, 1u, 1u)), state(nullptr), other(nullptr)
    {}

    // /+=======+\
    // ||Methods||
    // \+=======+/
protected:
    virtual constexpr static void getOwnMemoryRequirement(MemoryRequirement& state_requirement, MemoryRequirement& other_requirement, uint64_t& optimizables, MemoryRequirement& tmp_requirement) override {
        //1.: This layer
        state_nums = MemoryRequirement(sizeof(T) * outStateShape.prod() * batch_size, 16);                     //Output of layer (cublas wants alignment of at least 16 bytes)

        other_nums = MemoryRequirement(                                                             
            sizeof(T) * (uint64_t)outStateShape.prod() * (1ull + (uint64_t)layerBefore->outStateShape.prod()), //Bias + Weights
            16);                                                                                               //cublas wants alignment of at least 16 bytes
                                                                                                          
        optimizables = other_nums.num / sizeof(T);                                                             //Bias + Weights

        tmp_requirement = MemoryRequirement(sizeof(T) * layerBefore->outStateShape.prod() * batch_size, 16);   //For backprop (cublas wants alignment of at least 16 bytes)
    }

public:
    virtual constexpr static void getMemoryRequirement(MemoryRequirement& state_requirement, MemoryRequirement& other_requirement, uint64_t& optimizables, MemoryRequirement& tmp_requirement) override {
        //1.: Own requirements
        getOwnMemoryRequirement(state_requirement, other_requirement, optimizables, tmp_requirement);


        //2.: Requirements of activation
        MemoryRequirement other_req_activ, tmp_req_activ;
        act.getMemoryRequirements(other_req_activ, tmp_req_activ);

        other_nums += other_req_activ;
        tmp_requirements += tmp_req_activ;                    //In backpropagation, both layer and activation use "tmp" at the same time
    }

    //TODO/FIXIT: MAKE WORK AND DEPENDENT ON ACTIVATION FUNCTION
    virtual void initMem() override {
        //1.: Set memory of activation
        act.initMem();

        //2.: Useful variables
        T* bias    = other;
        T* weights = bias + outStateShape.prod();

        //3.: Initialize "other". "state" does not initialization
        set_random<T, true>(bias   , outStateShape.prod());
        set_random<T, true>(weights, layerBefore->outStateShape.prod(), outStateShape.prod());
    }

    virtual std::vector<T**> forwardProp(cudaGraph_t graph, Dependencies& depPrevState, cudaStream_t captureStream, T* tmp, bool after_dataset) override {
        //0.: Usefuls variables
        uint32_t outStateSize = outStateShape.prod();
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        //Split "other" pointer
        T* bias    = other;
        T* weights = bias + outStateShape.prod();

        //Dependencies
        Dependencies depState;

        //Node
        cudaGraphNode_t node;

        //1.: Bias
        void* biasArgs[] = {
            (void*)&state,
            (void*)&bias,
            (void*)&outStateSizeBatched,
            (void*)&outStateSize
        };
        cudaKernelNodeParams biasParam{
            (void*)set_repeating<T>,              //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&biasArgs,                    //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &biasParam);
        depState.apply<true>(node);
        
        //2.: Weight multiplication
        std::vector<T**> indirection_pointers;

        cudaGraph_t multGraph;
        if (after_dataset) {
            T** weights_gpu, state_gpu;                                           //Pointer to device pointer to data
            cudaMalloc(&weights_gpu, sizeof(T*));                                 //Each pointer holds memory for one device pointer
            cudaMalloc(&state_gpu  , sizeof(T*));                                 //Each pointer holds memory for one device pointer
                                                                                  
            cudaMemcpy(weights_gpu, &weights, cudaMemcpyHostToDevice);            //Device pointer point to data
            cudaMemcpy(state_gpu  , &state  , cudaMemcpyHostToDevice);            //Device pointer point to data
                           
            indirection_pointers.push_back(weights_gpu);
            indirection_pointers.push_back(state_gpu);

            multGraph = getMatmulGraphIndirection<T, false, false, false>(weights_gpu, (T**)layerBefore->state, state_gpu, outStateSize, layerBefore->outStateShape.prod(), batch_size, captureStream);
        }
        else {
            multGraph = getMatmulGraph<T, false, false, false>((T*)weights, (T*)layerBefore->state, (T*)state, outStateSize, layerBefore->outStateShape.prod(), batch_size, captureStream);
        }
        cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, multGraph);
        depPrevState.apply<false>(graph, node);
        depState.apply<true>(graph, node);     //This does clash with bias setting, as that is not atomic (theoreticaly bias could reset part of result of this)
    
        //3.: Activation
        act.addActivationNode(state, tmp, graph, depState, captureStream); //Manages dependencies and "out_node" incrementation itself

        //4.: Update parameters
        depPrevState = depState;

        //5.: Return
        return indirection_pointers;
    }

    virtual std::vector<T**> backwardProp(cudaGraph_t graph, Dependencies& depState, Optimizer<T, L>* opt, uint64_t& optimizable_index, cudaStream_t captureStream, T* tmp, bool after_dataset) override {
        //0.: Usefull variables
        uint32_t outStateSize = outStateShape.prod();
        uint32_t outStateSizeBatched = outStateSize * batch_size;

        //Split "other" pointer
        T* bias    = other;
        T* weights = bias + outStateSize;

        //Split "tmp" pointer. Needs alignment requirement of activation
        MemoryRequirement other_req_activ, tmp_req_activ;
        uint32_t nodes_activ;
        act.getMemoryRequirements(other_req_activ, tmp_req_activ, nodes_activ);

        T* tmp_delta = tmp;
        T* tmp_activ = tmp_delta + roundUpMultPow2(layerBefore->outStateShape.prod() * batch_size, tmp_req_activ.alignment);

        //Dependencies
        Dependencies depTmp_delta, depWeights, depBias, depPrevState;

        //Node
        cudaGraphNode_t node;


        //1.: Multiply deltas backwards (tmp = w^T * deltas.)
        if (!after_dataset) {
            cudaGraph_t multGraph = getMatmulGraph<T, true, false, true>(weights, state, tmp_delta, layerBefore->outStateShape.prod(), outStateShape.prod(), batch_size, captureStream);
            cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, multGraph);
            depState.apply<false>(graph, node);
            depWeights.apply<false>(graph, node);
            depTmp_delta.apply<true>(graph, node);
        }

        //2.: Apply optimizers to weights
        opt->addNodeWeights(weights, optimizable_index, state, layerBefore->state, outStateSize, layerBefore->outStateShape.prod(), batch_size, graph, depWeights, depState, depPrevState);

        //3.: Apply optimizer to bias
        opt->addNodeBias(bias, optimizable_index, state, outStateSize, batch_size, graph, depBias, depState);

        //4.: Write delta of previous layer
        if (!after_dataset)
            layerBefore->act.addActivationDerivNode(layerBefore->state, tmp_delta, tmp_activ, graph, depPrevState, depTmp_delta, captureStream);
        

        //5.: Update pointers
        depState = depPrevState;
    }

    virtual static LAYER_TYPE getLayerType() override {
        return LAYER_TYPE::FULLY_CONNECTED;
    }
};

//==========================================
//==================|Loss|==================
//==========================================

enum LOSS_TYPE : uint32_t { MSE, MAE, CROSS_ENTROPY };
template<typename T, typename L = T>
class Loss {
protected:
    Layer<T, L>* layer;
    T** target;
    T*  accumulator;

public:
    Loss(Layer<T, L>* layer = nullptr) :
        layer(layer)
    {
        cudaMalloc((void**)&target, sizeof(T*));
        cudaMalloc(&accumulator, sizeof(T));
    }
    
    /*
        Set the pointer to the gpu tile to a specified pointer.

        @param host_indirectionPointer: Must be a pointer allocated by "cudaMallocHost" that points to a pointer on the host that points to the correct gpu tile
        @param stream: The stream used for the memory transfer
    */
    void setTarget(T** host_indirection_pointer, cudaStream_t stream) {
        cudaMemcpyAsync(target, host_indirection_pointer, sizeof(T*), cudaMemcpyHostToDevice, stream);
    }

    /*
        Sets value of "accumulator" to zero

        @param stream: The stream used for the memory transfer
    */
    void clearAccumulator(cudaStream_t stream) {
        cudaMemsetAsync(accumulator, 0, sizeof(T), stream);
    }
    /*
        Returns the value of the accumulator to a host variable

        @param out   : Host pointer. Must have been allocated by "cudaMallocHost". The location where the value of accumulator will be written
        @param stream: The stream used for the memroy transfer
    */
    void getAccumulator(T* out, cudaStream_t stream) {
        cudaMemcpyAsync(out, accumulator, sizeof(T), cudaMemcpyDeviceToHost, stream);
    }

    virtual cudaGraph_t getLossGraph (cudaStream_t cap_stream);
    virtual cudaGraph_t getDeltaGraph(cudaStream_t cap_stream);
};

template<typename T, typename L>
class MSE_Loss : public Loss<T, L> {
    virtual cudaGraph_t getLossGraph(cudaStream_t cap_stream) {
        //1.: Create graph
        cudaGraph_t lossGraph;
        cudaGraphCreate(&lossGraph, 0);

        //2.: Add node
        cudaGraphNode_t node;
        
        uint32_t outStateSizeBatched = layer->outStateShape.prod() * layer->batch_size;
        constexpr auto ldb = []__device__(T in1, T in2) { return (in1 - in2) * (in1 - in2) / (T)2; };

        void* lossArgs[] = {
            (void*)&layer->state,
            (void*)&target,
            (void*)&accumulator,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams lossParams{
            (void*)transform_reduce_indirection<T, decltype(ldb), DIVISIBILITY::DIVISIBLE, DIVISIBLE::DIVISIBLE, false>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),                                                                        //Grid dimensions
            dim3(32, 1, 1),                                                                                              //Block dimensions
            0u,                                                                                                          //Dyn. shared-mem per block in bytes
            (void**)&lossArgs,                                                                                           //Array of pointers to individual kernel arguments
            nullptr                                                                                                      //Pointer to kernel arguments in the "extra" format
        };
        if (outStateSizeBatched % 32)
            lossParams.func = (void*)transform_reduce<T, decltype(ldb), DIVISIBILITY::NOT_DIVISIBLE, DIVISIBLE::NOT_DIVISIBLE, false>;
        cudaGraphAddKernelNode(&node, lossGraph, nullptr, 0, &lossParams);

        //3.: Return
        return lossGraph;
    }

    virtual cudaGraph_t getDeltaGraph(cudaStream_t cap_stream) {
        //1.: Create graph
        cudaGraph_t deltaGraph;
        cudaGraphCreate(&deltaGraph, 0);

        //2.: Add node
        cudaGraphNode_t node;

        uint32_t outStateSizeBatched = layer->outStateShape.prod() * layer->batch_size;
        constexpr auto ldb = []__device__(T in1, T in2) { return in1 - in2; };

        void* deltaArgs[] = {
            (void*)&layer->state,
            (void*)&target,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams deltaParams{
            (void*)transform_indirection<T, decltype(ldb)>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),           //Grid dimensions
            dim3(32, 1, 1),                                 //Block dimensions
            0u,                                             //Dyn. shared-mem per block in bytes
            (void**)&deltaArgs,                             //Array of pointers to individual kernel arguments
            nullptr                                         //Pointer to kernel arguments in the "extra" format
        };
        if (outStateSizeBatched % 32)
            lossParams.func = (void*)transform_reduce<T, decltype(ldb), DIVISIBILITY::NOT_DIVISIBLE, DIVISIBLE::NOT_DIVISIBLE, false>;
        cudaGraphAddKernelNode(&node, deltaGraph, nullptr, 0, &deltaParams);

        //3.: Return
        return deltaGraph;
    }
};

template<typename T, typename L>
class MAE_Loss : public Loss<T, L> {
    virtual cudaGraph_t getLossGraph(cudaStream_t cap_stream) {
        //1.: Create graph
        cudaGraph_t lossGraph;
        cudaGraphCreate(&lossGraph, 0);

        //2.: Add node
        cudaGraphNode_t node;

        uint32_t outStateSizeBatched = layer->outStateShape.prod() * layer->batch_size;
        constexpr auto ldb = []__device__(T in1, T in2) { return abs(in1 - in2); };

        void* lossArgs[] = {
            (void*)&layer->state,
            (void*)&target,
            (void*)&accumulator,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams lossParams{
            (void*)transform_reduce_indirection<T, decltype(ldb), DIVISIBILITY::DIVISIBLE, DIVISIBLE::DIVISIBLE, false>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),                                                                        //Grid dimensions
            dim3(32, 1, 1),                                                                                              //Block dimensions
            0u,                                                                                                          //Dyn. shared-mem per block in bytes
            (void**)&lossArgs,                                                                                           //Array of pointers to individual kernel arguments
            nullptr                                                                                                      //Pointer to kernel arguments in the "extra" format
        };
        if (outStateSizeBatched % 32)
            lossParams.func = (void*)transform_reduce<T, decltype(ldb), DIVISIBILITY::NOT_DIVISIBLE, DIVISIBLE::NOT_DIVISIBLE, false>;
        cudaGraphAddKernelNode(&node, lossGraph, nullptr, 0, &lossParams);

        //3.: Return
        return lossGraph;
    }

    virtual cudaGraph_t getDeltaGraph(cudaStream_t cap_stream) {
        //1.: Create graph
        cudaGraph_t deltaGraph;
        cudaGraphCreate(&deltaGraph, 0);

        //2.: Add node
        cudaGraphNode_t node;

        uint32_t outStateSizeBatched = layer->outStateShape.prod() * layer->batch_size;
        constexpr auto ldb = []__device__(T in1, T in2) { return sign(in1-in2); };

        void* deltaArgs[] = {
            (void*)&layer->state,
            (void*)&target,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams deltaParams{
            (void*)transform_indirection<T, decltype(ldb)>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),           //Grid dimensions
            dim3(32, 1, 1),                                 //Block dimensions
            0u,                                             //Dyn. shared-mem per block in bytes
            (void**)&deltaArgs,                             //Array of pointers to individual kernel arguments
            nullptr                                         //Pointer to kernel arguments in the "extra" format
        };
        if (outStateSizeBatched % 32)
            lossParams.func = (void*)transform_reduce<T, decltype(ldb), DIVISIBILITY::NOT_DIVISIBLE, DIVISIBLE::NOT_DIVISIBLE, false>;
        cudaGraphAddKernelNode(&node, deltaGraph, nullptr, 0, &deltaParams);

        //3.: Return
        return deltaGraph;
    }
};

template<typename T, typename L>
class CrossEntropy_Loss : public Loss<T, L> {
    virtual cudaGraph_t getLossGraph(cudaStream_t cap_stream) {
        //1.: Create graph
        cudaGraph_t lossGraph;
        cudaGraphCreate(&lossGraph, 0);

        //2.: Add node
        cudaGraphNode_t node;

        uint32_t outStateSizeBatched = layer->outStateShape.prod() * layer->batch_size;
        constexpr auto ldb = []__device__(T in1, T in2) { return in2 * logarithm<T>(in1); };

        void* lossArgs[] = {
            (void*)&layer->state,
            (void*)&target,
            (void*)&accumulator,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams lossParams{
            (void*)transform_reduce_indirection<T, decltype(ldb), DIVISIBILITY::DIVISIBLE, DIVISIBLE::DIVISIBLE, false>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),                                                                        //Grid dimensions
            dim3(32, 1, 1),                                                                                              //Block dimensions
            0u,                                                                                                          //Dyn. shared-mem per block in bytes
            (void**)&lossArgs,                                                                                           //Array of pointers to individual kernel arguments
            nullptr                                                                                                      //Pointer to kernel arguments in the "extra" format
        };
        if (outStateSizeBatched % 32)
            lossParams.func = (void*)transform_reduce<T, decltype(ldb), DIVISIBILITY::NOT_DIVISIBLE, DIVISIBLE::NOT_DIVISIBLE, false>;
        cudaGraphAddKernelNode(&node, lossGraph, nullptr, 0, &lossParams);

        //3.: Return
        return lossGraph;
    }

    virtual cudaGraph_t getDeltaGraph(cudaStream_t cap_stream) {
        //1.: Create graph
        cudaGraph_t deltaGraph;
        cudaGraphCreate(&deltaGraph, 0);

        //2.: Add node
        cudaGraphNode_t node;

        uint32_t outStateSizeBatched = layer->outStateShape.prod() * layer->batch_size;
        constexpr auto ldb = []__device__(T in1, T in2) { return in2 / in1; };

        void* deltaArgs[] = {
            (void*)&layer->state,
            (void*)&target,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams deltaParams{
            (void*)transform_indirection<T, decltype(ldb)>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),           //Grid dimensions
            dim3(32, 1, 1),                                 //Block dimensions
            0u,                                             //Dyn. shared-mem per block in bytes
            (void**)&deltaArgs,                             //Array of pointers to individual kernel arguments
            nullptr                                         //Pointer to kernel arguments in the "extra" format
        };
        if (outStateSizeBatched % 32)
            lossParams.func = (void*)transform_reduce<T, decltype(ldb), DIVISIBILITY::NOT_DIVISIBLE, DIVISIBLE::NOT_DIVISIBLE, false>;
        cudaGraphAddKernelNode(&node, deltaGraph, nullptr, 0, &deltaParams);

        //3.: Return
        return deltaGraph;
    }
};

//================================================
//==================|Network|=====================
//================================================

template<typename T, typename L = T>      //T is type of data, L is type of learning rates 
class NetworkBuilder {
    /*
        Checkpoint file structure:
         - 7 bytes signature ("JVCHECK")
         - 2 bytes library version
         - 4 bytes the used type (T)

         - num_layers
         - identifiers of each layer plus activation
         - identifiers of optimizer
         
         - dump of "other" and "params" for each layer (and activation)
         - dump of "optBuf" memory for optimizer 
    */
private:
    Layer<T,L>* layers;                         //Holds information on layers
    uint32_t num_layers;                   //The number of layers
                                           
    uint32_t mem_align;                    //Garanties that arguments of "setMem" are always aligned up to this number of bytes.
                                           
    uint32_t batch_size;                   //Numberos samples per batch
                                           
    Optimizer<T, L> opt;                    //Optimizer. Only information. Memory is stored at the end of "other_mem"
                                           
    T* state_mem;                          //Has to be realloced when batch size changes.
    T* other_mem;                          //Fixed size. Also memory of optimizer.
    T* tmp_mem;                            //Temporary buffer, has to be realloced when batch size changes.

    std::vector<T**> indirection_pointers; //Array to indirection pointers

    /*
        Returns the total memory requirements of all layers with all their activation functions.
    */
    void getLayerMemoryRequirements(MemoryRequirement& state_requirement, MemoryRequirement& other_requirement, uint64_t& optimizables, MemoryRequirement& tmp_requirement) {
        //0.: Accumulators
        state_requirement = MemoryRequirement(0ull, 1u);
        other_requirement = MemoryRequirement(0ull, 1u);
        optimizables      = 0;
        tmp_requirement   = MemoryRequirement(0ull, 1u);

        //1.: Requirements of layers
        for (uint32_t layer_index = 0; layer_index != num_layers; layer_index++) {
            MemoryRequirement s, o, t;
            uint64_t op, no;

            layers[layer_index].getMemoryRequirements(s, o, op, t, no);

            s.alignment = max(s.alignment, mem_align);                                           //Alignment has to be at least the minimum required alignment
            state_requirement += s;

            o.alignment = max(o.alignment, mem_align);                                           //Alignment has to be at least the minimum required alignment
            other_requirement += o;

            optimizables += op;

            t.alignment = max(t.alignment, mem_align);                                           //Alignment has to be at least the minimum required alignment                                            
            tmp_requirement = max(tmp_requirement, t);                                           //Temporary memory is reused
        }
    }

public:
    /*
        Call order:
        1.: First constructor + allocate + initialize     or      second constructor
        2.: Get scheduler
    */

    NetworkBuilder(Layer<T, L>* layers, uint32_t num_layers, uint32_t mem_align, uint32_t batch_size, Optimizer<T, L> opt) :
        layers(layers),
        num_layers(num_layers),
        mem_align(mem_align),
        batch_size(batch_size),
        opt(opt),
        state_mem(nullptr),
        other_mem(nullptr),
        tmp_mem(nullptr),
        indirection_pointers()
    {
        printf("[INFO] Creating NetworkBuilder from given layers\n");

        if (layers[0].getLayerType() != LAYER_TYPE::INPUT) {
            fprintf(stderr, "[ERROR] The first layer must be an input layer while you supplied a layer of type %u", (uint32_t)layers[0].getLayerType());
            exit(-1);
        }

        //1.: Connect the layers
        for(uint32_t ind = 1; ind < num_layers; ind++)
            l[ind].setLayerBefore(&l[ind-1])
    }
    
    /*
        Loads from a checkpoint file.

        @param save_file : Path to the checkpoint (has to be generated using "compress" method).
        @param mem_align : Sets internal variable
        @param batch_size: Sets internal variable
    */
    NetworkBuilder(char* save_file, uint32_t mem_align, uint32_t batch_size):
        mem_align(mem_align), batch_size(batch_size)
    {
        //1.: Open file
        FILE* file = fopen(save_file, "rb");

        //2.: Check signature
        printf("[INFO] Creating NetworkBuilder from save file. Parsing checkpoint file...\n");
        
        char sig[7];
        fread(+sig, sizeof(char), 7, file);
        if(strcmp(sig, "JVCHECK", 7) == 0)
            printf("[INFO] \t - Signature matches\n");
        else {
            printf("[Error] \t - Signature mismatch! Should be \"%s\" while the checkpoint file supplied \"%.7s\".", "JVCHECK", sig);
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
            fprintf(stderr, "[WARNING] \t - The checkpoint file is of a newer version (%u) than the library (%u)! There is a good chance that there are safe inconsistencies!\n", (uint32_t)ver, AI_VERSION);

        //4.: Type
        uint32_t type;
        fread(&type, sizeof(uint32_t), 1, file);
        if(type == type_hash<T>())
            printf("[INFO] \t - Type matches.\n");
        else{
            fprintf(stderr, "[ERROR] \t - Type mismatch! You supplied %u while the checkpoint uses %u", type_hash<T>(), type);
            exit(-1);
        }

        //5.: Read in information on layers
        fread(&num_layers, sizeof(num_layers), 1, file);
        layers = (Layer<T>*)malloc(num_layers * sizeof(Layer<T>));
        Layer<T>* layer_before = nullptr;
        for (uint32_t layer_index = 0; layer_index != num_layers; layer_index++) {
            Layer<T>::getLayerFromCompression(file, &layers[layer_index]);

            layers[layer_index].setLayerBefore(layer_before);
            layers[layer_index].setBatchSize  (batch_size);
            
            //No need to set "state","other" to a specific value

            layer_before = &layers[layer_index];    //Update last layer pointer
        }


        if (layers[0].getLayerType() != LAYER_TYPE::INPUT) {
            fprintf(stderr, "[ERROR] The first layer must be an input layer while the first saved layer is of type %u", (uint32_t)layers[0].getLayerType());
            exit(-1);
        }

        //6.: Read in information on optimizer
        Optimizer<T, L>::getOptimizerFromCompression(file, &opt);
        
        //7.: Allocate memory
        allocate();

        //8.: Fill memory of layers (other. state is left unitialized)
        for (uint32_t layer_index = 0; layer_index != num_layers; layer_index++)
            layers[layer_index].initMemFromCompression(file);

        //9.: Fill memory of optimizer
        opt.initMemFromCompression(file);
    }

    /*
        Allocates memory for optimizer and layers (and activations). Will set internal pointers of every layer and optimizer (and activations).
    */
    void allocate(){
        //0.: Initialize variables
        MemoryRequirement state_requirement, other_requirement, tmp_requirement;
        uint64_t optimizables;
        getLayerMemoryRequirements(state_requirement, other_requirement, optimizables, tmp_requirement);

        //2.: Get requirements of optimizer
        opt.setNumOptimizables(optimizables);

        MemoryRequirement opt_other_req;
        opt.getMemoryRequirements(opt_other_req);
        opt_other_req.alignment = max(opt_other_req.alignment, mem_align);

        other_requirement += opt_other_req;

        //3.: Allocation
        printf("[INFO] Trying to allocate %llumb on gpu for the network... ", (state_requirement.num_bytes + other_requirement.num_bytes + tmp_requirement.num_bytes) / (1024ull * 1024ull));

        gpuErrchk(cudaMallocAligned(&other_mem, state_requirement));
        gpuErrchk(cudaMallocAligned(&state_mem, other_requirement));
        gpuErrchk(cudaMallocAligned(&  tmp_mem,   tmp_requirement));
        if(other_tmp_mem != nullptr && state_mem != nullptr && tmp_mem != nullptr)
            printf("Success!\n");
        else {
            printf("Failure!\n");
            std::exit(-1);
        }

        //4.: Set pointers of layers (and activations)
        T* state_mem_ = state_mem;
        T* other_mem_ = other_mem;
        for(uint32_t ind = 0; ind != num_layers; ind++)
            layers[ind].setMem(state_mem_, other_mem_);

        //5.: Set pointer of optimizer
        opt.setMem(other_mem_);
    }

    /*
        Initialize the allocated memory
    */
    void initialize(){
        //1.: Initialize layers (and activations)
        for(uint32_t ind = 0; ind != num_layers; ind++)
            layers[ind].initMem();

        //2.: Initialize optimizer
        opt.initMem();
    }

    /*
        Resets the batch size and handels all reallocations.

        Note: This clears the state of each layer. Execution graphs need to be rebuild
    */
    void resetBatchSize(uint32_t new_batchSize){
        //1.: Reset variables
        batch_size = new_batchSize;
        for(uint32_t ind = 0, ind != num_layers; ind++)
            layers[ind].setBatchSize(new_batchSize);

        //2.: Reallocate
        MemoryRequirement state_requirement, other_requirement, tmp_requirement;
        uint64_t optimizables, num_nodes;
        getLayerMemoryRequirements(state_requirement, other_requirement, optimizables, tmp_requirement, num_nodes);

        //nothing changed for opt
        
        cudaFree(state_mem);
        cudaFree(tmp_mem);
        gpuErrchk(cudaMallocAligned(&state_mem, other_requirement));
        gpuErrchk(cudaMallocAligned(&  tmp_mem,   tmp_requirement));
        
        //3.: Set new memory
        T* state_mem_ = state_mem;
        T* other_mem_ = other_mem;
        for(uint32_t ind = 0; ind != num_layers; ind++)
            layers[ind].setMem(state_mem_, other_mem_);

        //optimizer should stay unchanged
    }

    /*
        Builds cuda execution graph
    */
    cudaGraph_t getForwardGraph(cudaStream_t recording_stream){
        //1.: Create graph
        cudaGraph_t forwardGraph;
        cudaGraphCreate(&forwardGraph, 0);

        //2.: Dependencies
        Dependencies depPrevState;

        //3.: Layer after dataset
        std::vector<T**> indirection_pointers_forward = layers[1].forwardProp(forwardGraph, depPrevState, recording_stream, tmp_mem, true);
        indirection_pointers.insert(std::end(indirection_pointers), std::begin(indirection_pointers_forward), std::end(indirection_pointers_forward)); //New indirection pointers

        //4.: Other layers
        for (uint32_t ind = 2; ind < num_layers; ind++)
            layers[ind].forwardProp(forwardGraph, depPrevState, recording_stream, tmp_mem, false);

        //5.: Return
        return forwardGraph;
    }

    /*
        Builds cuda execution graph
    */
    cudaGraph_t getBackwardsGraph(cudaStream_t recording_stream){
        //1.: Create graph
        cudaGraph_t backwardGraph;
        cudaGraphCreate(&backwardGraph, 0);

        //2.: Dependencies
        Dependencies depState;

        //3.: Other layers
        cudaGraphNode_t* depNode = nullptr;
        uint64_t optimizable_index = 0;
        for (uint32_t ind = num_layers - 1; ind > 1; ind--)
            layers[ind].backwardProp(backwardGraph, depState, &opt, optimizable_index, recording_stream, tmp_mem, false);

        //4.: Layer after dataset
        std::vector<T**> indirection_pointers_backwards = layers[1].backwardProp(backwardGraph, depState, &opt, optimizable_index, recording_stream, tmp_mem, true);
        indirection_pointers.insert(std::end(indirection_pointers), std::begin(indirection_pointers_backwards), std::end(indirection_pointers_backwards)); //New indirection pointers

        //5.: Return
        return backwardGraph;
    }

    void getFirstAndLastLayer(Layer<T, L>*& firstLayer, Layer<T, L>*& lastLayer) {
        firstLayer = &layers[0];
        lastLayer  = &layer[num_layers - 1];
    }

    /*
        Safes a checkpoint file.
    */
    void compress(char* out_file) {
        //1.: Open output file
        FILE* file = fopen(out_file, "wb");

        //2.: Write header
        char     signature[] = "JVCHECK";
        uint16_t version = AI_VERSION;
        uint32_t type = type_hash<T>();
        fwrite(+signature, sizeof(signature) - 1, 1, file); //Signature
        fwrite(&version, sizeof(version), 1, file); //Library version
        fwrite(&type, sizeof(type), 1, file); //The underlying type

        //3.: Information on layers
        fwrite(&num_layers, sizeof(num_layers), 1, file);
        for (uint32_t layer_index = 0; layer_index != num_layers; layer_index++)
            layers[layer_index].compress<false>(file);

        //4.: Information on optimizer
        opt.compress<false>(file);

        //5.: Memory of layers
        for (uint32_t layer_index = 0; layer_index != num_layers; layer_index++) 
            layers[layer_index].compress<true>(file);

        //6.: Memory of optimizer
        opt.compress<true>(file);

        //7.: Close file
        fclose(file);
    }
};

//==================================================
//==================|Scheduler|=====================
//==================================================
#define PI 3.141592653589793238462643383267502884197
enum LRSCHEDULE_TYPE : uint32_t { LINEAR, COSINE, DECAY, EXPONENTIAL, DEMON };
template<typename L>
struct LRSchedule {
    L lr_sta, lr_sto;
    uint32_t warm_restarts;

    virtual L getLrEpoch(uint32_t epoch, uint232_t num_epochs);
    virtual LRSCHEDULE_TYPE getType();
};

template<typename L>
struct Linear_LRSchedule : public LRSchedule<L> {
    virtual L getLrEpoch(uint32_t epoch, uint232_t num_epochs) override {
        double round;
        L frac = modf(epoch / (warm_restarts ? warm_restarts : num_epochs), &nullptr);

        return lr_sta - (lr_sta - lr_sto) * frac; 
    }
    virtual LRSCHEDULE_TYPE getType() { return LRSCHEDULE_TYPE::LINEAR; }
};

template<typename L>
struct Cosine_LRSchedule : public LRSchedule<L> {
    virtual L getLrEpoch(uint32_t epoch, uint232_t num_epochs) override {
        double round;
        L frac = modf(LRSchedule / (warm_restarts ? warm_restarts : num_epochs), &nullptr);

        return lr_sto + (lr_sta - lr_sto) * ((L)1 + (L)cos(PI * frac)) / (L)2; 
    }
    virtual LRSCHEDULE_TYPE getType() { return LRSCHEDULE_TYPE::COSINE; }
};

template<typename L>
struct Decay_LRSchedule : public LRSchedule<L> {
    virtual L getLrEpoch(uint32_t epoch, uint232_t num_epochs) override {
        double round;
        L frac = modf(epoch / (warm_restarts ? warm_restarts : num_epochs), &nullptr);

        return lr_sta * (L)1 / ((L)1 + ((lr_sta / lr_sto) - (L)1) * frac);
    }
    virtual LRSCHEDULE_TYPE getType() { return LRSCHEDULE_TYPE::DECAY; }
};

template<typename L>
struct Exponential_LRSchedule : public LRSchedule<T> {
    virtual L getLrEpoch(uint32_t epoch, uint232_t num_epochs) override {
        double round;
        L frac = modf(epoch / (warm_restarts ? warm_restarts : num_epochs), &nullptr);

        return lr_sta * pow(lr_sto / lr_sta, frac);
    }
    virtual LRSCHEDULE_TYPE getType() { return LRSCHEDULE_TYPE::EXPONENTIAL; }
};

template<typename L>
struct Demon_LRSchedule : public LRSchedule<T> {
    virtual L getLrEpoch(uint32_t epoch, uint232_t num_epochs) override {
        double round;
        L frac = modf(epoch / (warm_restarts ? warm_restarts : num_epochs), &nullptr);

        return (((L)1 - frac) * lr_sta) / ((L)1 - frac * lr_sta);
    }
    virtual LRSCHEDULE_TYPE getType() { return LRSCHEDULE_TYPE::DEMON; }
};

template<typename L>
struct LRScheduleComplete {
    LRSchedule<L> warmup, regular;
    uint32_t warmup_length;

    L getLrEpoch(uint32_t epoch, uint32_t num_epochs) {
        if (epoch < warmup_length)
            return warmup.getLrEpoch(epoch, warmup_length);
        else
            return regular.getLrEpoch(epoch - warmup_length, num_epochs - warmup_length);
    }
};

template<typename T, typename L = T>
class Scheduler{
    //Components
    NetworkBuilder<T, L>* network; //Also optimizer
    Loss<T>*              loss;
    DatasetHandler<T>*    dataset;

    //Hyperparameters
    uint32_t num_epochs;
    uint32_t steps_per_epoch;

    LRScheduleComplete<L> alpha_schedule, beta1_schedule, beta2_schedule;

    uint32_t plataue_start;
    T        plateau_threshold;
    uint32_t patienceLRChange, patienceEarlyStopping;   //0 to disable
    L        lrPlateauFactor;
    L        lrAccumulatedFactor;

    T        loss_goal;             //Stop early, when validation loss is under this goal
    
    //Execution stuff
    cudaStream_t execStream;

    //Monitoring
    std::vector<T> alpha_history, beta1_history, beta2_history;
    std::vector<L> loss_history;

public:
    Scheduler() = default;
    void setNumRuns(uint32_t num_epochs_, uint32_t steps_per_epoch_) {
        num_epochs = num_epochs_;
        steps_per_epoch = steps_per_epoch_;
    }

    void setLRSchedule(LRScheduleComplete<L> alpha_schedule_, LRScheduleComplete<L> beta1_schedule_, LRScheduleComplete<L> beta2_schedule_) {
        alpha_schedule = alpha_schedule_;
        beta1_schedule = beta1_schedule_;
        beta2_schedule = beta2_schedule_;
    }

    void setPlateau(uint32_t start, T threshold, uint32_t patienceLRChange_ = 0, L lrPlateauFactor_ = (L)0.1, uint32_t patienceEarlyStopping_ = 0) {
        if (0 < patienceEarlyStopping && patienceEarlyStopping < patienceLRChange)
            fprintf(stderr, "[WARNING] Early stopping will occur before the learning rate change. Thus, a learning rate change resulting of a plateau will never occur!\n");
        
        plateau_start         = start;
        plateau_thresholde    = threshold;
        patienceLRChange      = patienceLRChange_;
        lrPlateauFactor       = lrPlateauFactor_;
        patienceEarlyStopping = patienceEarlyStopping_;
    }

    void setLossGoal(T goal) {
        loss_goal = goal;
    }

    void launch(uint32_t dataset_workers, uint32_t dataset_streams, bool debug_window) {
        //0.: Test whether components match
        Image_Shape in_shape, out_shape;
        uint32_t training_samples, validation_samples;
        dataset->getAugmentedShapes(in_shape, out_shape);
        dataset->getNumSamples(training_samples, validation_samples);

        Layer<T, L>* firstLayer_, * lastLayer;
        network->getFirstAndLastLayer(firstLayer, lastLayer);

        if (firstLayer->getLayerType() != LAYER_TYPE::INPUT) {
            fprintf(stderr, "[ERROR] First layer of network has to be an input layer, yet it has type %u", firstLayer->getLayerType());
            exit(-1);
        }

        Input_Layer<T, L>* firstLayer = firstLayer_;

        if (firstLayer->outStateShape !=  in_shape) {
            if (firstLayer->outStateShape.prod() != in_shape.prod) {
                fprintf(stderr, "[WARNING] The input layer of the network has the right size, yet its shape(%u x %u x %u) is not the same as the sample from the dataset(%u x %u x %u)  ßn",
                    firstLayer->outStateShape.x, firstLayer->outStateShape.y, firstLayer->outStateShape.z,
                    in_shape.x, in_shape.y, in_shape.z
                );
            }
            else {
                fprintf(stderr, "[ERROR] The shape of the input layer(%u x %u x %u) does not match the shape of samples from the dataset(%u x %u x %u)",
                    firstLayer->outStateShape.x, firstLayer->outStateShape.y, firstLayer->outStateShape.z,
                    in_shape.x, in_shape.y, in_shape.z
                );
                exit(-1);
            }
        }
        if ( lastLayer->outStateShape != out_shape) {
            if (lastLayer->outStateShape.prod() != out_shape.prod) {
                fprintf(stderr, "[WARNING] The last layer of the network has the right size, yet its shape(%u x %u x %u) is not the same as the sample from the dataset(%u x %u x %u)           ßn",
                    lastLayer->outStateShape.x, lastLayer->outStateShape.y, lastLayer->outStateShape.z,
                    out_shape.x, out_shape.y, out_shape.z
                );
            }
            else {
                fprintf(stderr, "[ERROR] The shape of the last layer(%u x %u x %u) does not match the shape of samples from the dataset(%u x %u x %u)",
                    lastLayer->outStateShape.x, lastLayer->outStateShape.y, lastLayer->outStateShape.z,
                    out_shape.x, out_shape.y, out_shape.z
                );
                exit(-1);
            }
        }

        if (loss->layer != lastLayer)
            fprintf(stderr, "[WARNING] The loss is computed based of the output of an intermidiate layer (pointer %p instead of %p)\n", loss->layer, lastLayer);
        

        //1.: Build graphs
        cudaStreamCreateWithFlags(&execStream, cudaStreamNonBlocking);

        cudaGraph_t trainStep, validationStep;
        cudaGraphCreate(&trainStep, 0);
        cudaGraphCreate(&validationStep, 0);
        cudaGraphNode_t node, depNode;

        //1.1.: trainStep
        //Forward propagation
        cudaGraphAddChildGraphNode(&node, trainStep, nullptr, 0, network->getForwardGraph(execStream));
        depNode = node;

        //Last Deltas
        cudaGraphAddChildGraphNode(&node, trainStep, depNode, 1, loss->getDeltaGraph(execStream));
        depNode = node;

        //Backward propagation
        cudaGraphAddChildGraphNode(&node, trainStep, depNode, 1, network->getBackwardsGraph(execStream));
        depNode = node;

        //1.2.: validationStep
        //Forward propagation
        cudaGraphAddChildGraphNode(&node, validationStep, nullptr, 0, network->getForwardGraph(execStream));
        depNode = node;

        //Compute loss
        cudaGraphAddChildGraphNode(&node, validationStep, depNode, 1, loss->getLossGraph(execStream));
        depNode = node;


        //2.: Instantiate graphs
        cudaGraphExec_t trainExec, validationExec;
        char* errorBuf[512] = ",";
        cudaGraphNode_t errNode;

        cudaGraphInstantiate(&trainExec, trainStep, &errNode, +errorBuf, 512);
        if (errorBuf[0] != ',') {
            fprintf(stderr, "[ERROR] The following error arose during the instantiation of the training graph: %s", +errorBuf);
            exit(-1);
        }

        cudaGraphInstantiate(&validationExec, validationStep, &errNode, +errorBuf, 512);
        if (errorBuf[0] != ',') {
            fprintf(stderr, "[ERROR] The following error arose during the instantiation of the validation graph: %s", +errorBuf);
            exit(-1);
        }

        //3.: Run
        //Pointer to current gpu tile
        T** in, ** out, * loss_buf;
        L* alpha_buf, * beta1_buf, * beta2_buf;
        cudaMallocHost((void**)&in , sizeof(T*));
        cudaMallocHost((void**)&out, sizeof(T*));
        cudaMallocHost(&loss_buff, sizeof(T));
        cudaMallocHost(&alpha_buf, sizeof(L));
        cudaMallocHost(&beta1_buf, sizeof(L));
        cudaMallocHost(&beta2_buf, sizeof(L));

        //Start dataset workers
        printf("[INFO] Starting dataset workers\n");
        dataset->start_workers(dataset_workers, dataset_streams, WORKER_STATUS::TRAINING);

        printf("[INFO] Starting training loop\n");
        for (uint32_t epoch = 0; epoch != num_epochs; epoch++) {
            //Compute LRs
            *alpha_buf = alpha_schedule.getLrEpoch(epoch, num_epochs) * lrAccumulatedFactor;
            *beta1_buf = beta1_schedule.getLrEpoch(epoch, num_epochs) * lrAccumulatedFactor;
            *beta2_buf = beta2_schedule.getLrEpoch(epoch, num_epochs) * lrAccumulatedFactor;

            alpha_history.append(*alpha_buf);
            beta1_history.append(*beta1_buf);
            beta2_history.append(*beta2_buf);
            
            //Set LRs
            network->opt.setLR(alpha_buf, beta1_buf, beta2_buf, execStream);

            //Training
            for (uint32_t step = 0; step != steps_per_epoch; step++) {
                //Set input and output
                T* in_, *out_;
                dataset->advance<true>(in_, out_);
                *in  = in_;
                *out = out_;

                firstLayer->setInputPointer(in, execStream);
                loss->setTarget(out, execStream);

                cudaGraphLaunch(trainExec, execStream);
            }

            //Validation
            for (uint32_t step = 0; step != validation_samples; step++) {
                //Set input and output
                T* in_, *out_;
                dataset->advance<false>(in_, out_);
                *in  = in_;
                *out = out_;

                firstLayer->setInputPointer(in, execStream);
                loss->setTarget(out, execStream);

                cudaGraphLaunch(validationExec, execStream);
            }

            loss->getAccumulator(loss_buff, execStream);
            loss->clearAccumulator(execStream);
            loss_history.push_back(*loss_buff / (T)validation_samples);
        
            //Debugging
            if (debug) {
                printf("Epoch %u/%u | Loss %d |\n", epoch, num_epochs, loss_history.back());
            }

            //Inspect loss
            if (loss_history.back() < loss_goal) {
                printf("[INFO] Loss goal reached!\n");
                return;
            }

            if (patienceLRChange != 0 && patienceLRChange + plataue_start < epoch) {
                bool plateau = true;
                for (uint32_t pat = 1; pat <= patienceLRChange; pat++) {
                    if (loss_history.back() < plateau_threshold * loss_history[loss_history.size() - pat]) {
                        plateau = false;
                        break;
                    }
                }
                if (plateau) {
                    printf("[INFO] Plateau detected. Adapting learning rate\n");
                    lrAccumulatedFactor *= lrPlateauFactor;
                }
            }
            if (patienceEarlyStopping != 0 && patienceEarlyStopping + plataue_start < epoch) {
                bool plateau = true;
                for (uint32_t pat = 1; pat <= patienceEarlyStopping; pat++) {
                    if (loss_history.back() < plateau_threshold * loss_history[loss_history.size() - pat]) {
                        plateau = false;
                        break;
                    }
                }
                if (plateau) {
                    printf("[INFO] Early stopping triggered!\n");
                    return;
                }

            }
        }
    }

//https://arxiv.org/pdf/1812.01187.pdf
//https://arxiv.org/pdf/1608.03983.pdf
}

//Old implementation
#if 0
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
        @param negate: If this is true, multiply -f'(x) to o instead of f'(x)
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
#endif

//=============================================
//==================|MAIN|=====================
//=============================================

int main()
{
    CUBLAS_ERROR(cublasCreate(&cublas_handle));
    //Logging
    cublasSetAtomicsMode(cublas_handle, CUBLAS_ATOMICS_ALLOWED);
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    //Set cublas workspace

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

//Windows: clang++ Network.cpp -o Network.exe -I"D:\Librarys\CImg-2.9.2_pre070420" -I"D:\Librarys\VS-NuGet\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include" -I"D:\Librarys\GLFW\include" -I"D:\Librarys\glew-2.1.0\include" -I"D:\Librarys\freetype-2.10.3\include" -L"D:\Librarys\GLFW\lib" -L"D:\Librarys\glew-2.1.0\lib\Release\x64" -L"D:\Librarys\VS-NuGet\lib" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64" -L"D:\Librarys\freetype-2.10.3\objs" -O0 -march=native -m64 -std=c++17 -Wall -lzlib -llibpng16 -ljpeg -lkernel32 -luser32 -lgdi32 -lopengl32 -lglu32 -lglew32 -lglfw3dll -lpsapi -lwinspool -lcomdlg32 -ladvapi32 -lshell32 -lole32 -loleaut32 -luuid -lodbc32 -lodbccp32 -lcudart_static -lcublas -lfreetype -g -DDEBUG
