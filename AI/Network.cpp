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

#define CUDNN_LOGDEST_DBG stdout
#define CUDNN_LOGINFO_DBG 1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

//#include <mma.h>
//#include <cublasXt.h>

#include <inttypes.h>
#include <stdio.h>
#include <utility>
#include <vector>

#include "util.cpp" 

//Quick fix: TODO/FIXIT: Remove this
#define EXTENDED_CONSTEXPR
#define CPP20_CONSTEXPR

#if defined(_MSC_VER)                        //For Host side compilation
namespace cooperative_groups {};
#include "kernel.cu"
#endif


#define protected public
#define private public

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

//TODO: Convolutional layer after dataset
//TODO: Tensor operations for cudnn
//TODO: Padding
//TODO: Different computiation types
//TODO: Destructors
//TODO: Use cudnn frontend c++ api
//TODO: Adam
//TODO: Pooling layer
//TODO: Learn activation parameters
//TODO: Implement regularization and decay in optimizer (seperate for weights and biases as according to "Bag of tricks for Image classification using CNNs", biases should not be regularized)
//TODO: Dataset must load validation samples sequencially
//TODO: Implement serialization and other derived class methods in LRScheduler and Loss class
//TODO: constexpr the "getMemoryRequirements" functions. Also __restrict__ and const in other functions
//TODO: Expand Scheduler debugging with: Show lr, loss, episode, timings, in-&output, model, occupiancy ...
//TODO: Change batchSize in Scheduler, get Input_Layer from every layer, NetworkBuilder takes argument whether first layer is the dataset or not
//TODO: Check launch parameters and smem
//TODO: Check comments, move files
//TODO: CASH COHERENCE

//=========================================================
//==================|HELPER FUNCTIONS|=====================
//=========================================================

#define LAUNCH_PARAM(N) max(1, (int)(1. / (10. / ((double)(1 << 13)) + 32. / ((double)(N))))), 32
#define GRID_SIZE(N) max(1, (int)(1. / (10. / ((double)(1 << 13)) + 32. / ((double)(N))))), 1, 1
//#define STRIDES_ARRAY(dim) {dim[1] * dim[2] * dim[3] * dim[4] * dim[5],  dim[2] * dim[3] * dim[4] * dim[5], dim[3] * dim[4] * dim[5], dim[4] * dim[5], dim[5], 1} /*Assuming NCHW memory layout*/
#define STRIDES_ARRAY(dim) {dim[1] * dim[2] * dim[3],  dim[2] * dim[3], dim[3], 1} /*Assuming NCHW memory layout*/

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
template<typename T, bool trans_A, bool trans_B, bool overwrite>
inline cudaGraph_t getMatmulGraph(T* A, T* B, T* C, uint32_t y1, uint32_t x1, uint32_t x2, cudaStream_t captureStream) {
        //0.: Make sure T is a recognized type
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value || std::is_same<T, half>::value, "[Error] Matrix multiplication is not supported with this type!");
        
        //1.: Start stream capture
        cudaGraph_t capGraph;
        cudaStreamBeginCapture(captureStream, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal);
        
        //2.: Enqueue cublas kernel
        if constexpr (std::is_same<T, float>::value)
            cublasSgemm( cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, cublasConst.f[1], (float*) A, trans_A?x1:y1, (float*) B, trans_B?x2:x1, cublasConst.f[!overwrite], (float*)C, y1);
        if constexpr (std::is_same<T, double>::value)
            cublasDgemm( cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, cublasConst.d[1], (double*)A, trans_A?x1:y1, (double*)B, trans_B?x2:x1, cublasConst.d[!overwrite], (double*)C, y1);
        if constexpr (std::is_same<T, half>::value)
            cublasGemmEx(cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, cublasConst.h[1], (half*)  A, CUDA_R_16F, trans_A?x1:y1, (half*)B, CUDA_R_16F, trans_B?x2:x1, cublasConst.h[!overwrite], (half*)C, CUDA_R_16F, y1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
        //3.: Stop capture and return graph
        cudaStreamEndCapture(captureStream, &capGraph);
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
    cudaStreamBeginCapture(captureStream, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal);

    //2.: Enqueue cublas kernel
    if constexpr (std::is_same<T, float>::value)
        cublasSgemmBatched (cublas_handle, trans_A ? CUBLAS_OP_T : CUBLAS_OP_N, trans_B ? CUBLAS_OP_T : CUBLAS_OP_N, y1, x2, x1, cublasConst.f[1], (float**)A, trans_A ? x1 : y1, (float**)B, trans_B ? x2 : x1, cublasConst.f[!overwrite], (float**)C, y1, 1);
    if constexpr (std::is_same<T, double>::value)
        cublasDgemmBatched (cublas_handle, trans_A ? CUBLAS_OP_T : CUBLAS_OP_N, trans_B ? CUBLAS_OP_T : CUBLAS_OP_N, y1, x2, x1, cublasConst.d[1], (double**)A, trans_A ? x1 : y1, (double**)B, trans_B ? x2 : x1, cublasConst.d[!overwrite], (double**)C, y1, 1);
    if constexpr (std::is_same<T, half>::value)
        cublasGemmBatchedEx(cublas_handle, trans_A ? CUBLAS_OP_T : CUBLAS_OP_N, trans_B ? CUBLAS_OP_T : CUBLAS_OP_N, (int)y1, (int)x2, (int)x1, (void*)cublasConst.h[1], (void**)A, CUDA_R_16F, trans_A ? x1 : y1, (void**)B, CUDA_R_16F, trans_B ? x2 : x1, (void*)cublasConst.h[!overwrite], (void**)C, CUDA_R_16F, (int)y1, 1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    //3.: Stop capture and return graph
    cudaStreamEndCapture(captureStream, &capGraph);
    return capGraph;
}

template<typename T>
inline void addElementwiseMultNode(cudaGraph_t graph, T* A, T* B, uint32_t len, cudaGraphNode_t& node){
    EXTENDED_CONSTEXPR auto ldb = []__device__(T a, T b) EXTENDED_CONSTEXPR -> T { return a * b; };
    
    void* elemMultArgs[] = {
        (void*)& A,
        (void*)& B, 
        (void*)& len,
        (void*)& ldb
    };
    cudaKernelNodeParams elemMultParam {
        (void*)&transform2<T, decltype(ldb)>, //Function pointer
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

//Grid size has to be "batch_size"
template<typename T, uint32_t blockSize, bool nIsPow2, bool safe>
__global__ void softmax(T* mem, uint32_t outStateSize, uint32_t batch_size) {
    //Each block is supposed to do one batch.

    extern __shared__ T sdata[];
    T* in = mem + blockIdx.x * outStateSize;

    if constexpr (safe) {
        constexpr auto identity = []__device__(T in) constexpr -> T { return in; };
        transformReduceBlockMultipleElementsFragmented<T, decltype(identity), reduction_max<T>, false, blockSize, nIsPow2, 0u, 0u, 8u, false>(in, outStateSize, sdata, identity, reduction_max<T>(), sdata);
        __syncthreads();
        T maxBlock = sdata[blockIdx.x];

        auto ldb = [&maxBlock]__device__(T in) -> T { return exponential<T>(in - maxBlock); };
        transformReduceBlockMultipleElementsFragmented<T, decltype(ldb), reduction_add<T>, true, blockSize, nIsPow2, 0u, 0u, 8u, false>(in, outStateSize, sdata, ldb, reduction_add<T>(), sdata);
    }
    else {
        constexpr auto ldb = []__device__(T in) constexpr -> T { return exponential<T>(in); };
        transformReduceBlockMultipleElementsFragmented<T, decltype(ldb), reduction_add<T>, true, blockSize, nIsPow2, 0u, 0u, 8u, false>(in, outStateSize, sdata, ldb, reduction_add<T>(), sdata);
    }
    //sdata contains the sums
    __syncthreads();
    T accu = sdata[blockIdx.x];

    uint32_t i = threadIdx.x;
    while (i < outStateSize) {
        in[i] /= accu;

        i += blockSize;
    }
}

//Grid size has to be "batch_size"
//temp is device pointer
template<typename T, uint32_t blockSize, bool nIsPow2, bool safe>
__global__ void softmaxTemp(T* mem, uint32_t outStateSize, uint32_t batch_size, T* temp) {
    //Each block is supposed to do one batch.

    extern __shared__ T sdata[];
    T* in = mem + blockIdx.x * outStateSize;

    T temperature = *temp;

    if constexpr (safe) {
        constexpr auto identity = []__device__(T in) constexpr -> T { return in; };
        transformReduceBlockMultipleElementsFragmented<T, decltype(identity), reduction_max<T>, false, blockSize, nIsPow2, 0u, 0u, 8u, false>(in, outStateSize, sdata, identity, reduction_max<T>(), sdata);
        __syncthreads();
        T maxBlock = sdata[blockIdx.x];

        auto ldb = [&temperature, &maxBlock]__device__(T in) -> T { return exponential<T>(temperature * in - maxBlock); };
        transformReduceBlockMultipleElementsFragmented<T, decltype(ldb), reduction_add<T>, true, blockSize, nIsPow2, 0u, 0u, 8u, false>(in, outStateSize, sdata, ldb, reduction_add<T>(), sdata);
    }
    else {
        auto ldb = [&temperature]__device__(T in) -> T { return exponential<T>(temperature * in); };
        transformReduceBlockMultipleElementsFragmented<T, decltype(ldb), reduction_add<T>, true, blockSize, nIsPow2, 0u, 0u, 8u, false>(in, outStateSize, sdata, ldb, reduction_add<T>(), sdata);
    }
    //sdata contains the sums

    __syncthreads();
    T accu = sdata[blockIdx.x];

    uint32_t i = threadIdx.x;
    while (i < outStateSize) {
        in[i] /= accu;

        i += blockSize;
    }
}

template<typename T>
__global__ void softmax_deriv(T* mem, T* out, uint32_t outStateSize, uint32_t batch_size) {
    //Grid stride loop
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < outStateSize * outStateSize * batch_size) {
        uint32_t pos   = i / batch_size;
        uint32_t batch = i %  batch_size;
        uint32_t x = pos / outStateSize;
        uint32_t y = pos % outStateSize;

        //printf("%f %f %u\n", *(mem + outStateSize * batch + y), *(mem + outStateSize * batch + x), i);
        //printf("%p %p %p %u %u %u %u %u %u\n", out + outStateSize * outStateSize * batch + outStateSize * x + y, mem + outStateSize * batch + y, mem + outStateSize * batch + x, i, pos, batch, x, y, outStateSize);
        //printf("%u %u %f\n", x, y, *(mem + outStateSize * batch + y) * ((x == y) - *(mem + outStateSize * batch + x)));
        *(out + outStateSize * outStateSize * batch + outStateSize * x + y) = *(mem + outStateSize * batch + y) * ((x == y) - *(mem + outStateSize * batch + x));

        i += gridDim.x * blockDim.x;
    }
}

//temp is device pointer 
template<typename T>
__global__ void softmaxTemp_deriv(T* mem, T* out, uint32_t outStateSize, uint32_t batch_size, T* temp) {
    T temperature = *temp;

    //Grid stride loop
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < outStateSize * outStateSize * batch_size) {
        uint32_t pos = i / batch_size;
        uint32_t batch = i - pos * batch_size;
        uint32_t x = pos / outStateSize;
        uint32_t y = pos - x * outStateSize;

        *(out + outStateSize * outStateSize * batch + outStateSize * y + x) = temperature * *(mem + outStateSize * batch + y) * ((x == y) - *(mem + outStateSize * batch + x));

        i += gridDim.x * blockDim.x;
    }
}

template<typename T, typename F>
__global__ void transform_indirection(T* in1, T** in2_, uint32_t n, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T* in2 = *in2_;

    for (int i = idx; i < n / 4; i += blockDim.x * gridDim.x) {
        var4<T> val1 = reinterpret_cast<var4<T>*>(in1)[i];
        var4<T> val2 = reinterpret_cast<var4<T>*>(in2)[i];
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

template<typename T, typename F>
__global__ void transform4_indirection(T* in1, T** in2_, T* out, uint32_t n, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T* in2 = *in2_;

    for (int i = idx; i < n / 4; i += blockDim.x * gridDim.x) {
        var4<T> val1 = reinterpret_cast<var4<T>*>(in1)[i];
        var4<T> val2 = reinterpret_cast<var4<T>*>(in2)[i];
        val1.a = f(val1.a, val2.a);
        val1.b = f(val1.b, val2.b);
        val1.c = f(val1.c, val2.c);
        val1.d = f(val1.d, val2.d);
        reinterpret_cast<var4<T>*>(out)[i] = val1;
    }
    int i = idx + n / 4 * 4;
    if (i < n)
        out[i] = f(in1[i], in2[i]);
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
    int b = idx % batch_size;
    idx /= batch_size;
    int y = idx % w_y;
    int x = idx / w_y;

    if (idx < w_y * w_x * batch_size) {
        //printf("%d: \t %.16f \t %.16f \t %.16f \t %.16f \t %.16f\n", y + x * w_y, weigths[y + x * w_y], -*lrate, in[b * w_x + x], delta[b * w_y + y], -*lrate * in[b * w_x + x] * delta[b * w_y + y]);

        atomicAdd(&weigths[y + x * w_y], -*lrate * in[b * w_x + x] * delta[b * w_y + y]);
    } //index of "weigths" is "idx", compiler will optimize it away
}

//Same as before but "in" is a indirection pointer
template<typename T, typename L>
__global__ void sgdMul_indirection(T* weigths, T* delta, T** in_, L* lrate, uint32_t w_y, uint32_t w_x, uint32_t batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T* in = *in_;
    int b = idx % batch_size;
    idx /= batch_size;
    int y = idx % w_y;
    int x = idx / w_y;

    if (idx < w_y * w_x * batch_size) {

        //printf("%f %f %f %f | %u %u %u %u\n", weigths[y + x * w_y], -*lrate, in[b * w_x + x], delta[b * w_y + y], idx, b, x, y);

        atomicAdd(&weigths[y + x * w_y], -*lrate * in[b * w_x + x] * delta[b * w_y + y]);
    } //index of "weigths" is "idx", compiler will optimize it away
}

//Assumes that "in" only stores "1" (which is the case for the input of biases). Thus, the parameter can be ommited. Therefore only one size paramter is needed as well an "w_x" can be omitted. "w_y" is now the size of "weights".
template<typename T, typename L>
__global__ void sgdSimple(T* weigths, T* delta, L* _lrate, uint32_t w_y, uint32_t batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx % batch_size;
    idx /= batch_size;

    L lrate = -*_lrate;
    if (idx < w_y * batch_size) {
        atomicAdd(&weigths[idx], lrate * delta[b * w_y + idx]);
    } //index of "weigths" is "idx", compiler will optimize it away
}

template<typename T, typename L>
__global__ void sgdDebugMul(T* weigths, T* delta, T* in, L* lrate, uint32_t w_y, uint32_t w_x, uint32_t batch_size, uint32_t num_optimizables) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < w_y * w_x * batch_size) {
        int b = idx % batch_size;
        idx /= batch_size;
        int y = idx % w_y;
        int x = idx / w_y;

        //printf("%d: \t %.16f \t %.16f \t %.16f \t %.16f \t %.16f\n", y + x * w_y + num_optimizables * b, weigths[y + x * w_y + num_optimizables * b], -*lrate, in[b * w_x + x], delta[b * w_y + y], -*lrate * in[b * w_x + x] * delta[b * w_y + y]);
        atomicAdd(&weigths[y + x * w_y + num_optimizables * b], -*lrate * in[b * w_x + x] * delta[b * w_y + y]);
        //printf("%d: \t %.16f\n", y + x * w_y + num_optimizables * b, weigths[y + x * w_y + num_optimizables * b]);
    } //index of "weigths" is "idx", compiler will optimize it away
}

//Same as before but "in" is a indirection pointer
template<typename T, typename L>
__global__ void sgdDebugMul_indirection(T* weigths, T* delta, T** in_, L* lrate, uint32_t w_y, uint32_t w_x, uint32_t batch_size, uint32_t num_optimizables) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < w_y * w_x * batch_size) {
        T* in = *in_;
        int b = idx % batch_size;
        idx /= batch_size;
        int y = idx % w_y;
        int x = idx / w_y;

        atomicAdd(&weigths[y + x * w_y + num_optimizables * b], -*lrate * in[b * w_x + x] * delta[b * w_y + y]);
    }
}

//Assumes that "in" only stores "1" (which is the case for the input of biases). Thus, the parameter can be ommited. Therefore only one size paramter is needed as well an "w_x" can be omitted. "w_y" is now th esize of "weights".
template<typename T, typename L>
__global__ void sgdDebugSimple(T* weigths, T* delta, L* lrate, uint32_t w_y, uint32_t batch_size, uint32_t num_optimizables) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < w_y * batch_size) {
        int b = idx % batch_size;
        idx /= batch_size;

        atomicAdd(&weigths[idx + num_optimizables * b], -*lrate * delta[b * w_y + idx]);
    }
}

//===================================================
//==================|Optimizers|=====================
//===================================================

enum OPTIMIZER_TYPE: uint32_t {NONE=0, DBUG=1, SGD=2, ADAM=3};   //DON'T CHANGE THESE VALUES AS IT WILL BREAK OLD CHECKPOINT FILES!
template<typename T, typename L = T>             //T is type of data, L is type of learning rates
class Optimizer {
protected:
    uint64_t optimizables;                        //Number of T's this optimizer optimizes
    
    T* optBuf;                                    //Buffer that stores information on every variable it optimizes (e.g. momentum). Optimizer has no authority over layout.
    L* learningRates;                             //Learning rates (e.g. momentum decay, ...).    Variable names: alpha=learning rate; beta1,beta2=first and second order momentum decay.

    /*
        Returns the memory requirements of this optimizer excluding recursive calls.

        @param requirements: Out parameter. Appends first the requirements for "optBuf" and then for "learningRates" to this vector
    */
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements) const {
        fprintf(stderr, "[ERROR] You called \"getOwnMemoryRequirements\" on the optimizer base class. You should only use the derived classes!");
        std::exit(-1);
    }

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
    void setNumOptimizables(uint64_t num_optimizables) { 
        optimizables = num_optimizables; 
    }

    /*
        Returns the memory requirements of this optimizer including recursive calls.

        @param requirements: Out parameter. Appends first the requirements for "optBuf" and then for "learningRates" to this vector
    */
    virtual CPP20_CONSTEXPR void getMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements) const {
        getOwnMemoryRequirements(requirements);
    } 
    /*
        Sets the internal memory pointers of the optimizer. The passed pointer will be set to the first byte after the by this optimizer used memory 
        region. The pointer does not need to be aligned as first they will be padded so they satisfy the alignment requirement and then incrementen by the 
        space requirement of this optimizer, as specified in "getMemoryRequirement".

        @param mem: Pointer to enough free memory that this optimzier will use to store "optBuf" and "learningRates". Will be set after the region used for this.
    */
    void setMem(void**& mem) {
        //1.: Set buffers
        optBuf        = (T*)*mem++;
        learningRates = (L*)*mem++;
        
        //2.: Check alignments
        std::vector<MemoryRequirementLifetime> requirements;
        getOwnMemoryRequirements(requirements);

        if (!is_aligned(optBuf, requirements[0].alignment)) {
            fprintf(stderr, "[ERROR] \"setMem\" of the optimizer was called with a misaligned pointer for \"optBuf\"");
            std::exit(-1);
        }
        if (!is_aligned(learningRates, requirements[1].alignment)) {
            fprintf(stderr, "[ERROR] \"setMem\" of the optimizer was called with a misaligned pointer for \"learningRates\"");
            std::exit(-1);
        }
    }
    /*
        Allocates the memory required as reported in "getMemoryRequirements" and sets it using a call to "setMem"
    */
    void allocate() {
        //1.: Get own memory requirements
        std::vector<MemoryRequirementLifetime> requirements;
        getOwnMemoryRequirements(requirements);

        //2.: Allocate memory
        void* memory[2];
        gpuErrchk(cudaMallocAligned((void**)&memory[0], requirements[0].getMemoryRequirements()));
        gpuErrchk(cudaMallocAligned((void**)&memory[1], requirements[1].getMemoryRequirements()));
        
        //3.: Set memory
        void** mem = +memory;
        setMem(mem);
    }
    /*
        Initializes the optimization buffer
    */
    virtual void initMem() {
        fprintf(stderr, "[ERROR] You called \"initMem\" on the optimizer base class. You should only use the derived classes!");
        exit(-1);
    }
    
    /*
        Adds a node that optimizes a weights matrix multiplication to a graph.
        (Optimizes matrix "mem" of shape y*x. Gradient is product of respective elements from input, a vector of lenght x, with delta, a vector of lenght y.)
        ("mem" correspondes with weights, "input" with layerBefore->state and "delta" with the derivative of the loss with respect to state before activation)

        @param mem          : The memory that should be optimized (of the weights in this case). Matrix of shape y*x
        @param delta        : The delta of the output of this layer before acitivation to the loss. Vector of length y.              =delta o_n
        @param input        : The output of the layer before. Vector of length x. T* or T** when indirection=true                    =layerBefore->output
        @param y            : The y-dimension of "mem"                                                                               =height weights
        @param x            : The x-dimension of "mem"                                                                               =width weights
        @param batch_size   : The used batch size
        @param indirection  : If this is true, input does not contain a T* directly but a T** pointing to it.
        @param index        : The first element of "mem" is the index-ed value that this optimizer optimizes (used as a index to "optBuf"). Will be updated
        @param graph        : The graph the optimization node should be added to
        @param depMem       : The dependencies on "mem". Will be updated
        @param depDelta     : The dependencies on "delta". Will be updated
        @param depInput     : The dependencies on "input". Will be updated
        @param captureStream: Cuda stream that can be used for graph capture of nodes
    */
    virtual void addNodeGEMM(T* mem, T* delta, void* input, uint32_t y, uint32_t x, uint32_t batch_size, bool indirection, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput, cudaStream_t captureStream) {
        fprintf(stderr, "[ERROR] You called \"addNodeWeights\" on the optimizer base class. You should only use the derived classes!");
        std::exit(-1);
    }
    /*
        Adds a node that optimizes a bias addition to a graph.
        (Optimizes vector "mem" of lenght y. Gradient is repective element of "delta", another vector of lenght "y".)
        ("mem" corresponds to bias and "delta" to the derivative of the loss with respect to state before activation.)

        @param mem          : The memory that should be optimized (of the bias in this case). Vector of length y
        @param index        : The first element of "mem" is the index-ed value that this optimizer optimizes (used as a index to "optBuf"). Will be updated
        @param delta        : The delta of the output of this layer before acitivation to the loss. Vector of length y               =delta o_n
        @param y            : The length of "mem"                                                                                    =number of biases (in b_n)
        @param batch_size   : The used batch size
        @param graph        : The graph the optimization node should be added to
        @param depMem       : The dependencies on "mem". Will be updated
        @param depDelta     : The dependencies on "delta". Will be updated
        @param captureStream: Cuda stream that can be used for graph capture of nodes
    */
    virtual void addNodeBias(T* mem, Image_Shape bias_shape, T* delta, Image_Shape delta_shape, uint32_t batch_size, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, cudaStream_t captureStream) {
        fprintf(stderr, "[ERROR] You called \"addNodeBias\" on the optimizer base class. You should only use the derived classes!");
        std::exit(-1);
    }
    /*
        Adds a node that optimizes a convolution to a graph.
        (Optimizes kernel "mem" of shape kernel_size^2.)
        ("mem" correspondes with kernel, "input" with layerBefore->state and "delta" with the derivative of the loss with respect to state before activation)

        @param mem          : The memory that should be optimized (of the kernel in this case).
        @param kernel_shape : The shape of "mem"
        @param delta        : The delta of the output of this layer before acitivation to the loss.                                                            =delta o_n
        @param delta_shape  : The shape of "delta" (per sample)
        @param input        : The output of the layer before. Vector of length x. T* or T** when indirection=true                                              =layerBefore->output
        @param input_shape  : The shape of "input" (per sample)
        @param batch_size   : The used batch size
        @param indirection  : If this is true, input does not contain a T* directly but a T** pointing to it.
        @param index        : The first element of "mem" is the index-ed value that this optimizer optimizes (used as a index to "optBuf"). Will be updated
        @param graph        : The graph the optimization node should be added to
        @param depMem       : The dependencies on "mem".   Will be updated
        @param depDelta     : The dependencies on "delta". Will be updated
        @param depInput     : The dependencies on "input". Will be updated
        @param captureStream: Cuda stream that can be used for graph capture of nodes
    */
    virtual void addNodeConvolution(T* mem, Image_Shape kernel_shape, uint32_t dilation, uint32_t stride, T* delta, Image_Shape delta_shape, void* input, Image_Shape input_shape, uint32_t batch_size, bool indirection, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput, cudaStream_t captureStream) {
        fprintf(stderr, "[ERROR] You called \"addNodeConvolution\" on the optimizer base class. You should only use the derived classes!");
        std::exit(-1);
    }

    /*
        Sets learning rates. If some of the learning rates aren't actually used, the parameter is just ignored. To selectivly only change some of the learning rates, pass nullptr to the other parameters

        @param alpha : The normal learning rate to use. Host pointer created by "cudaMallocHost"
        @param beta1 : The first order momentum decay to use. Host pointer created by "cudaMallocHost"
        @param beta2 : The second order momentum decay to use. Host pointer created by "cudaMallocHost"
        @param stream: The stream to use when copying the previous parameters to the gpu
    */
    virtual void setLR(L* alpha, L* beta1, L* beta2, cudaStream_t stream) {
        fprintf(stderr, "[ERROR] You called \"setLR\" on the optimizer base class. You should only use the derived classes!");
        exit(-1);
    }

    /*
        Returns the sublcass of the optimizer.
    */
    virtual CPP20_CONSTEXPR OPTIMIZER_TYPE getOptimizerType() {
        fprintf(stderr, "[ERROR] You called \"getOptimizerType\" on the optimizer base class. You should only use the derived classes!");
        exit(-1);
    }
    /*
        Constructs an optimizer of a specific subclass at a specific memory location

        @param ot : The derived class of the optimizer to create
        @param out: The memory location where the optimizer should be created
    */
    static Optimizer<T, L>* getOptimizerOfType(OPTIMIZER_TYPE ot);

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
        std::vector<MemoryRequirementLifetime> requirements;
        getMemoryRequirements(requirements);

        fwrite(&requirements[0].num_bytes, sizeof(requirements[0].num_bytes), 1, file);
        fwrite((void*)optBuf, 1, requirements[0].num_bytes, file);

        fwrite(&requirements[1].num_bytes, sizeof(requirements[1].num_bytes), 1, file);
        fwrite((void*)learningRates, 1, requirements[1].num_bytes, file);
    }
    /*
        Deserialization according to deserialization rules
    */
    static Optimizer<T, L>* deserialize(FILE* file) {
        //1.: Create correct derived class
        OPTIMIZER_TYPE optimizer_type;
        fread(&optimizer_type, sizeof(OPTIMIZER_TYPE), 1, file);
        Optimizer<T, L>* ret = Optimizer<T, L>::getOptimizerOfType(optimizer_type);

        //2.: Read in variables
        fread(&ret->optimizables, sizeof(ret->optimizables), 1, file);

        //3.: Get memory requirements
        std::vector<MemoryRequirementLifetime> requirements;
        ret->getMemoryRequirements(requirements);

        //4.: Read in memory
        uint64_t buffer_bytes, lrates_bytes;
        fread(&buffer_bytes, sizeof(buffer_bytes), 1, file);
        if (requirements[0].num_bytes != buffer_bytes) {
            fprintf(stderr, "[ERROR] Trying to create a optimizer of type %llu with %llu state bytes, even though it requires %llu", (uint64_t)optimizer_type, (uint64_t)buffer_bytes, (uint64_t)requirements[0].num_bytes);
            exit(-1);
        }

        cudaMallocAligned((void**)&ret->optBuf, requirements[0].getMemoryRequirements());
        fread((void*)ret->optBuf, 1, buffer_bytes, file);

        fread(&lrates_bytes, sizeof(lrates_bytes), 1, file);
        if (requirements[1].num_bytes != lrates_bytes) {
            fprintf(stderr, "[ERROR] Trying to create a optimizer of type %llu with %llu other bytes, even though it requires %llu", (uint64_t)optimizer_type, (uint64_t)lrates_bytes, (uint64_t)requirements[1].num_bytes);
            exit(-1);
        }

        cudaMallocAligned((void**)&ret->learningRates, requirements[1].getMemoryRequirements());
        fread((void*)ret->learningRates, 1, lrates_bytes, file);

        //5.: Return
        return ret;
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
            std::vector<MemoryRequirementLifetime> requirements;
            getOwnMemoryRequirements(requirements);

            void* host_buf;
            cudaMallocHost(&host_buf, requirements[0].num_bytes);
            cudaMemcpy(host_buf, (void*)optBuf, requirements[0].num_bytes, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(NULL);
            fwrite(host_buf, 1, requirements[0].num_bytes, file);
            cudaFreeHost(host_buf);
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
    static Optimizer<T, L>* getOptimizerFromCompression(FILE* file) {
        OPTIMIZER_TYPE ot;
        fread(&ot, sizeof(OPTIMIZER_TYPE), 1, file);
        return getOptimizerOfType(ot);
    }
    /*
        Inverse of "compress<true>"
    */
    void initMemFromCompression(FILE* file) {
        std::vector<MemoryRequirementLifetime> requirements;
        getOwnMemoryRequirements(requirements);

        void* host_buf;
        cudaMallocHost(&host_buf, requirements[0].num_bytes);
        fread(host_buf, 1, requirements[0].num_bytes, file);
        cudaMemcpy((void*)optBuf, host_buf, requirements[0].num_bytes, cudaMemcpyHostToDevice);
        cudaStreamSynchronize(NULL);
        cudaFreeHost(host_buf);
    }
};

template<typename T, typename L = T>
class No_Optimizer : public Optimizer<T, L> {
    //learningRates = {}
protected:
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements) const override {
        requirements.push_back(MemoryRequirementLifetime()); //"optBuf" needs no memory
        requirements.push_back(MemoryRequirementLifetime()); //"learningRates" needs no memory
    }

public:
    No_Optimizer() = default;

    virtual void initMem() override {}     //No memory to initializes

    virtual CPP20_CONSTEXPR OPTIMIZER_TYPE getOptimizerType() override {
        return OPTIMIZER_TYPE::NONE;
    }

    virtual void setLR(L* alpha, L* beta1, L* beta2, cudaStream_t stream) override {}  //No learning rates need ot be stored

    virtual void addNodeGEMM(T* mem, T* delta, void* input, uint32_t y, uint32_t x, uint32_t batch_size, bool indirection, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput, cudaStream_t captureStream) override {    }

    virtual void addNodeBias(T* mem, Image_Shape bias_shape, T* delta, Image_Shape delta_shape, uint32_t batch_size, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, cudaStream_t captureStream) override{}

    virtual void addNodeConvolution(T* mem, Image_Shape kernel_shape, uint32_t dilation, uint32_t stride, T* delta, Image_Shape delta_shape, void* input, Image_Shape input_shape, uint32_t batch_size, bool indirection, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput, cudaStream_t captureStream) override {}
};

template<typename T, typename L = T>
class SGD_Optimizer : public Optimizer<T, L> {
    //learningRates = {alpha}
protected:
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements) const override {
        requirements.push_back(MemoryRequirementLifetime());                      //"optBuf" needs no memory
        requirements.push_back(MemoryRequirementLifetime(sizeof(L), alignof(L))); //"learningRates" needs memory for alpha
    }

public:
    SGD_Optimizer() = default;
    
    virtual void initMem() override {}     //No memory to initializes

    virtual CPP20_CONSTEXPR OPTIMIZER_TYPE getOptimizerType() override {
        return OPTIMIZER_TYPE::SGD;
    }
    
    virtual void setLR(L* alpha, L* beta1, L* beta2, cudaStream_t stream) override {
        gpuErrchk(cudaMemcpyAsync(this->learningRates, alpha, sizeof(L), cudaMemcpyHostToDevice, stream));  //SGD only uses alpha
    }


    virtual void addNodeGEMM(T* mem, T* delta, void* input, uint32_t y, uint32_t x, uint32_t batch_size, bool indirection, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput, cudaStream_t captureStream) override {
        //1.: Variables used
        uint32_t outStateSizeBatched = x * y * batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        void* sgdArgs[] = {
            (void*)&mem,
            (void*)&delta,
            (void*)&input,
            (void*)&this->learningRates,
            (void*)&y,
            (void*)&x,
            (void*)&batch_size
        };
        cudaKernelNodeParams sgdParam{
            (void*)sgdMul<T, L>,                         //Function pointer
            dim3((outStateSizeBatched + 31u)/32u, 1, 1), //Grid dimensions
            dim3(32, 1, 1),                              //Block dimensions
            0u,                                          //Dyn. shared-mem per block in bytes
            (void**)&sgdArgs,                            //Array of pointers to individual kernel arguments
            nullptr                                      //Pointer to kernel arguments in the "extra" format
        };
        if (indirection)
            sgdParam.func = (void*)sgdMul_indirection<T, L>;
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam);
        depMem.apply<true>(graph, node);
        depDelta.apply<false>(graph, node);
        depInput.apply<false>(graph, node);

        //3.: Update parameters
        index += x * y;
        //Dependencies were already updated
    }

    /*
    virtual void addNodeWeightsIndirection(T* mem, uint64_t& index, T* delta, T** input, uint32_t y, uint32_t x, uint32_t batch_size, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput) override {
        //1.: Variables used
        uint32_t outStateSizeBatched = x * y * batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        void* sgdArgs[] = {
            (void*)&mem,
            (void*)&delta,
            (void*)&input,
            (void*)&this->learningRates,
            (void*)&y,
            (void*)&x,
            (void*)&batch_size
        };
        cudaKernelNodeParams sgdParam{
            (void*)sgdMul_indirection<T, L>,               //Function pointer
            dim3((outStateSizeBatched + 31u) / 32u, 1, 1), //Grid dimensions
            dim3(32, 1, 1),                                //Block dimensions
            0u,                                            //Dyn. shared-mem per block in bytes
            (void**)&sgdArgs,                              //Array of pointers to individual kernel arguments
            nullptr                                        //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam);
        depMem.apply<true>(graph, node);
        depDelta.apply<false>(graph, node);
        depInput.apply<false>(graph, node);

        //3.: Update parameters
        index += x * y;
        //Dependencies were already updated
    }
    */

    virtual void addNodeBias(T* mem, Image_Shape bias_shape, T* delta, Image_Shape delta_shape, uint32_t batch_size, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, cudaStream_t captureStream) override {
        assert((bias_shape == delta_shape) || (bias_shape.x == 1 && bias_shape.y == 1)); //Currently only supports tied and untied bias

        if (bias_shape == delta_shape) { //Untied bias
            //1.: Variables used
            uint32_t outStateSize        = bias_shape.prod();
            uint32_t outStateSizeBatched = outStateSize * batch_size;

            cudaGraphNode_t node;

            //2.: Add node to graph
            void* sgdArgs[] = {
                (void*)&mem,
                (void*)&delta,
                (void*)&this->learningRates,
                (void*)&outStateSize,
                (void*)&batch_size
            };
            cudaKernelNodeParams sgdParam{
                (void*)sgdSimple<T, L>,                        //Function pointer
                dim3((outStateSizeBatched + 31u) / 32u, 1, 1), //Grid dimensions
                dim3(32, 1, 1),                                //Block dimensions
                0u,                                            //Dyn. shared-mem per block in bytes
                (void**)&sgdArgs,                              //Array of pointers to individual kernel arguments
                nullptr                                        //Pointer to kernel arguments in the "extra" format
            };
            gpuErrchk(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam));
            depMem.apply<true>(graph, node);
            depDelta.apply<false>(graph, node);

            //3.: Update parameters
            index += outStateSize;
            //Dependencies were already updated
        }
        else if (bias_shape.x == 1 && bias_shape.y == 1) { //Tied bias
            //0.: Variables used
            cudaGraphNode_t node;
            
            float alpha = -this->learningRates[0];
            float beta  = 1.f;
            cudnnDataType_t dtype = cudnnTypeOf<T>();

            //1.: Set up tensor descriptors
            cudnnTensorDescriptor_t dyDesc, dbDesc;

            cudnnCreateTensorDescriptor(&dyDesc);
            cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, dtype, batch_size, delta_shape.z, delta_shape.y, delta_shape.x);

            cudnnCreateTensorDescriptor(&dbDesc);
            cudnnSetTensor4dDescriptor(dbDesc, CUDNN_TENSOR_NCHW, dtype, 1, bias_shape.z, 1, 1);

            //2.: Record cudnn operation and add to graph
            cudaGraph_t cudnnGraph;
            cudaStreamBeginCapture(captureStream, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal);
            cudnnConvolutionBackwardBias(cudnn_handle, &alpha, dyDesc, (void*)delta, &beta, dbDesc, (void*)mem);
            cudaStreamEndCapture(captureStream, &cudnnGraph);
            gpuErrchk(cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, cudnnGraph));
            depMem.apply<true>(graph, node);
            depDelta.apply<false>(graph, node);

            //3.: Update parameters
            index += bias_shape.prod();
            //Dependencies were already updated
        }
        else {
            fprintf(stderr, "[ERROR] The SGD optimizer currently only supports tied or untied biases!");
            std::exit(-1);
        }
    }

    //TODO: Indirection
    virtual void addNodeConvolution(T* mem, Image_Shape kernel_shape, uint32_t dilation, uint32_t stride, T* delta, Image_Shape delta_shape, void* input, Image_Shape input_shape, uint32_t batch_size, bool indirection, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput, cudaStream_t captureStream) override {
        //0.: Variables used
        cudnnDataType_t dtype    = cudnnTypeOf<T>();
        cudnnDataType_t compType = cudnnTypeOf<T>();

        float alpha = -this->learningRates[0];
        float beta  = 1.f;

        cudaGraphNode_t node;

        //1.: Set up descriptors
        cudnnTensorDescriptor_t xDesc, dyDesc;
        cudnnFilterDescriptor_t dwDesc;
        cudnnConvolutionDescriptor_t convDesc;
        cudnnConvolutionBwdFilterAlgoPerf_t bestAlgo;

        PRINT_VAR(&xDesc);

        cudnnCreateTensorDescriptor(&xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, dtype, batch_size, input_shape.z, input_shape.y, input_shape.x);

        cudnnCreateTensorDescriptor(&dyDesc);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, dtype, batch_size, delta_shape.z, delta_shape.y, delta_shape.x);

        cudnnCreateFilterDescriptor(&dwDesc);
        cudnnSetFilter4dDescriptor(dwDesc, dtype, CUDNN_TENSOR_NCHW, delta_shape.z, kernel_shape.z, kernel_shape.y, kernel_shape.x);

        int32_t dilA[] = { dilation, dilation };
        int32_t strA[] = { stride  , stride };
        int32_t padA[] = { (kernel_shape.y * dilation) >> 1, (kernel_shape.y * dilation) >> 1 };
        cudnnCreateConvolutionDescriptor(&convDesc);
        cudnnSetConvolutionNdDescriptor(convDesc, 2, padA, strA, dilA, CUDNN_CONVOLUTION, compType);

        int32_t retCount;
        cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnn_handle, xDesc, (void*)input, dyDesc, (void*)delta, convDesc, dwDesc, (void*)mem, 1, &retCount, &bestAlgo, cudnn_workspace, cudnn_workspace_size);
        assert(retCount == 1);
        assert(bestAlgo.memory <= cudnn_workspace_size);

        //2.: Record cudnn operation and add to graph
        cudaGraph_t cudnnGraph;
        cudaStreamBeginCapture(captureStream, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal);
            cudnnConvolutionBackwardFilter(cudnn_handle, &alpha, xDesc, (void*)input, dyDesc, (void*)delta, convDesc, bestAlgo.algo, cudnn_workspace, cudnn_workspace_size, &beta, dwDesc, (void*)mem);
        cudaStreamEndCapture(captureStream, &cudnnGraph);
        gpuErrchk(cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, cudnnGraph));
        depMem.apply<true>(graph, node);
        depDelta.apply<false>(graph, node);
        depInput.apply<false>(graph, node);

        //3.: Update parameters
        index += kernel_shape.prod() * delta_shape.z;
        //Dependencies were already updated

        /*
        struct {
            float alpha, beta;
            cudnnTensorDescriptor_t xDesc, dyDesc;
            cudnnConvolutionDescriptor_t convDesc;
            cudnnConvolutionBwdFilterAlgo_t bestAlgo;
            cudnnFilterDescriptor_t dwDesc;
            T* delta, *mem;

            void exec(T** indirection_pointer) {
                cudnnConvolutionBackwardFilter(cudnn_handle, &alpha, xDesc, indirection_pointer, dyDesc, (void*)delta, convDesc, bestAlgo, cudnn_workspace, cudnn_workspace_size, &beta, dwDesc, (void*)mem);
            }
        }s;*/
    }
};

template<typename T, typename L = T> //SGD, but writes gradients to "optBuf" (each gradient, per batch)
class Debug_Optimizer : public Optimizer<T, L> {   
    //learningRates = {alpha}
public:
    uint32_t batch_size;

    Debug_Optimizer(uint32_t batch_size = 0) :
        batch_size(batch_size)
    {}

    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements) const override {
        requirements.push_back(MemoryRequirementLifetime(sizeof(T) * this->optimizables * batch_size, alignof(T), true)); //"optBuf" stores last gradient
        requirements.push_back(MemoryRequirementLifetime(sizeof(L), alignof(L)));                                         //"learningRates" needs no memory
    }


    virtual void initMem() override {
        cudaMemset(this->optBuf, 0, sizeof(T) * this->optimizables * batch_size);
    }

    virtual CPP20_CONSTEXPR OPTIMIZER_TYPE getOptimizerType() override {
        return OPTIMIZER_TYPE::DBUG;
    }

    virtual void setLR(L* alpha, L* beta1, L* beta2, cudaStream_t stream) override {
        gpuErrchk(cudaMemcpyAsync(this->learningRates, alpha, sizeof(L), cudaMemcpyHostToDevice, stream));  //SGD only uses alpha
    }

    virtual void addNodeGEMM(T* mem, T* delta, void* input, uint32_t y, uint32_t x, uint32_t batch_size, bool indirection, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput, cudaStream_t captureStream) override {
        //1.: Variables used
        uint32_t outStateSizeBatched = x * y * batch_size;

        cudaGraphNode_t node;

        //2.: Add debug node to graph
        T* optBuf_ind = &optBuf[index];
        void* sgdArgs_[] = {
            (void*)&optBuf_ind,                       //Write to "optBuf"
            (void*)&delta,
            (void*)&input,
            (void*)&this->learningRates,
            (void*)&y,
            (void*)&x,
            (void*)&batch_size,
            (void*)&this->optimizables
        };
        cudaKernelNodeParams sgdParam_{
            (void*)sgdDebugMul<T, L>,                      //Function pointer
            dim3((outStateSizeBatched + 31u) / 32u, 1, 1), //Grid dimensions
            dim3(32, 1, 1),                                //Block dimensions
            0u,                                            //Dyn. shared-mem per block in bytes
            (void**)&sgdArgs_,                             //Array of pointers to individual kernel arguments
            nullptr                                        //Pointer to kernel arguments in the "extra" format
        };
        if (indirection)
            sgdParam_.func = (void*)sgdDebugMul_indirection<T, L>;
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam_);
        depDelta.apply<false>(graph, node);
        depInput.apply<false>(graph, node);

        //3.: Add node to graph
        void* sgdArgs[] = {
            (void*)&mem,
            (void*)&delta,
            (void*)&input,
            (void*)&this->learningRates,
            (void*)&y,
            (void*)&x,
            (void*)&batch_size
        };
        cudaKernelNodeParams sgdParam{
            (void*)sgdMul<T, L>,                         //Function pointer
            dim3((outStateSizeBatched + 31u) / 32u, 1, 1), //Grid dimensions
            dim3(32, 1, 1),                              //Block dimensions
            0u,                                          //Dyn. shared-mem per block in bytes
            (void**)&sgdArgs,                            //Array of pointers to individual kernel arguments
            nullptr                                      //Pointer to kernel arguments in the "extra" format
        };
        if (indirection)
            sgdParam.func = (void*)sgdMul_indirection<T, L>;
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam);
        depMem.apply<true>(graph, node);
        depDelta.apply<false>(graph, node);
        depInput.apply<false>(graph, node);

        //4.: Update parameters
        index += x * y;
        //Dependencies were already updated
    }

    virtual void addNodeBias(T* mem, Image_Shape bias_shape, T* delta, Image_Shape delta_shape, uint32_t batch_size, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, cudaStream_t captureStream) override {
        assert((bias_shape == delta_shape) || (bias_shape.x == 1 && bias_shape.y == 1)); //Currently only supports tied and untied bias

        if (bias_shape == delta_shape) { //Untied bias
            //1.: Variables used
            uint32_t outStateSize = bias_shape.prod();
            uint32_t outStateSizeBatched = outStateSize * batch_size;

            cudaGraphNode_t node;

            //2.: Add debug node to graph
            T* optBuf_ind = &optBuf[index];
            void* sgdArgs_[] = {
                (void*)&optBuf_ind,
                (void*)&delta,
                (void*)&this->learningRates,
                (void*)&outStateSize,
                (void*)&batch_size,
                (void*)&this->optimizables
            };
            cudaKernelNodeParams sgdParam_{
                (void*)sgdDebugSimple<T, L>,                   //Function pointer
                dim3((outStateSizeBatched + 31u) / 32u, 1, 1), //Grid dimensions
                dim3(32, 1, 1),                                //Block dimensions
                0u,                                            //Dyn. shared-mem per block in bytes
                (void**)&sgdArgs_,                             //Array of pointers to individual kernel arguments
                nullptr                                        //Pointer to kernel arguments in the "extra" format
            };
            cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam_);
            depDelta.apply<false>(graph, node);

            //3.: Add node to graph
            void* sgdArgs[] = {
                (void*)&mem,
                (void*)&delta,
                (void*)&this->learningRates,
                (void*)&outStateSize,
                (void*)&batch_size
            };
            cudaKernelNodeParams sgdParam{
                (void*)sgdSimple<T, L>,                        //Function pointer
                dim3((outStateSizeBatched + 31u) / 32u, 1, 1), //Grid dimensions
                dim3(32, 1, 1),                                //Block dimensions
                0u,                                            //Dyn. shared-mem per block in bytes
                (void**)&sgdArgs,                              //Array of pointers to individual kernel arguments
                nullptr                                        //Pointer to kernel arguments in the "extra" format
            };
            gpuErrchk(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam));
            depMem.apply<true>(graph, node);
            depDelta.apply<false>(graph, node);

            //4.: Update parameters
            index += outStateSize;
            //Dependencies were already updated
        }
        else if (bias_shape.x == 1 && bias_shape.y == 1) { //Tied bias
            //0.: Variables used
            cudaGraphNode_t node;

            float alpha = -this->learningRates[0];
            float beta  = 1.f;
            float gamma = 0.f;
            cudnnDataType_t dtype = cudnnTypeOf<T>();

            //1.: Set up tensor descriptors
            cudnnTensorDescriptor_t dyDesc, dbDesc, optDesc;

            cudnnCreateTensorDescriptor(&dyDesc);
            cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, dtype, batch_size, delta_shape.z, delta_shape.y, delta_shape.x);

            cudnnCreateTensorDescriptor(&dbDesc);
            cudnnSetTensor4dDescriptor(dbDesc, CUDNN_TENSOR_NCHW, dtype, 1, bias_shape.z, 1, 1);

            cudnnCreateTensorDescriptor(&optDesc);
            cudnnSetTensor4dDescriptor(optDesc, CUDNN_TENSOR_NCHW, dtype, 1, bias_shape.z, 1, 1);

            //2.: Record cudnn operation and add to graph
            cudaGraph_t cudnnGraph;
            cudaStreamBeginCapture(captureStream, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal);
            cudnnConvolutionBackwardBias(cudnn_handle, &alpha, dyDesc, (void*)delta, &beta , dbDesc, (void*)mem);
            cudnnConvolutionBackwardBias(cudnn_handle, &alpha, dyDesc, (void*)delta, &gamma, dbDesc, (void*)&optBuf[index]);
            cudaStreamEndCapture(captureStream, &cudnnGraph);
            gpuErrchk(cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, cudnnGraph));
            depMem.apply<true>(graph, node);
            depDelta.apply<false>(graph, node);

            //3.: Update parameters
            index += bias_shape.prod();
            //Dependencies were already updated
        }
        else {
            fprintf(stderr, "[ERROR] The Debug optimizer currently only supports tied or untied biases!");
            std::exit(-1);
        }
    }

    virtual void addNodeConvolution(T* mem, Image_Shape kernel_shape, uint32_t dilation, uint32_t stride, T* delta, Image_Shape delta_shape, void* input, Image_Shape input_shape, uint32_t batch_size, bool indirection, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput, cudaStream_t captureStream) override {
        //0.: Variables used
        cudnnDataType_t dtype = cudnnTypeOf<T>();
        cudnnDataType_t compType = cudnnTypeOf<T>();

        float alpha = -this->learningRates[0];
        float beta = 1.f;

        cudaGraphNode_t node;

        //1.: Set up descriptors
        cudnnTensorDescriptor_t xDesc, dyDesc;
        cudnnFilterDescriptor_t dwDesc;
        cudnnConvolutionDescriptor_t convDesc;
        cudnnConvolutionBwdFilterAlgoPerf_t bestAlgo;


        cudnnCreateTensorDescriptor(&xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, dtype, batch_size, input_shape.z, input_shape.y, input_shape.x);

        cudnnCreateTensorDescriptor(&dyDesc);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, dtype, batch_size, delta_shape.z, delta_shape.y, delta_shape.x);

        cudnnCreateFilterDescriptor(&dwDesc);
        cudnnSetFilter4dDescriptor(dwDesc, dtype, CUDNN_TENSOR_NCHW, delta_shape.z, kernel_shape.z, kernel_shape.y, kernel_shape.x);

        int32_t dilA[] = { 1, dilation, dilation };
        int32_t strA[] = { 1, stride  , stride };
        int32_t padA[] = { 0, 0, 0 };
        cudnnCreateConvolutionDescriptor(&convDesc);
        cudnnSetConvolutionNdDescriptor(convDesc, 3, padA, strA, dilA, CUDNN_CONVOLUTION, compType);

        int32_t retCount;
        cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnn_handle, xDesc, (void*)input, dyDesc, (void*)delta, convDesc, dwDesc, (void*)mem, 1, &retCount, &bestAlgo, cudnn_workspace, cudnn_workspace_size);
        assert(retCount == 1);
        assert(bestAlgo.memory <= cudnn_workspace_size);

        //2.: Record cudnn operation and add to graph
        cudaGraph_t cudnnGraph;
        cudaStreamBeginCapture(captureStream, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal);
        cudnnConvolutionBackwardFilter(cudnn_handle, &alpha, xDesc, (void*)input, dyDesc, (void*)delta, convDesc, bestAlgo.algo, cudnn_workspace, cudnn_workspace_size, &beta, dwDesc, (void*)mem);
        cudnnConvolutionBackwardFilter(cudnn_handle, &alpha, xDesc, (void*)input, dyDesc, (void*)delta, convDesc, bestAlgo.algo, cudnn_workspace, cudnn_workspace_size, &beta, dwDesc, (void*)&this->optBuf[index]);
        cudaStreamEndCapture(captureStream, &cudnnGraph);
        gpuErrchk(cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, cudnnGraph));
        depMem.apply<true>(graph, node);
        depDelta.apply<false>(graph, node);
        depInput.apply<false>(graph, node);

        //3.: Update parameters
        index += kernel_shape.prod() * delta_shape.z;
        //Dependencies were already updated
    }

    T* getOptBuf() {
        return this->optBuf;
    }
};

//TODO
template<typename T, typename L = T>
class Adam_Optimizer : public Optimizer<T, L> {
    //learningRates = {alpha, beta1, beta2}
protected:
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements) const override { //TODO
        requirements.push_back(MemoryRequirementLifetime()); //"optBuf" needs no memory
        requirements.push_back(MemoryRequirementLifetime()); //"learningRates" needs no memory
    }

public:
    Adam_Optimizer() = default;

    virtual void initMem() override {
        //TODO
    }

    virtual CPP20_CONSTEXPR OPTIMIZER_TYPE getOptimizerType() override {
        return OPTIMIZER_TYPE::ADAM;
    }

    virtual void setLR(L* alpha, L* beta1, L* beta2, cudaStream_t stream) override {
        cudaMemcpyAsync(this->learningRates + 0, alpha, sizeof(L), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(this->learningRates + 1, beta1, sizeof(L), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(this->learningRates + 2, beta2, sizeof(L), cudaMemcpyHostToDevice, stream);
    }


    virtual void addNodeGEMM(T* mem, T* delta, void* input, uint32_t y, uint32_t x, uint32_t batch_size, bool indirection, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput, cudaStream_t captureStream) override {
        //1.: Variables used
        uint32_t outStateSizeBatched = x * y * batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        void* sgdArgs[] = {
            (void*)&mem,
            (void*)&delta,
            (void*)&input,
            (void*)&this->learningRates,
            (void*)&y,
            (void*)&x,
            (void*)&batch_size
        };
        cudaKernelNodeParams sgdParam{
            (void*)sgdMul<T, L>,                         //Function pointer
            dim3((outStateSizeBatched + 31u) / 32u, 1, 1), //Grid dimensions
            dim3(32, 1, 1),                              //Block dimensions
            0u,                                          //Dyn. shared-mem per block in bytes
            (void**)&sgdArgs,                            //Array of pointers to individual kernel arguments
            nullptr                                      //Pointer to kernel arguments in the "extra" format
        };
        if (indirection)
            sgdParam.func = (void*)sgdMul_indirection<T, L>;
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam);
        depMem.apply<true>(graph, node);
        depDelta.apply<false>(graph, node);
        depInput.apply<false>(graph, node);

        //3.: Update parameters
        index += x * y;
        //Dependencies were already updated
    }

    /*
    virtual void addNodeWeightsIndirection(T* mem, uint64_t& index, T* delta, T** input, uint32_t y, uint32_t x, uint32_t batch_size, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput) override {
        //1.: Variables used
        uint32_t outStateSizeBatched = x * y * batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        void* sgdArgs[] = {
            (void*)&mem,
            (void*)&delta,
            (void*)&input,
            (void*)&this->learningRates,
            (void*)&y,
            (void*)&x,
            (void*)&batch_size
        };
        cudaKernelNodeParams sgdParam{
            (void*)sgdMul_indirection<T, L>,               //Function pointer
            dim3((outStateSizeBatched + 31u) / 32u, 1, 1), //Grid dimensions
            dim3(32, 1, 1),                                //Block dimensions
            0u,                                            //Dyn. shared-mem per block in bytes
            (void**)&sgdArgs,                              //Array of pointers to individual kernel arguments
            nullptr                                        //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam);
        depMem.apply<true>(graph, node);
        depDelta.apply<false>(graph, node);
        depInput.apply<false>(graph, node);

        //3.: Update parameters
        index += x * y;
        //Dependencies were already updated
    }
    */

    virtual void addNodeBias(T* mem, Image_Shape bias_shape, T* delta, Image_Shape delta_shape, uint32_t batch_size, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, cudaStream_t captureStream) override {
        assert((bias_shape == delta_shape) || (bias_shape.x == 1 && bias_shape.y == 1)); //Currently only supports tied and untied bias

        if (bias_shape == delta_shape) { //Untied bias
            //1.: Variables used
            uint32_t outStateSize = bias_shape.prod();
            uint32_t outStateSizeBatched = outStateSize * batch_size;

            cudaGraphNode_t node;

            //2.: Add node to graph
            void* sgdArgs[] = {
                (void*)&mem,
                (void*)&delta,
                (void*)&this->learningRates,
                (void*)&outStateSize,
                (void*)&batch_size
            };
            cudaKernelNodeParams sgdParam{
                (void*)sgdSimple<T, L>,                        //Function pointer
                dim3((outStateSizeBatched + 31u) / 32u, 1, 1), //Grid dimensions
                dim3(32, 1, 1),                                //Block dimensions
                0u,                                            //Dyn. shared-mem per block in bytes
                (void**)&sgdArgs,                              //Array of pointers to individual kernel arguments
                nullptr                                        //Pointer to kernel arguments in the "extra" format
            };
            gpuErrchk(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sgdParam));
            depMem.apply<true>(graph, node);
            depDelta.apply<false>(graph, node);

            //3.: Update parameters
            index += outStateSize;
            //Dependencies were already updated
        }
        else if (bias_shape.x == 1 && bias_shape.y == 1) { //Tied bias
            //0.: Variables used
            cudaGraphNode_t node;

            float alpha = -this->learningRates[0];
            float beta = 1.f;
            cudnnDataType_t dtype = cudnnTypeOf<T>();

            //1.: Set up tensor descriptors
            cudnnTensorDescriptor_t dyDesc, dbDesc;

            cudnnCreateTensorDescriptor(&dyDesc);
            cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, dtype, batch_size, delta_shape.z, delta_shape.y, delta_shape.x);

            cudnnCreateTensorDescriptor(&dbDesc);
            cudnnSetTensor4dDescriptor(dbDesc, CUDNN_TENSOR_NCHW, dtype, 1, bias_shape.z, 1, 1);

            //2.: Record cudnn operation and add to graph
            cudaGraph_t cudnnGraph;
            cudaStreamBeginCapture(captureStream, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal);
            cudnnConvolutionBackwardBias(cudnn_handle, &alpha, dyDesc, (void*)delta, &beta, dbDesc, (void*)mem);
            cudaStreamEndCapture(captureStream, &cudnnGraph);
            gpuErrchk(cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, cudnnGraph));
            depMem.apply<true>(graph, node);
            depDelta.apply<false>(graph, node);

            //3.: Update parameters
            index += bias_shape.prod();
            //Dependencies were already updated
        }
        else {
            fprintf(stderr, "[ERROR] The SGD optimizer currently only supports tied or untied biases!");
            std::exit(-1);
        }
    }

    //TODO: Indirection
    virtual void addNodeConvolution(T* mem, Image_Shape kernel_shape, uint32_t dilation, uint32_t stride, T* delta, Image_Shape delta_shape, void* input, Image_Shape input_shape, uint32_t batch_size, bool indirection, uint64_t& index, cudaGraph_t graph, Dependencies& depMem, Dependencies& depDelta, Dependencies& depInput, cudaStream_t captureStream) override {
        //0.: Variables used
        cudnnDataType_t dtype = cudnnTypeOf<T>();
        cudnnDataType_t compType = cudnnTypeOf<T>();

        float alpha = -this->learningRates[0];
        float beta = 1.f;

        cudaGraphNode_t node;

        //1.: Set up descriptors
        cudnnTensorDescriptor_t xDesc, dyDesc;
        cudnnFilterDescriptor_t dwDesc;
        cudnnConvolutionDescriptor_t convDesc;
        cudnnConvolutionBwdFilterAlgoPerf_t bestAlgo;


        cudnnCreateTensorDescriptor(&xDesc);
        cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, dtype, batch_size, input_shape.z, input_shape.y, input_shape.x);

        cudnnCreateTensorDescriptor(&dyDesc);
        cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, dtype, batch_size, delta_shape.z, delta_shape.y, delta_shape.x);

        cudnnCreateFilterDescriptor(&dwDesc);
        cudnnSetFilter4dDescriptor(dwDesc, dtype, CUDNN_TENSOR_NCHW, delta_shape.z, kernel_shape.z, kernel_shape.y, kernel_shape.x);

        int32_t dilA[] = { 1, dilation, dilation };
        int32_t strA[] = { 1, stride  , stride };
        int32_t padA[] = { 0, 0, 0 };
        cudnnCreateConvolutionDescriptor(&convDesc);
        cudnnSetConvolutionNdDescriptor(convDesc, 3, padA, strA, dilA, CUDNN_CONVOLUTION, compType);

        int32_t retCount;
        cudnnFindConvolutionBackwardFilterAlgorithmEx(cudnn_handle, xDesc, (void*)input, dyDesc, (void*)delta, convDesc, dwDesc, (void*)mem, 1, &retCount, &bestAlgo, cudnn_workspace, cudnn_workspace_size);
        assert(retCount == 1);
        assert(bestAlgo.memory <= cudnn_workspace_size);

        //2.: Record cudnn operation and add to graph
        cudaGraph_t cudnnGraph;
        cudaStreamBeginCapture(captureStream, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal);
        cudnnConvolutionBackwardFilter(cudnn_handle, &alpha, xDesc, (void*)input, dyDesc, (void*)delta, convDesc, bestAlgo.algo, cudnn_workspace, cudnn_workspace_size, &beta, dwDesc, (void*)mem);
        cudaStreamEndCapture(captureStream, &cudnnGraph);
        gpuErrchk(cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, cudnnGraph));
        depMem.apply<true>(graph, node);
        depDelta.apply<false>(graph, node);
        depInput.apply<false>(graph, node);

        //3.: Update parameters
        index += kernel_shape.prod() * delta_shape.z;
        //Dependencies were already updated

        /*
        struct {
            float alpha, beta;
            cudnnTensorDescriptor_t xDesc, dyDesc;
            cudnnConvolutionDescriptor_t convDesc;
            cudnnConvolutionBwdFilterAlgo_t bestAlgo;
            cudnnFilterDescriptor_t dwDesc;
            T* delta, *mem;

            void exec(T** indirection_pointer) {
                cudnnConvolutionBackwardFilter(cudnn_handle, &alpha, xDesc, indirection_pointer, dyDesc, (void*)delta, convDesc, bestAlgo, cudnn_workspace, cudnn_workspace_size, &beta, dwDesc, (void*)mem);
            }
        }s;*/
    }

};



template<typename T, typename L>
static Optimizer<T, L>* Optimizer<T, L>::getOptimizerOfType(OPTIMIZER_TYPE ot) {
    switch (ot) {
    case OPTIMIZER_TYPE::NONE:
        return new No_Optimizer<T, L>();
    case OPTIMIZER_TYPE::DBUG:
        return new Debug_Optimizer<T, L>();
    case OPTIMIZER_TYPE::SGD:
        return new SGD_Optimizer<T, L>();
    case OPTIMIZER_TYPE::ADAM:
        return new Adam_Optimizer<T, L>();
    default:
        fprintf(stderr, "[ERROR] %llu is not a known optimizer type!", (uint64_t)ot);
        exit(-1);
    }
}

//==========================================
//==================|Loss|==================
//==========================================

enum LOSS_TYPE : uint32_t {MSE=0, MAE=1, CROSS_ENTROPY=2};
template<typename T, typename L = T>
class Loss {
protected:
    T* guess;                  //Device pointer (to output of last layer)
    Image_Shape outStateShape; //Per sample
    uint32_t batch_size;       //Number of samples per batch
    
    T* deltas;                 //Points to memory where the delta of the last layer will be writte (does not own this)

    T** target;                //Indirection pointer (to gpu tile)
    T*  accumulator;           //Accumulates loss

public:
    Loss() = default;
    Loss(T* guess, Image_Shape outStateShape, uint32_t batch_size) :
        guess(guess),
        outStateShape(outStateShape),
        batch_size(batch_size)
    {
        cudaMalloc((void**)&target, sizeof(T*));
        cudaMalloc(&accumulator, sizeof(T));
    }
    void setParameters(T* guess_, Image_Shape outStateShape_, uint32_t batch_size_, T* deltaMem) {
        guess = guess_;
        outStateShape = outStateShape_;
        batch_size = batch_size_;
        deltas = deltaMem;

        cudaMalloc((void**)&target, sizeof(T*));
        cudaMalloc(&accumulator, sizeof(T));
    }

    /*
        Set the pointer to the gpu tile to a specified pointer.

        @param host_indirectionPointer: Must be a pointer allocated by "cudaMallocHost" that points to a pointer on the host that points to the correct gpu tile
        @param stream: The stream used for the memory transfer
    */
    void setTarget(T** host_indirection_pointer, cudaStream_t stream) {
        gpuErrchk(cudaMemcpyAsync(target, host_indirection_pointer, sizeof(T*), cudaMemcpyHostToDevice, stream));
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

    virtual cudaGraph_t getLossGraph(cudaStream_t cap_stream) {
        fprintf(stderr, "[ERROR] You called \"getLossGraph\" on the loss base class. You should only use the derived classes!");
        exit(-1);
    }
    virtual cudaGraph_t getDeltaGraph(cudaStream_t cap_stream) {
        fprintf(stderr, "[ERROR] You called \"getDeltaGraph\" on the loss base class. You should only use the derived classes!");
        exit(-1);
    }
};

template<typename T, typename L = T>
class MSE_Loss : public Loss<T, L> {
public:
    virtual cudaGraph_t getLossGraph(cudaStream_t cap_stream) {
        //1.: Create graph
        cudaGraph_t lossGraph;
        cudaGraphCreate(&lossGraph, 0);

        //2.: Add node
        cudaGraphNode_t node;

        uint32_t outStateSizeBatched = outStateShape.prod() * batch_size;
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in1, T in2) EXTENDED_CONSTEXPR -> T { /*printf("Val: %f %f %f\n", in1, in2, (in1 - in2) * (in1 - in2) / (T)2);*/ return (in1 - in2) * (in1 - in2) / (T)2; };

        void* lossArgs[] = {
            (void*)&guess,
            (void*)&target,
            (void*)&accumulator,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams lossParams{
            (void*)&transform_reduce_indirection<T, decltype(ldb), DIVISIBILITY::DIVISIBLE, DIVISIBILITY::DIVISIBLE, false>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),                                                                            //Grid dimensions
            dim3(32, 1, 1),                                                                                                  //Block dimensions
            0u,                                                                                                              //Dyn. shared-mem per block in bytes
            (void**)&lossArgs,                                                                                               //Array of pointers to individual kernel arguments
            nullptr                                                                                                          //Pointer to kernel arguments in the "extra" format
        };
        if (outStateSizeBatched % 32)
            lossParams.func = (void*)&transform_reduce_indirection<T, decltype(ldb), DIVISIBILITY::NOT_DIVISIBLE, DIVISIBILITY::NOT_DIVISIBLE, false>;
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

        uint32_t outStateSizeBatched = outStateShape.prod() * batch_size;
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in1, T in2) EXTENDED_CONSTEXPR -> T { /*printf("Train: %f %f %f\n", in1, in2, in1-in2);*/ return in1 - in2; };

        void* deltaArgs[] = {
            (void*)&guess,
            (void*)&target,
            (void*)&deltas,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams deltaParams{
            (void*)&transform4_indirection<T, decltype(ldb)>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),             //Grid dimensions
            dim3(32, 1, 1),                                   //Block dimensions
            0u,                                               //Dyn. shared-mem per block in bytes
            (void**)&deltaArgs,                               //Array of pointers to individual kernel arguments
            nullptr                                           //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, deltaGraph, nullptr, 0, &deltaParams);

        //3.: Return
        return deltaGraph;
    }
};

template<typename T, typename L = T>
class MAE_Loss : public Loss<T, L> {
public:
    virtual cudaGraph_t getLossGraph(cudaStream_t cap_stream) {
        //1.: Create graph
        cudaGraph_t lossGraph;
        cudaGraphCreate(&lossGraph, 0);

        //2.: Add node
        cudaGraphNode_t node;

        uint32_t outStateSizeBatched = outStateShape.prod() * batch_size;
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in1, T in2) EXTENDED_CONSTEXPR-> T { return abs(in1 - in2); };

        void* lossArgs[] = {
            (void*)&guess,
            (void*)&target,
            (void*)&accumulator,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams lossParams{
            (void*)&transform_reduce_indirection<T, decltype(ldb), DIVISIBILITY::DIVISIBLE, DIVISIBILITY::DIVISIBLE, false>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),                                                                        //Grid dimensions
            dim3(32, 1, 1),                                                                                              //Block dimensions
            0u,                                                                                                          //Dyn. shared-mem per block in bytes
            (void**)&lossArgs,                                                                                           //Array of pointers to individual kernel arguments
            nullptr                                                                                                      //Pointer to kernel arguments in the "extra" format
        };
        if (outStateSizeBatched % 32)
            lossParams.func = (void*)&transform_reduce_indirection<T, decltype(ldb), DIVISIBILITY::NOT_DIVISIBLE, DIVISIBILITY::NOT_DIVISIBLE, false>;
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

        uint32_t outStateSizeBatched = outStateShape.prod() * batch_size;
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in1, T in2) EXTENDED_CONSTEXPR -> T { return sign(in1 - in2); };

        void* deltaArgs[] = {
            (void*)&guess,
            (void*)&target,
            (void*)&deltas,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams deltaParams{
            (void*)&transform4_indirection<T, decltype(ldb)>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),           //Grid dimensions
            dim3(32, 1, 1),                                 //Block dimensions
            0u,                                             //Dyn. shared-mem per block in bytes
            (void**)&deltaArgs,                             //Array of pointers to individual kernel arguments
            nullptr                                         //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, deltaGraph, nullptr, 0, &deltaParams);

        //3.: Return
        return deltaGraph;
    }
};

template<typename T, typename L = T>
class CrossEntropy_Loss : public Loss<T, L> {
public:
    virtual cudaGraph_t getLossGraph(cudaStream_t cap_stream) {
        //1.: Create graph
        cudaGraph_t lossGraph;
        cudaGraphCreate(&lossGraph, 0);

        //2.: Add node
        cudaGraphNode_t node;

        uint32_t outStateSizeBatched = outStateShape.prod() * batch_size;
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in1, T in2) EXTENDED_CONSTEXPR -> T { return in2 * logarithm<T>((T)0.0005 + in1) + ((T)1.0005 - in2) * logarithm<T>((T)1.0005 - in1); };

        void* lossArgs[] = {
            (void*)&guess,
            (void*)&target,
            (void*)&accumulator,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams lossParams{
            (void*)&transform_reduce_indirection<T, decltype(ldb), DIVISIBILITY::DIVISIBLE, DIVISIBILITY::DIVISIBLE, false>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),                                                                        //Grid dimensions
            dim3(32, 1, 1),                                                                                              //Block dimensions
            0u,                                                                                                          //Dyn. shared-mem per block in bytes
            (void**)&lossArgs,                                                                                           //Array of pointers to individual kernel arguments
            nullptr                                                                                                      //Pointer to kernel arguments in the "extra" format
        };
        if (outStateSizeBatched % 32)
            lossParams.func = (void*)&transform_reduce_indirection<T, decltype(ldb), DIVISIBILITY::NOT_DIVISIBLE, DIVISIBILITY::NOT_DIVISIBLE, false>;
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

        uint32_t outStateSizeBatched = outStateShape.prod() * batch_size;
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in1, T in2) EXTENDED_CONSTEXPR -> T { /*printf("T:%f G:%f -> %f\n", in2, in1, -(in2 + (T)0.0005) / (in1 + (T)0.0005) + ((T)1.0005 - in2) / ((T)1.0005 - in1));*/ return -(in2 + (T)0.0005) / (in1 + (T)0.0005) + ((T)1.0005 - in2) / ((T)1.0005 - in1); };

        void* deltaArgs[] = {
            (void*)&guess,
            (void*)&target,
            (void*)&deltas,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams deltaParams{
            (void*)&transform4_indirection<T, decltype(ldb)>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),             //Grid dimensions
            dim3(32, 1, 1),                                   //Block dimensions
            0u,                                               //Dyn. shared-mem per block in bytes
            (void**)&deltaArgs,                               //Array of pointers to individual kernel arguments
            nullptr                                           //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, deltaGraph, nullptr, 0, &deltaParams);

        //3.: Return
        return deltaGraph;
    }
};

//=============================================================
//==================|Activation Functions|=====================
//=============================================================

enum ACTIVATION_TYPE: uint32_t {IDENTITY=0, RELU=1, SOFTMAX=2, SOFTMAX_TEMP=3, SIGMOID=4, TANH=5, SOFTPLUS=6};  //DON'T CHANGE THESE VALUES AS IT WILL BREAK OLD CHECKPOINT FILES!
template<typename T>
class Activation {
protected:
    uint32_t sampleSize;
    uint32_t batch_size;

    T* params;      //TODO: At the moment, they are constant and can't be learned.

    /*
        Returns the memory requirements of this activation excluding every recursive calls

        @param requirements    : Out parameter. Add requirements for "params" to the vector
        @param tmp             : Out parameter. Overwrites variable with the number of bytes on the gpu needed as temporary storage for forward- and backpropagation
        @param num_optimizables: Out parameter. Add to this the number of T's that can be optimized in this layer
    */
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const {
        fprintf(stderr, "[ERROR] You called \"getOwnMemoryRequirements\" on the activation base class. You should only use the derived classes!");
        std::exit(-1);
    }

public:
    static constexpr bool cudnnBackendSupport = false;

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
    void setSizes(uint32_t sampleSize_, uint32_t batch_size_) {
        sampleSize = sampleSize_;
        batch_size = batch_size_;
    }
    /*
        Returns the memory requirements of this activation including every recursive calls

        @param requirements    : Out parameter. Add requirements for "params" to the vector
        @param tmp             : Out parameter. Overwrites variable with the number of bytes on the gpu needed as temporary storage for forward- and backpropagation
        @param num_optimizables: Out parameter. Adds to this the number of T's that can be optimized in this layer
    */
    virtual CPP20_CONSTEXPR void getMemoryRequirements(std::vector<MemoryRequirementLifetime>& other_requirement, MemoryRequirement& tmp_requirement, uint64_t& num_optimizables) const {
        getOwnMemoryRequirements(other_requirement, tmp_requirement, num_optimizables);
    }
    /*
        Sets the internal memory pointers of the actiavtion. The passed pointer will be set to the first byte after the by this activation used memory
        region. The pointers do not need to be aligned as first they will be padded so they satisfy the alignment requirement and then incrementen by the 
        space requirement of this activation, as specified in "getMemoryRequirement".

        Does not include memory for activation!

        @param param_memory: Pointer to enough free memory that this activation will use to store "params". Will be set after the region used for this.
    */
    void setMem(void**& param_memory) {
        //1.: Set memory
        params = (T*)*param_memory++;

#ifdef DEBUG
        //2.: Check alignment
        std::vector<MemoryRequirementLifetime> requirements;
        MemoryRequirement tmp;
        uint64_t num_optimizables = 0;
        getOwnMemoryRequirements(requirements, tmp, num_optimizables);

        assert(is_aligned((void*)params, requirements[0].alignment));
#endif
    }
    /*
        Initializes the memory pointed to by "params"
    */
    virtual void initMem() {
        fprintf(stderr, "[ERROR] You called \"initMem\" on the activation base class. You should only use the derived classes!");
        exit(-1);
    }

    /*
        Adds a graph node to a graph that performs forward propagation through this activation.

        @param mem         : The input memory that the activation should be applied to
        @param tmp         : Temporary memory satisfying the requirements returned in "getMemoryRequirements"
        @param graph       : The graph generated nodes will be added to
        @param depsMem     : The dependencies on "mem". Will be updated
        @param captureStream: The stream used to capture cublas operations
    */
    virtual void addActivationNode     (T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) const  {
        fprintf(stderr, "[ERROR] You called \"addActivationNode\" on the activation base class. You should only use the derived classes!");
        exit(-1);
    }
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
    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) const {
        fprintf(stderr, "[ERROR] You called \"addActivationDerivNode\" on the activation base class. You should only use the derived classes!");
        exit(-1);
    }
    /*
        Creates a cuDnn backend descrptor representing the activation operation.

        @param  inMemDesc: cuDnn backend tensor descriptor for the  input memory
        @param outMemDesc: cuDnn backend tensor descriptor for the output memory
    */
    virtual cudnnBackendDescriptor_t getCudnnOperationDescriptor(cudnnBackendDescriptor_t inMemDesc, cudnnBackendDescriptor_t outMemDesc) const {
        fprintf(stderr, "[ERROR] You called \"getCudnnOperationDescriptor\" on the activation base class. You should only use the derived classes!");
        exit(-1);
    }
    
    /*
        Returns the type of this activation
    */
    virtual CPP20_CONSTEXPR ACTIVATION_TYPE getActivationType() const {
        fprintf(stderr, "[ERROR] You called \"getActivationType\" on the activation base class. You should only use the derived classes!");
        exit(-1);
    }
    /*
        Creates an activation of the given derived class at a specific memory location.

        @param at:  The derived class to create
        @param out: The memory location the object should be created at
    */
    static Activation<T>* getActivationOfType(ACTIVATION_TYPE at);

    // /+=============+\
    // ||SERIALIZATION||
    // \+=============+/

    /*
        Serializes an object to a FILE*. It can be deserialized using the "deserialize" method.
        The serialize method should be called from the derived class. It is required to first write "getActivationType".
        Deserialization is invoked from this base class. It reads the layer type first and afterwards invokes the "deserialize"
        method of the corresponding derived class.
    */
    void serialize(FILE* file) const  {
        //1.: Write activation type
        ACTIVATION_TYPE activation_type = getActivationType();
        fwrite(&activation_type, sizeof(activation_type), 1, file);

        //2.: Write variables
        fwrite(&sampleSize, sizeof(sampleSize), 1, file);
        fwrite(&batch_size, sizeof(batch_size), 1, file);

        //3.: Write memory
        std::vector<MemoryRequirementLifetime> requirements;
        MemoryRequirement tmp;
        uint64_t num_optimizables = 0;
        getMemoryRequirements(requirements, tmp, num_optimizables);

        fwrite(&requirements[0].num_bytes, sizeof(uint64_t), 1, file);
        fwrite((void*)params, 1, requirements[0].num_bytes, file);
    }
    /*
        Deserializes an object from a FILE* that was serialized using the "serialize" method.
        The deserialize method should be called from the base class. It reads the activation type first and afterwards invokes the "deserialize"
        method of the corresponding derived class.
    */
    static Activation<T>* deserialize(FILE* file) {
        //1.: Create correct derived class
        ACTIVATION_TYPE activation_type;
        fread(&activation_type, sizeof(ACTIVATION_TYPE), 1, file);
        Activation<T>* ret = Activation<T>::getActivationOfType(activation_type);

        //2.: Get memory requirements
        std::vector<MemoryRequirementLifetime> requirements;
        MemoryRequirement tmp;
        uint64_t num_optimizables = 0;
        ret->getMemoryRequirements(requirements, tmp, num_optimizables);

        //3.: Read in variables
        fread(&ret->sampleSize, sizeof(sampleSize), 1, file);
        fread(&ret->batch_size, sizeof(batch_size), 1, file);

        //4.: Read in memory
        uint64_t num_bytes;
        fread(&num_bytes, sizeof(uint64_t), 1, file);
        if (requirements[0].num_bytes != num_bytes) {
            fprintf(stderr, "[ERROR] Trying to create a activation of type %llu with %llu parameter bytes, even though it requires %llu", (uint64_t)activation_type, (uint64_t)num_bytes, (uint64_t)requirements[0].num_bytes);
            std::exit(-1);
        }

        cudaMallocAligned((void**)&ret->params, requirements[0].getMemoryRequirements());
        fread((void*)ret->params, 1, other_bytes, file);

        //6.: Return
        return ret;
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
            std::vector<MemoryRequirementLifetime> requirements;
            MemoryRequirement tmp;
            uint64_t num_optimizables = 0;
            getMemoryRequirements(requirements, tmp, num_optimizables);

            void* host_buffer;
            cudaMallocHost(&host_buffer, requirements[0].num_bytes);
            cudaMemcpy(host_buffer, (void*)params, requirements[0].num_bytes, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(NULL);
            fwrite(host_buffer, 1, requirements[0].num_bytes, file);
            cudaFreeHost(host_buffer);
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
    static Activation<T>* getActivationFromCompression(FILE* file) {
        ACTIVATION_TYPE at;
        fread(&at, sizeof(ACTIVATION_TYPE), 1, file);
        return getActivationOfType(at);
    }
    /*
        Inverse of "compress<true>"
    */
    void initMemFromCompression(FILE* file) {
        std::vector<MemoryRequirementLifetime> requirements;
        MemoryRequirement tmp;
        uint64_t num_optimizables = 0;
        getMemoryRequirements(requirements, tmp, num_optimizables);

        if (requirements.size() && requirements[0].num_bytes) {
            void* host_buffer;
            cudaMallocHost(&host_buffer, requirements[0].num_bytes);
            fread(host_buffer, 1, requirements[0].num_bytes, file);
            cudaMemcpy((void*)params, host_buffer, requirements[0].num_bytes, cudaMemcpyHostToDevice);
            cudaStreamSynchronize(NULL);
            cudaFreeHost(host_buffer);
        }
    }
};

template<typename T>
class IDENTITY_Activation : public Activation<T> {
protected:
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const override {
        requirements.push_back(MemoryRequirementLifetime()); //No requirement to add
        tmp = MemoryRequirement();                           //No memory needed
        num_optimizables += 0;                               //No optimizables
    }

public:
    static constexpr bool cudnnBackendSupport = false;

    //1.: Constructors
    IDENTITY_Activation() = default;   //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual CPP20_CONSTEXPR ACTIVATION_TYPE getActivationType() const override {
        return ACTIVATION_TYPE::IDENTITY;
    }


    virtual void initMem() override {} //No memory to initialize

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) const override {}

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) const override {
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;

        cudaGraphNode_t node;

        //1.: Copy "deltas" to "mem"
        cudaGraphAddMemcpyNode1D(&node, graph, nullptr, 0, (void*)mem, (void*)deltas, sizeof(T) * outStateSizeBatched, cudaMemcpyDeviceToDevice);
        depsMem.apply<true>(graph, node);
        depsDeltas.apply<false>(graph, node);
    }

    virtual cudnnBackendDescriptor_t getCudnnOperationDescriptor(cudnnBackendDescriptor_t inMemDesc, cudnnBackendDescriptor_t outMemDesc) const override {
        fprintf(stderr, "[ERROR] There is currently no cudnn backend support for indentity activation");
        std::exit(-1);
    }
};

template<typename T>
class RELU_Activation : public Activation<T> {
protected:
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const override {
        requirements.push_back(MemoryRequirementLifetime()); //No requirement to add
        tmp = MemoryRequirement();                           //No memory needed
        num_optimizables += 0;                               //No optimizables
    }

public:
    static constexpr bool cudnnBackendSupport = true;

    //1.: Constructors
    RELU_Activation() = default;   //For "getActivationOfType" and user
    
    //2.: Overloaded functions
    virtual CPP20_CONSTEXPR ACTIVATION_TYPE getActivationType() const override {
        return ACTIVATION_TYPE::RELU;
    }

    virtual void initMem() override {} //No memory to initialize

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) const override {
        //1.: Variables used
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in) EXTENDED_CONSTEXPR -> T { return in > (T)0 ? in : (T)0; };
        void* reluArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams reluParam{
            (void*)&transform<T, decltype(ldb)>,   //Function pointer
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
    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) const override {
        //0: Variables
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;

        cudaGraphNode_t node;
        
        //1.: Calculate derivatives of output of activation with respect to input of activation
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in) EXTENDED_CONSTEXPR -> T { return in > (T)0 ? (T)1 : (T)0; };

        void* reluArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams reluParam{
            (void*)&transform<T, decltype(ldb)>,   //Function pointer
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

    virtual cudnnBackendDescriptor_t getCudnnOperationDescriptor(cudnnBackendDescriptor_t inMemDesc, cudnnBackendDescriptor_t outMemDesc) const override {
        cudnnBackendDescriptor_t op;
        cudnnPointwiseMode_t actOp = CUDNN_POINTWISE_RELU_FWD;
        float alpha1 = 1.f;

        CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &op));
        CUDNN_ERROR(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_POINTWISE_MODE    , 1, &actOp));
        CUDNN_ERROR(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC        , CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &inMemDesc));
        CUDNN_ERROR(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC        , CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &outMemDesc));
        CUDNN_ERROR(cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1       , CUDNN_TYPE_FLOAT             , 1, &alpha1));
        CUDNN_ERROR(cudnnBackendFinalize(op));

        return op;
    }
};

template<typename T>
class Softmax_Activation : public Activation<T> {
protected:
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const override {
        requirements.push_back(MemoryRequirementLifetime());                                              //No requirement to add
        tmp = MemoryRequirement(this->batch_size * this->sampleSize * this->sampleSize * sizeof(T), 16u); //For backprop
        num_optimizables += 0;                                                                            //No optimizables
    }

public:
    static constexpr bool cudnnBackendSupport = false;

    //1.: Constructors
    Softmax_Activation() = default; //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual CPP20_CONSTEXPR ACTIVATION_TYPE getActivationType() const override {
        return ACTIVATION_TYPE::SOFTMAX;
    }

    virtual void initMem() override {}  //No memory

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) const override {
        //1.: Variables
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;

        cudaGraphNode_t node;

        //2.: Kernel
        void* softmaxArgs[] = {
            (void*)&mem,
            (void*)&this->sampleSize,
            (void*)&this->batch_size
        };
        cudaKernelNodeParams softmaxParams{
            (void*)softmax<T, 256, false, true>,     //Function pointer
            dim3(this->batch_size, 1, 1),            //Grid dimensions
            dim3(256, 1, 1),                         //Block dimensions
            (256 / warp_size) * sizeof(T),           //Dyn. shared-mem per block in bytes
            (void**)softmaxArgs,                     //Array of pointers to individual kernel arguments
            nullptr                                  //Extra
        };        
        if ((this->sampleSize & (this->sampleSize - 1)) == 0)
            softmaxParams.func = (void*)softmax<T, 256, true, true>;
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &softmaxParams);
        depsMem.apply<true>(graph, node);
    }

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) const override {
        //1.: Variables
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;
        Dependencies depsTmp;

        cudaGraphNode_t node;

        //2.: Generate derivatives of softmax. tmp=derivs[batch_size][sampleSize][sampleSize].
        //derivs[b][x][y] (yep, column major) stores derivative of output of x with respect to input x in batch b.

        void* softmaxArgs[] = {
            (void*)&mem,
            (void*)&tmp,
            (void*)&this->sampleSize,
            (void*)&this->batch_size
        };
        cudaKernelNodeParams softmaxParams{
            (void*)softmax_deriv<T>,                 //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),    //Grid dimensions
            dim3(32, 1, 1),                          //Block dimensions
            0u,                                      //Dyn. shared-mem per block in bytes
            (void**)softmaxArgs,                     //Array of pointers to individual kernel arguments
            nullptr                                  //Extra
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &softmaxParams);
        depsMem.apply<false>(graph, node);
        depsTmp.apply<true> (graph, node);

        //3.: Multiply generated derivative matrix with deltas
        cudaGraph_t mulGraph = getMatmulGraph<T, false, false, true>(tmp, deltas, mem, this->sampleSize, this->sampleSize, this->batch_size, captureStream);
        cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, mulGraph);

        depsMem.apply<true> (graph, node);
        depsTmp.apply<false>(graph, node);
    }

    virtual cudnnBackendDescriptor_t getCudnnOperationDescriptor(cudnnBackendDescriptor_t inMemDesc, cudnnBackendDescriptor_t outMemDesc) const override {
        fprintf(stderr, "[ERROR] There is currently no cudnn backend support for softmax");
        std::exit(-1);
    }
};

template<typename T>
class SoftmaxTemp_Activation : public Activation<T> {
protected:
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const override {
        requirements.push_back(MemoryRequirementLifetime(sizeof(T), alignof(T)));                         //Temperature
        tmp = MemoryRequirement(this->batch_size * this->sampleSize * this->sampleSize * sizeof(T), 16u); //For backprop
        num_optimizables += 1;                                                                            //Temperature
    }

public:
    static constexpr bool cudnnBackendSupport = false;
    //1.: Constructors
    SoftmaxTemp_Activation() = default; //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual CPP20_CONSTEXPR ACTIVATION_TYPE getActivationType() const override {
        return ACTIVATION_TYPE::SOFTMAX_TEMP;
    }

    virtual void initMem() override {
        //Initialize temperature to 1
        T temp = (T)1;
        cudaMemcpy((void*)this->params, (void*)&temp, sizeof(T), cudaMemcpyHostToDevice);
    }

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) const override {
        //1.: Variables
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;

        cudaGraphNode_t node;

        //2.: Kernel
        void* softmaxArgs[] = {
            (void*)&mem,
            (void*)&this->sampleSize,
            (void*)&this->batch_size,
            (void*)&this->params
        };
        cudaKernelNodeParams softmaxParams{
            (void*)softmaxTemp<T, 256, false, true>, //Function pointer
            dim3(this->batch_size, 1, 1),            //Grid dimensions
            dim3(256, 1, 1),                         //Block dimensions
            (256 / warp_size) * sizeof(T),            //Dyn. shared-mem per block in bytes
            (void**)softmaxArgs,                     //Array of pointers to individual kernel arguments
            nullptr                                  //Extra
        }; 
        if ((this->sampleSize & (this->sampleSize - 1)) == 0)
            softmaxParams.func = (void*)softmaxTemp<T, 256, true, true>;
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &softmaxParams);
        depsMem.apply<true>(graph, node);
    }

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) const override {
        //1.: Variables
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;
        Dependencies depsTmp;

        cudaGraphNode_t node;

        //2.: Generate derivatives of softmax. tmp=derivs[batch_size][sampleSize][sampleSize].
        //derivs[b][x][y] (yep, column major) stores derivative of output of x with respect to input x in batch b.

        void* softmaxArgs[] = {
            (void*)&mem,
            (void*)&tmp,
            (void*)&this->sampleSize,
            (void*)&this->batch_size,
            (void*)&this->params
        };
        cudaKernelNodeParams softmaxParams{
            (void*)softmaxTemp_deriv<T>,             //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),    //Grid dimensions
            dim3(32, 1, 1),                          //Block dimensions
            0u,                                      //Dyn. shared-mem per block in bytes
            (void**)softmaxArgs,                     //Array of pointers to individual kernel arguments
            nullptr                                  //Extra
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &softmaxParams);
        depsMem.apply<false>(graph, node);
        depsTmp.apply<true>(graph, node);

        //3.: Multiply generated derivative matrix with deltas
        cudaGraph_t mulGraph = getMatmulGraph<T, false, false, true>(tmp, deltas, mem, this->sampleSize, this->sampleSize, 1u, captureStream);
        cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, mulGraph);
        depsMem.apply<true>(graph, node);
        depsTmp.apply<false>(graph, node);
    }

    virtual cudnnBackendDescriptor_t getCudnnOperationDescriptor(cudnnBackendDescriptor_t inMemDesc, cudnnBackendDescriptor_t outMemDesc) const override {
        fprintf(stderr, "[ERROR] There is currently no cudnn backend support for temperature softmax");
        std::exit(-1);
    }
};

template<typename T>
class Sigmoid_Activation : public Activation<T> {
protected:
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const override {
        requirements.push_back(MemoryRequirementLifetime()); //No requirement to add
        tmp = MemoryRequirement();                           //No memory needed
        num_optimizables += 0;                               //No optimizables
    }

public:
    static constexpr bool cudnnBackendSupport = true;
    //1.: Constructors
    Sigmoid_Activation() = default; //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual CPP20_CONSTEXPR ACTIVATION_TYPE getActivationType() const override {
        return ACTIVATION_TYPE::SIGMOID;
    }

    virtual void initMem() override {} //No memory to initialize

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) const override {
        //1.: Variables used
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in) EXTENDED_CONSTEXPR -> T { return (T)1 / ((T)1 + exponential<T>(-in)); };
        void* sigArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams sigParam{
            (void*)&transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&sigArgs,                     //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &sigParam);
        depsMem.apply<true>(graph, node);
    }

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) const override {
        //0: Variables
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;

        cudaGraphNode_t node;

        //1.: Calculate derivatives of output of activation with respect to input of activation
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in) EXTENDED_CONSTEXPR -> T { return in * ((T)1 - in); };

        void* sigArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams sigParam{
            (void*)&transform<T, decltype(ldb)>,   //Function pointer
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

    virtual cudnnBackendDescriptor_t getCudnnOperationDescriptor(cudnnBackendDescriptor_t inMemDesc, cudnnBackendDescriptor_t outMemDesc) const override {
        cudnnBackendDescriptor_t op;
        cudnnPointwiseMode_t actOp = CUDNN_POINTWISE_SIGMOID_FWD;
        float alpha1 = 1.f;

        cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &op);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_POINTWISE_MODE, 1, &actOp);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &inMemDesc);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &outMemDesc);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 1, &alpha1);
        cudnnBackendFinalize(op);

        return op;
    }
};

template<typename T>
class Tanh_Activation : public Activation<T> {
protected:
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const override {
        requirements.push_back(MemoryRequirementLifetime()); //No requirement to add
        tmp = MemoryRequirement();                           //No memory needed
        num_optimizables += 0;                               //No optimizables
    }

public:
    static constexpr bool cudnnBackendSupport = true;

    //1.: Constructors
    Tanh_Activation() = default; //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual CPP20_CONSTEXPR ACTIVATION_TYPE getActivationType() const override {
        return ACTIVATION_TYPE::TANH;
    }

    virtual void initMem() override {} //No memory to initialize

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) const override {
        //1.: Variables used
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        EXTENDED_CONSTEXPR auto ldb = []__device__(T x) EXTENDED_CONSTEXPR -> T { return tanh<T>(x); };
        void* tanhArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams tanhParam{
            (void*)&transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&tanhArgs,                    //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &tanhParam);
        depsMem.apply<true>(graph, node);
    }

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) const override {
        //0: Variables
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;

        cudaGraphNode_t node;

        //1.: Calculate derivatives of output of activation with respect to input of activation
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in) EXTENDED_CONSTEXPR -> T { return (T)1 - in * in; };

        void* tanhArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams tanhParam{
            (void*)&transform<T, decltype(ldb)>,   //Function pointer
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

    virtual cudnnBackendDescriptor_t getCudnnOperationDescriptor(cudnnBackendDescriptor_t inMemDesc, cudnnBackendDescriptor_t outMemDesc) const override {
        cudnnBackendDescriptor_t op;
        cudnnPointwiseMode_t actOp = CUDNN_POINTWISE_TANH_FWD;
        float alpha1 = 1.f;

        cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &op);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_POINTWISE_MODE, 1, &actOp);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &inMemDesc);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &outMemDesc);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 1, &alpha1);
        cudnnBackendFinalize(op);

        return op;
    }
};

template<typename T>
class Softplus_Activation : public Activation<T> {
protected:
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const override {
        requirements.push_back(MemoryRequirementLifetime()); //No requirement to add
        tmp = MemoryRequirement();                           //No memory needed
        num_optimizables += 0;                               //No optimizables
    }

public:
    static constexpr bool cudnnBackendSupport = true;

    //1.: Constructors
    Softplus_Activation() = default; //For "getActivationOfType" and user

    //2.: Overloaded functions
    virtual CPP20_CONSTEXPR ACTIVATION_TYPE getActivationType() const override {
        return ACTIVATION_TYPE::SOFTPLUS;
    }

    virtual void initMem() override {} //No memory to initialize

    virtual void addActivationNode(T* mem, T* tmp, cudaGraph_t graph, Dependencies& depsMem, cudaStream_t captureStream) const override {
        //1.: Variables used
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;

        cudaGraphNode_t node;

        //2.: Add node to graph
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in) EXTENDED_CONSTEXPR -> T { return logarithm<T>((T)1 + exponential<T>(-abs<T>(in))) + max<T>(in,0); };
        void* softplusArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams softplusParam{
            (void*)&transform<T, decltype(ldb)>,  //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&softplusArgs,                //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &softplusParam);
        depsMem.apply<true>(graph, node);

        //3.: Set out parameters     
        //depsMem reamains depsMem and thus remains the same.
    }

    virtual void addActivationDerivNode(T* mem, T* deltas, T* tmp, cudaGraph_t graph, Dependencies& depsMem, Dependencies& depsDeltas, cudaStream_t captureStream) const override {
        //0: Variables
        uint32_t outStateSizeBatched = this->sampleSize * this->batch_size;

        cudaGraphNode_t node;

        //1.: Calculate derivatives of output of activation with respect to input of activation
        EXTENDED_CONSTEXPR auto ldb = []__device__(T in) EXTENDED_CONSTEXPR -> T { return (T)1 - exponential<T>(-in); };

        void* softplusArgs[] = {
            (void*)&mem,
            (void*)&outStateSizeBatched,
            (void*)&ldb
        };
        cudaKernelNodeParams softplusParam{
            (void*)&transform<T, decltype(ldb)>,   //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)), //Grid dimensions
            dim3(32, 1, 1),                       //Block dimensions
            0u,                                   //Dyn. shared-mem per block in bytes
            (void**)&softplusArgs,                //Array of pointers to individual kernel arguments
            nullptr                               //Pointer to kernel arguments in the "extra" format
        };
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &softplusParam);
        depsMem.apply<true>(graph, node);

        //2.: Calculate derivatives of loss with respect to the input of the activation
        addElementwiseMultNode<T>(graph, (T*)mem, (T*)deltas, outStateSizeBatched, node);
        depsMem.apply<true>(graph, node);
        depsDeltas.apply<false>(graph, node);
    }

    virtual cudnnBackendDescriptor_t getCudnnOperationDescriptor(cudnnBackendDescriptor_t inMemDesc, cudnnBackendDescriptor_t outMemDesc) const override {
        cudnnBackendDescriptor_t op;
        cudnnPointwiseMode_t actOp = CUDNN_POINTWISE_SOFTPLUS_FWD;
        float alpha1 = 1.f;

        cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &op);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_POINTWISE_MODE, 1, &actOp);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &inMemDesc);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &outMemDesc);
        cudnnBackendSetAttribute(op, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1, CUDNN_TYPE_FLOAT, 1, &alpha1);
        cudnnBackendFinalize(op);

        return op;
    }
};



template<typename T>
static Activation<T>* Activation<T>::getActivationOfType(ACTIVATION_TYPE at) {
    switch (at) {
    case ACTIVATION_TYPE::IDENTITY:
        return new IDENTITY_Activation<T>();
    case ACTIVATION_TYPE::RELU:
        return new RELU_Activation<T>();
    case ACTIVATION_TYPE::SOFTMAX:
        return new Softmax_Activation<T>();
    case ACTIVATION_TYPE::SOFTMAX_TEMP:
        return new SoftmaxTemp_Activation<T>();
    case ACTIVATION_TYPE::SIGMOID:
        return new Sigmoid_Activation<T>();
    case ACTIVATION_TYPE::TANH:
        return new Tanh_Activation<T>();
    case ACTIVATION_TYPE::SOFTPLUS:
        return new Softplus_Activation<T>();
    default:
        fprintf(stderr, "[ERROR] %llu is not a known activation type!", (uint64_t)at);
        exit(-1);
    }
}

//============================================
//==================|Layers|==================
//============================================

template<typename T, typename L> class Scheduler; //Needed to declare friendship

enum LAYER_TYPE: uint32_t {INPT=0, FULLY_CONNECTED=1, CNN=2, RNN=3, LSTM=4, TRANSFORMER=5, POOL=6};  //DON'T CHANGE THESE VALUES AS IT WILL BREAK OLD CHECKPOINT FILES!
template<typename T, typename L = T>
class Layer {
    friend Scheduler<T, L>;

    /*
        A layer is a helper class that does not actually own any memory. It is only used by the NetworkBuilder to construct
        the execution graph.

        This is only the base class that should never be used itself except for static methods. Each actual layer type should be implemented as a
        new derived class that overwrites the respective methods. Each virtual method except "getMemoryRequirements" (has a default implementation) 
        needs to be overwritten.

        Each derived class is only allowed to have "trivial" constructors (should be initializable with other methods equally well) 
        and shall suppy a constructor without arguments that allocates "intern".

        A layer is only allowed to operate on "output" and "outputShape" of "layerBefore".
    */
public:
    T* output;                 //The output of this layer (device-pointer)
    Image_Shape outputShape;   //Shape of one output sample (in T's)


protected:
    Layer<T, L>* layerBefore;  //Pointer to the layer before this one. If this is the first layer, this has to be "nullptr"
    uint32_t     batch_size;   //Number of samples in a batch

    Activation<T>* act;        //The activation of this layer

    void** intern;             //Layers can use this however they want.
    static constexpr uint32_t numBuffersIntern = 0u;                                          //Only for use in derived class
    virtual CPP20_CONSTEXPR uint32_t getNumBuffersIntern() const { return numBuffersIntern; } //For use in base class

    /*
        Returns the memory requirements of this layer excluding the activation to the given variables.

        @param requirements    : Out parameter. Requirements for each internal buffer in the order they are stored in "intern" will be appended to this vector. The requirements of "output" have to be appended last
        @param tmp             : Out parameter. Overwrites variable with the number of bytes on the gpu needed as temporary storage for forward- and backpropagation 
        @param num_optimizables: Out parameter. Adds to this the number of T's that can be optimized in this layer
    */ 
    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const {
        fprintf(stderr, "[ERROR] You called \"getOwnMemoryRequirements\" on the layer base class. You should only use the derived classes!");
        exit(-1);
    }

public:
    /*
        Call methods in the order they are defined:
        1.: setLayerBefore
        2.: setBatchSize
        3.: getMemoryRequirements
        4.: setMem
        5.: initMem
        6.: forwardProp / backwardProp
    */
    
    /*
        Sets the variable holding a pointer to the layer right in front of this one in the network.
    */
    virtual void setLayerBefore(Layer<T, L>* l) { 
        layerBefore = l; 
    }
    /*
        Sets the batch size used by this layer. Note, that this does not reallocate state memory.
        This also passes the information on to "act".

        @param batch_size_: The new batch size.
    */
    void setBatchSize(uint32_t& batch_size_) {
        batch_size = batch_size_; 
        act->setSizes(outputShape.prod(), batch_size);
    }
    /*
        Returns the memory requirements of this layer including the activation to the given variables.

        @param requirements    : Out parameter. Requirements for each internal buffer in the order they are stored in "intern" will be appended to this vector. The requirements of "output" have to be appended last
        @param tmp             : Out parameter. Overwrites variable with the number of bytes on the gpu needed as temporary storage for forward- and backpropagation 
        @param num_optimizables: Out parameter. Adds to this the number of T's that can be optimized in this layer
     */
    virtual CPP20_CONSTEXPR void getMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const {
        //1.: Own requirements
        getOwnMemoryRequirements(requirements, tmp, num_optimizables);


        //2.: Requirements of activation
        MemoryRequirement tmp_req_activ;
        act->getMemoryRequirements(requirements, tmp_req_activ, num_optimizables);

        tmp = max(tmp, tmp_req_activ);
    }
    /*
        Sets the internal memory pointers of the layer and activation. The passed pointer will be set to the first byte after the used memory region.
        The pointers need to be aligned and be of the space required, as specified in "getMemoryRequirement".

        @param mem: Arrays of void* to set internal pointer with. They have the same order as the requirements returned on "getMemoryRequirements"
    */
    void setMem(void**& mem) {

        //1.: This layer
        uint32_t n_intern = getNumBuffersIntern();
        for (uint32_t buffer_ind = 0; buffer_ind < n_intern; buffer_ind++)
            intern[buffer_ind] = *mem++;

        output = (T*)(*mem++);

#ifdef DEBUG
        //2.: Check alignment
        std::vector<MemoryRequirementLifetime> requirements;
        MemoryRequirement tmp;
        uint64_t num_optimizables = 0;
        getOwnMemoryRequirements(requirements, tmp, num_optimizables);


        for (uint32_t buffer_ind = 0; buffer_ind < n_intern; buffer_ind++)
            assert(is_aligned(intern[buffer_ind], requirements[buffer_ind].alignment));

        assert(is_aligned(output, requirements[n_intern].alignment));
#endif

        //3.: Activation
        act->setMem(mem);
    }

    /* 
        Initializes "state" and "other" memory of the size returned in getMemoryRequirements.
        Also has to intialize memory of "act".
    */
    virtual void initMem() {
        fprintf(stderr, "[ERROR] You called \"initMem\" on the layer base class. You should only use the derived classes!");
        exit(-1);
    }
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
    virtual std::vector<T**> forwardProp(cudaGraph_t graph, Dependencies& depPrevState, cudaStream_t captureStream, T* tmp, bool after_dataset) const  {
        fprintf(stderr, "[ERROR] You called \"forwardProp\" on the layer base class. You should only use the derived classes!");
        exit(-1);
    }
    /*
        Adds backpropagation through this layer to an execution graph.
        This will update the internal parameters of the layer using the specified optimizer and specified gradients. Will generate the gradients of
        the previous layer as well.
    
        @param graph            : The execution graph to construct
        @param depState         : The dependencies of own output. Will be updated to the dependencies of output of layer before after this operation.
        @param opt              : Pointer optimizer used to change the internal parameters.
        @param optimizable_index: The index of the first element to the optimization buffer. Will be advanced (according to "getMemoryRequirements")
        @param deltas           : For each number in "output", this contains the derivative of the loss with respect to this number. If "after_dataset" is false, the deltas of the previous layer will be written in here. Thus, this buffer has to have enough space to accomondate these values.
        @apram depDeltas        : The dependencies of "deltas". Will be updated.
        @param captureStream    : The function can use stream capture to generate the call graph. This is the stream it will use for this.
        @param tmp              : Temporary storage of at least the size and alignment requested in "getMemoryRequirements".
        @param after_dataset    : True, when this layer is the first layer after the input layer that refers to the dataset
    */
    virtual std::vector<T**> backwardProp(cudaGraph_t graph, Dependencies& depState, Optimizer<T, L>* opt, uint64_t& optimizable_index, T* deltas, Dependencies& depDeltas, cudaStream_t captureStream, T* tmp, bool after_dataset) const  {
        fprintf(stderr, "[ERROR] You called \"backwardProp\" on the layer base class. You should only use the derived classes!");
        exit(-1);
    }
    
    /*
        Return the type of the layer.
    */
    virtual CPP20_CONSTEXPR LAYER_TYPE getLayerType() const  {
        fprintf(stderr, "[ERROR] You called \"getLayerType\" on the layer base class. You should only use the derived classes!");
        exit(-1);
    }
    /*
        This function calls new with default constructor on the correct derived class and returns the pointer to the object.

        @param lt: Specifies, which derived class to create;
    */
    static Layer<T, L>* getLayerOfType(LAYER_TYPE lt);
    /*
        This function can be called by user to create a wanted layer. However, the constructor of the corresponding derived class is to be prefered.
        Memory for object allocated by "new" and returned at the end.

        @param lt   : Speciefies the derived class of "Layer" to create
        @param shape
    */
    static Layer<T, L>* getLayerFromSpecifiers(LAYER_TYPE lt, Image_Shape shape, ACTIVATION_TYPE at) {
        Layer<T, L>* ret = getLayerOfType(lt);
        ret->outputShape = shape;
        ret->act         = Activation<T>::getActivationOfType(at);

        return ret;
    }
    
    // /+=============+\
    // ||SERIALIZATION||
    // \+=============+/

    /*
        Serialization according to the serialization rules.
    */
    void serialize(FILE* file) const  {
        //1.: Write layer type
        LAYER_TYPE layer_type = getLayerType();
        fwrite(&layer_type, sizeof(LAYER_TYPE), 1, file);

        //2.: Write variables
        fwrite(&layer_before, sizeof(layer_before), 1, file);
        fwrite(&batch_size  , sizeof(batch_size  ), 1, file);
        act->serialize(file);
        outputShape.serialize(file);

        //3.: Write memory
        std::vector<MemoryRequirementLifetime> requirements;
        MemoryRequirement tmp;
        uint64_t num_optimizables = 0;
        getOwnMemoryRequirements(requirements, tmp, num_optimizables);

        MemoryRequirement hostBuf = max(requirements);
        void* host_buf;
        cudaMallocHost(&host_buf, hostBuf.num_bytes);

        uint32_t n_intern = getNumBuffersIntern();
        fwrite(&n_intern, sizeof(n_intern), 1, file);
        for (uint32_t buf_ind = 0; buf_ind != n_intern; buf_ind++) {
            fwrite(&requirements[buf_ind].num_bytes, sizeof(uint64_t), 1, file);

            cudaMemcpy(host_buf, intern[buf_ind], requirements[buf_ind].num_bytes, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(NULL);
            fwrite(host_buf, 1, requirements[buf_ind].num_bytes, file);
        }

        uint32_t outputSize = requirements[n_intern].num_bytes;
        fwrite(&outputSize, sizeof(outputSize), 1, file);

        cudaMemcpy(host_buf, (void*)output, requirements[n_intern].num_bytes, cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(NULL);
        fwrite(host_buf, 1, requirements[n_intern].num_bytes, file);

        cudaFreeHost(host_buf);
    }
    /*
        Deserialization according to deserialization rules
    */
    static Layer<T, L>* deserialize(FILE* file) {
        //1.: Create correct derived class
        LAYER_TYPE layer_type;
        fread(&layer_type, sizeof(LAYER_TYPE), 1, file);
        Layer<T, L>* ret = Layer<T>::getLayerOfType(layer_type);

        //2.: Read in variables
        fread((void*)&ret->layer_before, sizeof(ret->layer_before), 1, file);
        fread((void*)&ret->batch_size  , sizeof(ret->batch_size), 1, file);
        ret->act = Activation<T>::deserialize(file);
        Image_Shape::deserialize(file, &ret->outputShape);

        //3.: Get memory requirements
        std::vector<MemoryRequirementLifetime> requirements;
        MemoryRequirement tmp;
        uint64_t num_optimizables = 0;
        ret->getMemoryRequirements(requirements, tmp, num_optimizables);

        MemoryRequirement hostBuf = max(requirements);
        void* host_buf;
        cudaMallocHost(&host_buf, hostBuf.num_bytes);

        //4.: Read in memory
        uint32_t n_intern;
        fread(&n_intern, sizeof(n_intern), 1, file);
        if (n_intern != ret->getNumBuffersIntern()) {
            fprintf(stderr, "[ERROR] Cannot deserialize layer as the wrong number of intern buffers were saved!");
            std::exit(-1);
        }

        for (uint32_t buf_ind = 0; buf_ind != n_intern; buf_ind++) {
            uint64_t num_bytes;
            fread(&num_bytes, sizeof(uint64_t), 1, file);

            if (num_bytes != requirements[buf_ind].num_bytes) {
                fprintf(stderr, "[ERROR] Cannot deserialize layer as internal buffer has the wrong size!");
                std::exit(-1);
            }

            cudaMallocAligned(&ret->intern[buf_ind], requirements[buf_ind].getMemoryRequirements());
            fread(host_buf, 1, num_bytes, file);
            cudaMemcpy(ret->intern[buf_ind], host_buf, num_bytes, cudaMemcpyHostToDevice);
            cudaStreamSynchronize(NULL);
        }

        uint32_t outputSize;
        fread(&outputSize, sizeof(outputSize), 1, file);

        if (outputSize != requirements[n_intern].num_bytes) {
            fprintf(stderr, "[ERROR] Cannot deserialize layer as output buffer has the wrong size!");
            std::exit(-1);
        }

        cudaMallocAligned((void**)&ret->output, requirements[n_intern].getMemoryRequirements());
        fread(host_buf, 1, outputSize, file);
        cudaMemcpy((void*)ret->output, host_buf, outputSize, cudaMemcpyHostToDevice);
        cudaStreamSynchronize(NULL);


        cudaFreeHost(host_buf);

        //6.: Return
        return ret;
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
            //1.: Write "intern"
            std::vector<MemoryRequirementLifetime> requirements;
            MemoryRequirement tmp;
            uint64_t num_optimizables = 0;
            getOwnMemoryRequirements(requirements, tmp, num_optimizables);
            requirements.pop_back();                                          //Requirement of "output" is not needed

            MemoryRequirement hostBuf = max(requirements);
            void* host_buf;
            cudaMallocHost(&host_buf, hostBuf.num_bytes);

            uint32_t n_intern = getNumBuffersIntern();
            for (uint32_t buf_ind = 0; buf_ind != n_intern; buf_ind++) {
                cudaMemcpy(host_buf, intern[buf_ind], requirements[buf_ind].num_bytes, cudaMemcpyDeviceToHost);
                cudaStreamSynchronize(NULL);
                fwrite(host_buf, 1, requirements[buf_ind].num_bytes, file);
            }

            cudaFreeHost(host_buf);

            //2.: Compress activation
            act->compress<true>(file);
        }
        else {
            //1.: Write specifiers
            LAYER_TYPE lt = getLayerType();
            fwrite(&lt, sizeof(LAYER_TYPE ), 1, file);
            outputShape.serialize(file);

            //2.: Write acitvation specifiers
            act->compress<false>(file);
        }
    }
    /*
        Inverse of "compress<false>"
    */
    static Layer<T, L>* getLayerFromCompression(FILE* file) {
        //1.: Layer specifiers
        LAYER_TYPE lt;
        fread(&lt, sizeof(LAYER_TYPE), 1, file);
        Layer<T, L>* ret = getLayerOfType(lt);

        Image_Shape::deserialize(file, &ret->outputShape);

        //2.: Activation
        ret->act = Activation<T>::getActivationFromCompression(file);

        //3.: Return
        return ret;
    }
    /*
        Inverse of "compress<true>". Should be calles after "setMem".
    */
    void initMemFromCompression(FILE* file) {
        //1.: Layer
        std::vector<MemoryRequirementLifetime> requirements;
        MemoryRequirement tmp;
        uint64_t num_optimizables = 0;
        getOwnMemoryRequirements(requirements, tmp, num_optimizables);
        requirements.pop_back();                                          //Requirement of "output" is not needed

        MemoryRequirement hostBuf = max(requirements);
        void* host_buf;
        cudaMallocHost(&host_buf, hostBuf.num_bytes);

        uint32_t n_intern = getNumBuffersIntern();
        for (uint32_t buf_ind = 0; buf_ind != n_intern; buf_ind++) {
            fread(host_buf, 1, requirements[buf_ind].num_bytes, file);
            cudaMemcpy(intern[buf_ind], host_buf, requirements[buf_ind].num_bytes, cudaMemcpyHostToDevice);
            cudaStreamSynchronize(NULL);
        }

        cudaFreeHost(host_buf);

        //2.: Activation
        act->initMemFromCompression(file);
    }
};

template<typename T, typename L = T>
class Input_Layer : public Layer<T, L> {
    //output={pointer to current gpu tile}
    //intern={}
protected:
    static constexpr uint32_t numBuffersIntern = 0u;                                         
    virtual CPP20_CONSTEXPR uint32_t getNumBuffersIntern() const override { return numBuffersIntern; } 

    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const override {
        //1.: This layer
        requirements.push_back(MemoryRequirementLifetime(sizeof(T*), alignof(T*))); //Output.  Alignment 16 for cublas and "transform"

        tmp = MemoryRequirement();                                                  //No computations, so no temporary memory needed

        num_optimizables += 0ull;                                                   //Nothing to optimizes
    }


public:
    void* output_host;

    //1.: Constructors
    //Used in "getLayerOfType"
    Input_Layer()        
    {
        this->layerBefore = nullptr;
        this->batch_size = 0;
        this->act = nullptr;

        this->output = nullptr;
        this->outputShape = Image_Shape(0u, 0u, 0u);

        this->intern = nullptr;
    }

    //Used by user
    Input_Layer(Image_Shape shape)
    {
        this->layerBefore = nullptr;
        this->batch_size = 0;
        this->act = Activation<T>::getActivationOfType(ACTIVATION_TYPE::IDENTITY);

        this->output = nullptr;
        this->outputShape = shape;

        this->intern = nullptr;
    }
    Input_Layer(uint32_t num_neurons)
    {
        this->layerBefore = nullptr;
        this->batch_size = 0;
        this->act = Activation<T>::getActivationOfType(ACTIVATION_TYPE::IDENTITY);

        this->output = nullptr;
        this->outputShape = Image_Shape(num_neurons, 1u, 1u);

        this->intern = nullptr;
    }

    virtual void initMem() override { 
        T* cur_gpu_tile = nullptr;
        gpuErrchk(cudaMemcpy(this->output, (void*)&cur_gpu_tile, sizeof(T*), cudaMemcpyHostToDevice)); //Write "nullptr" to state 
    }

    /*
        Set the pointer to the gpu tile to a specified pointer. Asynchronous.

        @param host_indirectionPointer: Must be a pointer allocated by "cudaMallocHost" that points to a pointer on the host that points to the correct gpu tile
        @param stream: The stream used for the memory transfer
    */
    void setInputPointer(T** host_indirectionPointer, cudaStream_t stream) {
        output_host = *host_indirectionPointer;
        cudaMemcpyAsync(this->output, host_indirectionPointer, sizeof(T*), cudaMemcpyHostToDevice, stream);
    }

    virtual std::vector<T**> forwardProp(cudaGraph_t graph, Dependencies& depPrevState, cudaStream_t captureStream, T* tmp, bool after_dataset) const override {
        fprintf(stderr, "[ERROR] You are trying to compute forward pass through input layer!");
        std::exit(-1);
    }

    virtual std::vector<T**> backwardProp(cudaGraph_t graph, Dependencies& depState, Optimizer<T, L>* opt, uint64_t& optimizable_index, T* deltas, Dependencies& depDeltas, cudaStream_t captureStream, T* tmp, bool after_dataset) const override {
        fprintf(stderr, "[ERROR] You are trying to compute backwards pass through input layer!");
        std::exit(-1);
    }

    virtual CPP20_CONSTEXPR LAYER_TYPE getLayerType() const override {
        return LAYER_TYPE::INPT;
    }
};

template<typename T, typename L = T>
class FullyConnected_Layer : public Layer<T, L> {
    //intern={bias|weights}
protected:
    static constexpr uint32_t numBuffersIntern = 2u;                                          //Only for use in derived class
    virtual CPP20_CONSTEXPR uint32_t getNumBuffersIntern() const override { return numBuffersIntern; } //For use in base class

    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const override {
        //1.: This layer
        requirements.push_back(MemoryRequirementLifetime(sizeof(T) * (uint64_t)this->outputShape.prod(), 16));                                                    //Bias.    Alignment 16 for cublas and "transform"
        requirements.push_back(MemoryRequirementLifetime(sizeof(T) * (uint64_t)this->outputShape.prod() * (uint64_t)this->layerBefore->outputShape.prod(), 16));  //Weights. Alignment 16 for cublas and "transform"
        requirements.push_back(MemoryRequirementLifetime(sizeof(T) * (uint64_t)this->outputShape.prod() * (uint64_t)this->batch_size, 16, true));                 //Output.  Alignment 16 for cublas and "transform"

        tmp  = MemoryRequirement();                                                                                                                               //No temporary memory needed

        num_optimizables += (uint64_t)this->outputShape.prod() * (1ull + (uint64_t)this->layerBefore->outputShape.prod());                                        //Bias + Weights
    }

public:
    // /+============+\
    // ||Constructors||
    // \+============+/
    //Used in "getLayerOfType":
    FullyConnected_Layer() 
    {
        this->layerBefore = nullptr; 
        this->batch_size  = 0;
        this->act         = nullptr;

        this->output      = nullptr;
        this->outputShape = Image_Shape(0u, 0u, 0u);

        this->intern      = (void**)malloc(sizeof(void*) * numBuffersIntern);   //bias and weights
    };
    //Used by user:
    FullyConnected_Layer(ACTIVATION_TYPE at  , uint32_t num_neurons)
    {
        this->layerBefore = nullptr;
        this->batch_size = 0;
        this->act = Activation<T>::getActivationOfType(at);

        this->output = nullptr;
        this->outputShape = Image_Shape(num_neurons, 1u, 1u);

        this->intern = (void**)malloc(sizeof(void*) * numBuffersIntern);       //bias and weights
    }
    FullyConnected_Layer(Activation<T>*  act_, uint32_t num_neurons)
    {
        this->layerBefore = nullptr;
        this->batch_size = 0;
        this->act = act_;

        this->output = nullptr;
        this->outputShape = Image_Shape(num_neurons, 1u, 1u);

        this->intern = (void**)malloc(sizeof(void*) * numBuffersIntern);       //bias and weights
    }

    FullyConnected_Layer(ACTIVATION_TYPE at, Image_Shape output_shape)
    {
        this->layerBefore = nullptr;
        this->batch_size = 0;
        this->act = Activation<T>::getActivationOfType(at);

        this->output = nullptr;
        this->outputShape = output_shape;

        this->intern = (void**)malloc(sizeof(void*) * numBuffersIntern);       //bias and weights
    }
    FullyConnected_Layer(Activation<T>* act_, Image_Shape output_shape)
    {
        this->layerBefore = nullptr;
        this->batch_size = 0;
        this->act = act_;

        this->output = nullptr;
        this->outputShape = output_shape;

        this->intern = (void**)malloc(sizeof(void*) * numBuffersIntern);       //bias and weights
    }

    // /+=======+\
    // ||Methods||
    // \+=======+/

    //TODO/FIXIT: MAKE WORK AND DEPENDENT ON ACTIVATION FUNCTION
    virtual void initMem() override {
        //1.: Set memory of activation
        this->act->initMem();

        //2.: Useful variables
        T* bias    = (T*)this->intern[0];
        T* weights = (T*)this->intern[1];

        //3.: Initialize "other". "state" does not initialization
        set_random<T, true>(bias   , this->outputShape.prod(), 1, LAUNCH_PARAM(this->outputShape.prod()));
        set_random<T, true>(weights, this->layerBefore->outputShape.prod(), this->outputShape.prod(), LAUNCH_PARAM(this->layerBefore->outputShape.prod() * this->outputShape.prod()));
    
        CHECK_CUDA_ERROR();
    }

    virtual std::vector<T**> forwardProp(cudaGraph_t graph, Dependencies& depPrevState, cudaStream_t captureStream, T* tmp, bool after_dataset) const override {
        //0.: Usefuls variables
        uint32_t outStateSize        = this->outputShape.prod();
        uint32_t outStateSizeBatched = outStateSize * this->batch_size;

        //Split "intern" pointer
        T* bias    = (T*)this->intern[0];
        T* weights = (T*)this->intern[1];

        //Dependencies
        Dependencies depState;

        //Node
        cudaGraphNode_t node;

        //1.: Bias
        void* biasArgs[] = {
            (void*)&this->output,
            (void*)&bias,
            (void*)&outStateSizeBatched,
            (void*)&outStateSize
        };
        cudaKernelNodeParams biasParam{
            (void*)&set_repeating<T, DIVISIBILITY::DIVISIBLE>, //Function pointer
            dim3(GRID_SIZE(outStateSizeBatched)),              //Grid dimensions
            dim3(32, 1, 1),                                    //Block dimensions
            0u,                                                //Dyn. shared-mem per block in bytes
            (void**)&biasArgs,                                 //Array of pointers to individual kernel arguments
            nullptr                                            //Pointer to kernel arguments in the "extra" format
        };
        if (outStateSizeBatched % 32)
            biasParam.func = (void*)&set_repeating<T, DIVISIBILITY::NOT_DIVISIBLE>;
        gpuErrchk(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &biasParam));
        depState.apply<true>(graph, node);
        
        //2.: Weight multiplication
        std::vector<T**> indirection_pointers;

        cudaGraph_t multGraph;
        if (after_dataset) {
            T** weights_gpu, **state_gpu;                                         //Pointer to device pointer to data
            cudaMalloc((void**)&weights_gpu, sizeof(T*));                         //Each pointer holds memory for one device pointer
            cudaMalloc((void**)&state_gpu  , sizeof(T*));                         //Each pointer holds memory for one device pointer
                                                                                  
            cudaMemcpy((void*)weights_gpu, &weights     , sizeof(T**), cudaMemcpyHostToDevice); //Device pointer point to data
            cudaMemcpy((void*)state_gpu  , &this->output, sizeof(T**), cudaMemcpyHostToDevice); //Device pointer point to data
                           
            indirection_pointers.push_back(weights_gpu);
            indirection_pointers.push_back(state_gpu);

            multGraph = getMatmulGraphIndirection<T, false, false, false>(weights_gpu, (T**)this->layerBefore->output, state_gpu, outStateSize, this->layerBefore->outputShape.prod(), this->batch_size, captureStream);
        }
        else {
            multGraph = getMatmulGraph<T, false, false, false>((T*)weights, (T*)this->layerBefore->output, (T*)this->output, outStateSize, this->layerBefore->outputShape.prod(), this->batch_size, captureStream);
        }
        gpuErrchk(cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, multGraph));
        depPrevState.apply<false>(graph, node);
        depState.apply<true>(graph, node);     //This does clash with bias setting, as that is not atomic (theoreticaly bias could reset part of result of this)

        //3.: Activation
        this->act->addActivationNode((T*)this->output, tmp, graph, depState, captureStream); //Manages dependencies and "out_node" incrementation itself

        //4.: Update parameters
        depPrevState = depState;

        //5.: Return
        return indirection_pointers;
    }

    virtual std::vector<T**> backwardProp(cudaGraph_t graph, Dependencies& depState, Optimizer<T, L>* opt, uint64_t& optimizable_index, T* deltas, Dependencies& depDeltas, cudaStream_t captureStream, T* tmp, bool after_dataset) const override {
        //0.: Usefull variables
        uint32_t outputSize = this->outputShape.prod();
        //uint32_t outStateSizeBatched = outStateSize * this->batch_size;

        //Split "intern" pointer
        T* bias    = (T*)this->intern[0];
        T* weights = (T*)this->intern[1];

        //Dependencies
        Dependencies depWeights, depBias, depPrevState;

        //Node
        cudaGraphNode_t node;


        //1.: Get deltas before activation
        this->act->addActivationDerivNode((T*)this->output, deltas, tmp, graph, depState, depDeltas, captureStream);          //Now, output contains the deltas before activation

        //2.: Compute deltas of previous layer after activation (deltas = w^T * output.)
        if (!after_dataset) {
            cudaGraph_t multGraph = getMatmulGraph<T, true, false, true>(weights, (T*)this->output, deltas, this->layerBefore->outputShape.prod(), outputSize, this->batch_size, captureStream);
            cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, multGraph);
            depWeights.apply<false>(graph, node);
            depState.apply<false>(graph, node);
            depDeltas.apply<true>(graph, node);
        }

        //3.: Apply optimizer to weights
        opt->addNodeGEMM(weights, (T*)this->output, (void*)this->layerBefore->output, outputSize, this->layerBefore->outputShape.prod(), this->batch_size, after_dataset, optimizable_index, graph, depWeights, depState, depPrevState, captureStream);

        //4.: Apply optimizer to bias
        opt->addNodeBias(bias, this->outputShape, (T*)this->output, this->outputShape, this->batch_size, optimizable_index, graph, depBias, depState, captureStream);

        //5.: Update parameters
        depState = depPrevState;

        //6.: Return
        return std::vector<T**>();
    }

    virtual CPP20_CONSTEXPR LAYER_TYPE getLayerType() const override {
        return LAYER_TYPE::FULLY_CONNECTED;
    }
};


//TODO: After dataset, Serialization, padding, alignment of pervious layers output and change its shape, better engine heuristic, layer overarching cudnn graphs, numerical notes of engines, c++ backend wrapper api
enum BIAS_MODE : uint32_t {NO = 0, TIED = 1, UNTIED = 2};
template<typename T, typename L = T>
class Convolution_Layer : public Layer<T, L> {
    //intern={kernels(|bias)}
protected:
    uint32_t  kernel_size;
    uint32_t  dilation;
    uint32_t  stride;
    BIAS_MODE bias_mode;

    virtual CPP20_CONSTEXPR uint32_t getNumBuffersIntern() const override { return 1 + (bias_mode != BIAS_MODE::NO); } //For use in both derived and base class

    virtual CPP20_CONSTEXPR void getOwnMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const override {
        //Kernels. Alignment 128 for cuDnn 
        requirements.push_back(MemoryRequirementLifetime(sizeof(T) * kernel_size * kernel_size * this->outputShape.z * this->layerBefore->outputShape.z, 128));

        //Bias. Alignment 128 for cuDnn
        switch (bias_mode) {
        case BIAS_MODE::NO:
            break;
        case BIAS_MODE::TIED:
            requirements.push_back(MemoryRequirementLifetime(sizeof(T) * this->outputShape.z, 128));      //One per channel
            break;
        case BIAS_MODE::UNTIED:
            requirements.push_back(MemoryRequirementLifetime(sizeof(T) * this->outputShape.prod(), 128)); //One per output
            break;
        default:
            assert(0 == 1);
        }

        //Output. Alignment 128 for cuDnn
        requirements.push_back(MemoryRequirementLifetime(sizeof(T) * (uint64_t)this->outputShape.prod() * (uint64_t)this->batch_size, 128, true));                           
                     

        tmp = MemoryRequirement();
              

        switch (bias_mode) {
        case BIAS_MODE::NO:
            break;
        case BIAS_MODE::TIED:
            num_optimizables += this->outputShape.z;       //One per channel
            break;
        case BIAS_MODE::UNTIED:
            num_optimizables += this->outputShape.prod();  //One per output
            break;
        }
        num_optimizables += kernel_size * kernel_size * this->outputShape.z * this->layerBefore->outputShape.z;
    }

private:
    /*
        Invokes a cudnn graph with a new variable pack
    */
    void after_dataset_callback(cudnnBackendDescriptor_t plan) {
        cudnnBackendDescriptor_t varpack;
        {
            cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varpack);
            if (cudnnBias) {
                void* dev_ptrs[4] = { ((Input_Layer<T, L>*)this->layerBefore)->output_host, this->intern[0], this->intern[1], this->output }; // device pointer
                int64_t uids[4] = { 'i', 'k', 'b', 'o3' };

                cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 4, dev_ptrs);
                cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 4, uids);
            }
            else {
                void* dev_ptrs[3] = { ((Input_Layer<T, L>*)this->layerBefore)->output_host, this->intern[0], this->output }; // device pointer
                int64_t uids[3] = { 'i', 'k', 'o3' };

                cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, dev_ptrs);
                cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
            }
            cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &cudnn_workspace);
            cudnnBackendFinalize(varpack);
        }


        cudnnBackendExecute(cudnn_handle, plan, varpack);
    }

public:
    // /+============+\
    // ||Constructors||
    // \+============+/
    //Used in "getLayerOfType":
    Convolution_Layer()
    {
        this->layerBefore = nullptr;
        this->batch_size = 0;
        this->act = nullptr;

        this->output = nullptr;
        this->outputShape = Image_Shape(0u, 0u, 0u);

        this->intern = (void**)malloc(sizeof(void*) * this->getNumBuffersIntern());
    };
    //Used by user:
    Convolution_Layer(ACTIVATION_TYPE at, uint32_t num_kernels, uint32_t kernel_size, uint32_t dilation = 1u, uint32_t stride = 1u, BIAS_MODE bias_mode = BIAS_MODE::NO):
        kernel_size(kernel_size), dilation(dilation), stride(stride), bias_mode(bias_mode)
    {
        this->layerBefore = nullptr;
        this->batch_size = 0;
        this->act = Activation<T>::getActivationOfType(at);

        this->output = nullptr;
        this->outputShape = Image_Shape(0u, 0u, num_kernels);
        
        this->intern = (void**)malloc(sizeof(void*) * this->getNumBuffersIntern());
    }
    
    virtual void setLayerBefore(Layer<T, L>* l) override {
        this->layerBefore = l;
        this->outputShape = Image_Shape(this->layerBefore->outputShape.x / stride, this->layerBefore->outputShape.y / stride, this->outputShape.z);
    }

    // /+=======+\
    // ||Methods||
    // \+=======+/

    //TODO/FIXIT: MAKE WORK AND DEPENDENT ON ACTIVATION FUNCTION
    virtual void initMem() override {
        //1.: Set memory of activation
        this->act->initMem();

        switch (bias_mode) {
        case BIAS_MODE::NO:
            break;
        case BIAS_MODE::TIED:
            set_random<T, false>((T*)this->intern[1], this->outputShape.z, 1, LAUNCH_PARAM(this->outputShape.z));
            break;
        case BIAS_MODE::UNTIED:
            set_random<T, false>((T*)this->intern[1], this->outputShape.prod(), 1, LAUNCH_PARAM(this->outputShape.prod()));
            break;
        default:
            assert(0 == 1);
        }

        uint32_t kernel_length = kernel_size * kernel_size * this->outputShape.z * this->layerBefore->outputShape.z;
        set_random<T, false>((T*)this->intern[0], kernel_length, 1, LAUNCH_PARAM(kernel_length));
        

        CHECK_CUDA_ERROR();
    }

    //Backend api is used for operand fusing
    virtual std::vector<T**> forwardProp(cudaGraph_t graph, Dependencies& depPrevState, cudaStream_t captureStream, T* tmp, bool after_dataset) const override {
        //0.: Usefuls variables
        cudnnDataType_t dtype    = cudnnTypeOf<T>();
        cudnnDataType_t compType = cudnnTypeOf<T>();
        bool cudnnBias = (bias_mode != BIAS_MODE::NO);
        bool cudnnAct  = this->act->cudnnBackendSupport;

        //0.1: Dependencies
        Dependencies depState;

        //0.2: Node
        cudaGraphNode_t node;

        //1.: Construct cuDnn graph
        //1.1: Generate Backend Tensor Descriptors
        cudnnBackendDescriptor_t inputMemDesc, outputMemDesc1, outputMemDesc2, outputMemDesc3, biasMemDesc, kernelMemDesc;
        CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &inputMemDesc  ));
        CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &biasMemDesc   ));
        CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &kernelMemDesc ));
        CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &outputMemDesc1));
        CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &outputMemDesc2));
        CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &outputMemDesc3));

        //Input
        {
            CUDNN_ERROR(cudnnBackendSetAttribute(inputMemDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));

            int64_t dimA[4] = { this->batch_size, this->layerBefore->outputShape.z, this->layerBefore->outputShape.y, this->layerBefore->outputShape.x }; //Always {n,c,h,w}
            int64_t strA[4] = STRIDES_ARRAY(dimA);
            int64_t uid = 'i';
            int64_t alignment = 4;          

            CUDNN_ERROR(cudnnBackendSetAttribute(inputMemDesc, CUDNN_ATTR_TENSOR_DIMENSIONS    , CUDNN_TYPE_INT64, 4, dimA      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(inputMemDesc, CUDNN_ATTR_TENSOR_STRIDES       , CUDNN_TYPE_INT64, 4, strA      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(inputMemDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID     , CUDNN_TYPE_INT64, 1, &uid      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(inputMemDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
            CUDNN_ERROR(cudnnBackendFinalize(inputMemDesc));
        }

        //Bias
        {
            CUDNN_ERROR(cudnnBackendSetAttribute(biasMemDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));
            int64_t uid = 'b';
            int64_t alignment = 16;
         
            switch (bias_mode) {
            case BIAS_MODE::NO:
                break;
            case BIAS_MODE::TIED:
                {
                    int64_t dimA[] = { 1, this->outputShape.z, 1, 1 };                         //Always {n,c,h,w}
                    int64_t strA[] = STRIDES_ARRAY(dimA);

                    CUDNN_ERROR(cudnnBackendSetAttribute(biasMemDesc, CUDNN_ATTR_TENSOR_DIMENSIONS    , CUDNN_TYPE_INT64, 4, dimA      ));
                    CUDNN_ERROR(cudnnBackendSetAttribute(biasMemDesc, CUDNN_ATTR_TENSOR_STRIDES       , CUDNN_TYPE_INT64, 4, strA      ));
                    CUDNN_ERROR(cudnnBackendSetAttribute(biasMemDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID     , CUDNN_TYPE_INT64, 1, &uid      ));
                    CUDNN_ERROR(cudnnBackendSetAttribute(biasMemDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
                }
                CUDNN_ERROR(cudnnBackendFinalize(biasMemDesc));
                break;
            case BIAS_MODE::UNTIED:
                {
                    int64_t dimA[4] = { 1, this->outputShape.z, this->outputShape.y, this->outputShape.x }; //Always {n,c,h,w}
                    int64_t strA[4] = STRIDES_ARRAY(dimA);

                    CUDNN_ERROR(cudnnBackendSetAttribute(biasMemDesc, CUDNN_ATTR_TENSOR_DIMENSIONS    , CUDNN_TYPE_INT64, 4, dimA      ));
                    CUDNN_ERROR(cudnnBackendSetAttribute(biasMemDesc, CUDNN_ATTR_TENSOR_STRIDES       , CUDNN_TYPE_INT64, 4, strA      ));
                    CUDNN_ERROR(cudnnBackendSetAttribute(biasMemDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID     , CUDNN_TYPE_INT64, 1, &uid      ));
                    CUDNN_ERROR(cudnnBackendSetAttribute(biasMemDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
                }
                CUDNN_ERROR(cudnnBackendFinalize(biasMemDesc));
                break;
            default:
                assert(0 == 1);
            }

        }

        //Kernel
        {
            CUDNN_ERROR(cudnnBackendSetAttribute(kernelMemDesc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype));

            int64_t dimA[4] = { this->outputShape.z, this->layerBefore->outputShape.z, kernel_size, kernel_size }; //{g,k,c,d,h,w}   (k=output channels, c=input channels)
            int64_t strA[4] = STRIDES_ARRAY(dimA);
            int64_t uid = 'k';
            int64_t alignment = 16;

            CUDNN_ERROR(cudnnBackendSetAttribute(kernelMemDesc, CUDNN_ATTR_TENSOR_DIMENSIONS    , CUDNN_TYPE_INT64, 4, dimA      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(kernelMemDesc, CUDNN_ATTR_TENSOR_STRIDES       , CUDNN_TYPE_INT64, 4, strA      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(kernelMemDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID     , CUDNN_TYPE_INT64, 1, &uid      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(kernelMemDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
            CUDNN_ERROR(cudnnBackendFinalize(kernelMemDesc));
        }

        //Output
        {
            int64_t dimA[4] = { this->batch_size, this->outputShape.z, this->outputShape.y, this->outputShape.x }; //Always {n,c,h,w}
            int64_t strA[4] = STRIDES_ARRAY(dimA);
            int64_t uid1 = 'o1';
            int64_t uid2 = 'o2';
            int64_t uid3 = 'o3';
            int64_t alignment = 16;
            bool isVirtual    = true;
            bool isNotVirtual = false;

            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc1, CUDNN_ATTR_TENSOR_DATA_TYPE     , CUDNN_TYPE_DATA_TYPE, 1, &dtype    ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc1, CUDNN_ATTR_TENSOR_IS_VIRTUAL    , CUDNN_TYPE_BOOLEAN  , 1, &isVirtual));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc1, CUDNN_ATTR_TENSOR_DIMENSIONS    , CUDNN_TYPE_INT64    , 4, dimA      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc1, CUDNN_ATTR_TENSOR_STRIDES       , CUDNN_TYPE_INT64    , 4, strA      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc1, CUDNN_ATTR_TENSOR_UNIQUE_ID     , CUDNN_TYPE_INT64    , 1, &uid1     ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc1, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64    , 1, &alignment));
            CUDNN_ERROR(cudnnBackendFinalize(outputMemDesc1));

            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc2, CUDNN_ATTR_TENSOR_DATA_TYPE     , CUDNN_TYPE_DATA_TYPE, 1, &dtype    ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc2, CUDNN_ATTR_TENSOR_IS_VIRTUAL    , CUDNN_TYPE_BOOLEAN  , 1, &isVirtual));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc2, CUDNN_ATTR_TENSOR_DIMENSIONS    , CUDNN_TYPE_INT64    , 4, dimA      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc2, CUDNN_ATTR_TENSOR_STRIDES       , CUDNN_TYPE_INT64    , 4, strA      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc2, CUDNN_ATTR_TENSOR_UNIQUE_ID     , CUDNN_TYPE_INT64    , 1, &uid2     ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc2, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64    , 1, &alignment));
            CUDNN_ERROR(cudnnBackendFinalize(outputMemDesc2));

            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc3, CUDNN_ATTR_TENSOR_DATA_TYPE     , CUDNN_TYPE_DATA_TYPE, 1, &dtype       ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc3, CUDNN_ATTR_TENSOR_IS_VIRTUAL    , CUDNN_TYPE_BOOLEAN  , 1, &isNotVirtual));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc3, CUDNN_ATTR_TENSOR_DIMENSIONS    , CUDNN_TYPE_INT64    , 4, dimA         ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc3, CUDNN_ATTR_TENSOR_STRIDES       , CUDNN_TYPE_INT64    , 4, strA         ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc3, CUDNN_ATTR_TENSOR_UNIQUE_ID     , CUDNN_TYPE_INT64    , 1, &uid3        ));
            CUDNN_ERROR(cudnnBackendSetAttribute(outputMemDesc3, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64    , 1, &alignment   ));
            CUDNN_ERROR(cudnnBackendFinalize(outputMemDesc3));
        }

        //1.2: Convolution
        cudnnBackendDescriptor_t convOpDesc;
        {
            cudnnBackendDescriptor_t convDesc;
            int64_t kernelDim = 2;
            cudnnConvolutionMode_t convMode = CUDNN_CONVOLUTION;
            int64_t dilA[] = { dilation, dilation };
            int64_t strA[] = { stride  , stride   };
            int64_t padA[] = { (dilation * kernel_size) >> 1, (dilation * kernel_size) >> 1 };

            CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &convDesc));
            CUDNN_ERROR(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE     , CUDNN_TYPE_DATA_TYPE       , 1        , &compType ));
            CUDNN_ERROR(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_CONV_MODE     , CUDNN_TYPE_CONVOLUTION_MODE, 1        , &convMode ));
            CUDNN_ERROR(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS  , CUDNN_TYPE_INT64           , 1        , &kernelDim));
            CUDNN_ERROR(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_DILATIONS     , CUDNN_TYPE_INT64           , kernelDim, dilA      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64           , kernelDim, strA      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS  , CUDNN_TYPE_INT64           , kernelDim, padA      ));
            CUDNN_ERROR(cudnnBackendSetAttribute(convDesc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS , CUDNN_TYPE_INT64           , kernelDim, padA      ));
            CUDNN_ERROR(cudnnBackendFinalize(convDesc));


            float alpha = 1.f;
            float beta  = 0.f;
            CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &convOpDesc));
            CUDNN_ERROR(cudnnBackendSetAttribute(convOpDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X        , CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &inputMemDesc ));
            CUDNN_ERROR(cudnnBackendSetAttribute(convOpDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W        , CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &kernelMemDesc));
            if (cudnnBias || cudnnAct){ //Virtual
                CUDNN_ERROR(cudnnBackendSetAttribute(convOpDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y    , CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &outputMemDesc1));
            }
            else {                      //Not virtual
                CUDNN_ERROR(cudnnBackendSetAttribute(convOpDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y    , CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &outputMemDesc3));
            }
            CUDNN_ERROR(cudnnBackendSetAttribute(convOpDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convDesc));
            CUDNN_ERROR(cudnnBackendSetAttribute(convOpDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA    , CUDNN_TYPE_FLOAT             , 1, &alpha   ));
            CUDNN_ERROR(cudnnBackendSetAttribute(convOpDesc, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA     , CUDNN_TYPE_FLOAT             , 1, &beta    ));
            CUDNN_ERROR(cudnnBackendFinalize(convOpDesc));
        }
        
        //1.3: Bias
        cudnnBackendDescriptor_t biasOpDesc;
        if (cudnnBias) {
            cudnnPointwiseMode_t biasOp = CUDNN_POINTWISE_ADD;
            float alpha1 = 1.f;
            float alpha2 = 1.f;

            CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &biasOpDesc));
            CUDNN_ERROR(cudnnBackendSetAttribute(biasOpDesc, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_POINTWISE_MODE    , 1, &biasOp        ));
            CUDNN_ERROR(cudnnBackendSetAttribute(biasOpDesc, CUDNN_ATTR_OPERATION_POINTWISE_XDESC        , CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &outputMemDesc1));
            CUDNN_ERROR(cudnnBackendSetAttribute(biasOpDesc, CUDNN_ATTR_OPERATION_POINTWISE_BDESC        , CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &biasMemDesc   ));
            if (cudnnAct) {
                CUDNN_ERROR(cudnnBackendSetAttribute(biasOpDesc, CUDNN_ATTR_OPERATION_POINTWISE_YDESC    , CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &outputMemDesc2));
            }
            else {
                CUDNN_ERROR(cudnnBackendSetAttribute(biasOpDesc, CUDNN_ATTR_OPERATION_POINTWISE_YDESC    , CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &outputMemDesc3));
            }
            CUDNN_ERROR(cudnnBackendSetAttribute(biasOpDesc, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1       , CUDNN_TYPE_FLOAT             , 1, &alpha1));
            CUDNN_ERROR(cudnnBackendSetAttribute(biasOpDesc, CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2       , CUDNN_TYPE_FLOAT             , 1, &alpha2));
            CUDNN_ERROR(cudnnBackendFinalize(biasOpDesc));
        }

        //1.4: Activation
        cudnnBackendDescriptor_t actOpDesc;
        if (cudnnAct) {
            if(cudnnBias)
                actOpDesc = this->act->getCudnnOperationDescriptor(outputMemDesc2, outputMemDesc3);
            else
                actOpDesc = this->act->getCudnnOperationDescriptor(outputMemDesc1, outputMemDesc3);
        }

        //1.5: Operation graph
        cudnnBackendDescriptor_t op_graph;
        uint32_t num_ops = 0;
        {
            cudnnBackendDescriptor_t ops[3];
            ops[num_ops++] = convOpDesc;
            if (cudnnBias)
                ops[num_ops++] = biasOpDesc;
            if (cudnnAct)
                ops[num_ops++] = actOpDesc;

            CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &op_graph));
            CUDNN_ERROR(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS   , CUDNN_TYPE_BACKEND_DESCRIPTOR, num_ops, ops          ));
            CUDNN_ERROR(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE            , 1      , &cudnn_handle));
            CUDNN_ERROR(cudnnBackendFinalize(op_graph));
        }

        //1.6: Determine engine using heuristics
        cudnnBackendDescriptor_t engineConfig;
        {
            cudnnBackendHeurMode_t heurMode = cudnnBackendHeurMode_t::CUDNN_HEUR_MODE_INSTANT;

            cudnnBackendDescriptor_t engineHeur;
            CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &engineHeur));
            CUDNN_ERROR(cudnnBackendSetAttribute(engineHeur, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph));
            CUDNN_ERROR(cudnnBackendSetAttribute(engineHeur, CUDNN_ATTR_ENGINEHEUR_MODE           , CUDNN_TYPE_HEUR_MODE         , 1, &heurMode));
            CUDNN_ERROR(cudnnBackendFinalize(engineHeur));

            int64_t ret = -1;
            CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engineConfig));
            CUDNN_ERROR(cudnnBackendGetAttribute(engineHeur, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ret, &engineConfig));
        }

        //1.7: Create operation plan
        cudnnBackendDescriptor_t plan; 
        int64_t neededWorkspaceSize;
        {
            CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
            CUDNN_ERROR(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE       , CUDNN_TYPE_HANDLE            , 1, &cudnn_handle));
            CUDNN_ERROR(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engineConfig));
            CUDNN_ERROR(cudnnBackendFinalize(plan));

            int64_t ret = -1;
            CUDNN_ERROR(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &ret, &neededWorkspaceSize));
        }
        if (neededWorkspaceSize > cudnn_workspace_size) {
            fprintf(stderr, "[ERROR] For this convolutional layer, cudnn requires a workspace size of %llu bytes while you only provided %llu bytes!\n", neededWorkspaceSize, cudnn_workspace_size);
            std::exit(-1);
        }

        //1.8: Create variable pack
        cudnnBackendDescriptor_t varpack;
        {
            CUDNN_ERROR(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varpack));
            if (cudnnBias) {
                void* dev_ptrs[4] = { this->layerBefore->output, this->intern[0], this->intern[1], this->output }; // device pointer
                int64_t uids[4] = { 'i', 'k', 'b', 'o3' };

                CUDNN_ERROR(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 4, dev_ptrs));
                CUDNN_ERROR(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS   , CUDNN_TYPE_INT64   , 4, uids    ));
            }
            else {
                void* dev_ptrs[3] = { this->layerBefore->output, this->intern[0], this->output }; // device pointer
                int64_t uids[3] = { 'i', 'k', 'o3' };

                CUDNN_ERROR(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, dev_ptrs));
                CUDNN_ERROR(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS   , CUDNN_TYPE_INT64   , 3, uids    ));
            }
            CUDNN_ERROR(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE    , CUDNN_TYPE_VOID_PTR, 1, &cudnn_workspace));
            CUDNN_ERROR(cudnnBackendFinalize(varpack));
        }

        //1.9: Execute and add to cuda graph
        if (after_dataset) {
            /*
            void* callbackArgs[] = {
                (void*)&plan
            };
            cudaHostNodeParams callbackParam{
                (void*)&Convolution_Layer<T, L>::after_dataset_callback, //Function pointer
                (void**)&callbackArgs,                                   //Array of pointers to individual function arguments
            };
            gpuErrchk(cudaGraphAddHostNode(&node, graph, nullptr, 0, &callbackParam));
            depState.apply<true>(graph, node);
            depPrevState.apply<false>(graph, node);
            */
            assert(0 == 1);
        }
        else {
            cudaGraph_t cudnnGraph;
            cudaStreamBeginCapture(captureStream, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal);
            cudnnBackendExecute(cudnn_handle, plan, varpack);
            cudaStreamEndCapture(captureStream, &cudnnGraph);
            gpuErrchk(cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, cudnnGraph));
            depState.apply<true>(graph, node);
            depPrevState.apply<false>(graph, node);
        }

        //2.: Activation when cudnnAct=false
        if (!cudnnAct)
            this->act->addActivationNode((T*)this->output, tmp, graph, depState, captureStream);


        //3.: Cleanup
        cudnnBackendDestroyDescriptor(convOpDesc);
        if(cudnnBias)
            cudnnBackendDestroyDescriptor(biasOpDesc);
        if(cudnnAct)
            cudnnBackendDestroyDescriptor(actOpDesc);
        cudnnBackendDestroyDescriptor(op_graph);
        cudnnBackendDestroyDescriptor(engineConfig);
        cudnnBackendDestroyDescriptor(varpack);
        if(!after_dataset)
            cudnnBackendDestroyDescriptor(plan);

        //4.: Update parameters
        depPrevState = depState;

        //5.: Return
        return {};
    }
    
    //No fusion possibilities, so frontend is used (for better heuristics and easier code)
    virtual std::vector<T**> backwardProp(cudaGraph_t graph, Dependencies& depState, Optimizer<T, L>* opt, uint64_t& optimizable_index, T* deltas, Dependencies& depDeltas, cudaStream_t captureStream, T* tmp, bool after_dataset) const override {
        //0.: Usefull variables
        cudnnDataType_t dtype    = cudnnTypeOf<T>();
        cudnnDataType_t compType = cudnnTypeOf<T>();
        bool cudnnBias = (bias_mode != BIAS_MODE::NO);
        bool cudnnAct  = this->act->cudnnBackendSupport;

        //Dependencies
        Dependencies depKernel, depBias, depPrevState;

        //Node
        cudaGraphNode_t node;

        //1.: Get deltas before activation
        this->act->addActivationDerivNode((T*)this->output, deltas, tmp, graph, depState, depDeltas, captureStream);      //Now, output contains the deltas before activation

        //2.: Compute deltas of previous layer after activation (deltas = backwards_convolution(output, kernel))
        if (!after_dataset) {
            float alpha = 1.f;
            float beta  = 0.f;

            cudnnFilterDescriptor_t kernelDesc;
            cudnnCreateFilterDescriptor(&kernelDesc);
            cudnnSetFilter4dDescriptor(kernelDesc, dtype, CUDNN_TENSOR_NCHW, this->outputShape.z, this->layerBefore->outputShape.z, this->kernel_size, this->kernel_size);

            cudnnTensorDescriptor_t outputDesc;        //This holds the deltas
            cudnnCreateTensorDescriptor(&outputDesc);
            cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, dtype, this->batch_size, this->outputShape.z, this->outputShape.y, this->outputShape.x);

            cudnnTensorDescriptor_t deltasDesc;        //This holds the results
            cudnnCreateTensorDescriptor(&deltasDesc);
            cudnnSetTensor4dDescriptor(deltasDesc, CUDNN_TENSOR_NCHW, dtype, this->batch_size, this->layerBefore->outputShape.z, this->layerBefore->outputShape.y, this->layerBefore->outputShape.x);

            int32_t dilA[] = { 1, dilation, dilation };
            int32_t strA[] = { 1, stride  , stride };
            int32_t padA[] = { 0, 0, 0 };
            cudnnConvolutionDescriptor_t convDesc;
            cudnnCreateConvolutionDescriptor(&convDesc);
            cudnnSetConvolutionNdDescriptor(convDesc, 3, padA, strA, dilA, CUDNN_CONVOLUTION, compType);


            int32_t retCount;
            cudnnConvolutionBwdDataAlgoPerf_t bestAlgo;
            cudnnFindConvolutionBackwardDataAlgorithmEx(cudnn_handle, kernelDesc, this->intern[0], outputDesc, this->output, convDesc, deltasDesc, deltas, 1, &retCount, &bestAlgo, cudnn_workspace, cudnn_workspace_size);
            assert(retCount == 1);
            assert(bestAlgo.memory <= cudnn_workspace_size);


            cudaGraph_t cudnnGraph;
            cudaStreamBeginCapture(captureStream, cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal);
            cudnnConvolutionBackwardData(cudnn_handle, &alpha, kernelDesc, this->intern[0], outputDesc, this->output, convDesc, bestAlgo.algo, cudnn_workspace, cudnn_workspace_size, &beta, deltasDesc, deltas);
            cudaStreamEndCapture(captureStream, &cudnnGraph);
            gpuErrchk(cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, cudnnGraph));
            depKernel.apply<false>(graph, node);
            depState.apply<false>(graph, node);
            depDeltas.apply<true>(graph, node);
        }

        //3.: Apply optimizer to kernel
        opt->addNodeConvolution((T*)this->intern[0], Image_Shape(kernel_size, kernel_size, this->outputShape.z), 
            this->dilation, this->stride,
            (T*)this->output, this->outputShape, 
            (T*)this->layerBefore->output, this->layerBefore->outputShape, 
            this->batch_size, after_dataset, optimizable_index, graph, depKernel, depState, depPrevState, captureStream);
        
        //4.: Apply optimizer to bias
        switch (bias_mode) {
        case BIAS_MODE::NO:
            break;
        case BIAS_MODE::TIED:
            opt->addNodeBias((T*)this->intern[1], Image_Shape(1u, 1u, this->outputShape.z), (T*)this->output, this->outputShape, this->batch_size, optimizable_index, graph, depBias, depState, captureStream);
            break;
        case BIAS_MODE::UNTIED:
            opt->addNodeBias((T*)this->intern[1], this->outputShape, (T*)this->output, this->outputShape, this->batch_size, optimizable_index, graph, depBias, depState, captureStream);
            break;
        }

        //5.: Update parameters
        depState = depPrevState;

        //6.: Return
        return std::vector<T**>();
    }

    virtual CPP20_CONSTEXPR LAYER_TYPE getLayerType() const override {
        return LAYER_TYPE::CNN;
    }
};



template<typename T, typename L>
static Layer<T, L>* Layer<T, L>::getLayerOfType(LAYER_TYPE lt) {
    switch (lt) {
    case LAYER_TYPE::INPT:
        return new Input_Layer<T, L>();
    case LAYER_TYPE::FULLY_CONNECTED:
        return new FullyConnected_Layer<T, L>();
    default:
        fprintf(stderr, "[ERROR] %llu is not a known layer type!", (uint64_t)lt);
        exit(-1);
    }
}
//================================================
//==================|Network|=====================
//================================================

template<typename T, typename L = T>       //T is type of data, L is type of learning rates 
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
    std::vector<Layer<T,L>*> layers;       //Holds information on layers
                                           
    uint32_t mem_align;                    //Garanties that arguments of "setMem" are always aligned up to this number of bytes.
                                           
    uint32_t batch_size;                   //Number of samples per batch
                                           
    T* batchDep_mem;                       //Has to be realloced when batch size changes.
    T* batchIndep_mem;                     //Fixed size.
    T* deltas_mem;                         //Holds the deltas for the layers (used in backpropagation). Batch dependend
    T* tmp_mem;                            //Temporary buffer, has to be realloced when batch size changes.

    void** member_pointer;                 //Pointers passed as "setMem"

    std::vector<T**> indirection_pointers; //Array to indirection pointers

    /*
        Returns the total memory requirements of all layers with all their activation functions.
    */
    inline void getLayerMemoryRequirements(std::vector<MemoryRequirementLifetime>& requirements, MemoryRequirement& tmp, uint64_t& num_optimizables) const  {
        //0.: Set initial values
        requirements.clear();
        MemoryRequirement tmp_;
        tmp = MemoryRequirement();
        num_optimizables = 0;
        
        //1.: Requirements of layers
        for (uint32_t layer_index = 0; layer_index != layers.size(); layer_index++) {
            layers[layer_index]->getMemoryRequirements(requirements, tmp_, num_optimizables);
            tmp = max(tmp, tmp_);                                       //Temporary memory is reused
        }
    }

public:
    /*
        Call order:
        1.: First constructor + allocate + initialize     or      second constructor
        2.: Get scheduler
    */

    NetworkBuilder(std::vector<Layer<T, L>*> layers, uint32_t mem_align, uint32_t batch_size) :
        layers(layers),
        mem_align(mem_align),
        batch_size(batch_size),
        batchDep_mem(nullptr),
        batchIndep_mem(nullptr),
        tmp_mem(nullptr),
        indirection_pointers()
    {
        printf("[INFO] Creating NetworkBuilder from given layers\n");

        //0.: Check arguments
        if (layers[0]->getLayerType() != LAYER_TYPE::INPT) {
            fprintf(stderr, "[ERROR] The first layer must be an input layer while you supplied a layer of type %u", (uint32_t)layers[0]->getLayerType());
            exit(-1);
        }

        //1.: Connect the layers
        connect();

        //2.: Set batch size
        for (uint32_t layer_index = 0; layer_index < layers.size(); layer_index++)
            layers[layer_index]->setBatchSize(batch_size);
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
        if(strncmp(sig, "JVCHECK", 7) == 0)
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
        uint32_t num_layers;
        fread(&num_layers, sizeof(uint32_t), 1, file);
        layers.reserve(num_layers);

        for (uint32_t layer_index = 0; layer_index != num_layers; layer_index++) {
            layers.push_back(Layer<T>::getLayerFromCompression(file));

            layers.back()->setBatchSize(batch_size);
            
            //No need to set "state","other" to a specific value
        }
        connect();

        if (layers[0]->getLayerType() != LAYER_TYPE::INPT) {
            fprintf(stderr, "[ERROR] The first layer must be an input layer while the first saved layer is of type %u", (uint32_t)layers[0]->getLayerType());
            exit(-1);
        }
                
        //6.: Allocate memory
        allocate();

        //7.: Fill memory of layers (intern)
        for (uint32_t layer_index = 1; layer_index != num_layers; layer_index++) //Exclude input layer
            layers[layer_index]->initMemFromCompression(file);
    }

    /*
        Connects the layers (set "layerBefore" pointers)
    */
    void connect() {
        //1.: First layer
        layers[0]->setLayerBefore(nullptr);

        //2.: Other layer
        for (uint32_t ind = 1; ind < layers.size(); ind++)
            layers[ind]->setLayerBefore(layers[ind - 1]);
    }

    /*
        Allocates memory for optimizer and layers (and activations). Will set internal pointers of every layer and optimizer (and activations).
    */
    void allocate(){
        //0.: Get requirements variables
        std::vector<MemoryRequirementLifetime> requirements;
        MemoryRequirement tmp;
        uint64_t num_optimizables;
        getLayerMemoryRequirements(requirements, tmp, num_optimizables);

        //1.: Combine requirements
        MemoryRequirement batch_dependend, batch_independend;
        for (uint32_t req_ind = 0; req_ind != requirements.size(); req_ind++) {
            MemoryRequirementLifetime req = requirements[req_ind];
            if (req.batchsize_dependend)
                batch_dependend += req.getMemoryRequirements();
            else
                batch_independend += req.getMemoryRequirements();
        }

        //2.: Get requirement of buffer for deltas
        uint32_t biggestOutputSize = 0;
        for (uint32_t layer_ind = 0; layer_ind != layers.size(); layer_ind++)
            biggestOutputSize = max(biggestOutputSize, layers[layer_ind]->outputShape.prod());
        MemoryRequirement delta_req = MemoryRequirement(sizeof(T) * biggestOutputSize * batch_size, max(16u, mem_align));

        //3.: Allocation
        printf("[INFO] Trying to allocate %llumb on gpu for the network... ", (batch_dependend.num_bytes + batch_independend.num_bytes + delta_req.num_bytes + tmp.num_bytes) / (1024ull * 1024ull));

        gpuErrchk(cudaMallocAligned((void**)&  batchDep_mem, batch_dependend));
        gpuErrchk(cudaMallocAligned((void**)&batchIndep_mem, batch_independend));
        gpuErrchk(cudaMallocAligned((void**)&    deltas_mem, delta_req));
        gpuErrchk(cudaMallocAligned((void**)&       tmp_mem, tmp));
        if((batchDep_mem != nullptr || batch_dependend.num_bytes == 0) && (batchIndep_mem != nullptr || batch_independend.num_bytes == 0) && (tmp_mem != nullptr || tmp.num_bytes == 0) && (deltas_mem != nullptr || delta_req.num_bytes == 0))
            printf("Success!\n");
        else {
            printf("Failure!\n");
            std::exit(-1);
        }


            //BUGP("tmp: ");
            //PRINT_VAR(tmp_mem);
            //BUGP("\t\t");
            //PRINT_VAR(tmp.num_bytes);
            //BUGP("\t\t");
            //PRINT_VAR((uint8_t*)tmp_mem + tmp.num_bytes);
            //BUGP("\n");


        //4.: Generate "setMem" pointers
        void** setMem_pointers = (void**)malloc(sizeof(void*) * requirements.size());
        uint8_t*   batchDep_mem_ = (uint8_t*)  batchDep_mem;
        uint8_t* batchIndep_mem_ = (uint8_t*)batchIndep_mem;

        for (uint32_t req_ind = 0; req_ind != requirements.size(); req_ind++) {
            MemoryRequirementLifetime req = requirements[req_ind];
            if (req.batchsize_dependend) {
                batchDep_mem_ = align_pointer_unsafe(batchDep_mem_, req.alignment);
                setMem_pointers[req_ind] = (void*)batchDep_mem_;
                batchDep_mem_ += req.num_bytes;
            }
            else {
                batchIndep_mem_ = align_pointer_unsafe(batchIndep_mem_, req.alignment);
                setMem_pointers[req_ind] = (void*)batchIndep_mem_;
                batchIndep_mem_ += req.num_bytes;
            }
        }

        //5.: Set pointers of layers (and activations)
        void** setMem_pointers_ = setMem_pointers;
        for(uint32_t ind = 0; ind != layers.size(); ind++)
            layers[ind]->setMem(setMem_pointers_);

        //6.: Cleanup
        free((void*)setMem_pointers);
    }

    /*
        Initialize the allocated memory
    */
    void initialize(){
        //1.: Initialize layers (and activations)
        for(uint32_t ind = 0; ind != layers.size(); ind++)
            layers[ind]->initMem();
    }

    /*
        Resets the batch size and handels all reallocations.

        Note: This clears the state of each layer. Execution graphs need to be rebuild
    */
    void resetBatchSize(uint32_t new_batchSize){
        //1.: Reset variables
        batch_size = new_batchSize;
        for(uint32_t ind = 0; ind != layers.size(); ind++)
            layers[ind]->setBatchSize(new_batchSize);

        //2.: Get requirements variables
        std::vector<MemoryRequirementLifetime> requirements;
        MemoryRequirement tmp;
        uint64_t num_optimizables;
        getLayerMemoryRequirements(requirements, tmp, num_optimizables);

        //3.: Combine requirements
        MemoryRequirement batch_dependend;
        for (uint32_t req_ind = 0; req_ind != requirements.size(); req_ind++) {
            MemoryRequirementLifetime req = requirements[req_ind];
            if (req.batchsize_dependend)
                batch_dependend += req.getMemoryRequirements();
        }
           

        //4.: Get requirement of buffer for deltas
        uint32_t biggestOutputSize = 0;
        for (uint32_t layer_ind = 0; layer_ind != layers.size(); layer_ind++)
            biggestOutputSize = max(biggestOutputSize, layers[layer_ind]->outputShape.prod());
        MemoryRequirement delta_req = MemoryRequirement(sizeof(T) * biggestOutputSize * batch_size, max(16u, mem_align));

        //4.: Reallocate
        cudaFree((void*)batchDep_mem);
        cudaFree((void*)deltas_mem);
        cudaFree((void*)tmp_mem);
        gpuErrchk(cudaMallocAligned(&batchDep_mem, batch_dependend));
        gpuErrchk(cudaMallocAligned(&  deltas_mem, delta_req));
        gpuErrchk(cudaMallocAligned(&     tmp_mem, tmp));
        
        //5.: Generate "setMem" pointers
        void** setMem_pointers = (void**)malloc(sizeof(void*) * requirements.size());
        uint8_t* batchDep_mem_ = (uint8_t*)batchDep_mem;
        uint8_t* batchIndep_mem_ = (uint8_t*)batchIndep_mem;

        for (uint32_t req_ind = 0; req_ind != requirements.size(); req_ind++) {
            MemoryRequirementLifetime req = requirements[req_ind];
            if (req.batchsize_dependend) {
                batchDep_mem_ = align_pointer_unsafe(batchDep_mem_, req.alignment);
                setMem_pointers[req_ind] = (void*)batchDep_mem_;
                batchDep_mem_ += req.num_bytes;
            }
            else {
                batchIndep_mem_ = align_pointer_unsafe(batchIndep_mem_, req.alignment);
                setMem_pointers[req_ind] = (void*)batchIndep_mem_;
                batchIndep_mem_ += req.num_bytes;
            }
        }

        //6.: Set pointers of layers (and activations)
        for (uint32_t ind = 0; ind != layers.size(); ind++)
            layers[ind]->setMem(setMem_pointers);

        //7.: Cleanup
        free((void*)setMem_pointers);
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
        std::vector<T**> indirection_pointers_forward = layers[1]->forwardProp(forwardGraph, depPrevState, recording_stream, tmp_mem, true);
        indirection_pointers.insert(std::end(indirection_pointers), std::begin(indirection_pointers_forward), std::end(indirection_pointers_forward)); //New indirection pointers
       
        //4.: Other layers
        for (uint32_t ind = 2; ind < layers.size(); ind++)
            layers[ind]->forwardProp(forwardGraph, depPrevState, recording_stream, tmp_mem, false);

        //5.: Return
        return forwardGraph;
    }

    /*
        Builds cuda execution graph
    */
    cudaGraph_t getBackwardsGraph(Optimizer<T, L>* opt, cudaStream_t recording_stream){
        //1.: Create graph
        cudaGraph_t backwardGraph;
        cudaGraphCreate(&backwardGraph, 0);

        //2.: Dependencies
        Dependencies depState, depDeltas;

        //3.: Other layers
        uint64_t optimizable_index = 0;
        for (uint32_t ind = layers.size() - 1; ind > 1; ind--)
            layers[ind]->backwardProp(backwardGraph, depState, opt, optimizable_index, deltas_mem, depDeltas, recording_stream, tmp_mem, false);

        //4.: Layer after dataset
        std::vector<T**> indirection_pointers_backwards = layers[1]->backwardProp(backwardGraph, depState, opt, optimizable_index, deltas_mem, depDeltas, recording_stream, tmp_mem, true);
        indirection_pointers.insert(std::end(indirection_pointers), std::begin(indirection_pointers_backwards), std::end(indirection_pointers_backwards)); //New indirection pointers

        //5.: Return
        return backwardGraph;
    }

    //Utility
    void getFirstAndLastLayer(Layer<T, L>*& firstLayer, Layer<T, L>*& lastLayer) {
        firstLayer = layers.front();
        lastLayer  = layers.back();
    }
    void* getDeltaMem() {
        return deltas_mem;
    }
    uint64_t getNumOptimizables() {
        std::vector<MemoryRequirementLifetime> requirements;
        MemoryRequirement tmp;
        uint64_t num_optimizables;
        getLayerMemoryRequirements(requirements, tmp, num_optimizables);

        return num_optimizables;
    }
    uint32_t getBatchSize() {
        return batch_size;
    }

    /*
        Safes a checkpoint file.
    */
    void compress(const char* out_file) {
        BUGP("Save triggered: ");
        printf("%s", out_file);
        BUGP("\n");
        //1.: Open output file
        FILE* file = fopen(out_file, "wb");

        //2.: Write header
        char     signature[] = "JVCHECK";
        uint16_t version = AI_VERSION;
        uint32_t type = type_hash<T>();
        fwrite(+signature, sizeof(signature) - 1, 1, file); //Signature
        fwrite(&version  , sizeof(version  )    , 1, file); //Library version
        fwrite(&type     , sizeof(type     )    , 1, file); //The underlying type

        //3.: Information on layers
        uint32_t num_layers = layers.size();
        fwrite(&num_layers, sizeof(uint32_t), 1, file);
        for (uint32_t layer_index = 0; layer_index != num_layers; layer_index++)
            layers[layer_index]->template compress<false>(file);

        //4.: Memory of layers
        for (uint32_t layer_index = 1; layer_index != num_layers; layer_index++) //Exclude input layer
            layers[layer_index]->template compress<true>(file);

        //5.: Close file
        fclose(file);
    }
};

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

#if 0
int main()
{
    cudaStream_t str;
    cudaStreamCreate(&str);

    cublasSetup(8ull * 1024ull * 1024ull, str); //8Mb workspace (default according to debugging with nsight compute, even though documentation says 4mb)

    Random::init_rand();

    //=======================================
    using T = float;
    using L = T;
    constexpr uint32_t N = 4; //Size of layers
    constexpr uint32_t B = 2;  //Batch size
    T alpha = (T)0.01 / (T)B;

    std::vector<Layer<T>*> layers;
    layers.push_back(new Input_Layer<T>(N));
    //layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::TANH, N));
    layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::SOFTMAX, N));
    //layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::RELU, N));

    NetworkBuilder<T, L> network(layers, 16, B);
    network.allocate();
    network.initialize();

    Optimizer<T>* opt = new Debug_Optimizer<T>(B);
    opt->setNumOptimizables(network.getNumOptimizables());
    opt->allocate();
    opt->initMem();
    opt->setLR(&alpha, nullptr, nullptr, str);

    Loss<T>* loss = new MSE_Loss<T>();
    loss->setParameters((T*)layers.back()->output, layers.back()->outputShape, B, (T*)network.getDeltaMem());

    //======
    cudaGraph_t forward  = network.getForwardGraph  (str);
    cudaGraph_t backward = network.getBackwardsGraph(opt, str);
    cudaGraph_t delta    = loss  ->getDeltaGraph    (str);
    
    cudaGraphExec_t forw, backw, delt;
    char errorBuf[512];
    cudaGraphNode_t errNode;

    cudaGraphInstantiate(&forw, forward, &errNode, +errorBuf, 512);
    if (errorBuf[0]) {
        fprintf(stderr, "[ERROR] The following error arose during the instantiation of the forward graph: %s", +errorBuf);
        exit(-1);
    }
    cudaGraphInstantiate(&backw, backward, &errNode, +errorBuf, 512);
    if (errorBuf[0]) {
        fprintf(stderr, "[ERROR] The following error arose during the instantiation of the backward graph: %s", +errorBuf);
        exit(-1);
    }
    cudaGraphInstantiate(&delt, delta, &errNode, +errorBuf, 512);
    if (errorBuf[0]) {
        fprintf(stderr, "[ERROR] The following error arose during the instantiation of the delta graph: %s", +errorBuf);
        exit(-1);
    }
    //========

    T* in_data, *out_data;
    cudaMalloc(&in_data , sizeof(T) * N * B);
    cudaMalloc(&out_data, sizeof(T) * N * B);
    
    T*  in_data_host = (T*)malloc(sizeof(T) * (N * max(N, B)));
    T* out_data_host = (T*)malloc(sizeof(T) * N * B);
    
    ((Input_Layer<T>*)layers[0])->setInputPointer(&in_data, str);
    loss->setTarget(&out_data, str);

    T* gradients = ((Debug_Optimizer<T>*)opt)->getOptBuf();
    T* deltas_mem = (T*)network.getDeltaMem();
    opt->initMem();
    cudaDeviceSynchronize();
    uint64_t num_opt = network.getNumOptimizables();

    //=======================================================================================================
    if(0)
    {
        BUGP("Bug checking:");
        for (uint32_t b = 1; b <= B; b++) {
            for (uint32_t ind = 0; ind != N; ind++) {
                in_data_host[(b - 1) * N + ind] = 1;
            }
        }
        //network.initialize();
        cudaMemcpy(in_data, in_data_host, sizeof(T) * N * B, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaGraphLaunch(forw, str);

        cudaDeviceSynchronize();
        BUGP("In-Data:\n");
        gpuErrchk(cudaMemcpy(in_data_host, in_data, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        ARR_PRINT(in_data_host, N, B);

        for (uint32_t l = 1; l < layers.size(); l++) {
            printf("\n\Bias Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->intern[0], sizeof(T) * N, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT_COLMAJ(in_data_host, N, 1);

            printf("\n\nWeights Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->intern[1], sizeof(T) * N * N, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT_COLMAJ(in_data_host, N, N);

            printf("\n\nState Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->output, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT(in_data_host, N, B);
        }

        cudaGraphLaunch(delt, str);
        cudaDeviceSynchronize();

        printf("\n\Loss Deltas:\n");
        gpuErrchk(cudaMemcpy(in_data_host, deltas_mem, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        ARR_PRINT(in_data_host, N, B);

        cudaGraphLaunch(backw, str);
        cudaDeviceSynchronize();

        for (uint32_t l = 1; l < layers.size(); l++) {
            printf("\n\Delta Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->output, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT(in_data_host, N, B);
        }

        //printf("\n\Gradients\n");
        //gpuErrchk(cudaMemcpy(in_data_host, gradients, sizeof(T)* num_opt* B, cudaMemcpyDeviceToHost));
        //cudaDeviceSynchronize();
        //ARR_PRINT_COLMAJ(in_data_host + 0, 4, 4);  //Weigts layer 2
        //ARR_PRINT_COLMAJ(in_data_host + 16, 4, 4); //Weigts layer 2
        //ARR_PRINT(in_data_host + 32, 4, 1);        //Bias   layer 2
        //ARR_PRINT(in_data_host + 36, 4, 1);        //Bias   layer 2
        //ARR_PRINT_COLMAJ(in_data_host + 40, 4, 4); //Weigts layer 1
        //ARR_PRINT_COLMAJ(in_data_host + 56, 4, 4); //Weigts layer 1
        //ARR_PRINT(in_data_host + 72, 4, 1);        //Bias   layer 1
        //ARR_PRINT(in_data_host + 76, 4, 1);        //Bias   layer 1
        //opt->initMem();
        //cudaDeviceSynchronize();




        for (uint32_t b = 1; b <= B; b++) {
            for (uint32_t ind = 0; ind != N; ind++) {
                in_data_host[(b - 1) * N + ind] = b;
            }
        }
        //network.initialize();
        cudaMemcpy(in_data, in_data_host, sizeof(T) * N * B, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        cudaGraphLaunch(forw, str);

        cudaDeviceSynchronize();
        BUGP("In-Data:\n");
        gpuErrchk(cudaMemcpy(in_data_host, in_data, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        ARR_PRINT(in_data_host, N, B);

        for (uint32_t l = 1; l < layers.size(); l++) {
            printf("\n\Bias Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->intern[0], sizeof(T) * N, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT_COLMAJ(in_data_host, N, 1);

            printf("\n\nWeights Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->intern[1], sizeof(T) * N * N, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT_COLMAJ(in_data_host, N, N);

            printf("\n\nState Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->output, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT(in_data_host, N, B);
        }

        cudaGraphLaunch(delt, str);
        cudaDeviceSynchronize();

        printf("\n\Loss Deltas:\n");
        gpuErrchk(cudaMemcpy(in_data_host, deltas_mem, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        ARR_PRINT(in_data_host, N, B);

        cudaGraphLaunch(backw, str);
        cudaDeviceSynchronize();

        for (uint32_t l = 1; l < layers.size(); l++) {
            printf("\n\Delta Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->output, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT(in_data_host, N, B);
        }

        //printf("\n\Gradients\n");
        //gpuErrchk(cudaMemcpy(in_data_host, gradients, sizeof(T)* num_opt* B, cudaMemcpyDeviceToHost));
        //cudaDeviceSynchronize();
        //ARR_PRINT_COLMAJ(in_data_host + 0, 4, 4);  //Weigts layer 2
        //ARR_PRINT_COLMAJ(in_data_host + 16, 4, 4); //Weigts layer 2
        //ARR_PRINT(in_data_host + 32, 4, 1);        //Bias   layer 2
        //ARR_PRINT(in_data_host + 36, 4, 1);        //Bias   layer 2
        //ARR_PRINT_COLMAJ(in_data_host + 40, 4, 4); //Weigts layer 1
        //ARR_PRINT_COLMAJ(in_data_host + 56, 4, 4); //Weigts layer 1
        //ARR_PRINT(in_data_host + 72, 4, 1);        //Bias   layer 1
        //ARR_PRINT(in_data_host + 76, 4, 1);        //Bias   layer 1
        //opt->initMem();
        //cudaDeviceSynchronize();



        BUGP("Done\n\n");
    }
    //=======================================================================================================




    for (uint32_t b = 1; b <= B; b++) {
        for (uint32_t ind = 0; ind != N; ind++) {
             in_data_host[(b - 1) * N + ind] = b;
            out_data_host[(b - 1) * N + ind] = ((ind+b)%N==0);
        }
    }

    cudaMemcpy( in_data,  in_data_host, sizeof(T) * N * B, cudaMemcpyHostToDevice);
    cudaMemcpy(out_data, out_data_host, sizeof(T) * N * B, cudaMemcpyHostToDevice);

    BUGP("Training... ");
    for (uint32_t i = 0; i < /*5*/000; i++) {
        cudaGraphLaunch(forw , str);
        cudaGraphLaunch(delt , str);
        cudaGraphLaunch(backw, str);
    }
    BUGP("Done\n");

    cudaDeviceSynchronize();
    

    while(getchar()!='e'){
        clear_console();
        cudaGraphLaunch(forw, str);

        //===========================
        //Printing
        cudaDeviceSynchronize();
        BUGP("In-Data:\n");
        gpuErrchk(cudaMemcpy(in_data_host, in_data, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        ARR_PRINT(in_data_host, N, B);
        
        for (uint32_t l = 1; l < layers.size(); l++) {
            printf("\n\Bias Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->intern[0], sizeof(T) * N, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT_COLMAJ(in_data_host, N, 1);
            
            printf("\n\nWeights Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->intern[1], sizeof(T) * N * N, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT_COLMAJ(in_data_host, N, N);
        
            printf("\n\nState Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->output, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT(in_data_host, N, B);
        }
        //=======================

        cudaGraphLaunch(delt, str);
        cudaDeviceSynchronize();

        
            printf("\n\Loss Deltas:\n");
            gpuErrchk(cudaMemcpy(in_data_host, deltas_mem, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT(in_data_host, N, B);


        cudaGraphLaunch(backw, str);
        cudaDeviceSynchronize();




        //======================================
        for (uint32_t l = 1; l < layers.size(); l++) {
            printf("\n\Delta Layer %u:\n", l);
            gpuErrchk(cudaMemcpy(in_data_host, layers[l]->output, sizeof(T) * N * B, cudaMemcpyDeviceToHost));
            cudaDeviceSynchronize();
            ARR_PRINT(in_data_host, N, B);
        }
        //
        //printf("\n\Gradients\n");
        //gpuErrchk(cudaMemcpy(in_data_host, gradients, sizeof(T)* num_opt* B, cudaMemcpyDeviceToHost));
        //cudaDeviceSynchronize();
        //ARR_PRINT_COLMAJ(in_data_host + 0, 4, 4);  //Weigts layer 2
        //ARR_PRINT_COLMAJ(in_data_host + 16, 4, 4); //Weigts layer 2
        //ARR_PRINT(in_data_host + 32, 4, 1);        //Bias   layer 2
        //ARR_PRINT(in_data_host + 36, 4, 1);        //Bias   layer 2
        //ARR_PRINT_COLMAJ(in_data_host + 40, 4, 4); //Weigts layer 1
        //ARR_PRINT_COLMAJ(in_data_host + 56, 4, 4); //Weigts layer 1
        //ARR_PRINT(in_data_host + 72, 4, 1);        //Bias   layer 1
        //ARR_PRINT(in_data_host + 76, 4, 1);        //Bias   layer 1
        //opt->initMem();
        //cudaDeviceSynchronize();
        //======================================



        CHECK_CUDA_ERROR();
    }
        

    BUGP("\n\nDone");

    CHECK_CUDA_ERROR();

    CUBLAS_ERROR(cublasDestroy(cublas_handle));

    return 0;
}
#endif

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

//Windows: clang++ Network.cpp -o Network.exe -I"D:\Librarys\CImg-2.9.2_pre070420" -I"D:\Librarys\VS-NuGet\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include" -I"D:\Librarys\GLFW\include" -I"D:\Librarys\glew-2.1.0\include" -I"D:\Librarys\freetype-2.10.3\include" -L"D:\Librarys\GLFW\lib" -L"D:\Librarys\glew-2.1.0\lib\Release\x64" -L"D:\Librarys\VS-NuGet\lib" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64" -L"D:\Librarys\freetype-2.10.3\objs" -O0 -march=native -m64 -std=c++2a -Wall -lzlib -llibpng16 -ljpeg -lkernel32 -luser32 -lgdi32 -lopengl32 -lglu32 -lglew32 -lglfw3dll -lpsapi -lwinspool -lcomdlg32 -ladvapi32 -lshell32 -lole32 -loleaut32 -luuid -lodbc32 -lodbccp32 -lcudart_static -lcublas -lfreetype -g -DDEBUG