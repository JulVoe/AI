#ifdef __NVCC__
#pragma warning( disable : 4514)
#pragma warning( disable : 4711)
#pragma warning( disable : 4710)
#pragma warning( disable : 5039)
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wwritable-strings"
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
//#include <mma.h>
//#include <cublasXt.h>

#include <inttypes.h>
#include <chrono>
#include <stdio.h>

#include "util.cpp"

/*
Notation conventions:
 - A matrix has size height*width.
 - layer[0] is the input layer
 - weight[i] are the weights between layer[i] and layer[i+1]
*/

#define LAUNCH_PARAM(N) (int)(1. / ((10. / ((double)(1 << 13)) + 32. / ((double)(N))))), 32
#define T float

cublasHandle_t cublas_handle;
class NT{
public:
    NT() = default;

    T* cublasConst;

    /* 
        Computes either the matrix multiplication C=trans_A(A)*trans_B(B) or C+=trans_A(A)*trans_B(B).
        All matrices (A,B,C) have to be stored column major.

        @param A: Left  factor of matrix product that will be computed.
        @param B: Right factor of matrix product that will be computed
        @param C: Where to store the result of the multiplication
        @param trans_A: Whether to transpose A before multiplication (swap height and width)
        @param trans_B: Whether to transpose B before multiplication (swap height and width)
        @param overwrite: If this is true, we overwrite C with the result of the matrix multiplication. If it is false, we add the result to the data in C
        @param y1, x1, x2: trans_A(A) has size y1*x1. trans_B(B) has size x1*x2. C has size y1*x2.
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
};


__global__ void kern(uint32_t val, uint32_t* gMem){
    gMem[threadIdx.x] = val;
}

//All arguments are read when node is added

int main()
{
    CUBLAS_ERROR(cublasCreate(&cublas_handle));
    //Logging
    cublasSetAtomicsMode(cublas_handle, CUBLAS_ATOMICS_ALLOWED);
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    //Block dimension

    //Set cuda stream
    //set vector, matrixes (async), pointer mode
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);

    //Initialize random
    Random::init_rand();

    //Maybe have to change indexing mode to start with 0 instead of 1 using #define IDX2C(i,j,ld) (((j)*(ld))+(i))

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    cudaGraphNode_t node;
    uint32_t arg = 4;
    uint32_t* gMem;
    cudaMalloc(&gMem, sizeof(uint32_t) * 32);        
    
    void* nodeArgs[] = {
        (void*)& arg,
        (void*)& gMem
    };
    cudaKernelNodeParams nodeParam {
        (void*)kern,                          //Function pointer
        dim3(1, 1, 1),                        //Grid dimensions
        dim3(32, 1, 1),                       //Block dimensions
        0u,                                   //Dyn. shared-mem per block in bytes */
        (void**)&nodeArgs,                    //Array of pointers to individual kernel arguments
        nullptr                               //Pointer to kernel arguments in the "extra" format
    };
    cudaGraphAddKernelNode(&node, graph, nullptr, 0, &nodeParam);

//==============================================
    uint32_t arg2 = 7;
    void* nodeArgs2[] = {
        (void*)& arg2,
        (void*)& gMem
    };
    //memcpy(nodeArgs, nodeArgs2, sizeof(void*) * 2);
    nodeParam.kernelParams = (void**)&nodeArgs2;
    memset((void*)&node, 0, sizeof(node));

//==============================================
    char buf[256];
    cudaGraphNode_t errNode;
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, &errNode, +buf, 256);

    printf("%s\n", +buf);
    cudaGraphLaunch(graphExec, nullptr);
    cudaMemcpy(+buf, gMem, sizeof(uint32_t) * 32, cudaMemcpyDefault);
    ARR_PRINT(((uint32_t*)buf), 32, 1);

    return 0;
        //2.: Weight multiplication        
        //cudaGraph_t multGraph = getMatmulNode<T, false, false, false>(other_weights, layerBefore->state, state, outStateSize, layerBefore->outStateSize, batch_size, captureStream);
        //cudaGraphAddChildGraphNode(output+1, graph, depNode, 1, multGraph);
















#if 0

    NT nt = NT();

    T *A, *B, *C;
    T *A_,*B_,*C_;

#define X1 5
#define Y1 3
#define X2 4
    A_ = (T*)malloc(sizeof(T) * Y1 * X1);
    B_ = (T*)malloc(sizeof(T) * X1 * X2);
    C_ = (T*)malloc(sizeof(T) * Y1 * X2);

    cudaMalloc(&A, sizeof(T) * Y1 * X1);
    cudaMalloc(&B, sizeof(T) * X1 * X2);
    cudaMalloc(&C, sizeof(T) * Y1 * X2);
    cudaMalloc(&(nt.cublasConst), sizeof(T) * 2);

    for(uint32_t i = 0; i != Y1*X1; i++)
        A_[i] = Random::rand_float(128.f);
    for(uint32_t i = 0; i != X1*X2; i++)
        B_[i] = Random::rand_float(128.f);
    T Const_[2] = {(T)0, (T)1};

    gpuErrchk(cudaMemcpy(A, A_, sizeof(T) * Y1 * X1, cudaMemcpyDefault));
    gpuErrchk(cudaMemcpy(B, B_, sizeof(T) * X1 * X2, cudaMemcpyDefault));
    gpuErrchk(cudaMemcpy(nt.cublasConst, +Const_, sizeof(T) * 2, cudaMemcpyDefault));

    
    ARR_PRINT_COLMAJ(A_, X1, Y1);
    BUGP("\n\n");
    ARR_PRINT_COLMAJ(B_, X2, X1);


    CHECK_CUDA_ERROR();
    
    nt.matmul<false, false, true>(A, B, C, Y1, X1, X2);

    cudaMemcpy(C_, C, sizeof(T) * Y1 * X2, cudaMemcpyDefault);
    BUGP("\n\n");
    ARR_PRINT_COLMAJ(C_, X2, Y1);
#endif



    CUBLAS_ERROR(cublasDestroy(cublas_handle));

    return 0;
}

//sudo clang++ testing.cpp -o testing.exe -I"/home/julian/Libs/CImg-2.9.2_pre070420" -I"/usr/local/cuda-10.0/include" -I"/home/julian/Libs/glfw-3.3.2/include" -I"/home/julian/Libs/glew-2.1.0/include" -I"/home/julian/Libs/freetype-2.10.3/include" -L"/usr/local/cuda-10.0/lib64" -O0 -march=native -m64 -std=c++17 -Wall -ldl -lrt -lpthread -lz -lpng -ljpeg -lGL -lGLU -lGLEW -lglfw3 -lfreetype -lX11 -lcudart_static -lcublas -lstdc++fs -g -DDEBUG -DEXPERIMENTAL_FILESYSTEM