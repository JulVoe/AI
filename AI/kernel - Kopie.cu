#define THE_VERSION_JULIAN_DID_NOT_SCREW_WITH
#ifdef __NVCC__
#pragma warning( disable : 4514)
#pragma warning( disable : 4711)
#pragma warning( disable : 4710)
#pragma warning( disable : 5039)
#endif
#include "stddef.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
//#include <mma.h>
//#include <cublasXt.h>

#include <algorithm>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <typeinfo>
#include <inttypes.h>

#include "omp.h"

namespace cg = cooperative_groups;

template<typename T> T min(T a, T b) { if (a < b) return a; else return b; }

//TODO: COPY WEIGHTS, BIAS, TRAIN TILE AND VALIDATIONS SAMPLES TO GPU

#define CUBLAS_ERROR(e); \
    if((e)!=CUBLAS_STATUS_SUCCESS){\
        printf("%d %d", __LINE__, e);\
    }

#define CHECK_CUDA_ERROR();\
    do{\
        auto error = cudaGetLastError(); \
        if (error != cudaSuccess) {\
            /* print the CUDA error message and exit*/\
            printf("CUDA error: %s\n", cudaGetErrorString(error)); \
        }\
    } while (0);

cublasHandle_t cublas_handle;

enum Activation {RELU=0, SIGMOID=1, SOFTMAX=2, SOFTPLUS=3};

template<typename T>
__device__ T exponential(T in) {
    __builtin_unreachable();
}
template<> __device__ float exponential<float>(float in) {
    return expf(in);
}
template<> __device__ double exponential<double>(double in) {
    return exp(in);
}
template<> __device__ half exponential<half>(half in) {
    return hexp(in);
}

template<typename T>
__device__ T logarithm(T in) {
    __builtin_unreachable();
}
template<> __device__ float logarithm<float>(float in) {
    return logf(in);
}
template<> __device__ double logarithm<double>(double in) {
    return log(in);
}
template<> __device__ half logarithm<half>(half in) {
    return hlog(in);
}

//=================================================================================================================

//Sizes are controlled by lunch parameters: mat=grid*block, vec=grid*1
template<typename T>
__global__ void add_vec_mat(T* mat, T* vec) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ind_bias = blockIdx.x;

    mat[idx] += vec[ind_bias];
}

template<typename T, bool nIsPow2>
__global__ void reduceKernel(T* g_idata, T* g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    
    T mySum = 0;
    
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    //printf("%d %f\n", blockIdx.x * blockDim.x + threadIdx.x, g_idata[blockIdx.x * blockDim.x + threadIdx.x]);
    while (i < n)
    {
        //T p1 = exponential<T>(g_idata[i]);
        //mySum += p1;
        //g_idata[i] = p1;
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockDim.x < n) {
            //T p2 = exponential<T>(g_idata[i + blockDim.x]);
            //mySum += p2;
            //g_idata[i + blockDim.x] = p2;
            mySum += g_idata[i + blockDim.x];
        }
    
        i += gridSize;
    }
    //printf("Tid: %d. Sum: %d.\n", blockIdx.x * blockDim.x + threadIdx.x, mySum);
    
    // each thread puts its local sum into shared memory
    cg::sync(cta);
    
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    
    //Reduce all warps of block using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
    {
        mySum += tile32.shfl_down(mySum, offset);
    }
    
    // write result for warp to global mem
    if ((tid & 31) == 0) atomicAdd(g_odata, mySum);
}

template<class T>
struct SharedMemory
{
    __device__ inline operator T* ()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T* () const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator double* ()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double* () const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T* g_idata, T* g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    T* sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i + blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    cg::sync(cta);

#if 0
    if ((blockSize >= 128) && (tid < 64))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 64];
    }

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >= 64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
        {
            mySum += tile32.shfl_down(mySum, offset);
        }
    }

    // write result for this block to global mem
    if (cta.thread_rank() == 0) atomicAdd(g_odata, mySum);
#else
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    //Reduce all warps of block using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
    {
        mySum += tile32.shfl_down(mySum, offset);
    }

    // write result for warp to global mem
    if ((tid & 31) == 0) atomicAdd(g_odata, mySum);
#endif
}

#define smem(BL_SIZE, T) ((BL_SIZE) <= 32) ? 2 * (BL_SIZE) * sizeof(T) : (BL_SIZE) * sizeof(T)
template<typename T>
void reduce(T* dat, uint32_t lenght, T* buf) {
    bool nIsPow2 = lenght & (lenght - 1);
    nIsPow2 = !nIsPow2;

    int smemSize = smem(BL_SIZE, T);

    //reduceKernel<T, BL_SIZE, nIsPow2><<<(lenght+ BL_SIZE-1)/ BL_SIZE, BL_SIZE, smemSize>>>(dat, buf, lenght);
}
//==================================================================================================================
template<typename T>
__global__ void relu(T* in, uint32_t n) {//Blocksize 256
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        T v = in[idx];
        in[idx] = (v > (T)0) ? v : (T)0;
    }
}

template<typename T>
__global__ void relu_deriv(T* in, uint32_t n) {//Blocksize 256
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        T v = in[idx];
        in[idx] = (v > (T)0) ? (T)1 : (T)0;
    }
}

template<typename T>
__global__ void sigmoid(T* in, uint32_t n) { //Blocksize 64/384
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        T v = in[idx];
        in[idx] = 1.0 / (1.0 + exponential<T>(v));
    }
}

template<typename T>
__global__ void sigmoid_deriv(T* in, uint32_t n) { //Blocksize 256
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        T v = in[idx];
        in[idx] = v * (1 - v);
    }
}

template<typename T>
__global__ void softplus(T* in, uint32_t n) { //Blocksize 256 / 128
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        T v = in[idx];
        in[idx] = logarithm<T>(1.0 + exponential<T>(v));
    }
}

template<typename T>
__global__ void softplus_deriv(T* in, uint32_t n) {//Blocksize 64 / 128
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        T v = in[idx];
        T w = exponential<T>(v);
        in[idx] = (v - 1) / v;
    }
}

template<typename T>
__global__ T softmax_helper1(T* in, uint32_t n, T* acc) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        in[idx] = exponential(in[idx]);
    }
}

template<typename T>
__global__ T softmax_helper2(T* in, uint32_t n, T acc) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        in[idx] /= acc;
    }
}

template<typename T>
void softmax(T* in, uint32_t n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    //__shared__ T sum = 0;

    if (idx < n) {
        softmax_helper(in, n);

    }
}

template<typename T, Activation ACT> //Activation for hidden layer, outputlayer use softmax by default
class MultiLayerPerceptron {
private:
    uint32_t num_layer;    //Number of layers
    uint32_t num_neurons;  //Number of neurons excluding the input layer
    uint32_t* layer_size;  //Number of neurons per layer
    float learning_factor; //Learning factor
    uint32_t batch_size;   //Number of samples per batch

    T** weights;  //Pointer to 2D-Array                                               | weights[i] has dimension layer_size[i+1] x layer_size[i]
    T** bias;     //Pointer to Array                                                  | bias[i]    has dimension layer_size[i+1] x 1 
    T** output;   //Pointer to Array, after activation                                | output[i]  has dimension layer_size[i+1] x batch_size
    T*  error[2]; //Memory is resused, only store error of current and last layer.    |

    int fd_in, fd_out;             //File descritors of files holding dataset
    T* train_input, train_output;  //Part of dataset for training
    uint32_t train_samples;        //Number of samples in this part
    T* val_input, val_output;      //Part of dataset for validation
    uint32_t validation_samples;   //Number of samples in this part
    uint32_t tile_size = 0;        //Number of training samples loaded into ram at once
    
    uint32_t* cur_in ;             //Pointer to current input data
    uint32_t* cur_out;             //Pointer to current output data

    /*  Multiplies B with weight matrix and stores in output.
        Because cuBLAS interprets the Matrixes as if they were column major, we simply switch order of multiplication
    */
          
    template<typename TYPE>  void weight_mul        (uint32_t ind, TYPE*   B, TYPE*   alpha, TYPE*   beta) {
        static_assert(typeid(TYPE) != typeid(TYPE), "The type passed to MultiLayerPerceptron is unsupported");
    }
    template<>               void weight_mul<float >(uint32_t ind, float*  B, float*  alpha, float*  beta) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, layer_size[ind+1], batch_size, layer_size[ind], alpha, weights[ind], layer_size[ind+1], B, layer_size[ind], beta, output[ind], layer_size[ind+1]);
    }
    template<>               void weight_mul<double>(uint32_t ind, double* B, double* alpha, double* beta) {
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, layer_size[ind + 1], batch_size, layer_size[ind], alpha, weights[ind], layer_size[ind + 1], B, layer_size[ind], beta, output[ind], layer_size[ind + 1]);
    }
    template<>               void weight_mul<half>  (uint32_t ind, half*   B, half*   alpha, half*   beta) {
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, layer_size[ind + 1], batch_size, layer_size[ind], alpha, weights[ind], CUDA_R_16F, layer_size[ind + 1], B, CUDA_R_16F, layer_size[ind], beta, output[ind], CUDA_R_16F, layer_size[ind + 1], CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    void add_bias(uint32_t ind) {
        add_vec_mat<<<layer_size[ind + 1], batch_size>>>(output[ind], bias[ind]);
        cudaDeviceSynchronize();
    }
    template<Activation a> void activate                      (uint32_t i) {
        static_assert(a != a, "Unknown activation");
    }
    template<>             void activate<Activation::RELU>    (uint32_t i) {
        uint32_t s = layer_size[i + 1];
        relu<<<(s + 255) / 256, 256>>>(bias[i], s);
        cudaDeviceSynchronize();
    }
    template<>             void activate<Activation::SIGMOID> (uint32_t i) {
        uint32_t s = layer_size[i + 1];
        sigmoid<<<(s + 63) / 64, 64>>>(bias[i], s);
        cudaDeviceSynchronize();
    }
    template<>             void activate<Activation::SOFTPLUS>(uint32_t i) {
        uint32_t s = layer_size[i + 1];
        softplus<<<(s + 255) / 256, 256>>>(bias[i], s);
        cudaDeviceSynchronize();
    }

public:
    MultiLayerPerceptron(uint32_t n, uint32_t* l)
        : num_layer(n), layer_size(l)
    {
        //Analyse input
        uint64_t num_weights = 0;
                 num_neurons = 0;
        uint32_t max_neurons = *std::max_element(layer_size, layer_size + num_layer);
        for (uint32_t layer = 1; layer != num_layer; layer++) {
            num_weights += layer_size[layer - 1] * layer_size[layer];
            num_neurons += layer_size[layer];
        }

        //Allocate memory
        uint64_t alloc_weights = num_weights;
        uint64_t alloc_bias = num_neurons;
        uint64_t alloc_error = max_neurons * 2;
        T* raw_mem = (T*)malloc(sizeof(T) * (alloc_weights + alloc_bias + alloc_error));

        T* weights_mem = raw_mem;
        raw_mem += alloc_weights;
        weights = (T**)malloc(sizeof(T*) * (num_layer - 1)); //First layer needs no weights
        weights[0] = weights_mem;
        for (uint32_t layer = 1; layer != num_layer - 1; layer++) {
            weights[layer] = weights[layer - 1] + layer_size[layer - 1] * layer_size[layer];
        }

        T* bias_mem = raw_mem;
        raw_mem += alloc_bias;
        bias = (T**)malloc(sizeof(T*) * (num_layer-1)); //First layer needs no bias
        bias[0] = bias_mem;
        for (uint32_t layer = 1; layer != num_layer - 1; layer++) {
            bias[layer] = bias[layer - 1] + layer_size[layer];
        }

        T* error_mem = raw_mem;
        error[0] = error_mem;
        error[1] = error_mem + max_neurons;


        output = (T**)calloc(sizeof(T*) * (num_layer- 1)); //First layer needs no output

        //Initialize memory
#ifdef SMART_WEIGHTS
        for (uint32_t layer = 0; layer != num_layer - 1; layer++) {
            uint32_t layer_size = layer_size[layer] * layer_size[layer + 1];
            double scale = sqrt(2.0 / ((double)layer_size));
            for (uint32_t ind = 0; ind != layer_size; ind++) {
                weights[layer][ind] = random(-scale, scale);
            }
        }
#else
        for (uint32_t i = 0; i != alloc_weights; i++) {
            weights_mem[i] = random(-1.0, 1.0);
        }
#endif

        for (uint32_t i = 0; i != alloc_bias; i++) {
            bias_mem[i] = random(0.0, 1.0);
        }
    }
    MultiLayerPerceptron(char* file) {
        load_parameters(file);
    }
    bool load_parameters(char* file) {
        //File-stuff
        int fd = open(file, O_RDONLY);
        
        std::type_info* type;                    //Actual value does not matter, only needed because std::type_info has no constructor
        //char type_[sizeof(std::type_info)];
        read(fd, type, sizeof(std::type_info));
        if (*type != typeid(T))
            return false;

        //Get dimensions
        read(fd, &num_layer, sizeof(uint32_t));
        layer_size = (uint32_t*)malloc(sizeof(uint32_t)*num_layer);
        read(fd, layer_size, sizeof(uint32_t) * num_layer);

        uint64_t num_weights = 0;
                 num_neurons = 0;
        uint32_t max_neurons = 0;
        for (uint32_t layer = 0; layer != num_layer; layer++) {
            if (layer != 0) num_weights += layer_size[layer - 1] * layer_size[layer];
            num_neurons += layer_size[layer];
            max_neurons = std::max(max_neurons, layer_size[layer]);
        }

        //Allocate memory
        uint64_t alloc_weights = num_weights;
        uint64_t alloc_bias = num_neurons;
        uint64_t alloc_error = max_neurons * 2;
        T* raw_mem = (T*)malloc(sizeof(T) * (alloc_weights + alloc_bias + alloc_error));

        T* weights_mem = raw_mem;
        raw_mem += alloc_weights;
        weights = (T**)malloc(sizeof(T*) * (num_layer - 1)); //First layer needs no weights
        weights[0] = weights_mem;
        for (uint32_t layer = 1; layer != num_layer - 1; layer++) {
            weights[layer] = weights[layer - 1] + layer_size[layer - 1] * layer_size[layer];
        }

        T* bias_mem = raw_mem;
        raw_mem += alloc_bias;
        bias = (T**)malloc(sizeof(T*) * (num_layer - 1)); //First layer needs no bias
        bias[0] = bias_mem;
        for (uint32_t layer = 1; layer != num_layer - 1; layer++) {
            bias[layer] = bias[layer - 1] + layer_size[layer];
        }

        T* error_mem = raw_mem;
        error[0] = error_mem;
        error[1] = error_mem + max_neurons;

        output = (T**)calloc(sizeof(T*) * (num_layer - 1)); //First layer needs no output

        //Initialize memory
        read(fd, weights_mem, alloc_weights * sizeof(T)); //These two reads could theoretically be merged
        read(fd, bias_mem, alloc_bias * sizeof(T));

        close(fd);
    }
    void store_parameters(char* file) {
        //File stuff
        int fd = open(file, O_WRONLY);

        write(fd, &(typeid(T)), sizeof(std::type_info));     //1.: Type
        write(fd, &(num_layers), sizeof(uint32_t));          //2.: num_layers
        write(fd, layer_size, sizeof(uint32_t) * num_layer); //3.: layer_size
        for (int i = 0; i != num_layer - 1; i++) {           //4.: weights
            write(fd, weights[i], sizeof(T) * layer_size[i] * layer_size[i+1]);
        }
        for (int i = 0; i != num_layer; i++) {               //5.: bias
            write(fd, bias[i], sizeof(T) * layer_size[i]);
        }
    
    }
   
    void unload_resources() {
        close(fd_in);
        close(fd_out);

        free(train_input);
        free(train_output);
        free(validation_input);
        free(validation_output);

        tile_size = 0;
    }
    void load_resources(char* in, char* out, uint32_t train_samples_, uint32_t validation_samples_, uint32_t tile_size_)
    {
        //Copy hyperparameters
        train_samples = train_samples_;
        validation_samples = validation_samples_;
        tile_size = tile_size_;

        //Open files
        fd_in  = open(in , O_RDONLY);
        fd_out = open(out, O_RDONLY);

        //Allocate ram
        uint64_t storage_needed = (layer_size[0] + layer_size[num_layer - 1]) * (tile_size + validation_samples);
        T* raw_mem = (T*)malloc(sizeof(T)*storage_needed);

        train_input = raw_mem;
        raw_mem += layer_size[0] * tile_size;
        train_output = raw_mem;
        raw_mem += layer_size[num_layer - 1] * tile_size;
        val_input = raw_mem;
        raw_mem += layer_size[0] * validation_samples;
        val_output = raw_mem;

        //Copy validation samples
        lseek(fd_in , sizeof(T) * train_samples * layer_size[0], SEEK_SET);
        lseek(fd_out, sizeof(T) * train_samples * layer_size[num_layer - 1], SEEK_SET);
        read(fd, val_input , sizeof(T) * validation_samples * layer_size[0]);
        read(fd, val_output, sizeof(T) * validation_samples * layer_size[num_layer - 1]);

        load_tile();
    }
    void load_tile() { 
        //Check for wrap around
        auto ofs = lseek(fd_in, 0, SEEK_CUR);
        int bytes_left = sizeof(T) * train_samples * layer_size[0] - ofs;
        if (sizeof(T) * tile_size * layer_size[0] > bytes_left) {
            int bytes_left_out = (bytes_left / layer_size[0]) * layer_size[num_layer - 1];

            read(fd_in , train_input , bytes_left);
            read(fd_out, train_output, bytes_left_out);
            lseek(fd_in , 0, SEEK_SET);
            lseek(fd_out, 0, SEEK_SET);
            read(fd_in , train_input  + (bytes_left     / sizeof(T)), sizeof(T) * tile_size * layer_size[0] - bytes_left);
            read(fd_out, train_output + (bytes_left_out / sizeof(T)), sizeof(T) * tile_size * layer_size[num_layer - 1] - bytes_left_out);
        }
        else {
            read(fd_in , train_input , sizeof(T) * tile_size * layer_size[0]);
            read(fd_out, train_output, sizeof(T) * tile_size * layer_size[num_layer - 1]);
        }

    } //Loads tile starting from current file pointer offset. Increments file pointer

    void set_batchSize(uint32_t batch_size_) {
        batch_size = batch_size_;
        assert(tile_size != 0 && tile_size % batch_size == 0 && batch_size <= 1024);

        //Free old output-arrays
        if ((uint64_t)output[0] != 0ull)
            free(output[0]);
        
        T* output_mem = (T*)malloc(sizeof(T) * num_neurons * batch_size); //First layer needs no output
        output[0] = output_mem;
        for (uint32_t layer = 1; layer != num_layer - 1; layer++) {
            output[layer] = output[layer - 1] + layer_size[layer] * batch_size;
        }
    } //Has to be called after a dataset was loaded
    void set_learningFactor(float f) {
        learning_factor = f;
    }

    void train(uint32_t num_batches) {
        T* train_in_end = train_input + tile_size * layer_size[0];

        for (int batch = 0; batch != num_batches; batch++) {
            if (cur_in == train_in_end) {
                cur_in = train_input;
                cur_out = train_output;
                load_tile();
            }
            forward_propagate(cur_in);
            backward_propagate(cur_out);

            cur_in += batch_size * layer_size[0];
            cur_out += batch_size * layer_size[num_layer - 1];
        }
    }
    void forward_propagate(uint32_t* in_data) {
        //Set up constants
        T scalars[2] = {(T)1, (T)0};
      
        weight_mul<T>(0, in_data, &scalars[0], &scalars[1]);
        add_bias(0);
        activate<ACT>(0);
        for (int l = 1; l != num_layer-1; l++) {
            weight_mul<T>(l, output[l-1], &scalars[0], &scalars[1]);
            add_bias(l);
            activate<ACT>(l);
        }
        weight_mul<T>(num_layer - 1, output[num_layer - 2], &scalars[0], &scalars[1]);
        add_bias(num_layer - 1);
        activate<Activation::SOFTMAX>(num_layer - 1);
    }
    void backward_propagate(uint32_t* ind) {
        
    }

    double get_error() {}
    char* get_info() {}
    T* get_output(uint32_t* in) {} //Has to reset batch_size
};


//=====================================================================================================
__global__ void set(float* mem, int l) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < l)
        mem[idx] = 1.0;
}
__global__ void inc(float* mem) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx == 0)
        *mem = (*mem) + 1.f;
}
__global__ void ex(float* i) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx == 0)
        *i = exponential<float>(*i);
}

#include <chrono>
#define BE(x, layer);\
    sta[layer] = std::chrono::high_resolution_clock::now();\
    x\
    sto = std::chrono::high_resolution_clock::now();\
    printf("%f\t%d\n", (sto-sta[layer]).count() / 1000000., __LINE__);



template<typename T> struct __device_builtin__ __builtin_align__(2 * sizeof(T)) var2 { T a, b; };
template<typename T> struct __device_builtin__ __builtin_align__(4 * sizeof(T)) var4 { T a, b, c, d; };

template<typename T>
__inline__ __device__
T multipleReduce1(T* in, T N, bool nIsPow2) {
    T sum = 0;

    for (uint32_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x; 
        i < N; 
        i += blockDim.x * 2 * gridDim.x)
    {
        sum += in[i];

        if (nIsPow2 || i + blockDim.x < N)
            sum += in[i + blockDim.x];

    }

    return sum;
}

template<typename T>
__inline__ __device__
T multipleReduce2(T* in, T N, bool nIsPow2) {
    //static_assert(typeid(T) == typeid(int) || typeid(T) == typeid(float) || typeid(T) == typeid(double), "Unknown reduction type");
    
    T sum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N / 2; i += blockDim.x * gridDim.x) {
        var2<T> val = reinterpret_cast<var2<T>*>(in)[i];
        sum += val.a + val.b;
    }
    int i = idx + N / 2 * 2;
    if (i < N)
        sum += in[i];

    return sum;
}

template<typename T>
__inline__ __device__
T multipleReduce3(T* in, T N, bool nIsPow2) {
    //static_assert(typeid(T) == typeid(int) || typeid(T) == typeid(float) || typeid(T) == typeid(double), "Unknown reduction type");

    T sum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N / 4; i += blockDim.x * gridDim.x) {
        var4<T> val = reinterpret_cast<var4<T>*>(in)[i];
        sum += (val.a + val.b) + (val.c + val.d);
    }
    int i = idx + N / 4 * 4;
    if (i < N)
        sum += in[i];

    return sum;
}

template<typename T, int mul_Algo>
__inline__ __device__ T multipleReduce(T* in, T N, bool nIsPow2) {
    if constexpr (mul_Algo == 0)
        return multipleReduce1<T>(in, N, nIsPow2);
    if constexpr (mul_Algo == 1) 
        return multipleReduce2<T>(in, N, nIsPow2);
    if constexpr (mul_Algo == 2)
        return multipleReduce3<T>(in, N, nIsPow2);
}

template<typename T>
__inline__ __device__
T warpReduceSum(T val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template<typename T, int blockSize>
__inline__ __device__
T blockReduceSum1(T val) {//2 Warp reduces, Best when Blocksize is 1024
    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockSize / 32) ? shared[lane] : 0;

    if (wid == 0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

template<typename T, int blockSize>
__inline__ __device__
T blockReduceSum2(T val, T* sdata) {//1 warp reduce + Manual reduce
    int tid = threadIdx.x;

    if constexpr(blockSize >= 1024)
    {
        if (tid < 512)
            sdata[tid] = val = val + sdata[tid + 512];
        __syncthreads();
    }

    if constexpr (blockSize >= 512)
    {
        if (tid < 256)
            sdata[tid] = val = val + sdata[tid + 256];
        __syncthreads();
    }
    
    if constexpr (blockSize >= 256)
    {
        if (tid < 128)
            sdata[tid] = val = val + sdata[tid + 218];
        __syncthreads();
    }
    
    if constexpr (blockSize >= 128)
    {
        if (tid < 64)
            sdata[tid] = val = val + sdata[tid + 64];
        __syncthreads();
    }

    if (tid < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if constexpr (blockSize >= 64) val += sdata[tid + 32];
        // Reduce final warp using shuffle
        val = warpReduceSum(val);
    }
    return val;
}

template<typename T>
__inline__ __device__
void blockReduceSum3(T val, T* out) {//1 warp reduce + Atomics
    T t = warpReduceSum(val);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, t);
}

template<typename T, int blockSize, int mul_Alg, bool atomic, int red_Algo>
__global__ void gridReduce(T* in, T* out, uint32_t N, bool nIsPow2) {
#ifdef __NVCC__
#pragma push
#pragma diag_suppress 177
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#endif
    __shared__ T sdata_[ 1024 * sizeof(T)];
    T* sdata = +sdata_;
#ifdef __NVCC__
#pragma pop
#else
#pragma clang diagnostic pop
#endif

    T sum;
    if constexpr(red_Algo == 1)                               //Setzt nIsPow2 voraus.
        sum = multipleReduce<T, mul_Alg>(in, N, true);
    else
        sum = multipleReduce<T, mul_Alg>(in, N, nIsPow2);

    if (threadIdx.x + blockIdx.x * blockDim.x + 32 < N) {     //Don't overflow
        if constexpr (red_Algo == 0) {
            sum = blockReduceSum1<T, blockSize>(sum);
        }
        else if constexpr (red_Algo == 1) {
            sdata[threadIdx.x] = sum;
            __syncthreads();
            sum = blockReduceSum2<T, blockSize>(sum, sdata);
        }
        else if constexpr (red_Algo == 2) {
            blockReduceSum3<T>(sum, out);
            return;
        }
    }
    
    if (threadIdx.x == 0) {
        if constexpr (atomic)
            atomicAdd(out, sum);
        else
            out[blockIdx.x] = sum;
    }
}

template<typename T, int blockSize, int gridSize, int mul_Alg, bool atomic, int red_Algo>
void gridReduceLauncher(T* in, T* out, uint32_t N, bool nIsPow2) {
    gridReduce<float, b_const, mul_Algo, atomic, red_Algo><<<g, b_const>>>(mem, mem + N, N, nIsPow2);
}

#define MAX_N (1u<<15)
#define ALGO(mul_Algo, atomic, red_Algo) (((red_Algo&((1<<15)-1))<<16)^((mul_Algo&((1<<15)-1))<<1)^(atomic&0b1))
#define UNHASH_Algo(hash) (hash>>16), ((hash>>1)&((1<<15)-1)), (hash&0b1)
//#define ALGO_HASH(mul_Algo, atomic, red_Algo) (mul_algo+3*red_algo+9*atomic)
#define TIME(mul_Algo, atomic, red_Algo, b_const);                                     \
    cudaEventRecord(start, 0);                                                         \
    gridReduce<float, b_const, g, mul_Algo, atomic, red_Algo>(mem, mem+N, N, nIsPow2); \
    cudaEventRecord(stop, 0);                                                          \
    cudaEventSynchronize(stop);                                                        \
    cudaEventElapsedTime(&time, start, stop);                                          \
    CHECK_CUDA_ERROR();                                                                \
    if(!atomic)                                                                        \
        time += global_timing[min(g, ((N+31)/32)&(~(31)))];                            \
    if(time < global_timing[N]){                                                       \
        global_timing[N] = time;                                                       \
        global_blocks[N] = b_const;                                                    \
        global_grids[N]  = (int)g;                                                     \
        global_algos[N]  = ALGO(mul_Algo, atomic, red_Algo);                           \
    }
#define BENCH(b_const);\
                  TIME(0, false, 0, b_const);   \
    if (nIsPow2){ TIME(0, false, 1, b_const); } \
                  TIME(0, false, 2, b_const);   \
                  TIME(0, true , 0, b_const);   \
    if (nIsPow2){ TIME(0, true , 1, b_const); } \
                  TIME(0, true , 2, b_const);   \
                                                \
                  TIME(1, false, 0, b_const);   \
    if (nIsPow2){ TIME(1, false, 1, b_const); } \
                  TIME(1, false, 2, b_const);   \
                  TIME(1, true , 0, b_const);   \
    if (nIsPow2){ TIME(1, true , 1, b_const); } \
                  TIME(1, true , 2, b_const);   \
                                                \
                  TIME(2, false, 0, b_const);   \
    if (nIsPow2){ TIME(2, false, 1, b_const); } \
                  TIME(2, false, 2, b_const);   \
                  TIME(2, true , 0, b_const);   \
    if (nIsPow2){ TIME(2, true , 1, b_const); } \
                  TIME(2, true , 2, b_const);
    
    
int main()
{
    CUBLAS_ERROR(cublasCreate(&cublas_handle));
    //Logging
    cublasSetAtomicsMode(cublas_handle, CUBLAS_ATOMICS_ALLOWED);
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

    //Block dimension

    //Set cuda stream
    //set vector, matrixes (async), pointer mode

    //Maybe have to change indexing mode to start with 0 instead of 1 using #define IDX2C(i,j,ld) (((j)*(ld))+(i))

    


    //Memory
    float* mem;
    cudaMalloc(&mem, 2ull * sizeof(float) * MAX_N);

    float* global_timing = (float*)malloc(sizeof(float) * (MAX_N + 1)); //Fastest Time
    int*   global_blocks = (int*)  malloc(sizeof(int)   * (MAX_N + 1)); //Block size for fastest time
    int*   global_grids  = (int*)  malloc(sizeof(int)   * (MAX_N + 1)); //Grid size for fastest time
    int*   global_algos  = (int*)  malloc(sizeof(int)   * (MAX_N + 1)); //Algorithm for fastest time
    for (int i = 0; i != 32; i++) {
        global_timing[i] =  0.03f;                          //Not right
        global_blocks[i] =  0;
        global_grids [i] =  0;
        global_algos [i] = -1;
    }

    //Timing stuff
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    //Debugging stuff
#ifdef __NVCC__
#pragma warning(push)
#pragma diag_suppress 177
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#endif
    std::chrono::high_resolution_clock::time_point sta[20], sto;
#ifdef __NVCC__
#pragma warning(pop)
#else
#pragma clang diagnostic pop
#endif

#pragma omp parallel for schedule(dynamic) num_threads(16)    
    for (int32_t N_ = 32; N_ <= MAX_N; N_++) {
        uint32_t N = (uint32_t)N_;
        global_timing[N] = 999999999999999999999999999999999.f;
        for (uint32_t g = 1; g <= min(2560u, 2*N/32); g++) {
            bool nIsPow2 = (N & (N - 1));
            BENCH(32  );
            BENCH(64  );
            BENCH(96  );
            BENCH(128 );
            BENCH(160 );
            BENCH(192 );
            BENCH(224 );
            BENCH(256 );
            BENCH(288 );
            BENCH(320 );
            BENCH(352 );
            BENCH(384 );
            BENCH(416 );
            BENCH(448 );
            BENCH(480 );
            BENCH(512 );
            BENCH(544 );
            BENCH(576 );
            BENCH(608 );
            BENCH(640 );
            BENCH(672 );
            BENCH(704 );
            BENCH(736 );
            BENCH(768 );
            BENCH(800 );
            BENCH(832 );
            BENCH(864 );
            BENCH(896 );
            BENCH(928 );
            BENCH(960 );
            BENCH(992 );
            BENCH(1024);
        }

        printf("Size: %d \t|Mul_Algo: %d \t|Atomics: %d \t|Red_Algo: %d \t|Block_Size: %d \t|Grid_Size: %d \t|Time: %f\n",
            N, UNHASH_Algo(global_algos[N]), global_blocks[N], global_grids[N], global_timing[N]);
    }



    CHECK_CUDA_ERROR();

    CUBLAS_ERROR(cublasDestroy(cublas_handle));

    return 0;
}


/*
#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)

#define MAX_N (1ull<<18)
#define ITER 8
#define OLD
#define OLD2
#define X 32
//#define HEURISTIK

#define CALL(b, g, n, a);  reduceKernelCall<a>(mem, b, g, n);

uint32_t pow2_less(uint32_t x)
{
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x ^ (x >> 1);
}

template<typename T, uint32_t blockSize, bool nIsPow2, uint32_t it>
__global__ void reduceKernel_(T* g_idata, T* g_odata, unsigned int n)
{
    for (uint32_t it_ = 0; it_ < it; it_++) {
        // Handle to thread block group
        cg::thread_block cta = cg::this_thread_block();
        extern __shared__ T sdata_[];
        T* sdata = +sdata_;

        // perform first level of reduction,
        // reading from global memory, writing to shared memory
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
        unsigned int gridSize = blockSize * 2 * gridDim.x;

        T mySum = 0;

        // we reduce multiple elements per thread.  The number is determined by the
        // number of active thread blocks (via gridDim).  More blocks will result
        // in a larger gridSize and therefore fewer elements per thread
        while (i < n)
        {
            T p1 = exponential<T>(g_idata[i]);
            mySum += p1;
            g_idata[i] = p1;

            // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
            if (nIsPow2 || i + blockSize < n) {
                T p2 = exponential<T>(g_idata[i + blockSize]);
                mySum += p2;
                g_idata[i + blockSize] = p2;
            }

            i += gridSize;
        }

        // each thread puts its local sum into shared memory
        sdata[tid] = mySum;
        cg::sync(cta);


        // do reduction in shared mem
        if ((blockSize >= 1024) && (tid < 512))
        {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }

        cg::sync(cta);

        if ((blockSize >= 512) && (tid < 256))
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockSize >= 256) && (tid < 128))
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockSize >= 128) && (tid < 64))
        {
            sdata[tid] = mySum = mySum + sdata[tid + 64];
        }

        cg::sync(cta);

        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32)
        {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >= 64) mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
            {
                mySum += tile32.shfl_down(mySum, offset);
            }
        }

        // write result for this block to global mem
        if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
        cg::sync(cta);
    }
}

template<typename T, uint32_t blockSize, bool nIsPow2, uint32_t it>
__global__ void reduceKernel_Atomic1_(T* g_idata, T* g_odata, unsigned int n)
{
    for (uint32_t it_ = 0; it_ != it; it_++) {
        // Handle to thread block group
        cg::thread_block cta = cg::this_thread_block();
        extern __shared__ T sdata_[];
        T* sdata = +sdata_;

        // perform first level of reduction,
        // reading from global memory, writing to shared memory
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
        unsigned int gridSize = blockSize * 2 * gridDim.x;

        T mySum = 0;

        // we reduce multiple elements per thread.  The number is determined by the
        // number of active thread blocks (via gridDim).  More blocks will result
        // in a larger gridSize and therefore fewer elements per thread
        while (i < n)
        {
            T p1 = exponential<T>(g_idata[i]);
            mySum += p1;
            g_idata[i] = p1;

            // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
            if (nIsPow2 || i + blockSize < n) {
                T p2 = exponential<T>(g_idata[i + blockSize]);
                mySum += p2;
                g_idata[i + blockSize] = p2;
            }

            i += gridSize;
        }

        // each thread puts its local sum into shared memory
        sdata[tid] = mySum;
        cg::sync(cta);


        // do reduction in shared mem
        if ((blockSize >= 1024) && (tid < 512))
        {
            sdata[tid] = mySum = mySum + sdata[tid + 512];
        }

        cg::sync(cta);

        if ((blockSize >= 512) && (tid < 256))
        {
            sdata[tid] = mySum = mySum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockSize >= 256) && (tid < 128))
        {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockSize >= 128) && (tid < 64))
        {
            sdata[tid] = mySum = mySum + sdata[tid + 64];
        }

        cg::sync(cta);

        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32)
        {
            // Fetch final intermediate sum from 2nd warp
            if (blockSize >= 64) mySum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
            {
                mySum += tile32.shfl_down(mySum, offset);
            }
        }

        // write result for this block to global mem
        if (cta.thread_rank() == 0) atomicAdd(g_odata, mySum);
    }
}

template<typename T, uint32_t blockSize, bool nIsPow2, uint32_t it>
__global__ void reduceKernel_Atomic2_(T* g_idata, T* g_odata, unsigned int n)
{
    for (uint32_t it_ = 0; it_ != it; it_++) {
        // Handle to thread block group
        cg::thread_block cta = cg::this_thread_block();

        // perform first level of reduction,
        // reading from global memory, writing to shared memory
        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
        unsigned int gridSize = blockSize * 2 * gridDim.x;

        T mySum = 0;

        // we reduce multiple elements per thread.  The number is determined by the
        // number of active thread blocks (via gridDim).  More blocks will result
        // in a larger gridSize and therefore fewer elements per thread
        while (i < n)
        {
            T p1 = exponential<T>(g_idata[i]);
            mySum += p1;
            g_idata[i] = p1;

            // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
            if (nIsPow2 || i + blockSize < n) {
                T p2 = exponential<T>(g_idata[i + blockSize]);
                mySum += p2;
                g_idata[i + blockSize] = p2;
            }

            i += gridSize;
        }

        // each thread puts its local sum into shared memory
        cg::sync(cta);


        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        //Reduce all warps of block using shuffle
        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
        {
            mySum += tile32.shfl_down(mySum, offset);
        }

        // write result for this warp to global mem
        if (tid & 31 == 0) atomicAdd(g_odata, mySum);
    }
}

template<uint32_t algo>
void reduceKernelCall(float* mem, uint32_t b, uint32_t g, uint32_t N) {
    constexpr bool p = false;//(0 == (N & (N - 1)));
    uint32_t m = smem(b, float);

    if constexpr (algo == 0) {
        switch (b) {
        case 1:
            reduceKernel_<float, 1, p, ITER> << <g, 1, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 2:
            reduceKernel_<float, 2, p, ITER> << <g, 2, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 4:
            reduceKernel_<float, 4, p, ITER> << <g, 4, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 8:
            reduceKernel_<float, 8, p, ITER> << <g, 8, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 16:
            reduceKernel_<float, 16, p, ITER> << <g, 16, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 32:
            reduceKernel_<float, 32, p, ITER> << <g, 32, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 64:
            reduceKernel_<float, 64, p, ITER> << <g, 64, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 128:
            reduceKernel_<float, 128, p, ITER> << <g, 128, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 256:
            reduceKernel_<float, 256, p, ITER> << <g, 256, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 512:
            reduceKernel_<float, 512, p, ITER> << <g, 512, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 1024:
            reduceKernel_<float, 1024, p, ITER> << <g, 1024, m >> > (mem, mem + N, (uint32_t)N);
            break;
        }
    }
    else if constexpr (algo == 1) {
        switch (b) {
        case 1:
            reduceKernel_Atomic1_<float, 1, p, ITER> << <g, 1, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 2:
            reduceKernel_Atomic1_<float, 2, p, ITER> << <g, 2, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 4:
            reduceKernel_Atomic1_<float, 4, p, ITER> << <g, 4, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 8:
            reduceKernel_Atomic1_<float, 8, p, ITER> << <g, 8, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 16:
            reduceKernel_Atomic1_<float, 16, p, ITER> << <g, 16, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 32:
            reduceKernel_Atomic1_<float, 32, p, ITER> << <g, 32, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 64:
            reduceKernel_Atomic1_<float, 64, p, ITER> << <g, 64, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 128:
            reduceKernel_Atomic1_<float, 128, p, ITER> << <g, 128, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 256:
            reduceKernel_Atomic1_<float, 256, p, ITER> << <g, 256, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 512:
            reduceKernel_Atomic1_<float, 512, p, ITER> << <g, 512, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 1024:
            reduceKernel_Atomic1_<float, 1024, p, ITER> << <g, 1024, m >> > (mem, mem + N, (uint32_t)N);
            break;
        }
    }
    else if constexpr (algo == 2) {
        switch (b) {
        case 1:
            reduceKernel_Atomic2_<float, 1, p, ITER> << <g, 1, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 2:
            reduceKernel_Atomic2_<float, 2, p, ITER> << <g, 2, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 4:
            reduceKernel_Atomic2_<float, 4, p, ITER> << <g, 4, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 8:
            reduceKernel_Atomic2_<float, 8, p, ITER> << <g, 8, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 16:
            reduceKernel_Atomic2_<float, 16, p, ITER> << <g, 16, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 32:
            reduceKernel_Atomic2_<float, 32, p, ITER> << <g, 32, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 64:
            reduceKernel_Atomic2_<float, 64, p, ITER> << <g, 64, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 128:
            reduceKernel_Atomic2_<float, 128, p, ITER> << <g, 128, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 256:
            reduceKernel_Atomic2_<float, 256, p, ITER> << <g, 256, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 512:
            reduceKernel_Atomic2_<float, 512, p, ITER> << <g, 512, m >> > (mem, mem + N, (uint32_t)N);
            break;
        case 1024:
            reduceKernel_Atomic2_<float, 1024, p, ITER> << <g, 1024, m >> > (mem, mem + N, (uint32_t)N);
            break;
        }
    }

}


    float* mem;
    cudaMalloc(&mem, 2ull * sizeof(float) * MAX_N);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    float* global_timing = (float*)malloc(sizeof(float) * (MAX_N + 1));
    int* global_blocks = (int*)malloc(sizeof(int) * (MAX_N + 1));
    int* global_grids = (int*)malloc(sizeof(int) * (MAX_N + 1));
    int* global_algos = (int*)malloc(sizeof(int) * (MAX_N + 1));

    global_timing[0] = 0;
    global_blocks[0] = 0;
    global_grids[0] = 0;
    global_algos[0] = -1;
    global_timing[1] = 0;
    global_blocks[1] = 0;
    global_grids[1] = 0;
    global_algos[1] = -1;

    float min_time;
    int best_grid_size;
    int algor;

    int num_bl[32] = { 0 };
    int num_al[3] = { 0 };

#ifdef HEURISTIK
    for (uint32_t N = 2; N <= 2047; N++) {
        uint32_t pow2 = pow2_less(N); //Highest power of two below or equal to N
        uint32_t dst = N - pow2;

        int b;
        if (dst <= pow2 / 2) {
            b = pow2;
        }
        else if (dst <= 7 * pow2 / 8) {
            b = pow2 / 2;
        }
        else {
            b = pow2 / 8;
        }

        min_time = 9999999999999999999999999.f;
        best_grid_size = 1;
        for (int g = 1; g <= N / b; g++) {
            cudaEventRecord(start, 0);
            CALL(b, g, N, 0);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            time += global_timing[g];
            CHECK_CUDA_ERROR();
            if (time < min_time) {
                min_time = time;
                best_grid_size = g;
            }

            cudaEventRecord(start, 0);
            CALL(b, g, N, 1);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            CHECK_CUDA_ERROR();
            if (time < min_time) {
                min_time = time;
                best_grid_size = g;
            }

            cudaEventRecord(start, 0);
            CALL(b, g, N, 2);
            cudaDeviceSynchronize();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            CHECK_CUDA_ERROR();
            if (time < min_time) {
                min_time = time;
                best_grid_size = g;
            }
        }

        global_blocks[N] = b;
        global_grids[N] = best_grid_size;
        global_timing[N] = min_time;
        printf("Size: %d \t| Block size: %d \t| Grid size: %d \t| Time: %f\n", N, global_blocks[N], global_grids[N], global_timing[N]);
    }
    for (uint32_t N = 2048; N <= MAX_N; N++) {
#else
    for (uint32_t N = 2; N <= MAX_N; N++) {
#endif
        float ti[X + 1] = { 0 };
        ti[0] = 999999999999999999999999999999999999.f;
        int grid_sizes[X + 1] = { 0 };
        int algors[X + 1] = { 0 };
        for (int b_ = 1; b_ <= X; b_++) {
            uint32_t b = b_ << 5;
            min_time = 9999999999999999999999999.f;
            best_grid_size = -1;
            algor = -1;
            for (int g = 1; g <= N / b; g++) {
#ifdef OLD
                cudaEventRecord(start, 0);
                CALL(b, g, N, 0);
                cudaDeviceSynchronize();
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time, start, stop);
                time += global_timing[g];
                CHECK_CUDA_ERROR();
                if (time < min_time) {
                    min_time = time;
                    best_grid_size = g;
                    algor = 0;
                }
#endif
                if (b >= 32) {
#ifdef OLD2
                    cudaEventRecord(start, 0);
                    CALL(b, g, N, 1);
                    cudaDeviceSynchronize();
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&time, start, stop);
                    CHECK_CUDA_ERROR();
                    if (time < min_time) {
                        min_time = time;
                        best_grid_size = g;
                        algor = 1;
                    }
#endif

                    cudaEventRecord(start, 0);
                    CALL(b, g, N, 2);
                    cudaDeviceSynchronize();
                    cudaEventRecord(stop, 0);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&time, start, stop);
                    CHECK_CUDA_ERROR();
                    if (time < min_time) {
                        min_time = time;
                        best_grid_size = g;
                        algor = 2;
                    }
                }
            }

            ti[b_] = min_time;
            grid_sizes[b_] = best_grid_size;
            algors[b_] = algor;
            //printf("Block size: %d \t| Grid size: %d \t| Time:  %f \t| Total time: %f\n", b, best_grid_size, min_time, ti[b_]);
            //return 0;
        }

        float* min_ = std::min_element(+ti, +ti + 11);
        int b = min_ - (+ti);

        global_blocks[N] = b << 5;
        global_grids[N] = grid_sizes[b];
        global_timing[N] = ti[b];
        global_algos[N] = algors[b];

        //printf("Size: %d \t| Block size: %d \t| Grid size: %d \t| Time: %f \t | Algorithm: %d\n", N, global_blocks[N], global_grids[N], global_timing[N], global_algos[N]);
        //printf("%d %d %d %f %d\n", N, global_blocks[N], global_grids[N], global_timing[N], global_algos[N]);

        num_bl[b]++;
        if(global_algos[N]>=0)
            num_al[global_algos[N]]++;
        if (N % 1024 == 0) {
            for (int u = 0; u != 32; u++) {
                printf("%d ", num_bl[u]);
            }
            printf("\n");
            for (int u = 0; u != 3; u++) {
                printf("%d ", num_al[u]);
            }
            printf("\n------------------------------------\n");
        }
    }
*/


/*
#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)

#define MAX_N 17
#define T1 64
#define T2 128
//#define VALIDATE

template<typename T>
__global__ void softplus_deriv2(T* in, uint32_t l) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < l) {
        T v = in[idx];
        T w = exponential<T>(v);
        in[idx] = (v - 1) / v;
    }
}

#define CALL(A,N); softplus_deriv2<<<((N)+(A)-1)/(A), (A)>>>(mem, (N));


    float* mem;
    cudaMalloc(&mem, sizeof(float) * (1<<MAX_N));

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float min_time;
    int best_block_size;

    uint32_t out[2] = { 0,0 };
    for (int n = 64; n <= (1<<MAX_N); n*=1.1) {
    min_time = 9999999999999999999999999.f;
    best_block_size = 1;
    float t1;
    float t2;
    for (int b = 1; b <= 1024; b++) {
        cudaEventRecord(start, 0);
        CALL(b, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);

        CHECK_CUDA_ERROR();
        if (time < min_time) {
            min_time = time;
            best_block_size = b;
        }

        if (b == T1)
            t1 = time;
        if (b == T2)
            t2 = time;

        //if( (b&(b-1))==0) printf("%d : %f\n", b, time);
    }
    printf("Elements: %d | Block size: %d | Time:  %f | " QUOTE(T1) ": %f | " QUOTE(T2) ": %f\n", n, best_block_size, min_time, t1, t2);
    out[t1 > t2]++;
#ifdef VALIDATE
    cudaEventRecord(start, 0);
    softplus << <n, 1 >> > (mem);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Elements: %d | Block size: %d | Time:  %f\n", n, n, time);
#endif
    }
    printf(QUOTE(T1)":%d, " QUOTE(T2) ":%d", out[0], out[1]);
*/