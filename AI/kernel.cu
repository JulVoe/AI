#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <typeinfo>
#include <inttypes.h>
#include <cublas_v2.h>
//#include <mma.h>
//#include <cublasXt.h>

//TODO: COPY WEIGHTS, BIAS, TRAIN TILE AND VALIDATIONS SAMPLES TO GPU

#define CUBLAS_ERROR(e); \
    if((e)!=CUBLAS_STATUS_SUCCESS){\
        printf("%d %d", __LINE__, e);\
    }

cublasHandle_t cublas_handle;

enum Activation {RELU=0, SIGMOID=1, SOFTMAX=2};

/*
__global__ void add_bias_1(uint32_t ind) { //output[ind] += bias[ind]
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int ind_bias = idx % layer_size[ind + 1];

        output[ind][idx] += bias[ind_bias];
    }
    __global__ void add_bias_2(uint32_t ind) { //output[ind] += bias[ind]
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int ind_bias = threadIdx.x;

        output[ind][idx] += bias[ind_bias];
    }
    __global__ void add_bias_3(uint32_t ind) { //output[ind] += bias[ind]
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int ind_bias = blockIdx.x;

        output[ind][idx] += bias[ind_bias];
    }
*/

__global__ void add_bias(uint32_t ind) {}
template<uint32_t i>
void activate(uint32_t) {}

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
    template<typename TYPE>        
    void weight_mul(uint32_t ind, TYPE* B, TYPE* alpha, TYPE* beta) {
        static_assert(typeid(TYPE) != typeid(TYPE), "The type passed to MultiLayerPerceptron is unsupported");
    }
    template<> void weight_mul<float>(uint32_t ind, float* B, float* alpha, float* beta) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, layer_size[ind+1], batch_size, layer_size[ind], alpha, weights[ind], layer_size[ind+1], B, layer_size[ind], beta, output[ind], layer_size[ind+1]);
    }
    template<> void weight_mul<double>(uint32_t ind, double* B, double* alpha, double* beta) {
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, layer_size[ind + 1], batch_size, layer_size[ind], alpha, weights[ind], layer_size[ind + 1], B, layer_size[ind], beta, output[ind], layer_size[ind + 1]);
    }
    template<> void weight_mul<half>(uint32_t ind, half* B, half* alpha, half* beta) {
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, layer_size[ind + 1], batch_size, layer_size[ind], alpha, weights[ind], CUDA_R_16F, layer_size[ind + 1], B, CUDA_R_16F, layer_size[ind], beta, output[ind], CUDA_R_16F, layer_size[ind + 1], CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
        assert(tile_size != 0 && tile_size % batch_size == 0);

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
        dim3 block_size = {1,0,0};
        dim3 grid_size = { 1,0,0 };

        weight_mul<T>(0, in_data, &scalars[0], &scalars[1]);
        add_bias<<<block_size, grid_size>>>(0);
        activate<ACT>(0);
        for (int l = 1; l != num_layer-1; l++) {
            weight_mul<T>(l, output[l-1], &scalars[0], &scalars[1]);
            add_bias<<<block_size, grid_size>>>(l);
            activate<ACT>(l);
        }
        weight_mul<T>(l, output[l - 1], &scalars[0], &scalars[1]);
        add_bias<<<block_size, grid_size>>>(l);
        activate<Activation::SOFTMAX>(l);
    }
    void backward_propagate(uint32_t* ind) {}

    double get_error() {}
    char* get_info() {}
    T* get_output(uint32_t* in) {} //Has to reset batch_size
};

#define X 32768
#define Y 32768

__global__ void add_bias_1(uint32_t* output, uint32_t* bias) { //output[ind] += bias[ind]
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < X * Y) {
        int ind_bias = idx % X;
        output[idx] += bias[ind_bias];
    }
}
__global__ void add_bias_2(uint32_t* output, uint32_t* bias) { //output[ind] += bias[ind]
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ind_bias = threadIdx.x;

    output[idx] += bias[ind_bias];
}
__global__ void add_bias_3(uint32_t* output, uint32_t* bias) { //output[ind] += bias[ind]
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ind_bias = blockIdx.x;

    output[idx] += bias[ind_bias];
}


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

   
    uint32_t *output, *bias, *output_cpu, *bias_cpu;
    output_cpu = (uint32_t*)malloc(X * Y * sizeof(uint32_t));
    bias_cpu   = (uint32_t*)malloc(X *     sizeof(uint32_t));
    cudaMalloc(&output, X * Y * sizeof(uint32_t));
    cudaMalloc(&bias  , X *     sizeof(uint32_t));

    for (int ind = 0; ind != X * Y; ind++)
        output_cpu[ind] = rand();
    for (int ind = 0; ind != X; ind++)
        bias_cpu[ind] = rand();

    cudaMemcpy(output, output_cpu, X * Y * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(bias  , bias_cpu  , X *     sizeof(uint32_t), cudaMemcpyHostToDevice);

    int blockSize;      // The launch configurator returned block size 
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
    int gridSize;       // The actual grid size needed, based on input size 

    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, add_bias_1, 0, X * Y);
    gridSize = (X*Y + blockSize - 1) / blockSize;

    printf("blockSize: %d %d\n", blockSize, gridSize);
   
    //Benchmark start here
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 bl1 = { (unsigned)blockSize, 1u, 1u };
    dim3 gs1 = { (unsigned)gridSize, 1u, 1u };
    cudaEventRecord(start, 0);
    add_bias_1<<<bl1, gs1>>>(output, bias);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time1: %f\n", time);

    dim3 bl2 = { (unsigned)X, 1u, 1u };
    dim3 gs2 = { (unsigned)Y, 1u, 1u };
    cudaEventRecord(start, 0);
    add_bias_2<<<bl2, gs2>>>(output, bias);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time2: %f\n", time);

    dim3 bl3 = { (unsigned)Y, 1u, 1u };
    dim3 gs3 = { (unsigned)X, 1u, 1u };
    cudaEventRecord(start, 0);
    add_bias_3<<<bl3, gs3>>>(output, bias);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time3: %f\n", time);

    CUBLAS_ERROR(cublasDestroy(cublas_handle));

    return 0;
}