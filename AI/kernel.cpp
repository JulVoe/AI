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

#include <algorithm>
#include <numeric>
#include <string>
#include <inttypes.h>
#include <chrono>

#include <atomic>
#include <future>
#include <thread>

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "omp.h"
#include <x86intrin.h>

#include "Dataset.cpp"

/*
Notation conventions:
 - A matrix has size height*width.
 - layer[0] is the input layer
 - weight[i] are the weights between layer[i] and layer[i+1]
*/




//TODOS (BOTH FILES!!)
//TODO: Block all kernel calls
//TODO: CONVOLUTIONS AND TILING DESIGN
//TODO: Unified allocator
//TODO: TYPE HANDLING
//TODO: REGULARIZATION AND NORMALIZATION

//=========================================================
//==================|HELPER FUNCITONS|=====================
//=========================================================


#define LAUNCH_PARAM(N) (int)(1. / ((10. / ((double)(1 << 13)) + 32. / ((double)(N))))), 32


//================================================
//==================|GLOBALS|=====================
//================================================

cublasHandle_t cublas_handle;

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
        static_assert(typeid(T)==typeid(float) || typeid(T)==typeid(double) || typeid(T)==typeid(half), "[Error] Matrix multiplication is not supported with this type!");
        
        if constexpr (typeid(T) == typeid(float))
            cublasSgemm(cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, &cublasConst[0], A, trans_A?x1:y1, B, trans_B?x2:x1, &cublasConst[!overwrite], C, y1);
        if constexpr (typeid(T) == typeid(double))
            cublasDgemm(cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, &cublasConst[0], A, trans_A?x1:y1, B, trans_B?x2:x1, &cublasConst[!overwrite], C, y1);
        if constexpr (typeid(T) == typeid(half))
            cublasGemmEx(cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, &cublasConst[0], A, CUDA_R_16F, trans_A?x1:y1, B, CUDA_R_16F, trans_B?x2:x1, &cublasConst[!overwrite], C, CUDA_R_16F, y1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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
        set_cublasMode(CublasMode::ADD);

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
        set_cublasMode(CublasMode::OVERWRITE);

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
