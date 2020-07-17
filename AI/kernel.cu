#define THE_VERSION_JULIAN_DID_NOT_SCREW_WITH
#ifdef __NVCC__
#pragma warning( disable : 4514)
#pragma warning( disable : 4711)
#pragma warning( disable : 4710)
#pragma warning( disable : 5039)
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
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

namespace cg = cooperative_groups;

#define CHECK_CUDA_ERROR();\
    do{\
        auto error = cudaGetLastError(); \
        if (error != cudaSuccess) {\
            /* print the CUDA error message and exit*/\
            printf("CUDA error: %s\n", cudaGetErrorString(error)); \
        }\
    } while (0);
//================================================
//==================|UTILITY|=====================
//================================================

template<typename T>
__device__ T exponential(T in) {
    __builtin_unreachable();
}
template<> __device__ float  exponential<float>(float in) {
    return expf(in);
}
template<> __device__ double exponential<double>(double in) {
    return exp(in);
}
template<> __device__ half   exponential<half>(half in) {
    return hexp(in);
}

template<typename T>
__device__ T logarithm(T in) {
    __builtin_unreachable();
}
template<> __device__ float  logarithm<float>(float in) {
    return logf(in);
}
template<> __device__ double logarithm<double>(double in) {
    return log(in);
}
template<> __device__ half   logarithm<half>(half in) {
    return hlog(in);
}

//========================================================
//==================|General Purpose|=====================
//========================================================

enum DIVISIBILITY { UNKNOWN = -1, DIVISIBLE = 1, NOT_DIVISIBLE = 2 };
template<typename T> struct __device_builtin__ __builtin_align__(2 * sizeof(T)) var2 { T a, b; };
template<typename T> struct __device_builtin__ __builtin_align__(4 * sizeof(T)) var4 { T a, b, c, d; };

/*Fill out with repeting copies of in.
@param out: pointer to memory to fill
@param in: memory used for filling
@param n_out: lenght of out
@param n_in: lenght of in
@template div: is n_out divisible by blockDim.x?
*/
template<typename T, DIVISIBILITY div>
__global__ void set_repeating(T* out, T* in, uint32_t n_out, uint32_t n_in) {
    bool div_;
    if constexpr (div == DIVISIBILITY::UNKNOWN)
        div_ = ((n_out % blockDim.x) == 0);

    for (uint32_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
        i < n_out;
        i += blockDim.x * 2 * gridDim.x)
    {
        out[i] = in[i % n_in];

        if constexpr (N_divisible_blocksize == DIVISIBILITY::DIVISIBLE)
            out[i + blockDim.x] = in[(i + blockDim.x) % n_in];
        if constexpr (N_divisible_blocksize == DIVISIBILITY::NOT_DIVISIBLE) {
            if (i + blockDim.x < n_out)
                out[i + blockDim.x] = in[(i + blockDim.x) % n_in];
        }
        if constexpr (N_divisible_blocksize == DIVISIBILITY::UNKNOWN) {
            if (div_ || i + blockDim.x < n_out) //Compiler will automaticly eliminate lopp-independet condition N_divisible_blocksize_
                out[i + blockDim.x] = in[(i + blockDim.x) % n_in];
        }
    }
}

//Same as above, but with constexpr n_in to eliminate slow modulo
template<typename T, DIVISIBILITY div, uint32_t n_in>
__global__ void set_repeating(T* out, T* in, uint32_t n_out) {
    bool div_;
    if constexpr (div == DIVISIBILITY::UNKNOWN)
        div_ = ((n_out % blockDim.x) == 0);

    for (uint32_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
        i < n_out;
        i += blockDim.x * 2 * gridDim.x)
    {
        out[i] = in[i % n_in];

        if constexpr (N_divisible_blocksize == DIVISIBILITY::DIVISIBLE)
            out[i + blockDim.x] = in[(i + blockDim.x) % n_in];
        if constexpr (N_divisible_blocksize == DIVISIBILITY::NOT_DIVISIBLE) {
            if (i + blockDim.x < n_out)
                out[i + blockDim.x] = in[(i + blockDim.x) % n_in];
        }
        if constexpr (N_divisible_blocksize == DIVISIBILITY::UNKNOWN) {
            if (div_ || i + blockDim.x < n_out) //Compiler will automaticly eliminate lopp-independet condition N_divisible_blocksize_
                out[i + blockDim.x] = in[(i + blockDim.x) % n_in];
        }
    }
}

//Adds all elements of in[0:N] to out[0]
template<typename T, DIVISIBILITY N_divisible_blocksize, DIVISIBILITY N_divisible_32>
__global__ void reduce(T* in, T* out, uint32_t N) {
    //0.: Compute unknown divisibilities
    bool N_divisible_blocksize_, N_divisible_32_;
    if constexpr (N_divisible_blocksize == DIVISIBILITY::UNKNOWN)
        N_divisible_blocksize_ = ((N % blockDim.x) == 0);
    if constexpr (N_divisible_32 == DIVISIBILITY::UNKNOWN)
        N_divisible_32_ = ((N % 32) == 0);

    //1.: Initialize variables
    T sum = 0;

    //2.: Reduce multiple elements per thread
    for (uint32_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
        i < N;
        i += blockDim.x * 2 * gridDim.x)
    {
        sum += in[i];

        if constexpr (N_divisible_blocksize == DIVISIBILITY::DIVISIBLE)
            sum += in[i + blockDim.x];
        if constexpr (N_divisible_blocksize == DIVISIBILITY::NOT_DIVISIBLE) {
            if (i + blockDim.x < N)
                sum += in[i + blockDim.x];
        }
        if constexpr (N_divisible_blocksize == DIVISIBILITY::UNKNOWN) {
            if (N_divisible_blocksize_ || i + blockDim.x < N) //Compiler will automaticly eliminate lopp-independet condition N_divisible_blocksize_
                sum += in[i + blockDim.x];
        }
    }
    
    //3.: Store results
    if constexpr (N_divisible_32 == DIVISIBILITY::DIVISIBLE) {
        for (int offset = 32 / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

        if ((threadIdx.x & 31) == 0) atomicAdd(out, sum);
    }
    if constexpr (N_divisible_32 == DIVISIBILITY::NOT_DIVISIBLE) {
        if (((threadIdx.x + blockIdx.x * blockDim.x)&~(0b11111)) + 32 < N) {
            for (int offset = 32 / 2; offset > 0; offset /= 2)
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

            if ((threadIdx.x & 31) == 0) atomicAdd(out, sum);
        }
        else {
            atomicAdd(out, sum);
        }
    }
    if constexpr (N_divisible_32 == DIVISIBILITY::UNKNOWN) {
        if (N_divisible_32_ || ((threadIdx.x + blockIdx.x * blockDim.x) & ~(0b11111)) + 32 < N) {
            for (int offset = 32 / 2; offset > 0; offset /= 2)
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

            if ((threadIdx.x & 31) == 0) atomicAdd(out, sum);
        }
        else {
            atomicAdd(out, sum);
        }
    }
}

template<typename T, typename F, typename G, typename V>
__global__ void set(T* in, uint32_t n, F f, G g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    V var;
    g(&var, idx);

    for (int i = idx; i < n / 4; i += blockDim.x * gridDim.x) {
        var4<T> val;
        val.a = f(var);
        val.b = f(var);
        val.c = f(var);
        val.d = f(var);
        reinterpret_cast<var4<T>*>(in)[i] = val;
    }
    int i = idx + n / 4 * 4;
    if (i < n)
        in[i] = f(var);
}

//Applys f elementwise to in[0:n]
template<typename T, typename F>
__global__ void transform(T* in, uint32_t n, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < n / 4; i += blockDim.x * gridDim.x) {
        var4<T> val = reinterpret_cast<var4<T>*>(in)[i];
        val.a = f(val.a);
        val.b = f(val.b);
        val.c = f(val.c);
        val.d = f(val.d);
        reinterpret_cast<var4<T>*>(in)[i] = val;
    }
    int i = idx + n / 4 * 4;
    if (i < n)
        in[i] = f(in[i]);
}

//Same as above, but calls g at the beginning to initialize additional variable
template<typename T, typename F, typename G, typename V>
__global__ void transform(T* in, uint32_t n, F f, G g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    V var;
    g(&var, idx);

    for (int i = idx; i < n / 4; i += blockDim.x * gridDim.x) {
        var4<T> val = reinterpret_cast<var4<T>*>(in)[i];
        val.a = f(val.a, var);
        val.b = f(val.b, var);
        val.c = f(val.c, var);
        val.d = f(val.d, var);
        reinterpret_cast<var4<T>*>(in)[i] = val;
    }
    int i = idx + n / 4 * 4;
    if (i < n)
        in[i] = f(in[i], var);
}

//in1[i] = f(in1[i], in2[i]); 0<=n<n
template<typename T, typename F>
__global__ void transform(T* in1, T* in2, uint32_t n, F f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
__global__ void transform_reduce(T* in, T* out, uint32_t n, F f) {
    //0.: Compute unknown divisibilities
    bool N_divisible_blocksize_, N_divisible_32_;
    if constexpr (N_divisible_blocksize == DIVISIBILITY::UNKNOWN)
        N_divisible_blocksize_ = ((n % blockDim.x) == 0);
    if constexpr (N_divisible_32 == DIVISIBILITY::UNKNOWN)
        N_divisible_32_ = ((n % 32) == 0);

    //1.: Initialize variables
    T sum = 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    //2.: Reduce multiple elements per thread
    for (int i = idx; i < n / 4; i += blockDim.x * gridDim.x) {
        var4<T> val = reinterpret_cast<var4<T>*>(in)[i];
        val.a = f(val.a);
        val.b = f(val.b);
        val.c = f(val.c);
        val.d = f(val.d);
        sum += (val.a + val.b) + (val.c + val.d);
        if constexpr (write) { reinterpret_cast<var4<T>*>(in)[i] = val; }
    }
    int i = idx + n / 4 * 4;
    if (i < n) {
        T tmp = f(in[i]);
        sum += tmp;
        if constexpr (write) in[i] = tmp;
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

template<typename T, typename F, DIVISIBILITY N_divisible_blocksize, DIVISIBILITY N_divisible_32, bool write>
__global__ void transform_reduce(T* in1, T* in2, T* out, uint32_t n, F f) {
    //0.: Compute unknown divisibilities
    bool N_divisible_blocksize_, N_divisible_32_;
    if constexpr (N_divisible_blocksize == DIVISIBILITY::UNKNOWN)
        N_divisible_blocksize_ = ((n % blockDim.x) == 0);
    if constexpr (N_divisible_32 == DIVISIBILITY::UNKNOWN)
        N_divisible_32_ = ((n % 32) == 0);

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

template<typename T, DIVISIBILITY N_divisible_blocksize, DIVISIBILITY N_divisible_32>
void reduceLauncher(T* in, T* out, uint32_t N) {
    reduce<T, N_divisible_blocksize, N_divisible_32><<<(int)(1./((10./((double)(1<<13)) + 32./((double)N)))), 32>>>(in, out, N);
}

//A *= B (elementwise multiplication)
template<typename T>
void multiplyElementwise(T* A, T* B, uint32_t n, uint32_t b, uint32_t g) {
    constexpr auto ldb = []__device__(T a, T b) { return a * b; };
    transform<T, decltype(ldb)><<<g, b>>>(A, B, n, ldb)
}

//===============================================
//==================|RANDOM|=====================
//===============================================
//All these functions generate a new curand state per thread. This is inefficient
//however it is not feasible to store a state for every thread as this would require an unknown memory lenght

/*
    @param b: If false, use completly random values.
*/
template<typename T, bool b>
void set_random(T* in, uint32_t s1, uint32_t s2 = 1) {
    auto init = []__device__(curandState_t* s, int idx) { curand_init(1337, idx, 0, s); }

    float mul = sqrt(2.f / s1);
    auto ldb = [mul]__device__(curandState_t* s) { return  mul * curand_normal(s) };
    set<T, decltype(ldb), decltype(init), curandState_t>(in, s1 * s2, ldb, init);
}

template<typename T>
void random_noise(T* in, uint32_t n, T devi) {
    auto init = []__device__(curandState_t* s, int idx) { curand_init(1337, idx, 0, s); }
    auto ldb = [devi]__device__(T in, curandState_t* s) { return in + devi * curand_normal(s) };
    transform<T , decltype(ldb), decltype(init), curandState_t>(in, n, ldb, init);
}

template<typename T>
void random_dropout(T* in, uint32_t n, float prob) {
    auto init = []__device__(curandState_t* s, int idx) { curand_init(1337, idx, 0, s); }
    float rec = 1.f - prob;
    auto ldb = [rec]__device__(T in, curandState_t* s) { return in * (uint32_t)(rec + curand_uniform(s)) }; //It makes no difference, if cast is to int or unsigned: cvt.rzi.s32.f32 and cvt.rzi.u32.f32 are generated respectivly
    transform<T, decltype(ldb), decltype(init), curandState_t>(in, n, ldb, init);
}

//=====================================================
//==================|HIDDEN LAYER|=====================
//=====================================================

template<typename T>
void relu(T* in, uint32_t n, uint32_t b, uint32_t g) { //in[i] = f(in[i])
    constexpr auto ldb = []__device__(T in) { return in > (T)0 ? in : (T)0; };
    transform<T, decltype(ldb)><<<g, b>>>(in, n, ldb);
}

template<typename T>
void relu_deriv(T* in, uint32_t n, uint32_t g, uint32_t b) {//in[i] = f'(f^-1(in[i]))
    constexpr auto ldb = []__device__(T in) { return in > (T)0 ? (T)1 : (T)0; };
    transform<T, decltype(ldb)><<<g, b>>>(in, n, ldb);
}

template<typename T, bool negate>
void relu_deriv_mul(T* in, T* out, uint32_t n, uint32_t g, uint32_t b) { //out[i] *= f'(f^-1(in[i]))
    constexpr auto ldb   = []__device__(T o, T i) { return in > (T)0 ? (T)o  : (T)0; };
    constexpr auto ldb_n = []__device__(T o, T i) { return in > (T)0 ? (T)-o : (T)0; };
    
    if constexpr (negate)
        transform<T, decltype(ldb  )><<<g, b>>>(out, in, n, ldb  );
    else
        transform<T, decltype(ldb_n)><<<g, b>>>(out, in, n, ldb_n);
}

template<typename T>
void sigmoid(T* in, uint32_t n, uint32_t g, uint32_t b) { //Blocksize 64/384
    constexpr auto ldb = []__device__(T in) { return (T)1 / ((T)1 + exponential<T>(in)); };
    transform<T, decltype(ldb)><<<g, b>>>(in, n, ldb);
}

template<typename T>
void sigmoid_deriv(T* in, uint32_t n, uint32_t g, uint32_t b) {
    constexpr auto ldb = []__device__(T in) { return in * ((T)1 - in); }
    transform<T, decltype(ldb)><<<g, b>>>(in, n, ldb);
}

template<typename T, bool negate>
void sigmoid_deriv_mul(T* in, T* out, uint32_t n, uint32_t g, uint32_t b) { //out[i] *= f'(f^-1(in[i]))
    constexpr auto ldb   = []__device__(T o, T i) { return o * (in * ((T)1 - in)); };
    constexpr auto ldb_n = []__device__(T o, T i) { return o * (in * (in - (T)1)); };
    
    if constexpr (negate)
        transform<T, decltype(ldb_n)><<<g, b>>>(out, in, n, ldb_n);
    else
        transform<T, decltype(ldb  )><<<g, b>>>(out, in, n, ldb  );
}

template<typename T>
void softplus(T* in, uint32_t n, uint32_t g, uint32_t b) {
    constexpr auto ldb = []__device__(T in) { return logarithm<T>((T)1 + exponential<T>(in)); };
    transform<T, decltype(ldb)><<<g, b>>>(in, n, ldb);
}

template<typename T>
void softplus_deriv(T* in, uint32_t n, uint32_t g, uint32_t b) {
    constexpr auto ldb = []__device__(T in) { T e = exponential<T>(in); return (e - (T)1) / e; };
    transform<T, decltype(ldb)><<<g, b>>>(in, n, ldb);
}

template<typename T, bool negate>
void softplus_deriv_mul(T* in, T* out, uint32_t n, uint32_t g, uint32_t b) { //out[i] *= f'(f^-1(in[i]))
    constexpr auto ldb   = []__device__(T o, T i) { T e = exponential<T>(i); return o * (e - (T)1) / e; };
    constexpr auto ldb_n = []__device__(T o, T i) { T e = exponential<T>(i); return o * ((T)1 - e) / e; };
    
    if constexpr (negate)
        transform<T, decltype(ldb_n)><<<g, b>>>(out, in, n, ldb_n);
    else    
        transform<T, decltype(ldb  )><<<g, b>>>(out, in, n, ldb  );
}

//===================================================
//==================|LAST LAYER|=====================
//===================================================
template<typename T>
void softmax(T* in, uint32_t n, uint32_t g, uint32_t b) {
    T acc;
    auto ldb = []__device__(T in) { return exponential<T>(in); };

    transform_reduce<T, decltype(ldb), DIVISIBILITY::UNKNOWN, DIVISIBILITY::UNKNOWN, true><<<g, b>>>(in, &acc, n, ldb);
    acc = (T)1 / acc;
    transform<<<g, b>>>(in, n, [acc]__device__(T in) { return in * acc; });
}

template<typename T>
T cross_entropy_loss(T* in, T* expected, uint32_t n, uint32_t g, uint32_t b) {
    T* ret;
    cudaMalloc(&ret, sizeof(T));
    cudaMemset(ret, 0, sizeof(T));

    auto ldb = []__device__(T o, T expec) { return expec * logarithm<float>(o); };
    transform_reduce<T, decltype(ldb), DIVISIBILITY::UNKNOWN, DIVISIBILITY::UNKNOWN, false><<<g, b>>>(in, expected, ret, n, ldb);
    
    T ret_;
    cudaMemcpy(&ret_, ret, sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(ret);
    return ret_;
}

/*
Computes delta for every neuron in the output layer
Assumes that all neurons were activated using softmax and that expected has a sum of one. The loss function is cross entropy
\frac{\partial L}{\partial x_i} &= -\sum_{k=1}^n ŷ_k \frac{\partial}{\partial x_i}ln(y_k)\\
&= -\sum_{k=1}^n \frac{ŷ_k}{y_k} \frac{\partial y_k}{\partial x_i}\\
&= -\frac{ŷ_i}{y_i}(y_i(1-y_i))-\sum_{k\neq i} \frac{ŷ_k}{y_k}(y_k(-y_i))\\
&= y_iŷ_i-ŷ_i + y_i\sum{k\neq i} ŷ_k\\
&= y_iŷ_i-ŷ_i + y_i(1-ŷ_i)\\
&= y_i - ŷ_i 
where L is loss(cross entropy), y_i is output of neuron, x_i is before activation(softmax) and ŷ_i is the expected value

@param in: Pointer to output of softmax
@param expected: Expected output
@param n: Number of neurons in output layer
@param g,b: Launch parameters
*/
template<typename T>
void softmax_cross_entropy_deriv(T* in, T* expected, uint32_t n, uint32_t g, uint32_t b) {
    auto ldb = []__device__(T o, T expec) { return o - expec; };
    transform<T, decltype(ldb)>(in, expected, n, f);
}

//==================================================
//==================|OPTIMIZER|=====================
//==================================================
template<typename T>
__global__ void sgd(T* w, T* delta, T* in, T scalar, uint32_t w_y, uint32_t w_x, uint32_t batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < w_y * w_x * batch_size) {
        int b = idx % batch_size;
        idx /= batch_size;
        int y = idx % w_y;
        int x = idx / w_y;

        atomicAdd(weight[y + x * w_y], scalar * in[b * w_x + x] * delta[b * w_y + y]]));
    } //index of weight is idx, compiler will optimize it away
}

template<typename T>
__global__ void adam(T* w, T* delta, T* in, T* mom1, T* mom2, T neg_lr, T b1, T b2, T e, uint32_t w_y, uint32_t w_x, uint32_t batch_size, uint32_t t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < w_y * w_x * batch_size) {
        int b = idx % batch_size;
        idx /= batch_size;
        int y = idx % w_y;
        int x = idx / w_y;

        T grad = in[b * w_x + x] * delta[b * w_y + y]] / batch_size;
        T tmp1 = mom1[y + x * w_y];
        T tmp2 = mom2[y + x * w_y];
        tmp1 = (tmp1 * b1) + ((1 - b1) * grad); 
        tmp2 = (tmp2 * b1) + ((1 - b1) * grad * grad);
        mom1[y + x * w_y] = tmp1;
        mom2[y + x * w_y] = tmp2;
        
        tmp1 /= (1 - pow(b1, t));
        tmp2 /= (1 - pow(b2, t));

        atomicAdd(weight[y + x * w_y], neg_lr * tmp1 / sqrt(tmp2 + e));
    } //index of weight is idx, compiler will optimize it away
}

template<typename T>
void sgd(T* w, T* delta, T* in, T learning_factor, uint32_t w_y, uint32_t w_x, uint32_t batch_size) {
    uint32_t s = w_x * w_y * batch_size;
    sgd<T><<<(s+31u)/32u , 32)>>>(w, delta, in, -learning_factor/batch_size, w_y, w_x, batch_size)
}

template<typename T>
void adam(T* w, T* delta, T* in, T* mom1, T* mom2, T neg_lr, T b1, T b2, T e, uint32_t w_y, uint32_t w_x, uint32_t batch_size, uint32_t t) {
    uint32_t s = w_x * w_y * batch_size;
    adam<T><<<(s+31u)/32u, 32>>>(w, delta, in, mom1, mom2, neg_lr, b1, b2, e, w_y, w_x, batch_size, t);
}
//=====================================================================================================
//==========================OLD BENCHMARKS======================
#if 0
void BENCHMARK(float* in, float* out, uint32_t N, int b, int g, float* time, int& best_size) {
    float t;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    gridReduce<float, DIVISIBILITY::UNKNOWN, DIVISIBILITY::UNKNOWN> << <g, b >> > (in, out, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t, start, stop);
    CHECK_CUDA_ERROR();
    if (t < *time) {
        *time = t;
        best_size = (g << 16) + b;
    }
}

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
T blockReduceSum0(T val) {//2 Warp reduces, Best when Blocksize is 1024
    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x & 0b11111;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < (blockSize >> 5)) ? shared[lane] : 0;

    if (wid == 0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

template<typename T, int blockSize>
__inline__ __device__
T blockReduceSum1(T val, T* sdata) {//1 warp reduce + Manual reduce
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
void blockReduceSum2(T val, T* out) {//1 warp reduce + Atomics
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
            sum = blockReduceSum0<T, blockSize>(sum);
        }
        else if constexpr (red_Algo == 1) {
            sdata[threadIdx.x] = sum;
            __syncthreads();
            sum = blockReduceSum1<T, blockSize>(sum, sdata);
        }
        else if constexpr (red_Algo == 2) {
            blockReduceSum2<T>(sum, out);
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

#define ALGO(mul_Algo, atomic, red_Algo) (((red_Algo&((1<<15)-1))<<16)^((mul_Algo&((1<<15)-1))<<1)^(atomic&0b1))
#define TIME(mul_Algo, atomic, red_Algo, b_const);                                                 \
    cudaEventRecord(start, 0);                                                                     \
    gridReduce<float, b_const, mul_Algo, atomic, red_Algo><<<g, b_const>>>(mem, mem+N, N, nIsPow2);\
    cudaDeviceSynchronize();                                                                       \
    cudaEventRecord(stop, 0);                                                                      \
    cudaEventSynchronize(stop);                                                                    \
    cudaEventElapsedTime(&time, start, stop);                                                      \
    CHECK_CUDA_ERROR();                                                                            \
    if(!atomic)                                                                                    \
        time += global_timing[min(g, ((N+31)/32)&(~(31)))];                                        \
    if(time < global_timing[N]){                                                                   \
        global_timing[N] = time;                                                                   \
        global_blocks[N] = b_const;                                                                \
        global_grids[N]  = (int)g;                                                                 \
        global_algos[N]  = ALGO(mul_Algo, atomic, red_Algo);                                       \
    }
#define BENCH_AT_RED(b_const, at, red); \
     TIME(0, at, red, b_const);         \
     TIME(1, at, red, b_const);         \
     TIME(2, at, red, b_const);
#define BENCH_AT(b_const, at);                   \
    BENCH_AT_RED(b_const, at, 0);                \
    if(nIsPow2) { BENCH_AT_RED(b_const, at, 1); }\
    BENCH_AT_RED(b_const, at, 2);
#define BENCH(b_const); BENCH_AT(b_const, true); BENCH_AT(b_const, false);
#define BENCH_ALL();\
    BENCH(32);  \
    BENCH(64);  \
    BENCH(96);  \
    BENCH(128); \
    BENCH(160); \
    BENCH(192); \
    BENCH(224); \
    BENCH(256); \
    BENCH(288); \
    BENCH(320); \
    BENCH(352); \
    BENCH(384); \
    BENCH(416); \
    BENCH(448); \
    BENCH(480); \
    BENCH(512); \
    BENCH(544); \
    BENCH(576); \
    BENCH(608); \
    BENCH(640); \
    BENCH(672); \
    BENCH(704); \
    BENCH(736); \
    BENCH(768); \
    BENCH(800); \
    BENCH(832); \
    BENCH(864); \
    BENCH(896); \
    BENCH(928); \
    BENCH(960); \
    BENCH(992); \
    BENCH(1024);
void DO_BENCH(int N, uint32_t g, float* global_timing, int* global_blocks, int* global_grids, int* global_algos, float* mem) {
    bool nIsPow2 = !(N & (N - 1));

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    BENCH_ALL();
}

#define B1(bl, m, a);\
    gridReduceLauncher<float, bl, m, a, 0>(in, out, N, nIsPow2, gridSize);\
    gridReduceLauncher<float, bl, m, a, 1>(in, out, N, nIsPow2, gridSize);\
    gridReduceLauncher<float, bl, m, a, 2>(in, out, N, nIsPow2, gridSize);
#define B2(bl, m);\
    B1(bl, m, false);\
    B1(bl, m, true);
#define B3(bl);\
    B2(bl, 0);\
    B2(bl, 1);\
    B2(bl, 2);
#define B4();\
    B3(32);  \
    B3(64);  \
    B3(96);  \
    B3(128); \
    B3(160); \
    B3(192); \
    B3(224); \
    B3(256); \
    B3(288); \
    B3(320); \
    B3(352); \
    B3(384); \
    B3(416); \
    B3(448); \
    B3(480); \
    B3(512); \
    B3(544); \
    B3(576); \
    B3(608); \
    B3(640); \
    B3(672); \
    B3(704); \
    B3(736); \
    B3(768); \
    B3(800); \
    B3(832); \
    B3(864); \
    B3(896); \
    B3(928); \
    B3(960); \
    B3(992); \
    B3(1024);
//========================================================================================
template<typename T> struct __device_builtin__ __builtin_align__(2 * sizeof(T)) var2 { T a, b; };
template<typename T> struct __device_builtin__ __builtin_align__(4 * sizeof(T)) var4 { T a, b, c, d; };
template<typename T> __global__ void b1(T* in, T N, bool nIsPow2) {
    if (nIsPow2) {
        for (uint32_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x; i < N; i += blockDim.x * 2 * gridDim.x) {
            in[i] = exponential<T>(in[i]);
            in[i + blockDim.x] = exponential<T>(in[i + blockDim.x]);

        }
    }
    else {
        for (uint32_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x; i < N; i += blockDim.x * 2 * gridDim.x) {
            in[i] = exponential<T>(in[i]);
            if (i + blockDim.x < N) in[i + blockDim.x] = exponential<T>(in[i + blockDim.x]);
        }
    }
}

template<typename T> __global__ void b2(T* in, T N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N / 2; i += blockDim.x * gridDim.x) {
        var2<T> val = reinterpret_cast<var2<T>*>(in)[i];
        val.a = exponential<T>(val.a);
        val.b = exponential<T>(val.b);
        reinterpret_cast<var2<T>*>(in)[i] = val;
    }
    int i = idx + N / 2 * 2;
    if (i < N)
        in[i] = exponential<T>(in[i]);
}

template<typename T> __global__ void b3(T* in, T N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N / 4; i += blockDim.x * gridDim.x) {
        var4<T> val = reinterpret_cast<var4<T>*>(in)[i];
        val.a = exponential<T>(val.a);
        val.b = exponential<T>(val.b);
        val.c = exponential<T>(val.c);
        val.d = exponential<T>(val.d);
        reinterpret_cast<var4<T>*>(in)[i] = val;
    }
    int i = idx + N / 4 * 4;
    if (i < N)
        in[i] = exponential<T>(in[i]);
}

int BENCHMARK2(float* in, float* out, uint32_t N, int b, int g) {
    float t[3];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    b1<float> << <g, b >> > (in, N, false);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t[0], start, stop);
    CHECK_CUDA_ERROR();

    cudaEventRecord(start, 0);
    b2<float> << <g, b >> > (in, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t[1], start, stop);
    CHECK_CUDA_ERROR();

    cudaEventRecord(start, 0);
    b3<float> << <g, b >> > (in, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t[2], start, stop);
    CHECK_CUDA_ERROR();

    out[0] += t[0];
    out[1] += t[1];
    out[2] += t[2];

    return t[0] < t[1] ? (t[0] < t[2] ? 0 : 2) : (t[1] < t[2] ? 1 : 2);
}

//=============================================================================================
#define typeof decltype
int BENCHMARK(float* in, float* out, uint32_t N, int g, int b, float* time) {
    float t[4];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    auto lambda1 = []__device__(float in) { return exponential<float>(in); };
    auto lambda2 = [out]__device__(float in, int i) { return out[i] * logarithm<float>(in); };

    cudaEventRecord(start, 0);
    transform_reduce<float, typeof(lambda2), DIVISIBILITY::UNKNOWN, DIVISIBILITY::UNKNOWN, false, 0> << <g, b >> > (in, out, N, lambda2);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t[0], start, stop);
    CHECK_CUDA_ERROR();

    cudaEventRecord(start, 0);
    transform_reduce<float, typeof(lambda2), DIVISIBILITY::UNKNOWN, DIVISIBILITY::UNKNOWN, false, 1> << <g, b >> > (in, out, N, lambda2);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t[1], start, stop);
    CHECK_CUDA_ERROR();

    cudaEventRecord(start, 0);
    transform_reduce<float, typeof(lambda2), DIVISIBILITY::UNKNOWN, DIVISIBILITY::UNKNOWN, false, 2> << <g, b >> > (in, out, N, lambda2);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t[2], start, stop);
    CHECK_CUDA_ERROR();

    cudaEventRecord(start, 0);
    transform<float, typeof(lambda1)> << <g, b >> > (in, N, lambda1);
    reduce<float, DIVISIBILITY::UNKNOWN, DIVISIBILITY::UNKNOWN> << <g, b >> > (in, out, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&t[3], start, stop);
    CHECK_CUDA_ERROR();

    time[0] += t[0];
    time[1] += t[1];
    time[2] += t[2];
    time[3] += t[3];

    int m1 = t[0] < t[1] ? 0 : 1;
    int m2 = t[2] < t[3] ? 2 : 3;
    return t[m1] < t[m2] ? m1 : m2;
}
#endif