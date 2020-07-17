//#define THE_VERSION_JULIAN_DID_NOT_SCREW_WITH
#ifdef __NVCC__
#pragma warning( disable : 4514)
#pragma warning( disable : 4711)
#pragma warning( disable : 4710)
#pragma warning( disable : 5039)
#endif

#ifndef DEBUG
#define NDBUG
#endif

#define _LARGEFILE_SOURCE
#define _FILE_OFFSET_BITS 64

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


#include "util.cpp"
using namespace Random;
using namespace Image;
using namespace DatasetAssemble;

//TODOS (BOTH FILES!!)
//TODO: ADD STREAMING, MULTITHREADING AND AVX
//TODO: Block all kernel calls
//TODO: Check initialization (Globals)
//TODO: CONVOLUTIONS AND TILING DESIGN
//TODO: Unified allocator
//TODO: TYPE HANDLING
//TODO: Different input, output and augmentation type for Dataset
//TODO: ADD SUPPORT FOR NON_IMAGES
//TODO: SWITCH WORKER THREADS TO VALIDATION
//TODO: REGULARIZATION AND NORMALIZATION
//TODO: MEMORY MAP FILES

//=========================================================
//==================|HELPER FUNCITONS|=====================
//=========================================================

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

#if defined(GCC) or defined(CLANG)
#define ALIGN_PREF
#define ALGIN_SUFF __attribute__(align(32))
#else
#define ALIGN_PREF __declspec(align(32))
#define ALGIN_SUFF 
#endif

#define LAUNCH_PARAM(N) (int)(1. / ((10. / ((double)(1 << 13)) + 32. / ((double)(N))))), 32

template<typename T> T min(T a, T b) { if (a < b) return a; else return b; }
template<typename T> abs(T x) { return (x >= (T)0) ? x : -x; }
/*union {
        float f;
        int32_t i;
    } b = { .f = x };
    b.i &= ~(1 << 31);

    return b.f;*/
template<typename T> void align_val(T& in, T al) { in /= al; in *= al; }

//================================================
//==================|GLOBALS|=====================
//================================================

cublasHandle_t cublas_handle;
cudaStream_t   data_stream;

//======================================================
//==================|DEEP LEARNING|=====================
//======================================================

enum Activation { RELU = 0, SIGMOID = 1, SOFTMAX = 2, SOFTPLUS = 3 };
enum Optimizer_Type { SGD, ADAM, DEMON_ADAM };
enum CublasMode { OVERWRITE = 0, ADD = 1 };                                //Don't change values!!
enum WORKER_STATUS { VALIDATION = -3, TRAINING = -2, IDLE = -1, STOP = 0}; //Don't change values!!

struct OptVariables {
    Optimizer_Type opt_t;
    float learningRate;
    float beta1;
    float beta2;
    float betaInit;

    uint32_t t;
    uint32_t T;
};

struct AugmentationInfo2D {
    float aspect_in;             //Input  vector is understood as an image with channels_in channels and aspect ration aspect_in
    float aspect_out;            //Output vector is understood as an image of shape (sqrt(sample_size[1]/channels_out * (1/aspect_out)), sqrt(sample_size[1]/channels_out * (aspect_out)), channels_out)
    uint32_t channels_in;        //The input vector is understood as an Image of the shape (dim_x, dim_y, channels). Augmentations are applied on this image
    uint32_t channels_out;       //The input vector is understood as an Image of the shape (dim_x, dim_y, channels). Augmentations are applied on this image

    //The "if not zero" part is redundant and only used for understanding, that incase of 0 nothing happens.
    uint32_t blur_range;         //If not zero, applies gaussian blur with this distance 
    bool  flip;                  //If true    , the Image has a 50% chance of getting mirrored horitontaly
    float rotate;                //If not zero, the Image is rotated by an random angle with mean 0 and this standart deviation
    float random_noise;          //If not zero, to every channel a random number with mean 0 and this standart deviation stored
    float random_dropout;        //If not zero, this stores the probability for each pixel to individually become black
    float random_saturation;     //If not zero, the saturation of each pixel is multiplies by a constant random number with mean 0 and this standart deviation
    float random_brightness;     //If not zero, the brightness of each pixel is multiplies by a constant random number with mean 0 and this standart deviation
                                 
    Offset2D<float> crop;        //Crops a random part of shape (crop[0]*dim_x, crop[1]*dim_y, channels) from the input and output. <0 indicates to skip this step
    Offset2D<int32_t> resize[2]; //Resizes input to shape (resize[0], channels_in) and output to (resize[1], channels_out). resize[0].x=-1 indicates to skip this step
};

template<typename T>
class DatasetHandler {
    /*
        This holds and processes datasets.

        To initialize a dataset, you need one file for output and one for input.
        The data is read into ram (where it is cached) and augmented by the cpu 
        (augmentation are not cached, as to not reuse them to prevent overfitting). 
        Therefore, the dataset needs no preprocessing and samples can have a 
        different shapes than the input of the networks, thus it can be universal.
        Afterwards, the augmentated samples are batched and sent to a vram tile,
        ready to be used by the gpu to train the networks.
    */
    /*
        Low level description of how this class operates
        ------------------------------------------------

        Files (fd_in and fd_out):
         - Start with the following header |type_hash<T>() -> uint32_t|sample_size -> uint32_t|
         - After that raw input data that will be fit to networks (for example no .jpg file but rgb data -> no headers, no encoding, no checksums, ...)

        Loading and processing schemes:
         - An as big as possible part of the dataset is loaded into cpu_tiles from fd_in and fd_out. This is done sequencially so that training data is loaded first
            - The data in cpu_tiles is persistent and not getting reloaded (as there is simply no reason to)
         - Whenever there is an element in ready_cpu that is false, a new sample is loaded into aug_tiles the following way:
            - We generate batch_size random indices into the dataset and load one sample at a time into the tile and augment it
            - If the need samples are in cpu_tiles, we take the data from there. Otherwise we need to go to main memory
            - This way we achieve a maximum caching effect while haveing a completly random shuffle (important according to https://arxiv.org/pdf/1810.04509.pdf)
         - Whenever advance is called, we load a new batch from aug_tiles to cur_gpu and afterwards advance cur_gpu and return those

    */
    /*
        Usage
        -----

        1.: Constructor
        2.: "set_augmentation"
        3.: "set_batch_size"
        4.: "start_workers"
        5.: Loop "advance". Step 2+3+4 can be repeated anytime in between
    */
private:
    int fd_in, fd_out;                  //File descritors of files holding dataset
    uint32_t sample_size[4];            //How many number are contained in an input/output sample in the dataset ([0],[1]) and in augmentated samples ([2],[3])

    bool fit_in_ram;                    //True if the whole dataset fits into RAM
    uint32_t train_samples;             //Number of all training samples
    uint32_t validation_samples;        //Number of all validation samples
    uint32_t batch_size;                //Number of samples to return to neural network per call. Has to be multiple of 8 (for AVX)

    uint32_t tile_samples_cpu;          //Number of samples per tile.
    uint32_t augmentation_batches_cpu;  //Number of augmented batches stored in ram
    uint32_t augmentation_batches_gpu;  //Number of augmented batches stored in vram

    AugmentationInfo2D agi;             //How will the dataset be augmented
    uint32_t num_workers;               //Number of threads launched for augmentation
    std::thread* workers;               //Handle to threads launched for augmentation   (nullptr, when no workers are started)
    std::atomic<int32_t> worker_status; //-3=validation, -2=training, -1=idle, >=0: workers should exit, this is the number of threads that already did

    T* cpu_tiles[2];                    //Start of tiles in ram that hold raw dataset ( ={input, output} )
    T* aug_tiles[2];                    //Start of tiles in ram that hold augmented batches of dataset ( ={input, output} )
    uint32_t aug_padding[2];            //Padding to add to aug tiles to be able to accomodate copy of size sample_size[0/1] and intermediates
    std::atomic<bool>* ready_cpu;       //Stores for each element of aug_tiles, whether it contains a new unused batch
    
    T* gpu_tiles[2];                    //Holds start of tiles (gpu) = {input, output}  
    T* cur_gpu[2];                      //Holds last used element in tiles (gpu) = {input, output}. If gpu_tiles was reloaded, cur_gpu[0]=nullptr
    cudaEvent_t* gpu_sync;              //Events to tell when memory-transfer to gpu are finished

#define SAMPLE_VAL(s);                                                                       \
    do() {                                                                                   \
        T* aug_in  = aug_tiles[0][(b + s) * (batch_size * sample_size[2] + aug_padding[0])]; \
        T* aug_out = aug_tiles[1][(b + s) * (batch_size * sample_size[3] + aug_padding[1])]; \
                                                                                             \
        /*Copy to augmentation tile*/                                                        \
        if (fit_in_ram || ((in_mask >> (4 * s)) & 0b1) { /*In Ram tile*/                     \
            memcpy(aug_in , cpu_tiles[0][off[s    ]], sample_sizes[0]);                      \
            memcpy(aug_out, cpu_tiles[1][off[s + 8]], sample_sizes[1]);                      \
        }                                                                                    \
        else { /*Go to disk*/                                                                \
            lseek(fd_in , sizeof(T) * off[s    ] + 2 * sizeof(uin32_t), SEEK_SET);           \
            lseek(fd_out, sizeof(T) * off[s + 8] + 2 * sizeof(uin32_t), SEEK_SET);           \
            read(fd_in , aug_in , sample_size[0]);                                           \
            read(fd_out, aug_out, sample_size[1]);                                           \
        }                                                                                    \
                                                                                             \
        /*Resize to input size*/                                                             \
        if (agi.resize[0] != -1) {                                                           \
            resize<T>(aug_in ,  in_shape, resize[0]);                                        \
            resize<T>(aug_out, out_shape, resize[1]);                                        \
        }                                                                                    \
    } while(0);
#define SAMPLE_TRAIN(s);                                                                     \
    do() {                                                                                   \
        T* aug_in  = aug_tiles[0][(b + s) * (batch_size * sample_size[2] + aug_padding[0])]; \
        T* aug_out = aug_tiles[1][(b + s) * (batch_size * sample_size[3] + aug_padding[1])]; \
                                                                                             \
        /*Copy to augmentation tile*/                                                        \
        if (fit_in_ram || (in_mask >> (4 * s)) & 0b1) { /*In Ram tile*/                      \
            memcpy(aug_in , cpu_tiles[0][off[s    ]], sample_sizes[0]);                      \
            memcpy(aug_out, cpu_tiles[1][off[s + 8]], sample_sizes[1]);                      \
        }                                                                                    \
        else { /*Go to disk*/                                                                \
            lseek(fd_in , sizeof(T) * off[s    ] + 2 * sizeof(uin32_t), SEEK_SET);           \
            lseek(fd_out, sizeof(T) * off[s + 8] + 2 * sizeof(uin32_t), SEEK_SET);           \
            read(fd_in , aug_in , sample_size[0]);                                           \
            read(fd_out, aug_out, sample_size[1]);                                           \
        }                                                                                    \
                                                                                             \
        /*Augment*/                                                                          \
        if(agi.blur_range != 0.f)                                                            \
            blur<T>(aug_in, in_shape, abs(rand_normal(agi.blur_range));                      \
        if (agi.flip && rand_prob(0.5f)) { /*50% chance*/                                    \
            flip<T>(aug_in ,  in_shape);                                                     \
            flip<T>(aug_out, out_shape);                                                     \
        }                                                                                    \
        if (agi.random_noise != 0.f) {                                                       \
            random_noise<T>(aug_in, in_shape, agi.random_noise);                             \
        }                                                                                    \
        if (agi.random_dropout != 0.f) {                                                     \
            random_dropout<T>(aug_in, in_shape, agi.random_dropout);                         \
        }                                                                                    \
        if (agi.random_saturation != 0.f) {                                                  \
            random_saturation<T>(aug_in, in_shape, rand_normal(agi.random_saturation));      \
        }                                                                                    \
        if (agi.random_brightness != 0.f) {                                                  \
            random_brightness<T>(aug_in, in_shape, rand_normal(agi.random_brightness));      \
        }                                                                                    \
        if (agi.rotate != 0.f) {                                                             \
            float rot_deg = rand_normal(agi.rotate);                                         \
            rotate<T>(aug_in ,  in_shape, rot_deg);                                          \
            rotate<T>(aug_out, out_shape, rot_deg);                                          \
        }                                                                                    \
        if (agi.crop[0] >= 0.f) {                                                            \
            in_shape  *= agi.crop;                                                           \
            out_shape *= agi.crop;                                                           \
                                                                                             \
            Offset2D<uint32_t> pos  = {rand_float(1.f), rand_float(1.f) };                   \
            Offset2D<uint32_t> map1 =  in_shape_.getOffset2D() -  in_shape.getOffset2D();    \
            Offset2D<uint32_t> map2 = out_shape_.getOffset2D() - out_shape.getOffset2D();    \
                                                                                             \
            crop<T>(aug_in ,  in_shape_, pos * map1,  in_shape);                             \
            crop<T>(aug_out, out_shape_, pos * map2, out_shape);                             \
        }                                                                                    \
        if (agi.resize[0] != -1) {                                                           \
            resize<T>(aug_in ,  in_shape, resize[0]);                                        \
            resize<T>(aug_out, out_shape, resize[1]);                                        \
        }                                                                                    \
        in_shape =  in_shape_;                                                               \
        out_shape = out_shape_;                                                              \
    } while(0);

    void worker_function(uint64_t rand_init) {
        //1.: Generate a deterministic random key
        Key my_key;
        uint64_t r = (763808623281539ull * rand_init) ^ (2009741990ull * rand_init);
        avx_xorshift128plus_init(r ^ (r << 5) ^ (r >> 19), r ^ (r << 32) ^ (r >> 32), my_key);

        //2.: Set AVX constants
        __m256  rand_normalize1      = _mm256_set1_ps((float)train_samples / (float)(1 << 32));
        __m256  rand_adder1          = _mm256_set1_ps(train_samples >> 1);

        __m256  rand_normalize2      = _mm256_set1_ps((float)validation_samples / (float)(1 << 32));
        __m256  rand_adder2          = _mm256_set1_ps(train_samples + (validation_samples >> 1));

        __m256i tile_samples_cpu_vec = _mm256_set1_epi32(tile_samples_cpu);
        __m256i sample_size_0        = _mm256_set1_epi32(sample_size[0]);
        __m256i sample_size_1        = _mm256_set1_epi32(sample_size[1]);

        ALIGN_PREF uint32_t off[16] ALIGN_SUFF;                                                       //Store offset of T's to start of sample

        //3.: Calculate sizes for augmentation
        Image_Shape in_shape (sample_size[0], agi.aspect_in , agi.channels_in );                      // in_shape store the size of the  input sample at the moment during the calculations
        Image_Shape out_shape(sample_size[1], agi.aspect_out, agi.channels_out);                      //out_shape store the size of the output sample at the moment during the calculations

        Image_Shape  in_shape_ =  in_shape;                                                           //This is the backup value to restore  in_shape after calculations are finished
        Image_Shape out_shape_ = out_shape;                                                           //This is the backup value to restore out_shape after calculations are finished

        //4.: Work loop
        uint32_t stat;
        while (stat = worker_status.load() < WORKER_STATUS::STOP) {                                   //Break if exit signal is reached
            if (stat == WORKER_STATUS::TRAINING) {                                                                         //Testing
                for (uint32_t b = 0; b != augmentation_batches_cpu; b++) {                            //Go through all batches in augmentation tile
                    if (!ready_cpu[b].load()) {                                                       //If a batch is not ready, work on it
                        //Reload batch b in augmentation tile                                       
                        for (uint32_t sample = 0; sample != batch_size; sample += 8) {                //Loop is unrolled 8 times
                            __m256 random = __m256_cvtepi32_ps(avx_xorshift128plus(my_key));          //[-2^31, 2^31]
                            random = _mm256_add_ps(_mm256_mul_ps(gen, rand_normalize1), rand_adder1); //[0, train_samples]

                            __m256   in_tile = _mm256_cmpgt_epi32(tile_samples_cpu_vec, random);      //True, if samples is in ram tile
                            uint32_t in_mask = _mm256_movemask_epi8(in_tile);                         //Most significant bits

                            __m256 ind1 = _mm256_mullo_epi32(_mm256_cvtps_epi32(gen), sample_size_0); //Compute input  indices
                            __m256 ind2 = _mm256_mullo_epi32(_mm256_cvtps_epi32(gen), sample_size_1); //Compute output indices
                            _mm256_store_si256((__m256i) & off[0], ind1);
                            _mm256_store_si256((__m256i) & off[8], ind2);

                            //Load and process all the samples
                            SAMPLE_TRAIN(0);
                            SAMPLE_TRAIN(1);
                            SAMPLE_TRAIN(2);
                            SAMPLE_TRAIN(3);
                            SAMPLE_TRAIN(4);
                            SAMPLE_TRAIN(5);
                            SAMPLE_TRAIN(6);
                            SAMPLE_TRAIN(7);
                        }
                        //Now, the whole batch is loaded. Currently, operations on the whole batch are not supported

                        //Mark batch as ready
                        ready_cpu[b].store(true);
                        break;                                                                        //When finished, check again for exit signal and start at the beginning
                    }
                }
            }
            else if (stat == WORKER_STATUS::VALIDATION) {
                for (uint32_t b = 0; b != augmentation_batches_cpu; b++) {                            //Go through all batches in augmentation tile
                    if (!ready_cpu[b].load()) {                                                       //If a batch is not ready, work on it
                        //Reload batch b in augmentation tile                                       
                        for (uint32_t sample = 0; sample != batch_size; sample += 8) {                //Loop is unrolled 8 times
                            __m256 random = __m256_cvtepi32_ps(avx_xorshift128plus(my_key));          //[-2^31, 2^31]
                            random = _mm256_add_ps(_mm256_mul_ps(gen, rand_normalize2), rand_adder2); //[train_samples, train_samples + validation_samples]

                            __m256 in_tile;
                            uint32_t in_mask;
                            if (!fit_in_ram) {                                                        //Theses values are only needed, when part of dataset is still on disk
                                __m256   in_tile = _mm256_cmpgt_epi32(tile_samples_cpu_vec, random);  //True, if samples is in ram tile
                                uint32_t in_mask = _mm256_movemask_epi8(in_tile);                     //Most significant bits
                            }

                            __m256 ind1 = _mm256_mullo_epi32(_mm256_cvtps_epi32(gen), sample_size_0); //Compute input  indices
                            __m256 ind2 = _mm256_mullo_epi32(_mm256_cvtps_epi32(gen), sample_size_1); //Compute output indices
                            _mm256_store_si256((__m256i) & off[0], ind1);
                            _mm256_store_si256((__m256i) & off[8], ind2);

                            //Load and process all the samples
                            SAMPLE_VAL(0);
                            SAMPLE_VAL(1);
                            SAMPLE_VAL(2);
                            SAMPLE_VAL(3);
                            SAMPLE_VAL(4);
                            SAMPLE_VAL(5);
                            SAMPLE_VAL(6);
                            SAMPLE_VAL(7);
                        }
                        //Now, the whole batch is loaded. Currently, operations on the whole batch are not supported

                        //Mark batch as ready
                        ready_cpu[b].store(true);
                        break;                                                                        //When finished, check again for exit signal and start at the beginning
                    }
                }
            }
        }
        worker_status++;
    }
    inline void load_batch_to(T* in, T* out) {                                 //in and out have to point to gpu_tiles[0] and gpu_tiles[1], respectifly
        assert(workers != nullptr);                                            //If there are no workers active, this could be an infinite loop
        
        do() {
            for (uint32_t b = 0; b != augmentation_batches_cpu; b++) {         //Go through all batches in augmentation tile
                if (ready_cpu[b].load()) {                                     //Fresh batch was found
                    //Load batch
                    cudaMemcpyAsync(in , aug_tiles[0][b * (batch_size * sample_size[2] + aug_padding[0])], batch_size * sample_size[2] * sizeof(T), cudaMemcpyHostToDevice, data_stream);
                    cudaMemcpyAsync(out, aug_tiles[1][b * (batch_size * sample_size[3] + aug_padding[1])], batch_size * sample_size[3] * sizeof(T), cudaMemcpyHostToDevice, data_stream);
                    
                    //Add ability to synchronize
                    cudaEvent_t sync = gpu_sync[(in - gpu_tiles[0]) / (batch_size * sample_size[2])];
                    cudaEventRecord(sync, data_stream);
                    
                    //Tell workers to generate new batch in augmentation tile after the datatransfer finished
                    std::thread([&sync, &b] {cudaEventSynchronize(sync);  ready_cpu[b].store(false)};).detach();
                    return;                                                    //Reloaded tile
                }
            }
        } while (true);                                                        //No fresh batch was found (worker not finished). Retry.
    }

public:
    /*
        Allocates and fills cpu_tiles. Sets all related member variables.

        @param in, out: File name for files that contain input and output for neural network in sequencial order
        @param train_split: Which percentile of samples should be used for training (has to be between 0.f and 1.f)
        @param batch_size_: The batch size
        @param mb_cpu: How many megabytes of RAM this class is allowed to use
    */
    DatasetHandler(char* in, char* out, float train_split, uint32_t mb_cpu) {
        DatasetHandler(open(in, O_RDONLY), open(out, O_RDONLY), train_split, mb_cpu);
    }
    DatasetHandler(int fd_in_, int fd_out_, float train_split, uint32_t mb_cpu)
        :fd_in(fd_in_), fd_out(fd_out_), 
        batch_size(0u), 
        augmentation_batches_cpu(0u), augmentation_batches_gpu(0u),
        num_workers(0u), workers(nullptr), worker_status(-1),
        ready_cpu(nullptr),
        gpu_sync(nullptr)
    {
        //-1.: Finish initialization
        sample_size[2] = sample_size[3] = 0u;
        aug_tiles = { nullptr, nullptr }; aug_padding = { 0u, 0u };
        gp_tiles = { nullptr, nullptr }; cur_gpu = { nullptr, nullptr };

        //0.: Check parameters
        assert(0.f <= train_split && train_split <= 1.0f && mb_cpu != 0 && fd_in_ != -1 && fd_out_ != -1);

        //1.: File headers
        uint32_t types[2];
        read(fd_in , types[0], sizeof(uint32_t));
        read(fd_out, types[1], sizeof(uint32_t));
        assert(types[0] == types[1] && types[1] = type_hash<T>());                       //Make sure the type of dataset and this class match up
        
        read(fd_in , sample_size[0], sizeof(uint32_t));
        read(fd_out, sample_size[1], sizeof(uint32_t));

        //2.: Size of input and output file
        uint64_t bytes_in, bytes_out;
        struct stat st;
        fstat(fd_in, &st);
        bytes_in = st.st_size - 2 * sizeof(uint32_t);                                    //Get rid of header
        fstat(fd_out, &st);                                            
        bytes_out = st.st_size - 2 * sizeof(uint32_t);                                   //Get rid of header
                                                                              
        //3.: Number of samples
        assert((bytes_in % sample_size[0] == 0) && (bytes_out % sample_size[1] == 0));
        uint32_t samples = bytes_in / sample_size[0];
        uint32_t samples_out = bytes_out / sample_size[1];
        assert(samples == samples_out);

        //4.: Split samples in training and validation
        train_samples = ((uint32_t)(samples * train_split)) & ~(uint32_t)0b1;            //Alignment of 2
        validation_samples = samples - train_samples;

        //5.: Check how many samples fit in a ram tile to fulfill the memory requirements
        uint32_t bytes_per_sample = (sample_size[0] + sample_size[1]) * sizeof(T);
        tile_samples_cpu = (mb_cpu * 1024ull * 1024ull) / bytes_per_sample;
        
        //6.: Does the whole dataset fit into RAM?
        if (tile_samples_cpu >= samples) {
            fit_in_ram = true;
            tile_samples_cpu = samples;                                                  //Do not allocate more than needed
            printf("The whole dataset fits into ram\n");
        }
        else {
            printf("Dataset does not fit into ram. This might degrade performance. Additional %d mb are needed\n", ((samples - tile_samples_cpu) * (uint64_t)bytes_per_sample) / (1024ull * 1024ull));
            fit_in_ram = false:
        }
        assert(tile_samples_cpu != 0);

        //8.: Allocate memory
        uint64_t aloc_size = tile_samples_cpu * (bytes_per_sample);
        printf("Trying to allocate %d mb: ", aloc_size / (1024 * 1024));
        T *cpu_mem = (T*)malloc(aloc_size);
        if (cpu_mem == NULL) {
            printf("Failure");
            return;
        }
        else
            printf("Success");

        cpu_tiles[0] = cpu_mem;
        cpu_tiles[1] = cpu_tiles[0] + sizeof(T) * sample_size[0] * tile_samples_cpu;

        //9.: Fill memory
        read(fd_in , cpu_tiles[0], sizeof(T) * sample_size[0] * tile_samples_cpu);
        read(fd_out, cpu_tiles[1], sizeof(T) * sample_size[1] * tile_samples_cpu);

        //10.: Debugging
#ifdef DEBUG
        printf("sample_size[0]: %u|sample_size[1]: %u|train_samples: %u|validation_samples: %u|tile_samples_cpu\n", 
            sample_size[0], sample_size[1], train_samples, validation_samples, tile_samples_cpu);
#endif
    }
    
    /*
        Sets augmenation and infers size of augmentated samples (sample_size[2] and sample_size[3])
    */
    inline void set_augmentation(AugmentationInfo2D agi_) {
        //0.: Check parameters
        //If the aspect ratio and channels are valid will be checked by worker threads!
        assert(agi_.random_brightness >= 0.f && agi_.random_dropout >= 0.f && agi_.random_dropout <= 1.f && agi_.random_noise >= 0.f && agi_.random_saturation >= 0.f && agi_.rotate >= 0); //Standart deviation has to be >=0, probability is in ]0, 1[.
        assert(agi_.crop.x <= 1 && agi_.crop.y <= 1);
        assert(agi_.resize[0].x == -1 || (agi_.resize[0].x > 0 && agi_.resize[0].y > 0 && agi_.resize[1].x > 0 && agi_.resize[1].y > 0));

        //1.: Set variables
        agi = agi_;

        //2.: Infer augmentation size
        if (agi.resize[0] == -1) {
            if (agi.crop[0] == -1) {
                //No change
                sample_size[2] = sample_size[0];
                sample_size[3] = sample_size[1];
            }
            else {
                //Crop controlls size
                sample_size[2] = sample_size[3] = agi.crop[0] * agi.crop[1];
            }
        }
        else {
            //Resize controls size
            sample_size[2] = sample_size[3] = agi.resize[0] * agi.resize[1];
        }
        aug_padding[0] = max(sample_size[0], sample_size[2]) - sample_size[0];
        aug_padding[1] = max(sample_size[1], sample_size[3]) - sample_size[1];
    }
    inline AugmentationInfo2D get_augmentation() {
        return agi;
    }

    /*
        Allocates aug_tiles and gpu_tiles.
    */
    inline void set_batching(uint32_t b, uint32_t augmentation_batches_cpu_, uint32_t augmentation_batches_gpu_) {
        //0.: Check parameters
        assert((b != 0) && (augmentation_batches_cpu_ > 1) && (augmentation_batches_gpu_ > 1));
        assert(b % 8 == 0);

        //1.: Free previous memory
        if (augmentation_batches_cpu != 0) { //Make sure this funtion is not called for the first time
            cudaFreeHost(aug_tiles[0]);
            cudaFree(gpu_tiles[0]);
            free(ready_cpu);
            free(gpu_sync);
        }

        //2.: Set variables
        batch_size = b;
        augmentation_batches_cpu = augmentation_batches_cpu_;
        augmentation_batches_gpu = augmentation_batches_gpu_;

        //3.: Allocate aug_tiles and ready_cpu
        T* aug_mem;
        cudaMallocHost(&aug_mem, sizeof(T) * ((sample_size[3] + sample_size[4]) * batch_size + aug_padding[0] + aug_padding[1]) * augmentation_batches_cpu);

        aug_tiles[0] = aug_mem;
        aug_tiles[1] = aug_tiles[0] + sizeof(T) * (sample_size[3] * batch_size + aug_padding[0]) * augmentation_batches_cpu);
        
        ready_cpu = (std::atomic<bool>*)malloc(sizeof(std::atomic<bool>) * augmentation_batches_cpu);
        for (uint32_t u = 0; u != augmentation_batches_cpu; u++) {
            ready_cpu[0] = std::atomic<bool>(false);
        }

        //4.: Allocate gpu_tiles and gpu_sync
        T* gpu_mem;
        cudaMalloc(&gpu_mem, sizeof(T) * (sample_size[3] + sample_size[4]) * batch_size * augmentation_batches_gpu);

        gpu_tiles[0] = gpu_mem;
        gpu_tiles[1] = gpu_tiles[0] + sizeof(T) * sample_size[3] * batch_size * augmentation_batches_cpu);

        gpu_sync = (cudaEvent_t*)malloc(sizeof(cudaEvent_t) * augmentation_batches_gpu);
        for (uint32_t u = 0; u != augmentation_batches_gpu; u++) {
            cudaEventCreate(&gpu_sync[u]);
        }
    }
    inline uint32_t get_batch_size() {
        return batch_size;
    }

    /*
        Start the threads that copy and augment data from dataset ram tile to augmentation tile

        @param n: Number of workers
        @param stat: Initialize worker status
    */
    inline void start_workers(uint32_t n, WORKER_STATUS stat = WORKER_STATUS::IDLE) {
        //0.: Check parameters
        assert(n > 0);
        assert(stat < WORKER_STATUS::IDLE);

        //1.: Copy variables
        num_workers = n;

        //2.: Reset stopping signal
        worker_status.store(stat);

        //3.: Start num_workers new worker threads
        workers = (std::thread*)malloc(num_workers * sizeof(std::thread));
        for (uint32_t u = 0; u != num_workers; u++) {
            workers[u] = std::thread(worker_function, u);
            workers[u].detach();
        }
    }
    template<bool training> void setDataInput() {
        //1.: Sent signal to worker threads
        if constexpr (training)
            worker_status = WORKER_STATUS::TRAINING;
        else
            worker_status = WORKER_STATUS::VALIDATION;

        //2.: Reload cpu augmentation tile
        for (uint32_t u = 0; u != augmentation_batches_cpu; u++)
            ready_cpu.store(false);

        //3.: Reload gpu tile
        fill_gpu_tile();
    }
    inline void stop_workers() {
        //0.: Check whether there even are thread running
        assert(worker != nullptr);

        //1.: Set exit signal
        worker_status.store(WORKER_STATUS::STOP);
        
        //2.: Destroy handles
        while (worker_status.load() != num_workers) {}; //Wait until all threads exited
        for (uint32_t u = 0; u != num_workers; u++)
            ~workers[u];

        //3.: Free memory of handles
        free(workers);
        workers = nullptr;
    }

    /*
        Fills complete gpu_tiles with new batches. Set cur_gpu[0]=nullptr

        NOTES: Currently, only one datastream is supported. Multiple streams will improve the performance if each copy is small
    */
    inline void fill_gpu_tile() {
        cur_gpu[0] = nulltpr;                                     //Indicate, that the last used batch is not in memory again

        for (uint32_t b = 0; b != augmentation_batches_gpu; b++)
            load_batch_to(gpu_tiles[0][b * batch_size * sample_size[2]], gpu_tiles[0][b * batch_size * sample_size[2]]);
    }

    template<bool training> void advance(T*& in, T*& out) {
        //0.: Check, if we switched from training to validation or the other way around
        int32_t stat = worker_status.load();
        assert(stat < WORKER_STATUS::STOP);
        
        if constexpr (training) {
            if (stat == WORKER_STATUS::VALIDATION || stat == WORKER_STATUS::IDLE)
                setDataInput<true>();
        }
        else {
            if (stat == WORKER_STATUS::TRAINING || stat == WORKER_STATUS::IDLE)
                setDataInput<false>();
        }

        //1.: Reload last used batch
        if (cur_gpu[0] != nullptr)
            load_batch_to(cur_gpu[0], cur_gpu[1]);

        //2.: Increment cur_gpu
        if (cur_gpu[0] == nullptr) {
            cur_gpu[0] = gpu_tiles[0];
            cur_gpu[1] = gpu_tiles[1];
        }
        else if (cur_gpu[0] != gpu_tiles[0] + batch_size * sample_size[2] * (augmentation_batches_gpu - 1)) { //Whole tile was used
            cur_gpu[0] = gpu_tiles[0];                      //Start at beginning again
            cur_gpu[1] = gpu_tiles[1];                      //Start at beginning again
        }
        else {
            cur_gpu[0] += batch_size * sample_size[2];
            cur_gpu[1] += batch_size * sample_size[3];
        }

        //3.: Set output
        in  = cur_gpu[0];
        out = cur_gpu[1];

        //4.: Synchronize
        cudaEventSynchronize(gpu_sync[(in - gpu_tiles[0]) / (batch_size * sample_size[2])]); //Make sure the copy to in and out are finished
    }

    ~DatasetHandler() {
        if (workers != nullptr)
            stop_workers();

        close(fd_in);
        close(fd_out);

        free(cpu_tiles[0]);
        cudaFreeHost(aug_tiles[0]);
        cudaFree(gpu_tiles[0]);

        free(ready_cpu);
        free(gpu_sync);
    }

    //Old implemetation to load tiles serially
#if 0
    template<bool training> inline void load_tile_cpu() {
        uint32_t bytes_to_read_i = tile_samples_cpu * sizeof(T) * i;
        uint32_t bytes_to_read_o = tile_samples_cpu * sizeof(T) * o;

        uint32_t byte_pos_i = lseek(fd_in, 0, SEEK_CUR);
        uint32_t byte_pos_o = lseek(fd_out, 0, SEEK_CUR);

        if constexpr (training) {
            if (byte_pos_i >= train_samples * sizeof(T) * i) { //File pointer is in validation section
                //Start at begining of training section
                lseek(fd_in, 0, SEEK_SET);
                lseek(fd_out, 0, SEEK_SET);

                read(fd_in, cpu_tiles[0], bytes_to_read_i);
                read(fd_out, cpu_tiles[1], bytes_to_read_o);
            }
            else { //File pointer is already in training section
                uint32_t bytes_left_i = train_samples * sizeof(T) * i - byte_pos_i;
                uint32_t bytes_left_o = train_samples * sizeof(T) * o - byte_pos_o;

                if (bytes_left_i < bytes_to_read_i) { //We have to wrap around
                    read(fd_in, cpu_tiles[0], bytes_left_i);
                    read(fd_out, cpu_tiles[1], bytes_left_o);

                    lseek(fd_in, 0, SEEK_SET);
                    lseek(fd_out, 0, SEEK_SET);

                    read(fd_in, cpu_tiles[0], bytes_to_read_i - bytes_left_i);
                    read(fd_out, cpu_tiles[1], bytes_to_read_o - bytes_left_o);
                }
            }

            cur_cpu[0] = gpu_tiles[0];
            cur_cpu[1] = gpu_tiles[1];
        }
        else {
            if (byte_pos_i < train_samples * sizeof(T) * i) { //File pointer is in testing section
                //Start at begining of validation section
                lseek(fd_in, train_samples * sizeof(T) * i, SEEK_SET);
                lseek(fd_out, train_samples * sizeof(T) * o, SEEK_SET);

                read(fd_in, cpu_tiles[2], bytes_to_read_i);
                read(fd_out, cpu_tiles[3], bytes_to_read_o);
            }
            else { //File pointer is already in validation section
                uint32_t bytes_left_i = (train_samples + validation_samples) * sizeof(T) * i - byte_pos_i;
                uint32_t bytes_left_o = (train_samples + validation_samples) * sizeof(T) * o - byte_pos_o;

                if (bytes_left_i < bytes_to_read_i) { //We have to wrap around
                    read(fd_in, cpu_tiles[2], bytes_left_i);
                    read(fd_out, cpu_tiles[3], bytes_left_o);

                    lseek(fd_in, train_samples * sizeof(T) * i, SEEK_SET);
                    lseek(fd_out, train_samples * sizeof(T) * o, SEEK_SET);

                    read(fd_in, cpu_tiles[2], bytes_to_read_i - bytes_left_i);
                    read(fd_out, cpu_tiles[3], bytes_to_read_o - bytes_left_o);
                }
            }

            cur_cpu[2] = gpu_tiles[2];
            cur_cpu[3] = gpu_tiles[3];
        }

        cpu_sample_reuse_counter = 0;
    }
    template<bool training> inline void load_tile_gpu() { //Loads new gpu tiles from current cpu tiles
        T* from1, * from2;
        if constexpr (training) {
            from1 = cpu_tiles[0];
            from2 = cpu_tiles[1];

            cur_gpu[0] = gpu_tiles[0];
            cur_gpu[1] = gpu_tiles[1];
        }
        else {
            from1 = cpu_tiles[2];
            from2 = cpu_tiles[3];

            cur_gpu[2] = gpu_tiles[2];
            cur_gpu[3] = gpu_tiles[3];
        }
        T* to1 = gpu_tiles[0];
        T* to2 = gpu_tiles[1];
        uint64_t lenght = (uint64_t)tile_samples_gpu * i * sizeof(T);

        //1.: Copy elements over
        if (!agi.random_order) {
            cudaMemcpyAsync(to1, from1, lenght, cudaMemcpyHostToDevice);
            cudaMemcpyAsync(to2, from2, lenght, cudaMemcpyHostToDevice);
        }
        else {//TODO: HEURISTIC BECAUSE random_shuffle MIGHT HAVE TO DO BLOCK LEVEL SHUFFLE -> 1024 ELEMENTS MAXIMUM SHUFFELING DISTANCE 
            if constexpr ((i + o) * sizeof(T) < 8192) { //If i or o are small, the overhead of cudaMemcpyAsync is too high. In this case, transfer in order and shuffle on gpu 
                cudaMemcpyAsync(to1, from1, lenght, cudaMemcpyHostToDevice);
                cudaMemcpyAsync(to2, from2, lenght, cudaMemcpyHostToDevice);

                cudaDeviceSynchronize();

                random_shuffle<T>(to1, to2, lenght, i); //TODO: https://stackoverflow.com/questions/12653995/how-to-generate-random-permutations-with-cuda
            }
            else { //Big elements, overhead of cudaMemcpyAsync is not existent. Now, we can shuffle the reads and thus achive bigger shuffelings
                __m256  c1 = _mm256_set1_ps((float)tile_samples_gpu / (float)(1 << 32));
                __m256  c2 = _mm256_set1_ps(tile_samples_gpu >> 1); //This does not round because 8 divides tile_samples_gpu
                __m256i c3 = _mm256_set1_epi32(i);
                ALIGN_PREF uint32_t ind[8] ALIGN_SUFF;

                for (uint64_t off = 0; n != lenght; n += 8 * i) {
                    __m256 gen = __m256_cvtepi32_ps(avx_xorshift128plus(random_key)); //[-2^31, 2^31]
                    gen = _mm256_add_ps(_mm256_mul_ps(gen, c1), c2); //[0, tile_samples_gpu]
                    __m256 ind_ = _mm256_mullo_epi32(_mm256_cvtps_epi32(gen), c3);
                    _mm256_store_si256((__m256i) & ind[0], ind_);

                    cudaMemcpyAsync(to1[ind[0]], to2[off + 0 * i], i * sizeof(T), cudaMemcpyHostToDevice);
                    cudaMemcpyAsync(to1[ind[1]], to2[off + 1 * i], i * sizeof(T), cudaMemcpyHostToDevice);
                    cudaMemcpyAsync(to1[ind[2]], to2[off + 2 * i], i * sizeof(T), cudaMemcpyHostToDevice);
                    cudaMemcpyAsync(to1[ind[3]], to2[off + 3 * i], i * sizeof(T), cudaMemcpyHostToDevice);
                    cudaMemcpyAsync(to1[ind[4]], to2[off + 4 * i], i * sizeof(T), cudaMemcpyHostToDevice);
                    cudaMemcpyAsync(to1[ind[5]], to2[off + 5 * i], i * sizeof(T), cudaMemcpyHostToDevice);
                    cudaMemcpyAsync(to1[ind[6]], to2[off + 6 * i], i * sizeof(T), cudaMemcpyHostToDevice);
                    cudaMemcpyAsync(to1[ind[7]], to2[off + 7 * i], i * sizeof(T), cudaMemcpyHostToDevice);
                }
            }
        }

        //2.: Augment according to agi
        if (agi.random_noise > 0.f) {
            cudaDeviceSynchronize();

            random_noise(to1, lenght, agi.random_noise, curand_state);
        }

        if (agi.random_dropout > 0.f) {
            cudaDeviceSynchronize();

            random_dropout(to1, lenght, agi.random_dropout, curand_state);
        }
    }


    template<bool training> void advance(T*& in, T*& out) {
        if (cur_gpu[0] == gpu_tiles[0] + i * tile_samples_gpu) {//GPU tile was completly used
            if (!fit_in_ram && cpu_sample_reuse_counter >= cpu_sample_reuses * tile_samples_cpu) {
                load_tile_cpu<training>();
                cpu_sample_reuse_counter = 0;
            }
            load_tile_gpu<training>();
        }

        in = cur_gpu[0];
        out = cur_gpu[1];

        cur_gpu[0] += i * batch_size;
        cur_gpu[1] += o * batch_size;

        cpu_sample_reuse_counter += batch_size;
    }
#endif
};

template<typename T, Activation ACT> //Activation for hidden layer (not softmax), output layer use softmax by default
class MultiLayerPerceptron {
private:
    uint32_t  num_layer;   //Number of layers
    uint32_t* layer_size;  //Number of neurons per layer
    uint32_t  batch_size;  //Number of samples per batch

    uint64_t num_neurons;  //Number of neurons not in the input layer
    uint64_t num_weights;  //Number of weights

    T** weights;  //Pointer to 2D-Array                                 | weights[i] has dimension layer_size[i+1] * layer_size[i], column major
    T** bias;     //Pointer to Array                                    | bias[i]    has dimension layer_size[i+1] * 1            , column major 
    T** output;   //Pointer to Array, after activation                  | output[i]  has dimension layer_size[i+1] * batch_size   , column major

    OptVariables<T> opt_var; //Hold information on what optimizer to use
    T*  opt_buf;             //Size of biggest output[] buffer. Temporary storage for optimizer
    T** opt_momentum_buf[2]; //Stores the momentum of all weights, if opt\in\{ADAM, DEMON_ADAM\}. Otherwise should be set to nullptr
    
    DatasetHandler<T> dataset;     //Dataset to use.
    T* cur_in;                     //Pointer to input data to use (testing, validation or custom)
    T* cur_out;                    //Pointer to output data to use (testing, validation or custom)

    CublasMode cublasMode;
    T* cublasConst;               //={1,cublasMode}. In GPU memory

    //==========================================================
    //========================|FUNCTION|========================
    //==========================================================
    inline void allocate() {//Allocates memory for weights and according to num_layer and layer_size in a contiuos array
        //Analyse input
        num_weights = 0;
        num_neurons = 0;
        for (uint32_t layer = 1; layer != num_layer; layer++) {
            num_weights += layer_size[layer - 1] * layer_size[layer];
            num_neurons += layer_size[layer];
        }

        //Allocate memory
        T* raw_mem;
        cudaMalloc(raw_mem, sizeof(T) * (num_weights + num_neurons));

        T* weights_mem = raw_mem;
        cudaMallocHost(&weights, sizeof(T*) * (num_layer - 1)); //First layer needs no weights
        weights[0] = weights_mem;
        for (uint32_t layer = 1; layer != num_layer - 1; layer++) {
            weights[layer] = weights[layer - 1] + layer_size[layer - 1] * layer_size[layer];
        }

        T* bias_mem = raw_mem + num_weights;
        cudaMallocHost(&bias, sizeof(T*) * (num_layer - 1)); //First layer needs no bias
        bias[0] = bias_mem;
        for (uint32_t layer = 1; layer != num_layer - 1; layer++) {
            bias[layer] = bias[layer - 1] + layer_size[layer];
        }
    }

    inline void set_cublasMode(bool b) {//true=add, false=overwrite
        cublasMode = b;
        T mode_cast = (T)cublasMode;
        cudaMemcpy(&cublasConst[0], mode_cast, sizeof(T), cudaMemcpyHostToDevice);
    }

    /* Matrix multiplication: C += A * B
       Sizes: trans_A(A)=y1*x1, trans_B(B)=x1*x2, C=y1*x2  | trans_A swaps height and with of matrix
       All matrices have to be stored column major!
    */
    template<bool trans_A, bool trans_B>
    void matmul(T* A, T* B, T* C, uint32_t y1, uint32_t x1, uint32_t x2) {
        static_assert(typeid(T)==typeid(float) || typeid(T)==typeid(double) || typeid(T)==typeid(half), "Matrix multiplication is not supported with this type!");
        
        if constexpr (typeid(T) == typeid(float))
            cublasSgemm(cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, &cublasConst[0], A, trans_A?x1:y1, B, trans_B?x2:x1, &cublasConst[1], C, y1);
        if constexpr (typeid(T) == typeid(double))
            cublasDgemm(cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, &cublasConst[0], A, trans_A?x1:y1, B, trans_B?x2:x1, &cublasConst[1], C, y1);
        if constexpr (typeid(T) == typeid(half))
            cublasGemmEx(cublas_handle, trans_A?CUBLAS_OP_T:CUBLAS_OP_N, trans_B?CUBLAS_OP_T:CUBLAS_OP_N, y1, x2, x1, &cublasConst[0], A, CUDA_R_16F, trans_A?x1:y1, B, CUDA_R_16F, trans_B?x2:x1, &cublasConst[1], C, CUDA_R_16F, y1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    
    inline void forward_weight_mul(uint32_t ind, T* in) {//output[ind] += weights[ind] * in.
        assert(cublasMode == CublasMode::ADD);
        matmul<false, false>(weights[ind], in, output[ind], layer_size[ind + 1], layer_size[ind], batch_size);
    }
    inline void backward_delta_mul(uint32_t ind, T* out) {//out = weights[ind+1]^T * output[ind+1]
        assert(cublasMode == CublasMode::OVERWRITE);
        matmul<true, false>(weights[ind + 1, output[ind + 1], out, layer_size[ind + 1], layer_size[ind + 2], batch_size);
    }
    inline void set_bias(uint32_t ind) {//Copy bias[i] to output[i] for all samples in batch
        uint32_t size_out = layer_size[ind + 1] * batch_size;
        set_repeating<T><<<LAUNCH_PARAM(size_out)>>>(output[ind], bias[ind], size_out, layer_size[ind + 1]);
    }

    template<Activation a> void activate(uint32_t i) {
        static_assert(a == Activation::RELU || a == Activation::SIGMOID || a == Activation::SOFTPLUS || a == Activation::SOFTMAX, "Unknown activation");

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
    inline void activate_derivative(uint32_t i) {//Replaces f(x) with f'(x)
        static_assert(ACT == Activation::RELU || ACT == Activation::SIGMOID || ACT == Activation::SOFTPLUS, "Unknown activation for hidden layer");
        
        uint32_t s = layer_size[i + 1] * batch_size;
        if constexpr (ACT == Activation::RELU)     relu_deriv<T>(output[i], s, LAUNCH_PARAM(s));
        if constexpr (ACT == Activation::SIGMOID)  sigmoid_deriv<T>(output[i], s, LAUNCH_PARAM(s));
        if constexpr (ACT == Activation::SOFTPLUS) softplus_deriv<T>(output[i], s, LAUNCH_PARAM(s));
    }
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
    cudaStreamCreate(&data_stream);
    //set vector, matrixes (async), pointer mode

    //Initialize random
    init_rand();


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