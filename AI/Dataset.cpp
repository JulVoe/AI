#ifndef DEBUG
#define NDEBUG
#endif

#ifdef __clang__
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wwritable-strings"
#pragma clang diagnostic ignored "-Wnull-conversion"
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#ifdef WIN32
#include <gl\GLU.h>
#endif

#include <inttypes.h>
#include <utility>

#include <atomic>
#include <thread>

#include <stdio.h>
#include <cstdlib>

#if defined(__GNUC__) or defined(_clang__)
#include <x86intrin.h>
#else
#include <immintrin.h>
#endif

#include "util.cpp"
using namespace Random;
using namespace Image;
using namespace DatasetAssemble;

//TODO: Memadvise, mmap
//TODO: Thread Priority
//TODO: Sampler and self tuning
//==============================================================================================================================================================
enum WORKER_STATUS : int32_t { VALIDATION = -3, TRAINING = -2, IDLE = -1, STOP = 0};    //Don't change values!! All values >= 0 are considered as "STOP".
enum BATCH_STATUS  : int32_t { USED = -2, READY = -1, LOADING = 0, DEPRECATED = 1<<30}; //Don't change values!! All values >= 0 are considered as "LOADING". DEPRECATED bit is used to determin, when during the loading of a batch the input source changed. When a spawned thread sets the status to any loading status, it is not allowed to change it to a different loading status itself until it set the status to a not-loading status before

/*
    Stores information on how input should be augmented.
*/
struct AugmentationInfo2D_IN {
   DATA_FORMAT format;          //Transform data to this format

    //The "if not zero" part is redundant and only used for understanding, that incase of 0 nothing happens.
    uint32_t blur_range;         //If not zero, applies gaussian blur with this distance 
    bool  flip;                  //If true    , the Image has a 50% chance of getting mirrored horitontaly
    float rotate;                //If not zero, the Image is rotated by an random angle with mean 0 and this standart deviation
    float random_noise;          //If not zero, to every channel a random number with mean 0 and this standart deviation stored
    float random_dropout;        //If not zero, this stores the probability for each pixel to individually become black
    float random_saturation;     //If not zero, the saturation of each pixel is multiplies by a constant random number with mean 1 and this standart deviation
    float random_brightness;     //If not zero, the brightness of each pixel is multiplies by a constant random number with mean 1 and this standart deviation
                                 
    Offset2D<float> crop;        //Crops a random part of shape (crop.x*dim_x, crop.y*dim_y, channels) from the input and output. crop.x=-1 indicates to skip this step
    Offset2D<int32_t> resize;    //Resizes input to shape (resize.x, resize.y channels_in). resize.x=-1 indicates to skip this step

    AugmentationInfo2D_IN(DATA_FORMAT format, uint32_t blur_range, bool  flip, float rotate, float random_noise, float random_dropout, float random_saturation, float random_brightness, Offset2D<float> crop, Offset2D<int32_t> resize) :
        format(format),
        blur_range(blur_range),
        flip(flip),
        rotate(rotate),
        random_noise(random_noise),
        random_dropout(random_dropout),
        random_saturation(random_saturation),
        random_brightness(random_brightness),
        crop(crop),
        resize(resize)
    {}
    void output() {
        printf("%u %g %u %d %f %f %f %f %f %f %f %d %d\n",format.distribution,format.range,blur_range,flip,rotate,random_noise,random_dropout,random_saturation,random_brightness,crop.x,crop.y,resize.x,resize.y);
    }
};

/*
    Stores information on how output should be augmented.
*/
struct AugmentationInfo2D_OUT {
   DATA_FORMAT format;          //Transform data to this format

    Offset2D<int32_t> resize;    //Resizes output to (resize.x, resize.y, channels_out). resize.x=-1 indicates to skip this step
    float label_smoothing;       //Apply label smoothing

    bool do_input_aug;           //If true, performs rotate, flip and crop from input augmentation also on output (useful for segmentation but not classification)

    AugmentationInfo2D_OUT(DATA_FORMAT format, Offset2D<int32_t> resize, float label_smoothing, bool do_input_aug) :
        format(format),
        resize(resize),
        label_smoothing(label_smoothing),
        do_input_aug(do_input_aug)
    {}
};

//==============================================================================================================================================================

#ifdef DEBUG
/*
    Key handler for the debugger window. Has to be defined outside DatasetHandler, as otherwise one could not pass its pointer
*/
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)        //ESC: close window
        glfwSetWindowShouldClose(window, GLFW_TRUE);

    if (key == GLFW_KEY_R && action == GLFW_PRESS) {           //R: reset workers_tile
        int32_t* workers_tile = (int32_t*)glfwGetWindowUserPointer(window);
        PRINT_VAR((void*)workers_tile);


        for (uint32_t ind = 0; ind != workers_tile[0]; ind++) {
            workers_tile[ind + 1] = -1;
        }
    }
}
#endif

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
        3.: "set_batching"
        4.: "start_workers"
        (5.: "setDataInput")
        6.: Loop "advance". Step 2+3+4+5 can be repeated anytime in between
    */
private:
    FILE* fd_in, *fd_out;                 //File descritors of files holding dataset
    HEADER_V1 in_header, out_header;      //Header of files containing the datasets
    uint32_t header_offset[2];            //Number of bytes for both files, until data starts
    Image_Shape sample_shape[4];          //The shape of unaugmented input/output samples in dataset ([0],[1]) and of augmented input/output samples([2], [3])
    uint32_t sample_size[4];              //How many number are contained in an input/output sample in the dataset ([0],[1]) and in augmentated samples ([2],[3])
                                              
    bool fit_in_ram;                      //True if the whole dataset fits into RAM
    uint32_t train_samples;               //Number of all training samples
    uint32_t validation_samples;          //Number of all validation samples
    uint32_t batch_size;                  //Number of samples to return to neural network per call. Has to be multiple of 8 (for AVX)
                                          
    uint32_t tile_samples_cpu;            //Number of samples per tile.
    uint32_t augmentation_batches_cpu;    //Number of augmented batches stored in ram
    uint32_t augmentation_batches_gpu;    //Number of augmented batches stored in vram
                                          
    AugmentationInfo2D_IN  agi_in ;       //Augmentations performed on input
    AugmentationInfo2D_OUT agi_out;       //Augmentations performed on output
    uint32_t num_workers;                 //Number of threads launched for augmentation
    std::thread* workers;                 //Handle to threads launched for augmentation   (nullptr, when no workers are started)
    std::thread  cpy_thread;              //Handle to thread that copies batches from cpu to gpu
    std::atomic<int32_t> thread_status;   //Tells all spawned threads what to do. Actual type is WORKER_STATUS. int32_t is only used to enable arithmetics
                                          
    T* cpu_tiles[2];                      //Start of tiles in ram that hold raw dataset ( ={input, output} )
    T* aug_tiles[2];                      //Start of tiles in ram that hold augmented batches of dataset ( ={input, output} )
    uint32_t aug_padding[2];              //Padding to add to aug tiles to be able to accomodate copy of size sample_size[0/1] and intermediates
    std::atomic<BATCH_STATUS>* ready_cpu; //Stores the status for ech batch in aug_tiles
    
    T* gpu_tiles[2];                      //Holds start of tiles (gpu) = {input, output}. Owns memory
    T* cur_gpu[2];                        //Holds last used element in tiles (gpu) = {input, output}. If gpu_tiles was reloaded, cur_gpu[0]=nullptr
    std::atomic<BATCH_STATUS>* ready_gpu; //Stores the status for ech batch in gpu_tiles

#ifdef DEBUG
    //Debugging variables
    int32_t* workers_tile;                //Array that stores for each worker thread, which augmentation tile it last worked on. First element is num_workers
#endif

    // /+===========+\
    // ||Cpu workers||
    // \+===========+/
    /*
        Loads a sample from the ram tile to the augmentation tile.

        @param aug_in : destination of  input sample (in augmentation tile)
        @param aug_out: destination of output sample (in augmentation tile)
        @param sample_off_in : the offset in the cpu tile to get to the source  input image
        @param sample_off_out: the offset in the cpu tile to get to the source output image
        @param in_ram: if true, the source is in the ram tile. else, go to disk.
    */
    inline void load_sample_augmentation(T* aug_in, T* aug_out, uint32_t sample_off_in, uint32_t sample_off_out, bool in_ram){
        //printf("\n%p %p %u %u %d\n", aug_in, aug_out, sample_off_in, sample_off_out, (int)in_ram);
        if (in_ram) { /*In Ram tile*/    
            memcpy(aug_in , &cpu_tiles[0][sample_off_in ], sample_size[0] * sizeof(T));
            memcpy(aug_out, &cpu_tiles[1][sample_off_out], sample_size[1] * sizeof(T));
        }                                                                                    
        else { /*Go to disk*/                                                                
            fseek(fd_in , sizeof(T) * sample_off_in  + header_offset[0], SEEK_SET);   
            fseek(fd_out, sizeof(T) * sample_off_out + header_offset[1], SEEK_SET);
            fread(aug_in , sizeof(T), sample_size[0], fd_in );
            fread(aug_out, sizeof(T), sample_size[1], fd_out);
        }
    }

    /*
        Applies the augmetations specified in agi_in and agi_out to the data in aug_in and aug_out with shapes in_shape and out_shape.

        @param aug_in : pointer to  input sample to augment
        @param aug_out: pointer to output sample to augment
        @param  in_shape: shape of  input sample
        @param out_shape: shape of output sample
        @param training: if true, use all augmentations. if false, only remap and resize
    */
    inline void apply_augmentations(T* aug_in, T* aug_out, bool training){        
        if (training){
            //Input:  Everything
            //Output: Resize + label_smoothing + remap

            Image_Shape  in_shape_ = sample_shape[0];  //The shape of the input and output can change because of augmentation. This holds the actual shape at every moment
            Image_Shape out_shape_ = sample_shape[1];  //The shape of the input and output can change because of augmentation. This holds the actual shape at every moment

            //Input
            if (agi_in.crop.x != -1.f) {
                in_shape_  *= agi_in.crop;
                if (agi_out.do_input_aug) out_shape_ *= agi_in.crop;
            
                Offset2D<float> pos  = {rand_float(1.f), rand_float(1.f) };
            
                crop<T>(aug_in , sample_shape[0], (sample_shape[0].getOffset2D() -  in_shape_.getOffset2D()) * pos,  in_shape_.getOffset2D(),  in_header.order);
                if(agi_out.do_input_aug) crop<T>(aug_out, sample_shape[1], (sample_shape[1].getOffset2D() - out_shape_.getOffset2D()) * pos, out_shape_.getOffset2D(), out_header.order);
            }
            if (agi_in.blur_range != 0.f) {
                pseudo_gausblur<T, PADDING::ZERO_PADDING_NORMAL>(aug_in, in_shape_, abs(rand_normal(agi_in.blur_range)), 3, in_header.order); //TODO:PADDING AND N
            }
            if (agi_in.flip && rand_prob(0.5f)) { /*50% chance*/
                flip<T>(aug_in ,  in_shape_,  in_header.order);
                if (agi_out.do_input_aug) flip<T>(aug_out, out_shape_, out_header.order);
            }
            if (agi_in.random_noise != 0.f) {
                random_noise<T>(aug_in, in_shape_, agi_in.random_noise);
            }
            if (agi_in.random_dropout != 0.f) {                                                     
                random_dropout<T>(aug_in, in_shape_, agi_in.random_dropout, in_header.order);
            }
            if (agi_in.random_saturation != 0.f) {                                                  
                mul_saturation<T>(aug_in, in_shape_, 1.f + rand_normal(agi_in.random_saturation), in_header.format, in_header.order);
            }
            if (agi_in.random_brightness != 0.f) {
                mul_brightness<T>(aug_in, in_shape_, 1.f + rand_normal(agi_in.random_brightness), in_header.format, in_header.order);
            }
            if (agi_in.rotate != 0.f) {
                float rot_deg = rand_normal(agi_in.rotate);
                rotate<T>(aug_in ,  in_shape_, rot_deg);
                if (agi_out.do_input_aug) rotate<T>(aug_out, out_shape_, rot_deg);
            }
            if (agi_in.resize.x != -1) {          
                assert(agi_in.resize.x > 0 && agi_in.resize.y > 0);
                resize<T>(aug_in , in_shape_, agi_in.resize.convert<uint32_t>());
                in_shape_.setOffset2D(agi_in.resize);
            }
                {
                    int32_t shiftX = rand_normal(4.f);
                    int32_t shiftY = rand_normal(4.f);

                    translate<T>(aug_in, in_shape_, Offset2D<int32_t>(shiftX, shiftY), in_header.order);
                }
            if (agi_in.format != in_header.format){
                remap_format<T>(aug_in, in_shape_, in_header.format, agi_in.format, in_header.order);
            }


            //Output
            if (agi_out.resize.x != -1) {                                                           
                resize<T>(aug_out, out_shape_, agi_out.resize.convert<uint32_t>());
                out_shape_.setOffset2D(agi_out.resize);
            }
            if (agi_out.label_smoothing != 0.f){
                label_smoothing<T>(aug_out, out_shape_, agi_out.label_smoothing);
            }
            if (agi_out.format != out_header.format){
                remap_format<T>(aug_out, out_shape_, out_header.format, agi_out.format, out_header.order);
            }
        } 
        else{
            // Input: Remap + resize
            //Output: Remap + resize
          
            Image_Shape  in_shape_ = sample_shape[0];  //The shape of the input and output can change because of augmentation. This holds the actual shape at every moment
            Image_Shape out_shape_ = sample_shape[1];  //The shape of the input and output can change because of augmentation. This holds the actual shape at every moment
        
            //Input
            if (agi_in.resize.x != -1) {                                                           
                resize<T>(aug_in , in_shape_, agi_in.resize.convert<uint32_t>());
                in_shape_.setOffset2D(agi_in.resize);
            }
            if(agi_in.format != in_header.format){
                remap_format<T>(aug_in, in_shape_, in_header.format, agi_in.format, in_header.order);
            }

            //Output
            if (agi_out.resize.x != -1) {                                                           
                resize<T>(aug_out, out_shape_, agi_out.resize.convert<uint32_t>());
                out_shape_.setOffset2D(agi_out.resize);
            }
            if(agi_out.format != out_header.format){
                remap_format<T>(aug_out, out_shape_, out_header.format, agi_out.format, out_header.order);
            }
        }
    }
  
    /*
        The function executed by the worker threads. It searches for augmentation tiles with status USED and loads and generates a new augmented batch in it.
        When it does not find a used batch, it buisy-waits.

        @param id: Each worker thread gets a unique id. It is used to seed random number generator as samples are loaded in random order.
    */
    void worker_thread(uint64_t id) {
        //1.: Generate a deterministic random key
        Key my_key;
        uint64_t r = (763808623281539ull * (id + 1ull)) ^ (2009741990ull * (id + 1ull));                     //Use id to initialize random number generator
        avx_xorshift128plus_init(r ^ (r << 5) ^ (r >> 19), r ^ (r << 39) ^ (r >> 27), my_key);

        //2.: Set AVX constants
        __m256  rand_normalize1      = _mm256_set1_ps((float)(train_samples) / (float)(1ull<<32));           //]-2^31, 2^31-1[ -> ]-train_samples/2, train_samples/2[
        __m256  rand_adder1          = _mm256_set1_ps((float)(train_samples) * 0.5f);                        //]-train_samples/2, train_samples/2[ -> ]0, train_samples[

        __m256  rand_normalize2      = _mm256_set1_ps((float)(validation_samples) / (float)(1ull << 32));    //]-2^31, 2^31-1[ -> ]-validation_samples/2, validation_samples/2[ 
        __m256  rand_adder2          = _mm256_set1_ps(train_samples + ((float)(validation_samples) * 0.5f)); //]-validation_samples/2, validation_samples/2[ -> ]train_samples, train_samples+validation_samples[

        __m256i tile_samples_cpu_vec = _mm256_set1_epi32(tile_samples_cpu);
        __m256i sample_size_0        = _mm256_set1_epi32(sample_size[0]);
        __m256i sample_size_1        = _mm256_set1_epi32(sample_size[1]);

        uint32_t* off = (uint32_t*)_mm_malloc(16 * sizeof(uint32_t), 32);                         //Store offset of T's to start of sample

        //3.: Work loop 
        while (thread_status.load() < WORKER_STATUS::STOP) {                                      //Break if exit signal is reached
            //0.: Check for idle
            if (thread_status.load() == WORKER_STATUS::IDLE)
                continue;

            for (uint32_t b = 0; b != augmentation_batches_cpu; b++) {                            //Go through all batches in augmentation tile
                BATCH_STATUS expected = BATCH_STATUS::USED;
                ready_cpu[b].compare_exchange_strong(expected, BATCH_STATUS::LOADING);            //If ready_cpu[b] was "USED", it no is "LOADING". Otherwise, only expected was changed to old value 
                if (expected == BATCH_STATUS::USED) {                                             //Since expected was not changed, batch was used and is now getting reloaded
                    //Reload batch b in augmentation tile                                       
                    int32_t training = thread_status.load();                                      //Input source before batch is loaded
                    expected = BATCH_STATUS::LOADING;                                             //The status of this batch before batch is loaded. When it has a different status afterwards, we now the input source changed
                    start:

#ifdef DEBUG        //If debugging is enabled, tell debugger that this thread has last worked on tile "b"
                    workers_tile[id + 1] = b;
                    //printf("Thread %llu doing tile %d\n", id, b); fflush(stdout);
                    //printf("Thread %llu set status of tile %d to %d\n", id, b, (int32_t)ready_cpu[b].load()); fflush(stdout);
#endif
                    
                    T* aug_in  = &aug_tiles[0][b * (aug_padding[0] + batch_size * sample_size[2])];
                    T* aug_out = &aug_tiles[1][b * (aug_padding[1] + batch_size * sample_size[3])];
                    for (uint32_t sample = 0; sample != batch_size; sample += 8) {                //Loop is unrolled 8 times
                        __m256 random = _mm256_cvtepi32_ps(avx_xorshift128plus(my_key));          //]-2^31, 2^31-1[
                        random = _mm256_fmadd_ps(                                                      \
                            random,                                                                    \
                            (training == WORKER_STATUS::TRAINING) ? rand_normalize1 : rand_normalize2, \
                            (training == WORKER_STATUS::TRAINING) ? rand_adder1     : rand_adder2);    //]0, train_samples] or ]train_samples, train_samples + validation_samples]
                        __m256i gen = _mm256_cvttps_epi32(random);

                        __m256i  in_tile = _mm256_cmpgt_epi32(tile_samples_cpu_vec, gen);         //True, if samples is in ram tile
                        uint32_t in_mask = _mm256_movemask_epi8(in_tile);                         //Most significant bits
                            
                        __m256i ind1 = _mm256_mullo_epi32(gen, sample_size_0);                    //Compute input  indices
                        __m256i ind2 = _mm256_mullo_epi32(gen, sample_size_1);                    //Compute output indices
                        _mm256_store_si256((__m256i*) &off[0], ind1);
                        _mm256_store_si256((__m256i*) &off[8], ind2);

                        //Load and process all the samples
                        load_sample_augmentation(aug_in, aug_out, off[0], off[8 ], fit_in_ram || ((in_mask >> (4 * 0)) & 0b1));
                        apply_augmentations(aug_in, aug_out, training == WORKER_STATUS::TRAINING);
                        aug_in  += sample_size[2];
                        aug_out += sample_size[3];

                        load_sample_augmentation(aug_in, aug_out, off[1], off[9 ], fit_in_ram || ((in_mask >> (4 * 1)) & 0b1));
                        apply_augmentations(aug_in, aug_out, training == WORKER_STATUS::TRAINING);
                        aug_in  += sample_size[2];
                        aug_out += sample_size[3];
                        
                        load_sample_augmentation(aug_in, aug_out, off[2], off[10], fit_in_ram || ((in_mask >> (4 * 2)) & 0b1));
                        apply_augmentations(aug_in, aug_out, training == WORKER_STATUS::TRAINING);
                        aug_in  += sample_size[2];
                        aug_out += sample_size[3];
                        
                        load_sample_augmentation(aug_in, aug_out, off[3], off[11], fit_in_ram || ((in_mask >> (4 * 3)) & 0b1));
                        apply_augmentations(aug_in, aug_out, training == WORKER_STATUS::TRAINING);
                        aug_in  += sample_size[2];
                        aug_out += sample_size[3];
                        
                        load_sample_augmentation(aug_in, aug_out, off[4], off[12], fit_in_ram || ((in_mask >> (4 * 4)) & 0b1));
                        apply_augmentations(aug_in, aug_out, training == WORKER_STATUS::TRAINING);
                        aug_in  += sample_size[2];
                        aug_out += sample_size[3];
                        
                        load_sample_augmentation(aug_in, aug_out, off[5], off[13], fit_in_ram || ((in_mask >> (4 * 5)) & 0b1));
                        apply_augmentations(aug_in, aug_out, training == WORKER_STATUS::TRAINING);
                        aug_in  += sample_size[2];
                        aug_out += sample_size[3];

                        load_sample_augmentation(aug_in, aug_out, off[6], off[14], fit_in_ram || ((in_mask >> (4 * 6)) & 0b1));
                        apply_augmentations(aug_in, aug_out, training == WORKER_STATUS::TRAINING);
                        aug_in  += sample_size[2];
                        aug_out += sample_size[3];
                        
                        load_sample_augmentation(aug_in, aug_out, off[7], off[15], fit_in_ram || ((in_mask >> (4 * 7)) & 0b1));
                        apply_augmentations(aug_in, aug_out, training == WORKER_STATUS::TRAINING);
                        aug_in  += sample_size[2];
                        aug_out += sample_size[3];
                    }
                    //Now, the whole batch is loaded. Currently, operations on the whole batch are not supported

                    //Mark batch as ready
                    BATCH_STATUS old_expected = expected;
                    ready_cpu[b].compare_exchange_strong(expected, BATCH_STATUS::READY);          //When the input source changes, the status of this batch will be changed. If it is still the same, batch is ready
                    if (expected == old_expected)                                                 //Input source did not changed
                        break;                                                                    //When finished, check again for exit signal and start at the beginning
                    else {                                                                        //Input source did change. expected is now set to the new batch status, as we will reload the batch again and this is now the status before we begin
                        //As the input source changed, this is now the new input source
                        training = thread_status.load();
                        goto start;                                                               //Reload this batch again with new input source and new expected
                    }
                }
            }
            //If execution reaches this point, no USED batch was found. TODO: detmerine whether to suspend thread as busy waiting takes many system resources
        }
        thread_status++;
    }

    /*
        The function executed by the memcpy thread

        @param num_streams: The number of cuda streams to use for host->device memcpies
    */
    void memcpy_thread(uint32_t num_streams){
        //1.: Create cuda variables
        cudaEvent_t* gpu_sync = (cudaEvent_t*)malloc(augmentation_batches_gpu * sizeof(cudaEvent_t)); //Events to tell when memory-transfer to gpu are finished
        for (uint32_t u = 0; u != augmentation_batches_gpu; u++) {
            gpuErrchk(cudaEventCreateWithFlags(&gpu_sync[u], cudaEventDisableTiming));
        }

        cudaStream_t* data_stream = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
        for(uint32_t u = 0; u != num_streams; u++) {
            gpuErrchk(cudaStreamCreateWithFlags(&data_stream[u], cudaStreamNonBlocking));
        }
        cudaStream_t*  cur_stream = data_stream;
        cudaStream_t* last_stream = data_stream + (num_streams - 1);


        bool found;
        //Main loop
        while(thread_status.load() < WORKER_STATUS::STOP){
            //BUGP("Copier working\n");

            //0.: Check for idle
            if(thread_status.load() == WORKER_STATUS::IDLE)
                continue;
          
            //1.: Launch new copies
            for(uint32_t ind = 0; ind != augmentation_batches_gpu; ind++){
                if(ready_gpu[ind].load() == BATCH_STATUS::USED) {
                    //Load next ready batch to gpu tile ind
                    found = false;
                    for(uint32_t cpu_ind = 0; cpu_ind != augmentation_batches_cpu; cpu_ind++){
                        BATCH_STATUS expected = BATCH_STATUS::READY;                                   //Check whether cpu tile is READY
                        ready_cpu[cpu_ind].compare_exchange_strong(expected, BATCH_STATUS::LOADING);   //If tile is READY, set it to LOADING. TODO: could also set this to ind?! Maybe add a COPYING status to differentiate from cpu work

                        if(expected == BATCH_STATUS::READY){                                           //Cpu tile was READY
                            //Found a loadable augmentation tile
                            T* in  = gpu_tiles[0] + ind * batch_size * sample_size[2];
                            T* out = gpu_tiles[1] + ind * batch_size * sample_size[3]; 
                            
                                //PRINT_VAR(in);
                                //BUGP("\t\t");
                                //PRINT_VAR(&aug_tiles[0][cpu_ind * (aug_padding[0] + batch_size * sample_size[2])]);
                                //BUGP("\n");
                                //PRINT_VAR(out);
                                //BUGP("\t\t");
                                //PRINT_VAR(&aug_tiles[1][cpu_ind * (aug_padding[1] + batch_size * sample_size[3])]);
                                //BUGP("\n");

                            gpuErrchk(cudaMemcpyAsync(in , &aug_tiles[0][cpu_ind * (aug_padding[0] + batch_size * sample_size[2])], batch_size * sample_size[2] * sizeof(T), cudaMemcpyHostToDevice, *cur_stream));
                            gpuErrchk(cudaMemcpyAsync(out, &aug_tiles[1][cpu_ind * (aug_padding[1] + batch_size * sample_size[3])], batch_size * sample_size[3] * sizeof(T), cudaMemcpyHostToDevice, *cur_stream));
                            gpuErrchk(cudaEventRecord(gpu_sync[ind], *cur_stream));

                            if (cur_stream == last_stream)
                                cur_stream = data_stream;
                            else
                                cur_stream++;

                            ready_gpu[ind].store((BATCH_STATUS)cpu_ind);

                            //Make sure, only one batch is copied
                            found = true;

                            CHECK_CUDA_ERROR();
                            break;
                        }
                    }

                    if (!found) {//No ready augmentation tiles. If this happens, there is no point in searching for used gpu tiles. A performance problem exists!
                       //BUGP("No tile was found!\n");
                    } 
                }
            }

            //2.: Check which copies finished
            for(uint32_t ind = 0; ind != augmentation_batches_gpu; ind++){
                int32_t gpu_batch_status = ready_gpu[ind].load();
                if(gpu_batch_status >= 0 && cudaEventQuery(gpu_sync[ind]) == cudaSuccess) {                                   //Was status "LOADING" and the copy finished?
                    BATCH_STATUS cpu_origin_InChanged = (BATCH_STATUS)(gpu_batch_status | (int32_t)BATCH_STATUS::DEPRECATED); //When the input source changed, ready_gpu[ind] has this value
                    ready_gpu[ind].compare_exchange_strong(cpu_origin_InChanged, BATCH_STATUS::USED);                         //If ready_gpu[ind]==cpu_origin_InChanged, the input source changed and the batch is marked as USED
                    if (cpu_origin_InChanged != (BATCH_STATUS)(gpu_batch_status | (int32_t)BATCH_STATUS::DEPRECATED)) {       //If ready_gpu[ind]==cpu_origin_InChanged is the case, cpu_origin_InChanged will not change. If it did, this is not the case and thus the input source stayed the same
                        //The copy has finished, the input source stayed the same. Mark augmentation tile as used and gpu tile as ready
                        ready_cpu[gpu_batch_status].store(BATCH_STATUS::USED);
                        ready_gpu[ind].store(BATCH_STATUS::READY);
                    }
                    else
                        ready_cpu[(cpu_origin_InChanged ^ (int32_t)BATCH_STATUS::DEPRECATED)].store(BATCH_STATUS::USED);
                }
            }
        }

        //Stopping signal was received
        for (uint32_t u = 0; u != augmentation_batches_gpu; u++) {
            cudaEventDestroy(gpu_sync[u]);
        }
        for(uint32_t u = 0; u != num_streams; u++) {
            cudaStreamDestroy(data_stream[u]);
        }
        
        thread_status++;
    }

    // /+=========+\
    // ||Debugging||
    // \+=========+/
#ifdef DEBUG // Check, whether the DEBUG makro was defined during compilation. This is needed to get information from worker threads
#ifdef WIN32
    static constexpr char*    FONT_PATH = "C:/Windows/Fonts/Arial.ttf";
    static constexpr uint32_t YSTART    = 0;
#else
    static constexpr char* FONT_PATH = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf";
    static constexpr uint32_t YSTART = 300;
#endif
    static constexpr uint32_t XRES    = 3440;
    static constexpr uint32_t YRES    = 1440;
    static constexpr uint32_t XIMG    = 100;
    static constexpr uint32_t YIMG    = 100;
    static constexpr uint32_t PADDING = 15;
    static constexpr uint32_t GAP     = 50;
    static constexpr uint32_t PSIZE   = 6;
    /*
        Uses opengl to draw a window which always shows the pixel data
    */
    void debug_thread() {
        //1.: Create Window and initialize GLFW
        GLFWwindow* window;
            //if (!glfwInit())
            //    return;
        window = glfwCreateWindow(XRES, YRES, "Debugger", NULL, NULL);
        if (!window) {
            glfwTerminate();
            return;
        }
        glfwMakeContextCurrent(window);

        //2.: Initialize GLEW
        glewExperimental = GL_TRUE;
        GLenum err = glewInit();
        if (GLEW_OK != err)
        {
            fprintf(stderr, "[Error]: When trying to open the debugger window in line %u, the following error was encountered: %s\n", __LINE__, glewGetErrorString(err));
            glfwTerminate();
            return;
        }

        //3.: Key handler
        glfwSetWindowUserPointer(window, (void*)workers_tile);                //Key handler needs to access this, as it resets these values when "R" is pressed
        glfwSetKeyCallback(window, key_callback);
        
        //4.: Debugging
        if (glDebugMessageControlARB != NULL) {
            printf("[INFO] Setting up Opengl-Debugging\n");
            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
            glDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
            glDebugMessageCallbackARB((GLDEBUGPROCARB)ETB_GL_ERROR_CALLBACK, NULL);
        }

        //4.5: FreeType initialization and blending
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        TextRenderer::initFreeType(FONT_PATH, 100);

        //5.: Refresh speed
        glfwSwapInterval(1);

        //6.: Set up projection
        glClearColor(0.2265, 0.2344, 0.2617, 1.0);
        glDisable(GL_DEPTH_TEST);
        glShadeModel(GL_SMOOTH);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0, XRES, YRES, 0);
        glEnable(GL_TEXTURE_2D);

        //7.: Setting up textures
        GLuint tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, XIMG, YIMG);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        //8.: Setting up buffer
        T* buffer;
        cudaMallocHost(&buffer, sizeof(T) * sample_size[2]);
        //buffer = (T*)malloc(sizeof(T) * sample_size[3]);

        //9.: Display-Loop
        while (thread_status.load() < WORKER_STATUS::STOP && !glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            uint32_t x = PADDING;
            uint32_t y = PADDING + YSTART;

            for (uint32_t tile = 0; tile != augmentation_batches_cpu; tile++) {glBindTexture(GL_TEXTURE_2D, 0);
                int32_t val = ready_cpu[tile];
                glBegin(GL_QUADS);
                glColor3f((float)(val == BATCH_STATUS::USED), (float)(val == BATCH_STATUS::READY), (float)(val >= BATCH_STATUS::LOADING));
                glVertex2f(x + XIMG / 2 - PSIZE / 2, y - PADDING / 2 - PSIZE / 2);
                glVertex2f(x + XIMG / 2 - PSIZE / 2, y - PADDING / 2 + PSIZE / 2);
                glVertex2f(x + XIMG / 2 + PSIZE / 2, y - PADDING / 2 + PSIZE / 2);
                glVertex2f(x + XIMG / 2 + PSIZE / 2, y - PADDING / 2 - PSIZE / 2);
                glEnd();
                glColor3f(1.0, 1.0, 1.0);
                glBindTexture(GL_TEXTURE_2D, tex);

                for(uint32_t sample = 0; sample != batch_size; sample++){
                    shrinkToGL<T>(&aug_tiles[0][tile * (aug_padding[0] + batch_size * sample_size[2]) + sample * sample_size[2]], sample_shape[2], (uint8_t*)buffer);
                    //Image::show<uint8_t, true>((uint8_t*)buffer, Image_Shape(sample_shape[2].x, sample_shape[2].y, 3u), Image::CHANNEL_ORDER::CHANNELS_LAST);
                    Image::resize<uint8_t, Image::CHANNEL_ORDER::CHANNELS_LAST>((uint8_t*)buffer, Image_Shape(sample_shape[2].x, sample_shape[2].y, 3u), Offset2D<uint32_t>(XIMG, YIMG));
                    //Image::show<uint8_t, true>((uint8_t*)buffer, Image_Shape((uint32_t)XIMG, (uint32_t)YIMG, 3u), Image::CHANNEL_ORDER::CHANNELS_LAST);

                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, XIMG, YIMG, GL_RGB, GL_UNSIGNED_BYTE, (void*)buffer);

                    glBegin(GL_QUADS);
                    glTexCoord2f(0.f, 0.f); glVertex2i(x, y);
                    glTexCoord2f(0.f, 1.f); glVertex2i(x, y + YIMG);
                    glTexCoord2f(1.f, 1.f); glVertex2i(x + XIMG, y + YIMG);
                    glTexCoord2f(1.f, 0.f); glVertex2i(x + XIMG, y);
                    glEnd();

                    x += XIMG + PADDING;
                }
            }

            x = PADDING;
            y += YIMG + PADDING + GAP;
            for (uint32_t ind = 0; ind != augmentation_batches_gpu * batch_size; ind++) {
                    //printf("\n%p %u", gpu_tiles[0] + (ind * sample_size[2]), sizeof(T) * in_shape.x * in_shape.y * in_shape.z);
                cudaMemcpy(buffer, gpu_tiles[0] + (ind * sample_size[2]), sizeof(T) * sample_shape[2].x * sample_shape[2].y * sample_shape[2].z, cudaMemcpyDefault);
                cudaStreamSynchronize((cudaStream_t)0);
                    //std::this_thread::sleep_for(std::chrono::seconds(1));
                    //PRINT_VAR(sample_shape[2].z);
                    //Image::show<T, true>(buffer, sample_shape[2]);
                shrinkToGL<T>(buffer, sample_shape[2], (uint8_t*)buffer);
                    //Image::show<uint8_t, true>((uint8_t*)buffer, sample_shape[2], Image::CHANNEL_ORDER::CHANNELS_LAST);
                Image::resize<uint8_t, Image::CHANNEL_ORDER::CHANNELS_LAST>((uint8_t*)buffer, Image_Shape(sample_shape[2].x, sample_shape[2].y, 3u), Offset2D<uint32_t>(XIMG, YIMG));
                    //Image::show<uint8_t, true>((uint8_t*)buffer, Image_Shape((uint32_t)XIMG, (uint32_t)YIMG, 3u), Image::CHANNEL_ORDER::CHANNELS_LAST);

                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, XIMG, YIMG, GL_RGB, GL_UNSIGNED_BYTE, (void*)buffer);

                glBegin(GL_QUADS);
                glTexCoord2f(0.f, 0.f); glVertex2i(x, y);
                glTexCoord2f(0.f, 1.f); glVertex2i(x, y + YIMG);
                glTexCoord2f(1.f, 1.f); glVertex2i(x + XIMG, y + YIMG);
                glTexCoord2f(1.f, 0.f); glVertex2i(x + XIMG, y);
                glEnd();


                if (ind % batch_size == 0) {
                    glBindTexture(GL_TEXTURE_2D, 0);

                    int32_t val = ready_gpu[ind / batch_size];
                    glBegin(GL_QUADS);
                    glColor3f((float)(val == BATCH_STATUS::USED), (float)(val == BATCH_STATUS::READY), (float)(val >= BATCH_STATUS::LOADING));
                    glVertex2f(x + XIMG / 2 - PSIZE / 2, y - PADDING / 2 - PSIZE / 2);
                    glVertex2f(x + XIMG / 2 - PSIZE / 2, y - PADDING / 2 + PSIZE / 2);
                    glVertex2f(x + XIMG / 2 + PSIZE / 2, y - PADDING / 2 + PSIZE / 2);
                    glVertex2f(x + XIMG / 2 + PSIZE / 2, y - PADDING / 2 - PSIZE / 2);
                    glEnd();

                    glColor3f(1.0, 1.0, 1.0);
                    if (val >= 0) {
                        glBegin(GL_LINES);
                        glVertex2f(val * batch_size * (XIMG + PADDING) + PADDING + XIMG / 2, y - PADDING - GAP);
                        glVertex2f(x + XIMG / 2, y);
                        glEnd();
                    }                

                    glBindTexture(GL_TEXTURE_2D, tex);
                }

                x += XIMG + PADDING;
            }


            //Print text
            TextRenderer::renderText("Usage: Press \"ESC\" to exit and \"R\" to reset the last used values", 1000, 1280 - PADDING, 0.3f);

            for (uint32_t ind = 0; ind != num_workers; ind++) {
                char text[255];
                sprintf(text, "Worker %d last worked on augmentation tile %d", ind, workers_tile[ind + 1]);
                TextRenderer::renderText(+text, PADDING, y + 400 +  50 * ind, 0.3f);
            }
            glBindTexture(GL_TEXTURE_2D, tex);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }


        glfwDestroyWindow(window);
    }
#else
    void debug_thread() {
        fprintf(stderr, "[ERROR] If you want to use the debugging window, compile with -DDEBUG! %s %d\n", __FILE__, __LINE__);
    }
#endif

    
    // /+================+\
    // ||Helper functions||
    // \+================+/
    inline HEADER_V1 parseInputFile(FILE* fd, Image_Shape& shape, uint32_t& sample_size, uint32_t& header_off, uint32_t& num_samples){
        //1.: Signature
        printf("[INFO] Parsing dataset file...\n");
        char sig[6];
        fread(+sig, sizeof(char), 6, fd);
        assert(strncmp(sig, "JVDATA", 6)==0);
        printf("[INFO] \t - Signature matches\n");
        
        //2.: Version
        uint16_t ver;
        fread(&ver, sizeof(uint16_t), 1, fd);
        assert(ver <= AI_VERSION);
        if (ver < AI_VERSION)
            printf("[INFO] \t - File version: %u. This is an old version, since this is library version %u\n", (uint32_t)ver, AI_VERSION);
        else
            printf("[INFO] \t - File version: %u. This is the recent version\n", (uint32_t)ver);
        
        //3.: Switch according to version
        if ( ver == 1 ) {
            //4.: Read in header
            HEADER_V1 ret = HEADER_V1::deserialize(fd);

            //5.: Set shape and sample_size
            shape = Image_Shape(ret.x, ret.y, ret.z);
            sample_size = shape.prod();

            //6.: Compute num_samples
            header_off = ftell(fd);
            fseek(fd, 0, SEEK_END);
            uint64_t num_bytes = ftell(fd) - header_off;
            fseek(fd, header_off, SEEK_SET);
            
            assert(num_bytes % (sizeOfType(ret.type) * sample_size) == 0);
            num_samples = num_bytes / (sizeOfType(ret.type) * sample_size);

            return ret;
        }

        fprintf(stderr, "[ERROR] The dataset file is of a version for which \"parseInputFile\" does not implement a header conversion! file %s, line %u", __FILE__, __LINE__);
        std::exit(-1);
    }

public:
    /*
        Allocates and fills cpu_tiles. Sets all related member variables.

        @param in, out: File name for files that contain input and output for neural network in sequencial order
        @param train_split: Which percentile of samples should be used for training (has to be between 0.f and 1.f)
        @param batch_size_: The batch size
        @param mb_cpu: How many megabytes of RAM this class is allowed to use
    */
    DatasetHandler(char* in, char* out, float train_split, uint32_t mb_cpu) :
        DatasetHandler(fopen(in, "rb"), fopen(out, "rb"), train_split, mb_cpu)
    {}
    DatasetHandler(FILE* fd_in_, FILE* fd_out_, float train_split, uint32_t mb_cpu) :
        batch_size(0u),
        augmentation_batches_cpu(0u), augmentation_batches_gpu(0u),
        num_workers(0u), workers(nullptr), thread_status(WORKER_STATUS::IDLE),
        ready_cpu(nullptr),
        ready_gpu(nullptr),
         in_header(),
        out_header(),
        agi_in (DATA_FORMAT(DISTRIBUTION::UNIFORM, -1.f), (uint32_t)-1, false, NULL, NULL, NULL, NULL, NULL, Offset2D<float>(-1.f, -1.f), Offset2D<int32_t>(-1, -1) ),//TODO: SUCKS
        agi_out(DATA_FORMAT(DISTRIBUTION::UNIFORM, -1.f), Offset2D<int32_t>((uint32_t)-1, (uint32_t)-1), -1.f, false )//TODO: SUCKS
    {
        //-1.: Finish initialization (except cpy_thread)
        fd_in = fd_in_;
        fd_out = fd_out_;
        sample_shape[2] = sample_shape[3] = Image_Shape(0u, 0u, 0u); sample_size[2] = sample_size[3] = 0u;
        aug_tiles[0] = aug_tiles[1] = nullptr; aug_padding[0] = aug_padding[1] = 0u;
        gpu_tiles[0] = gpu_tiles[1] = nullptr; cur_gpu[0] = cur_gpu[1] = nullptr;
#ifdef DEBUG
        workers_tile = nullptr;
#endif

        //0.: Check parameters
        assert(0.f <= train_split && train_split <= 1.0f && mb_cpu != 0 && fd_in_ != NULL && fd_out_ != NULL);
        printf("[INFO] Constructing DatasetHandler\n");

        //1.: Read in file
        uint32_t samples, samples1, samples2;
         in_header = HEADER_V1(parseInputFile(fd_in , sample_shape[0], sample_size[0], header_offset[0], samples1));
        out_header = HEADER_V1(parseInputFile(fd_out, sample_shape[1], sample_size[1], header_offset[0], samples2));
        if(samples1 != samples2)
            printf("[WARN] The dataset files contain a different number of samples (%u vs %u)!\n", samples1, samples2);
        samples = min(samples1, samples2);
        printf("[INFO] Using %u samples.\n", samples);
        
        //2.: Split samples in training and validation
        train_samples = ((uint32_t)(samples * train_split)) & ~(uint32_t)0b1;            //Alignment of 2
        validation_samples = samples - train_samples;
        printf("[INFO] Splitted dataset into %u training and %u validation.\n", train_samples, validation_samples);
        
        //3.: Check how many samples fit in a ram tile to fulfill the memory requirements
        uint32_t bytes_per_sample = (sample_size[0] + sample_size[1]) * sizeof(T);
        tile_samples_cpu = (mb_cpu * 1024ull * 1024ull) / bytes_per_sample;
        
        //4.: Does the whole dataset fit into RAM?
        if (tile_samples_cpu >= samples) {
            fit_in_ram = true;
            tile_samples_cpu = samples;                                                  //Do not allocate more than needed
            printf("[INFO] The whole dataset fits into ram.\n");
        }
        else {
            printf("[WARN] Dataset does not fit into ram. This might degrade performance. Additional %llu mb are needed.\n", ((samples - tile_samples_cpu) * (uint64_t)bytes_per_sample) / (1024ull * 1024ull));
            fit_in_ram = false;
        }
        assert(tile_samples_cpu != 0);

        //5.: Allocate memory
        uint64_t alloc_size = tile_samples_cpu * (bytes_per_sample);
        T *cpu_mem = (T*)malloc(alloc_size);

        assert(cpu_mem != NULL && cpu_mem != nullptr);
        printf("[INFO] Successfully allocated %llu mb.\n", alloc_size / (1024ull * 1024ull));

        cpu_tiles[0] = cpu_mem;
        cpu_tiles[1] = cpu_tiles[0] + sample_size[0] * tile_samples_cpu;

        //6.: Fill memory
        printf("[INFO] Reading dataset into ram.\n");
        fread(cpu_tiles[0], sizeof(T), sample_size[0] * tile_samples_cpu, fd_in );
        fread(cpu_tiles[1], sizeof(T), sample_size[1] * tile_samples_cpu, fd_out);
    }
    
    /*
        Returns the shapes of augmented in- an output samples
    */
    inline void getAugmentedShapes(Image_Shape& in_shape, Image_Shape& out_shape) {
        in_shape  = sample_shape[2];
        out_shape = sample_shape[3];
    }
    inline void getNumSamples(uint32_t& n_train_samples, uint32_t& n_validation_sample) {
        n_train_samples = train_samples;
        n_validation_sample = validation_samples;
    }

    /*
        Sets augmenation and infers size of augmentated samples (sample_size[2] and sample_size[3])
    */
    inline void set_augmentation(struct AugmentationInfo2D_IN& agi_in_, struct AugmentationInfo2D_OUT& agi_out_) {
        //0.: Check parameters
        //Whether the aspect ratio and channels are valid will be checked by worker threads! Blur range remains unchecked
        assert(agi_in_.format.distribution == Image::DISTRIBUTION::UNIFORM && agi_in_.rotate >= 0.f && agi_in_.random_noise >= 0.f & agi_in_.random_dropout >= 0.f && agi_in_.random_dropout <= 1.f && agi_in_.random_saturation >= 0.f && agi_in_.random_brightness >= 0.f); //Standart deviation has to be >=0, probability is in ]0, 1[.
        assert(((agi_in_.crop.x >= 0.f && agi_in_.crop.y >= 0.f) || agi_in_.crop.x == -1)  && agi_in_.crop.x <= 1.f && agi_in_.crop.y <= 1.f);
        assert(agi_in_.resize.x == -1 || (agi_in_.resize.x > 0 && agi_in_.resize.y > 0));

        assert(agi_out_.format.distribution == Image::DISTRIBUTION::UNIFORM && agi_out_.label_smoothing >= 0.f);
        assert(agi_out_.resize.x == -1 || (agi_out_.resize.x > 0 && agi_out_.resize.y > 0));

        
        //1.: Set variables
        agi_in  = agi_in_ ;
        agi_out = agi_out_;

        
        //2.: Compute shape lenght of augmented batches
        //sample_shape[2]
        if (agi_in.resize.x == -1) {
            if (agi_in.crop.x == -1)                          //No change
                sample_shape[2] = sample_shape[0];
            else                                              //Crop controlls size
                sample_shape[2] = Image_Shape((uint32_t)(sample_shape[0].x * agi_in.crop.x), (uint32_t)(sample_shape[0].y * agi_in.crop.y), sample_shape[0].z);
        }
        else                                                  //Resize controls size
          sample_shape[2] = Image_Shape((uint32_t)agi_in.resize.x, (uint32_t)agi_in.resize.y, sample_shape[0].z);

        //sample_shape[3]
        if (agi_out.resize.x == -1) {
            if (agi_in.crop.x == -1 || !agi_out.do_input_aug) //No change
                sample_shape[3] = sample_shape[1];
            else                                              //Crop controlls size
                sample_shape[3] =  Image_Shape((uint32_t)(sample_shape[1].x * agi_in.crop.x), (uint32_t)(sample_shape[1].y * agi_in.crop.y), sample_shape[1].z);
        }
        else                                                  //Resize controls size
            sample_shape[3] = Image_Shape((uint32_t)agi_out.resize.x, (uint32_t)agi_out.resize.y, sample_shape[1].z);

        //Sizes
        sample_size[2] = sample_shape[2].x * sample_shape[2].y * sample_shape[2].z;
        sample_size[3] = sample_shape[3].x * sample_shape[3].y * sample_shape[3].z;

        //3.: Compute padding
        aug_padding[0] = max(sample_size[0], sample_size[2]) - sample_size[2];
        aug_padding[1] = max(sample_size[1], sample_size[3]) - sample_size[3];
    }
    inline void get_augmentation(AugmentationInfo2D_IN& agi_in_, AugmentationInfo2D_OUT& agi_out_) {
      agi_in_  = agi_in ;
      agi_out_ = agi_out;
    }

    /*
        Allocates aug_tiles and gpu_tiles.

        @param b: batch size
    */
    inline void set_batching(uint32_t b, uint32_t augmentation_batches_cpu_, uint32_t augmentation_batches_gpu_) {
        //0.: Check parameters
        assert((b != 0) && (augmentation_batches_cpu_ > 1) && (augmentation_batches_gpu_ > 1));
        assert(b % 8 == 0);  //Batch size need to be multiple of 8 for avx purposes

        //1.: Free previous memory
        if (augmentation_batches_cpu != 0) { //Make sure this funtion is not called for the first time
            cudaFreeHost(aug_tiles[0]);
            cudaFree(gpu_tiles[0]);
            free(ready_cpu);
            free(ready_gpu);
        }

        //2.: Set variables
        batch_size = b;
        augmentation_batches_cpu = augmentation_batches_cpu_;
        augmentation_batches_gpu = augmentation_batches_gpu_;

        //3.: Allocate aug_tiles and ready_cpu
        T* aug_mem;
        cudaMallocHost(&aug_mem, sizeof(T) * ((sample_size[2] + sample_size[3]) * batch_size + aug_padding[0] + aug_padding[1]) * augmentation_batches_cpu);

        aug_tiles[0] = aug_mem;
        aug_tiles[1] = aug_tiles[0] + (sample_size[2] * batch_size + aug_padding[0]) * augmentation_batches_cpu;

            //BUGP("Cpu mem: ");
            //PRINT_VAR(aug_tiles[0]);
            //BUGP("\t\t");
            //PRINT_VAR(aug_tiles[1]);
            //BUGP("\n");

        ready_cpu = new std::atomic<BATCH_STATUS>[augmentation_batches_cpu];
        for (uint32_t u = 0; u != augmentation_batches_cpu; u++)
            ready_cpu[u].store(BATCH_STATUS::USED);

        //4.: Allocate gpu_tiles and gpu_sync
        T* gpu_mem;
        cudaMalloc(&gpu_mem, sizeof(T) * (sample_size[2] + sample_size[3]) * batch_size * augmentation_batches_gpu);

        gpu_tiles[0] = gpu_mem;
        gpu_tiles[1] = gpu_tiles[0] + sample_size[2] * batch_size * augmentation_batches_cpu;

            //BUGP("GPU mem: ");
            //PRINT_VAR(gpu_tiles[0]);
            //BUGP("\t\t");
            //PRINT_VAR(gpu_tiles[1]);
            //BUGP("\n");

        ready_gpu = new std::atomic<BATCH_STATUS>[augmentation_batches_gpu];
        for (uint32_t u = 0; u != augmentation_batches_gpu; u++)
            ready_gpu[u].store(BATCH_STATUS::USED);
    }
    inline uint32_t get_batch_size() {
        return batch_size;
    }

    /*
        Start the threads that copy and augment data from dataset ram tile to augmentation tile

        @param n: Number of workers
        @param num_streams: number of cuda streams to use to copy samples from augmentation tiles to gpu
        @param stat: Initialize worker status
    */
    inline void start_workers(uint32_t n, uint32_t num_streams, WORKER_STATUS stat = WORKER_STATUS::IDLE) {
        //0.: Check parameters
        assert(n > 0);
        assert(stat < WORKER_STATUS::IDLE && workers == nullptr);

        //1.: Copy variables
        num_workers = n;

#ifdef DEBUG
        //1.5.: If debugging is enabled, initialize the debug-variables
        workers_tile = (int32_t*)malloc(sizeof(int32_t) * (n + 1));
        workers_tile[0] = num_workers;
        for (uint32_t ind = 0; ind != num_workers; ind++) {
            workers_tile[ind + 1] = -1;
        }
#endif

        //2.: Reset stopping signal
        thread_status.store(stat);

        //3.: Start num_workers new worker threads
        workers = (std::thread*)malloc(num_workers * sizeof(std::thread));
        for (uint32_t u = 0; u != num_workers; u++) {
            new (&workers[u]) std::thread(&DatasetHandler::worker_thread, this, u);    //This uses the "std::invoke" syntax
            //workers[u] = std::thread(&DatasetHandler::worker_thread, this, u+1);
            workers[u].detach();
        }

        //4.: Start one memcpy thread
        cpy_thread = std::thread(&DatasetHandler::memcpy_thread, this, num_streams);
        cpy_thread.detach();
    }
    /*
        Opens a windows which shows the content of every augmentation tile as well as its status
    */
    inline void start_debugWindow() {
        std::thread debugger = std::thread(&DatasetHandler::debug_thread, this);
        debugger.detach();
    }

    /*
        Switches input to training or validation. Invalidates all gpu samples.
    */
    template<bool training> void setDataInput() {
        //1.: Sent signal to worker threads
        if constexpr (training)
            thread_status.store(WORKER_STATUS::TRAINING);
        else
            thread_status.store(WORKER_STATUS::VALIDATION);

        //2.: Reload cpu and gpu tiles.
        for (uint32_t u = 0; u != augmentation_batches_cpu; u++) {
#if 0 //TODO: DECIDE which version is better. The second allows worker threads to use loading bits and only uses one atomic operation. Is it working, though?
            //If the status is LOADING, flip its DEPRECATED bit
            BATCH_STATUS expected = BATCH_STATUS::LOADING;
            ready_cpu[u].compare_exchange_strong(expected, BATCH_STATUS::DEPRECATED);       //If DEPRECATED bit was not set, it now is (and only it, which is not a problem this for the cpu all other bits are 0 while LOADING)

            //expected = BATCH_STATUS::DEPRECATED;
            //ready_cpu[u].compare_exchange_strong(expected, BATCH_STATUS::LOADING);        //If DEPRECATED bit was set, it now isn't (as well as all other, which is not a problem this for the cpu all other bits are 0 while LOADING)

            //If the batch is READY, set it to USED
            expected = BATCH_STATUS::READY;
            ready_cpu[u].compare_exchange_strong(expected, BATCH_STATUS::USED);
#else
            //If the status is LOADING, flip its DEPRECATED bit
            BATCH_STATUS stored_value = ready_cpu[u].load();

            BATCH_STATUS stored_loading     = (BATCH_STATUS)(abs((int32_t)stored_value));         //stored_value but highest bit is not set. When stored value was positiv (exactly when LOADING status), it stayes the same. Otherwise, it changes
            BATCH_STATUS toggled_deprecated = (BATCH_STATUS)((int32_t)stored_value ^ (int32_t)BATCH_STATUS::DEPRECATED); //stored_value with toggled DEPRECATED bit
            ready_cpu[u].compare_exchange_strong(stored_loading, toggled_deprecated);             //If ready_gpu[u] is a loading status, its deprecated value gets changed

            //If the batch is READY, set it to USED
            BATCH_STATUS expected = BATCH_STATUS::READY;
            ready_cpu[u].compare_exchange_strong(expected, BATCH_STATUS::USED);
#endif        
        }
        for (uint32_t u = 0; u != augmentation_batches_gpu; u++) {
            //If the status is LOADING, flip its DEPRECATED bit
            BATCH_STATUS stored_value = ready_gpu[u].load();
            
            BATCH_STATUS stored_loading     = (BATCH_STATUS)(abs((int32_t)stored_value));          //stored_value but highest bit is not set. When stored value was positiv (exactly when LOADING status), it stayes the same. Otherwise, it changes
            BATCH_STATUS toggled_deprecated = (BATCH_STATUS)((int32_t)stored_value ^ (int32_t)BATCH_STATUS::DEPRECATED); //stored_value with toggled DEPRECATED bit
            ready_gpu[u].compare_exchange_strong(stored_loading, toggled_deprecated);             //If ready_gpu[u] is a loading status, its deprecated value gets changed

            //If the batch is ready, set it to used
            BATCH_STATUS expected = BATCH_STATUS::READY;
            ready_gpu[u].compare_exchange_strong(expected, BATCH_STATUS::USED);
        }

        //3.: Reset cur_gpu (to avoid reloading one batch two times)
        cur_gpu[0] = cur_gpu[1] = nullptr;
    }
    inline void stop_workers() {
        //0.: Check whether there even are threads running
        assert(workers != nullptr);

        //1.: Set exit signal
        thread_status.store(WORKER_STATUS::STOP);
        
        //2.: Destroy handles
        while (thread_status.load() != num_workers + 1) {}; //Wait until all workers and memcpy thread exited
        //TODO: HOW TO DO THIS
        //for (uint32_t u = 0; u != num_workers; u++)
        //    ~workers[u]();
        //~cpy_thread();

        //3.: Free memory of handles
        free(workers);
        workers = nullptr;
    }

    //Returns to *in and *out
    template<bool training> void advance(T** in, T** out) {
        //0.: Check, if we switched from training to validation or the other way around
        int32_t stat = thread_status.load();
        assert(stat < WORKER_STATUS::STOP);                    //Could run into infinite loop

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
            ready_gpu[(cur_gpu[0] - gpu_tiles[0]) / (batch_size * sample_size[2])].store(BATCH_STATUS::USED);

        //2.: Increment cur_gpu
        if (cur_gpu[0] == nullptr) {
            cur_gpu[0] = gpu_tiles[0];
            cur_gpu[1] = gpu_tiles[1];
        }
        else if (cur_gpu[0] == gpu_tiles[0] + batch_size * sample_size[2] * (augmentation_batches_gpu - 1)) { //Whole tile was used
            cur_gpu[0] = gpu_tiles[0];                      //Start at beginning again
            cur_gpu[1] = gpu_tiles[1];                      //Start at beginning again
        }
        else {
            cur_gpu[0] += batch_size * sample_size[2];
            cur_gpu[1] += batch_size * sample_size[3];
        }

        //3.: Set output
        *in  = cur_gpu[0];
        *out = cur_gpu[1];

        //4.: Synchronize
        while (ready_gpu[(cur_gpu[0] - gpu_tiles[0]) / (batch_size * sample_size[2])].load() != BATCH_STATUS::READY) {} //Wait until data is ready
    }

    ~DatasetHandler() {
        if (workers != nullptr)
            stop_workers();

        fclose(fd_in);
        fclose(fd_out);

        free(cpu_tiles[0]);
        cudaFreeHost(aug_tiles[0]);
        cudaFree(gpu_tiles[0]);

        free(ready_cpu);
        free(ready_gpu);
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

//Old main function
#if 0
using namespace DatasetAssemble;

#define P_IN    "C:/Users/julia/source/repos/AI/Datasets/1/Raw/In"
#define P_OUT   "C:/Users/julia/source/repos/AI/Datasets/1/Raw/Out"
#define P_D_IN  "C:/Users/julia/source/repos/AI/Datasets/1/in.jvdata"
#define P_D_OUT "C:/Users/julia/source/repos/AI/Datasets/1/out.jvdata"

int main() {
    //Cuda Device options
    //Set up priorities and stuff
    Random::init_rand();

    //1.: Build a dataset
    Offset2D<uint32_t> size(100, 100);
    generateDatasetFile_Image<float>(P_IN, P_D_IN, size);
    generateDatasetFile_Raw<float>(P_OUT, P_D_OUT);

    //2.: Build a dataset handler
    AugmentationInfo2D_IN  agi_in (DATA_FORMAT(DISTRIBUTION::UNIFORM, 1.f), 1, false, 0.f, 0.05f, 0.2f, 0.1f, 0.1f, Offset2D<float>(0.95f, 0.95f), Offset2D<int32_t>(200, 200));
    AugmentationInfo2D_OUT agi_out(DATA_FORMAT(DISTRIBUTION::UNIFORM, 1.f), Offset2D<int32_t>(-1, -1), 0.1f, false);

    DatasetHandler<float> handler(P_D_IN, P_D_OUT, 0.7f, 8192);
    handler.set_augmentation(agi_in, agi_out);
    handler.set_batching(8, 4, 4);
    handler.start_workers(3, 2, WORKER_STATUS::TRAINING);

    handler.start_debugWindow();
    //STALL();
#if 0
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1s);
    float* gpu_dat = handler.gpu_tiles[0];
    float* cpu_dat = (float*)malloc(sizeof(float) * 200u * 200u * 3u);
    for (uint32_t u = 0; u != 8 * 2; u++) {
        cudaMemcpy(cpu_dat, gpu_dat, sizeof(float) * 200u * 200u * 3u, cudaMemcpyDeviceToHost);
        Image::show<float, true>(cpu_dat, Image_Shape(200u, 200u, 3u));
        gpu_dat += 200u * 200u * 3u;
    }
#elif 1
    //using namespace std::chrono_literals;
    //std::this_thread::sleep_for(1s);
    float* gpu_dat, * __g__;

    Image_Shape in_shape, out_shape;
    handler.getAugmentedShapes(in_shape, out_shape);
    uint32_t batch_size = handler.get_batch_size();

    float* save_in, * save_out;
    cudaMalloc(&save_in ,  in_shape.prod() * batch_size * sizeof(float));
    cudaMalloc(&save_out, out_shape.prod() * batch_size * sizeof(float));

    cudaStream_t save_stream;
    cudaStreamCreateWithFlags(&save_stream, cudaStreamNonBlocking);
    
    float* cpu_dat = (float*)malloc(max(in_shape.prod(), out_shape.prod()) * batch_size * sizeof(float));
    
    uint32_t n = 0; bool b = true;
    while (true) {
        if (++n % 8 == 0)
            b = !b;
        if (b)
            handler.advance<true>(&gpu_dat, &__g__);
        else
            handler.advance<false>(&gpu_dat, &__g__);

        

        //STALL();
        //std::this_thread::sleep_for(1ms);

        cudaMemcpyAsync(save_in , gpu_dat,  in_shape.prod() * batch_size * sizeof(float), cudaMemcpyDeviceToDevice, save_stream);
        cudaMemcpyAsync(save_out, __g__  , out_shape.prod() * batch_size * sizeof(float), cudaMemcpyDeviceToDevice, save_stream);
        cudaStreamSynchronize(save_stream);

        if(GetKeyState('T') & 0x8000)
        {
            cudaMemcpy(cpu_dat, save_in, in_shape.prod() * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
            Image::show<float, true>(cpu_dat + 0 * in_shape.prod(), in_shape);

            cudaMemcpy(cpu_dat, save_out, out_shape.prod() * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
            ARR_PRINT(cpu_dat + 7 * out_shape.prod(), out_shape.x, out_shape.y);
        }
        



        //getchar();
    }
#endif
  

    //while (true) {}
    getchar();
    //3.: Test augmentation tile


}


//Linux:   sudo clang++ Dataset.cpp -o Dataset.exe -I"/home/julian/Libs/CImg-2.9.2_pre070420" -I"/usr/local/cuda-10.0/include" -I"/home/julian/Libs/glfw-3.3.2/include" -I"/home/julian/Libs/glew-2.1.0/include" -I"/home/julian/Libs/freetype-2.10.3/include" -L"/usr/local/cuda-10.0/lib64" -O0 -march=native -m64 -std=c++17 -Wall -ldl -lrt -lpthread -lz -lpng -ljpeg -lGL -lGLU -lGLEW -lglfw3 -lfreetype -lX11 -lcudart_static -lstdc++fs -g -DDEBUG -DMAIN -DEXPERIMENTAL_FILESYSTEM
//Cygwin:  g++ Dataset.cpp -o Dataset.exe -I"D:\Librarys\CImg-2.9.2_pre070420" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin" -O3 -march=native -std=c++17 -Wall -lpthread -lz -ldl -lpng -ljpeg -lX11 -g
//Windows: clang++ Dataset.cpp -o Dataset.exe -I"D:\Librarys\CImg-2.9.2_pre070420" -I"D:\Librarys\VS-NuGet\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include" -I"D:\Librarys\GLFW\include" -I"D:\Librarys\glew-2.1.0\include" -I"D:\Librarys\freetype-2.10.3\include" -L"D:\Librarys\GLFW\lib" -L"D:\Librarys\glew-2.1.0\lib\Release\x64" -L"D:\Librarys\VS-NuGet\lib" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64" -L"D:\Librarys\freetype-2.10.3\objs" -O0 -march=native -m64 -std=c++17 -Wall -lzlib -llibpng16 -ljpeg -lkernel32 -luser32 -lgdi32 -lopengl32 -lglu32 -lglew32 -lglfw3dll -lpsapi -lwinspool -lcomdlg32 -ladvapi32 -lshell32 -lole32 -loleaut32 -luuid -lodbc32 -lodbccp32 -lcudart_static -lfreetype -g -DDEBUG
#endif
