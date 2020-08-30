#ifndef DEBUG
#define NDBUG
#endif

#define _LARGEFILE_SOURCE
#define _FILE_OFFSET_BITS 64

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>

#include <algorithm>
#include <numeric>
#include <string>
#include <inttypes.h>
#include <chrono>

#include <atomic>
#include <thread>

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <x86intrin.h>

#include <stdlib.h>

#include "util.cpp"
using namespace Random;
using namespace Image;
using namespace DatasetAssemble;


//TODO: CHECK STACK-RETURNS
//========================================================================================
enum WORKER_STATUS : int32_t { VALIDATION = -3, TRAINING = -2, IDLE = -1, STOP = 0}; //Don't change values!!
enum BATCH_STATUS : int32_t {USED = -2, READY = -1, LOADING = 0};                      //Don't change values!!
#define CPU_BATCH_BLOCK (1 << (8 * sizeof(BATCH_STATUS) - 2)) /*Used to block other worker threads. Has to be one bit not in BATCH_STATUS*/

struct AugmentationInfo2D_IN {
    float aspect;                //Input  vector is understood as an image with channels_in channels and aspect ration aspect_in
    uint32_t channels;           //The input vector is understood as an Image of the shape (dim_x, dim_y, channels). Augmentations are applied on this image
    DATA_FORMAT format;          //Transform data to this format

    //The "if not zero" part is redundant and only used for understanding, that incase of 0 nothing happens.
    uint32_t blur_range;         //If not zero, applies gaussian blur with this distance 
    bool  flip;                  //If true    , the Image has a 50% chance of getting mirrored horitontaly
    float rotate;                //If not zero, the Image is rotated by an random angle with mean 0 and this standart deviation
    float random_noise;          //If not zero, to every channel a random number with mean 0 and this standart deviation stored
    float random_dropout;        //If not zero, this stores the probability for each pixel to individually become black
    float random_saturation;     //If not zero, the saturation of each pixel is multiplies by a constant random number with mean 0 and this standart deviation
    float random_brightness;     //If not zero, the brightness of each pixel is multiplies by a constant random number with mean 0 and this standart deviation
                                 
    Offset2D<float> crop;        //Crops a random part of shape (crop.x*dim_x, crop.y*dim_y, channels) from the input and output. <0 indicates to skip this step
    Offset2D<int32_t> resize;    //Resizes input to shape (resize.x, resize.y channels_in). resize.x=-1 indicates to skip this step
};

struct AugmentationInfo2D_OUT {
    float aspect;                //Output vector is understood as an image of shape (sqrt(sample_size[1]/channels_out * (1/aspect_out)), sqrt(sample_size[1]/channels_out * (aspect_out)), channels_out)
    uint32_t channels;           //The output vector is understood as an Image of the shape (dim_x, dim_y, channels). Augmentations are applied on this image
    DATA_FORMAT format;          //Transform data to this format

    Offset2D<int32_t> resize;    //Resizes output to (resize.x, resize.y, channels_out). resize.x=-1 indicates to skip this step
    float label_smoothing;       //Apply label smoothing
};

//========================================================================================================================

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
    HEADER in_header, out_header;       //Header of files containing the datasets 
    uint32_t sample_size[4];            //How many number are contained in an input/output sample in the dataset ([0],[1]) and in augmentated samples ([2],[3])

    bool fit_in_ram;                    //True if the whole dataset fits into RAM
    uint32_t train_samples;             //Number of all training samples
    uint32_t validation_samples;        //Number of all validation samples
    uint32_t batch_size;                //Number of samples to return to neural network per call. Has to be multiple of 8 (for AVX)

    uint32_t tile_samples_cpu;          //Number of samples per tile.
    uint32_t augmentation_batches_cpu;  //Number of augmented batches stored in ram
    uint32_t augmentation_batches_gpu;  //Number of augmented batches stored in vram

    AugmentationInfo2D_IN  agi_in ;     //Augmentations performed on input
    AugmentationInfo2D_OUT agi_out;     //Augmentations performed on output
    uint32_t num_workers;               //Number of threads launched for augmentation
    std::thread* workers;               //Handle to threads launched for augmentation   (nullptr, when no workers are started)
    std::thread  cpy_thread;            //Handle to thread that copies batches from cpu to gpu
    std::atomic<int32_t> thread_status; //-3=validation, -2=training, -1=idle, >=0: threads should exit, this is the number of threads that already did

    T* cpu_tiles[2];                    //Start of tiles in ram that hold raw dataset ( ={input, output} )
    T* aug_tiles[2];                    //Start of tiles in ram that hold augmented batches of dataset ( ={input, output} )
    uint32_t aug_padding[2];            //Padding to add to aug tiles to be able to accomodate copy of size sample_size[0/1] and intermediates
    std::atomic<int32_t>* ready_cpu;    //Stores for each element of aug_tiles, whether it contains a new unused batch
    
    T* gpu_tiles[2];                    //Holds start of tiles (gpu) = {input, output}  
    T* cur_gpu[2];                      //Holds last used element in tiles (gpu) = {input, output}. If gpu_tiles was reloaded, cur_gpu[0]=nullptr
    std::atomic<int32_t>* ready_gpu;    //-1=ready, -2=used, >=0 means copy is taking place from augmentation tile of this index


    inline void load_sample_augmentation(T* aug_in, T* aug_out, uint32_t sample_off_in, uint32_t sample_off_out, bool in_ram){
        if (in_ram) { /*In Ram tile*/                     
            memcpy(aug_in , cpu_tiles[0][sample_off_in ], sample_size[0] * sizeof(T));                      
            memcpy(aug_out, cpu_tiles[1][sample_off_out], sample_size[1] * sizeof(T));                      
        }                                                                                    
        else { /*Go to disk*/                                                                
            lseek(fd_in , sizeof(T) * sample_off_in  +  in_header.bytes, SEEK_SET);   
            lseek(fd_out, sizeof(T) * sample_off_out + out_header.bytes, SEEK_SET);           
            read(fd_in , aug_in , sample_size[0] * sizeof(T));                                     
            read(fd_out, aug_out, sample_size[1] * sizeof(T));                                           
        }
    }

    
    inline void apply_augmentations(T* aug_in, T* aug_out, Image_Shape in_shape, Image_Shape out_shape, bool training){
        if (training){
            //Input: Everything
            //Output: Resize + label_smoothing + remap

            Image_Shape  in_shape_ =  in_shape;  //The shape of the input and output can change because of augmentation. This holds the actual shape at every moment
            Image_Shape out_shape_ = out_shape;  //The shape of the input and output can change because of augmentation. This holds the actual shape at every moment

            //Input
            if (agi_in.crop.x != -1) {
                in_shape_  *= agi_in.crop;
                out_shape_ *= agi_in.crop;
            
                Offset2D<float> pos  = {rand_float(1.f), rand_float(1.f) };
            
                crop<T>(aug_in ,  in_shape_, ( in_shape.getOffset2D() -  in_shape_.getOffset2D()) * pos,  in_shape_,  in_header.order);
                crop<T>(aug_out, out_shape_, (out_shape.getOffset2D() - out_shape_.getOffset2D()) * pos, out_shape_, out_header.order);
            }
            if(agi_in.blur_range != 0.f) {
                pseudo_gausblur<T, PADDING::ZERO_PADDING_NORMAL>(aug_in, in_shape_, abs(rand_normal(agi_in.blur_range)), 3, in_header.order); //TODO:PADDING AND N
            }
            if (agi_in.flip && rand_prob(0.5f)) { /*50% chance*/
                flip<T>(aug_in ,  in_shape_,  in_header.order);
                flip<T>(aug_out, out_shape_, out_header.order);
            }
            if (agi_in.random_noise != 0.f) {
                random_noise<T>(aug_in, in_shape_, agi_in.random_noise);
            }
            if (agi_in.random_dropout != 0.f) {                                                     
                random_dropout<T>(aug_in, in_shape_, agi_in.random_dropout, in_header.order);
            }
            if (agi_in.random_saturation != 0.f) {                                                  
                mul_saturation<T>(aug_in, in_shape_, rand_normal(agi_in.random_saturation), in_header.format, in_header.order);
            }
            if (agi_in.random_brightness != 0.f) {
                mul_brightness<T>(aug_in, in_shape_, rand_normal(agi_in.random_brightness), in_header.format, in_header.order);
            }
            if (agi_in.rotate != 0.f) {
                float rot_deg = rand_normal(agi_in.rotate);
                rotate<T>(aug_in ,  in_shape_, rot_deg);
                rotate<T>(aug_out, out_shape_, rot_deg); 
            }
            if (agi_in.resize.x != -1) {                                                           
                resize<T>(aug_in , in_shape_, agi_in.resize);
                in_shape_.setOffset2D(agi_in.resize);
            }
            if(agi_in.format != in_header.format){
                remap_format<T>(aug_in, in_shape_, in_header.format, agi_in.format, in_header.order);
            }

            //Output
            if (agi_out.resize.x != -1) {                                                           
                resize<T>(aug_out, out_shape_, agi_out.resize);
                out_shape_.setOffset2D(agi_out.resize);
            }
            if(agi_out.label_smoothing != 0.f){
                label_smoothing<T>(aug_out, out_shape_, agi_out.label_smoothing);
            }
            if(agi_out.format != out_header.format){
                remap_format<T>(aug_out, out_shape_, out_header.format, agi_out.format, out_header.order);
            }
        } else{
            //Input: Remap + resize
            //Output: Remap + resize
          
            Image_Shape  in_shape_ =  in_shape;  //The shape of the input and output can change because of augmentation. This holds the actual shape at every moment
            Image_Shape out_shape_ = out_shape;  //The shape of the input and output can change because of augmentation. This holds the actual shape at every moment
        
            //Input
            if (agi_in.resize.x != -1) {                                                           
                resize<T>(aug_in , in_shape_, agi_in.resize);
                in_shape_.setOffset2D(agi_in.resize);
            }
            if(agi_in.format != in_header.format){
                remap_format<T>(aug_in, in_shape_, in_header.format, agi_in.format, in_header.order);
            }

            //Output
            if (agi_out.resize.x != -1) {                                                           
                resize<T>(aug_out, out_shape_, agi_out.resize);
                out_shape_.setOffset2D(agi_out.resize);
            }
            if(agi_out.format != out_header.format){
                remap_format<T>(aug_out, out_shape_, out_header.format, agi_out.format, out_header.order);
            }
        }
    }
  
    void worker_function(uint64_t rand_init) {
        //1.: Generate a deterministic random key
        Key my_key;
        uint64_t r = (763808623281539ull * rand_init) ^ (2009741990ull * rand_init);
        avx_xorshift128plus_init(r ^ (r << 5) ^ (r >> 19), r ^ (r << 32) ^ (r >> 32), my_key);

        //2.: Set AVX constants
        __m256  rand_normalize1      = _mm256_set1_ps((float)train_samples / (float)(1 << 31));
        __m256  rand_adder1          = _mm256_set1_ps(train_samples >> 1);

        __m256  rand_normalize2      = _mm256_set1_ps((float)validation_samples / (float)(1 << 31));
        __m256  rand_adder2          = _mm256_set1_ps(train_samples + (validation_samples >> 1));

        __m256i tile_samples_cpu_vec = _mm256_set1_epi32(tile_samples_cpu);
        __m256i sample_size_0        = _mm256_set1_epi32(sample_size[0]);
        __m256i sample_size_1        = _mm256_set1_epi32(sample_size[1]);

        uint32_t* off = (uint32_t*)aligned_alloc(32, 16 * sizeof(uint32_t));                      //Store offset of T's to start of sample

        //3.: Calculate sizes for augmentation
        Image_Shape in_shape (sample_size[0],  agi_in.aspect,  agi_in.channels);                  // in_shape store the size of the  input sample at the moment during the calculations
        Image_Shape out_shape(sample_size[1], agi_out.aspect, agi_out.channels);                  //out_shape store the size of the output sample at the moment during the calculations

        //4.: Work loop 
        while (thread_status.load() < WORKER_STATUS::STOP) {                                      //Break if exit signal is reached
            for (uint32_t b = 0; b != augmentation_batches_cpu; b++) {                            //Go through all batches in augmentation tile
                int32_t s = ready_cpu[b].fetch_or(CPU_BATCH_BLOCK);                               //Block this value so it is not "used" for any other thread => no race condition
                if (s == BATCH_STATUS::USED) {                                                    //If a batch is not ready, work on it
                    //Reload batch b in augmentation tile                                       
                    bool training = (thread_status.load() == WORKER_STATUS::TRAINING);
                    for (uint32_t sample = 0; sample != batch_size; sample += 8) {                //Loop is unrolled 8 times
                        __m256 random = _mm256_cvtepi32_ps(avx_xorshift128plus(my_key));          //[-2^31, 2^31]
                        random = _mm256_add_ps(_mm256_mul_ps(random,                              \
                                    training ? rand_normalize1 : rand_normalize2),                \
                                    training ? rand_adder1     : rand_adder2);                    //[0, train_samples] or [train_samples, train_samples + validation_samples]
                        __m256i gen = _mm256_cvtps_epi32(random);
                        
                        //TODO: SWITCH INPUT TO VALIDATION AND CHANGE TO rand_normalize2 AND rand_adder2
                        
                        __m256i  in_tile = _mm256_cmpgt_epi32(tile_samples_cpu_vec, gen);         //True, if samples is in ram tile
                        uint32_t in_mask = _mm256_movemask_epi8(in_tile);                         //Most significant bits
                            
                        __m256i ind1 = _mm256_mullo_epi32(gen, sample_size_0);                    //Compute input  indices
                        __m256i ind2 = _mm256_mullo_epi32(gen, sample_size_1);                    //Compute output indices
                        _mm256_store_si256((__m256i*) &off[0], ind1);
                        _mm256_store_si256((__m256i*) &off[8], ind2);
                        
                        //Load and process all the samples                        
                        T* aug_in  = aug_tiles[0][b * (batch_size * sample_size[2] + aug_padding[0])]; \
                        T* aug_out = aug_tiles[1][b * (batch_size * sample_size[3] + aug_padding[1])];
                        load_sample_augmentation(aug_in, aug_out, off[0], off[8], fit_in_ram || ((in_mask >> (4 * 0)) & 0b1));
                        apply_augmentations(aug_in, aug_out, in_shape, out_shape, training);
                        
                        aug_in  += batch_size * sample_size[2];
                        aug_out += batch_size * sample_size[3];
                        load_sample_augmentation(aug_in, aug_out, off[1], off[9], fit_in_ram || ((in_mask >> (4 * 1)) & 0b1));
                        apply_augmentations(aug_in, aug_out, in_shape, out_shape, training);
                        
                        aug_in  += batch_size * sample_size[2];
                        aug_out += batch_size * sample_size[3];
                        load_sample_augmentation(aug_in, aug_out, off[2], off[10], fit_in_ram || ((in_mask >> (4 * 2)) & 0b1));
                        apply_augmentations(aug_in, aug_out, in_shape, out_shape, training);
                        
                        aug_in  += batch_size * sample_size[2];
                        aug_out += batch_size * sample_size[3];
                        load_sample_augmentation(aug_in, aug_out, off[3], off[11], fit_in_ram || ((in_mask >> (4 * 3)) & 0b1));
                        apply_augmentations(aug_in, aug_out, in_shape, out_shape, training);
                        
                        aug_in  += batch_size * sample_size[2];
                        aug_out += batch_size * sample_size[3];
                        load_sample_augmentation(aug_in, aug_out, off[4], off[12], fit_in_ram || ((in_mask >> (4 * 4)) & 0b1));
                        apply_augmentations(aug_in, aug_out, in_shape, out_shape, training);
                        
                        aug_in  += batch_size * sample_size[2];
                        aug_out += batch_size * sample_size[3];
                        load_sample_augmentation(aug_in, aug_out, off[5], off[13], fit_in_ram || ((in_mask >> (4 * 5)) & 0b1));
                        apply_augmentations(aug_in, aug_out, in_shape, out_shape, training);
                        
                        aug_in  += batch_size * sample_size[2];
                        aug_out += batch_size * sample_size[3];
                        load_sample_augmentation(aug_in, aug_out, off[6], off[14], fit_in_ram || ((in_mask >> (4 * 6)) & 0b1));
                        apply_augmentations(aug_in, aug_out, in_shape, out_shape, training);
                        
                        aug_in  += batch_size * sample_size[2];
                        aug_out += batch_size * sample_size[3];
                        load_sample_augmentation(aug_in, aug_out, off[7], off[15], fit_in_ram || ((in_mask >> (4 * 7)) & 0b1));
                        apply_augmentations(aug_in, aug_out, in_shape, out_shape, training);
                    }
                    //Now, the whole batch is loaded. Currently, operations on the whole batch are not supported

                    //Mark batch as ready
                    ready_cpu[b].store(BATCH_STATUS::READY);
                    break;                                                                        //When finished, check again for exit signal and start at the beginning
                }
                ready_cpu[b] &= ~CPU_BATCH_BLOCK;
            }
        }
        thread_status++;
    }

    void memcpy_thread(uint32_t num_streams){
        //1.: Create cuda variables
        cudaEvent_t gpu_sync[augmentation_batches_gpu];                        //Events to tell when memory-transfer to gpu are finished
        for (uint32_t u = 0; u != augmentation_batches_gpu; u++) {
            cudaEventCreateWithFlags(&gpu_sync[u], cudaEventDisableTiming);
        }

        cudaStream_t data_stream[num_streams];
        for(uint32_t u = 0; u != num_streams; u++) {
            cudaStreamCreateWithFlags(&data_stream[u], cudaStreamNonBlocking);
        }
        cudaStream_t*  cur_stream = +data_stream;
        cudaStream_t* last_stream = +data_stream + (num_streams - 1);
        
        
        bool found;
        while(thread_status.load() < WORKER_STATUS::STOP){
            //0.: Check for idle
            if(thread_status.load() != WORKER_STATUS::IDLE)
                continue;
          
            //1.: Launch new copies
            for(uint32_t ind = 0; ind != augmentation_batches_gpu; ind++){
                int32_t gpu_stat = ready_gpu[ind].load();
                if(gpu_stat == BATCH_STATUS::USED) {
                    //Load next ready batch to gpu tile ind
                    found = false;
                    for(uint32_t cpu_ind = 0; cpu_ind != augmentation_batches_cpu; cpu_ind++){
                        if(ready_cpu[cpu_ind].load() == BATCH_STATUS::READY){
                            //Found a loadable augmentation tile
                            T* in  = gpu_tiles[0] + ind * batch_size * sample_size[2];
                            T* out = gpu_tiles[1] + ind * batch_size * sample_size[3]; 
   
                            cudaMemcpyAsync(in , aug_tiles[0][cpu_ind * (batch_size * sample_size[2] + aug_padding[0])], batch_size * sample_size[2] * sizeof(T), cudaMemcpyHostToDevice, *cur_stream);
                            if(cur_stream == last_stream)
                                cur_stream = +data_stream;
                            else
                                cur_stream++;

                            cudaMemcpyAsync(out, aug_tiles[1][cpu_ind * (batch_size * sample_size[3] + aug_padding[1])], batch_size * sample_size[3] * sizeof(T), cudaMemcpyHostToDevice, *cur_stream);
                            if(cur_stream == last_stream)
                                cur_stream = +data_stream;
                            else
                                cur_stream++;

                            
                            cudaEventRecord(gpu_sync[ind], data_stream);
                          
                            found = true;
                            ready_gpu[ind].store(cpu_ind);
                        }
                    }

                    //if (!found)      //No ready augmentation tiles. If this happens, there is no point in searching for used gpu tiles. A performance problem exists!
                      break;   
                }
            }

            //2.: Check which copies finished
            for(uint32_t ind = 0; ind != augmentation_batches_gpu; ind++){
                int32_t gpu_stat = ready_gpu[ind].load();
                if(gpu_stat >= 0 && cudaEventQuerry(gpu_sync[ind]) == cudaSuccess) {
                    //The copy has finished. Mark augmentation tile as used and gpu tile as unused
                    ready_cpu[gpu_stat] = BATCH_STATUS::USED;
                    ready_gpu[ind]      = BATCH_STATUS::READY;
                }
            }
        }

        //Stopping signal was received
        for (uint32_t u = 0; u != augmentation_batches_gpu; u++) {
            cudaEventDestroy(&gpu_sync[u]);
        }
        for(uint32_t u = 0; u != num_streams; u++) {
            cudaStreamDestroy(&data_stream[u]);
        }
        
        thread_status++;
    }



    inline void parseInputFile(int fd, HEADER& header, uint32_t& sample_size, uint32_t& num_samples){
        //1.: Signature
        printf("[INFO] Parsing file...\n");
        char sig[6];
        read(fd, +sig, 6);
        assert(strncmp(sig, "JVDATA", 6));
        printf("[INFO] \t - Header matches\n");
        
        //2.: Version
        uint16_t ver;
        read(fd, &ver, 2);
        assert(ver <= AI_VERSION);
        if (ver < AI_VERSION)
            printf("[INFO] \t - File version: %u. This is an old version, since this is library version %u\n", (uint32_t)ver, AI_VERSION);
        else
            printf("[INFO] \t - File version: %u. This is the recent version\n", (uint32_t)ver);
        //3.: Lenght
        uint16_t len;
        read(fd, &len, 2);
        
        //4.: Read in header
        if ( ver == 1) {
            HEADER_V1 header_;
            read(fd, &header_, sizeof(header_));
            header = header_.toHEADER(len);
        }
    
        //5.: Compute sample_size
        sample_size = header.x * header.y * header.z;

        //6.: Compute num_samples
        uint64_t num_bytes;
        struct stat st;
        fstat(fd, &st);
        num_bytes = st.st_size - len;
        
        assert(num_bytes % sample_size == 0);
        num_samples = num_bytes / sample_size;
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
        num_workers(0u), workers(nullptr), thread_status(WORKER_STATUS::IDLE),
        ready_cpu(nullptr),
        ready_gpu(nullptr)
    {
        //-1.: Finish initialization (except agi_in, agi_out, cpy_thread)
        sample_size[2] = sample_size[3] = 0u;
        aug_tiles = { nullptr, nullptr }; aug_padding = { 0u, 0u };
        gpu_tiles = { nullptr, nullptr }; cur_gpu = { nullptr, nullptr };

        //0.: Check parameters
        assert(0.f <= train_split && train_split <= 1.0f && mb_cpu != 0 && fd_in_ != -1 && fd_out_ != -1);
        printf("[INFO] Constructing DatasetHandler\n");

        //1.: Read in file
        uint32_t samples, samples1, samples2;
        parseInputFile(fd_in , in_header , sample_size[0], samples1);
        parseInputFile(fd_out, out_header, sample_size[1], samples2);
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
            printf("[WARN] Dataset does not fit into ram. This might degrade performance. Additional %d mb are needed.\n", ((samples - tile_samples_cpu) * (uint64_t)bytes_per_sample) / (1024ull * 1024ull));
            fit_in_ram = false;
        }
        assert(tile_samples_cpu != 0);

        //5.: Allocate memory
        uint64_t alloc_size = tile_samples_cpu * (bytes_per_sample);
        T *cpu_mem = (T*)malloc(alloc_size);

        assert(cpu_mem != NULL && cpu_mem != nullptr);
        printf("[INFO] Successfully allocated %u mb.\n", alloc_size / (1024ull * 1024ull));

        cpu_tiles[0] = cpu_mem;
        cpu_tiles[1] = cpu_tiles[0] + sample_size[0] * tile_samples_cpu;

        //6.: Fill memory
        printf("[INFO] Reading dataset into ram.\n");
        read(fd_in , &cpu_tiles[0], sizeof(T) * sample_size[0] * tile_samples_cpu);
        read(fd_out, &cpu_tiles[1], sizeof(T) * sample_size[1] * tile_samples_cpu);
    }
    
    /*
        Sets augmenation and infers size of augmentated samples (sample_size[2] and sample_size[3])
    */
    inline void set_augmentation(AugmentationInfo2D_IN agi_in_, AugmentationInfo2D_OUT agi_out_) {
        //0.: Check parameters
        //Whether the aspect ratio and channels are valid will be checked by worker threads! Blur range remains unchecked
        assert(agi_in_.format.distribution == Image::DISTRIBUTION::UNIFORM && agi_in_.rotate >= 0.f && agi_in_.random_noise >= 0.f & agi_in_.random_dropout >= 0.f && agi_in_.random_dropout <= 1.f && agi_in_.random_saturation >= 0.f && agi_in_.random_brightness >= 0.f); //Standart deviation has to be >=0, probability is in ]0, 1[.
        assert(agi_in_.crop.x >= 0.f && agi_in_.crop.y >= 0.f  && agi_in_.crop.x <= 1.f && agi_in_.crop.y <= 1.f);
        assert(agi_in_.resize.x == -1 || (agi_in_.resize.x > 0 && agi_in_.resize.y > 0));

        assert(agi_out_.format.distribution == Image::DISTRIBUTION::UNIFORM && agi_out_.label_smoothing >= 0.f);
        assert(agi_out_.resize.x == -1 || (agi_out_.resize.x > 0 && agi_out_.resize.y > 0));

        Image_Shape in_shape (sample_size[0], agi_in_ .aspect, agi_in_ .channels);
        Image_Shape out_shape(sample_size[1], agi_out_.aspect, agi_out_.channels); 
        
        //1.: Set variables
        agi_in  = agi_in_ ;
        agi_out = agi_out_;

        //2.: Infer augmentation size
        if (agi_in.resize.x == -1) {
            if (agi_in.crop.x == -1)     //No change
                sample_size[2] = sample_size[0];
            else                         //Crop controlls size
                sample_size[2] = ((uint32_t)(in_shape.x * agi_in.crop.x)) * ((uint32_t)(in_shape.y * agi_in.crop.y));
        }
        else                             //Resize controls size
            sample_size[2] = agi_in.resize.x * agi_in.resize.y;

        
        if (agi_out.resize.x == -1) {
            if (agi_in.crop.x == -1)     //No change
                sample_size[3] = sample_size[1];
            else                         //Crop controlls size
                sample_size[3] = ((uint32_t)(out_shape.x * agi_in.crop.x)) * ((uint32_t)(out_shape.y * agi_in.crop.y));
        }
        else                             //Resize controls size
            sample_size[3] = agi_out.resize.x * agi_out.resize.y;
        
        
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
        assert(b % 8 == 0);

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
        cudaMallocHost(&aug_mem, sizeof(T) * ((sample_size[3] + sample_size[4]) * batch_size + aug_padding[0] + aug_padding[1]) * augmentation_batches_cpu);

        aug_tiles[0] = aug_mem;
        aug_tiles[1] = aug_tiles[0] + (sample_size[3] * batch_size + aug_padding[0]) * augmentation_batches_cpu;
        
        ready_cpu = (std::atomic<int32_t>*)malloc(sizeof(std::atomic<int32_t>) * augmentation_batches_cpu);
        for (uint32_t u = 0; u != augmentation_batches_cpu; u++) {
          ready_cpu[u] = std::atomic<int32_t>(BATCH_STATUS::USED);
        }

        //4.: Allocate gpu_tiles and gpu_sync
        T* gpu_mem;
        cudaMalloc(&gpu_mem, sizeof(T) * (sample_size[3] + sample_size[4]) * batch_size * augmentation_batches_gpu);

        gpu_tiles[0] = gpu_mem;
        gpu_tiles[1] = gpu_tiles[0] + sample_size[3] * batch_size * augmentation_batches_cpu;

        ready_gpu = (std::atomic<int32_t>*)malloc(sizeof(std::atomic<int32_t>) * augmentation_batches_gpu);
        for (uint32_t u = 0; u != augmentation_batches_gpu; u++) {
            ready_gpu[u] = std::atomic<int32_t>(BATCH_STATUS::USED);
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
    inline void start_workers(uint32_t n, uint32_t num_streams, WORKER_STATUS stat = WORKER_STATUS::IDLE) {
        //0.: Check parameters
        assert(n > 0);
        assert(stat < WORKER_STATUS::IDLE && workers == nullptr);

        //1.: Copy variables
        num_workers = n;

        //2.: Reset stopping signal
        thread_status.store(stat);

        //3.: Start num_workers new worker threads
        workers = (std::thread*)malloc(num_workers * sizeof(std::thread));
        for (uint32_t u = 0; u != num_workers; u++) {
            workers[u] = std::thread(worker_function, u);
            workers[u].detach();
        }

        //4.: Start one memcpy thread
        cpy_thread = std::thread(memcpy_thread, num_streams);
        cpy_thread.detach();
    }
    /*
        Switches input to training or validation. Invalidates all gpu samples.
    */
    template<bool training> void setDataInput() {
        //1.: Sent signal to worker threads
        if constexpr (training)
            thread_status = WORKER_STATUS::TRAINING;
        else
            thread_status = WORKER_STATUS::VALIDATION;

        //2.: Reload cpu and gpu tile
        for (uint32_t u = 0; u != augmentation_batches_cpu; u++)
            ready_cpu[u].store(BATCH_STATUS::USED);
        for (uint32_t u = 0; u != augmentation_batches_gpu; u++)
            ready_gpu[u].store(BATCH_STATUS::USED);

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

    template<bool training> void advance(T*& in, T*& out) {
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
        uint32_t ind = (cur_gpu[0] - gpu_tiles[0]) / (batch_size * sample_size[2]);
        if (cur_gpu[0] != nullptr)
          ready_gpu[ind].store(BATCH_STATUS::USED);

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
        in  = cur_gpu[0];
        out = cur_gpu[1];

        //4.: Synchronize
        while(ready_gpu[ind].load() != BATCH_STATUS::READY) {} //Wait until data is ready
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

using namespace DatasetAssemble;
int main(){
  //1.: Build a dataset
  Offset2D<uint32_t> size(224, 224);
  generateDatasetFile_Image<float>("/home/julian/Datasets/1/RAW/In", "/home/julian/Dataset/1/dataset.jvdata", size);

  
  //2.: Build a dataset handler
  
  //3.: Test augmentation tile


}


//sudo g++ /home/julian/cuda-workspace/AI-master/AI/Dataset.cpp -o /home/julian/cuda-workspace/AI-master/AI/Dataset.exe -I"/home/julian/Downloads/CImg-2.9.2_pre070420" -I"/usr/local/cuda-10.0/include" -O3 -march=native -std=c++17 -Wall -lpthread -lz -ldl -lpng -ljpeg -lX11 -g
