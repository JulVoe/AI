//==============================================
//==================|Types|=====================
//==============================================

#include <typeinfo>
#include <inttypes.h>
#include <cuda_fp16.h>
template<typename T> uint32_t type_hash() {
    static_assert(typeid(T) == typeid(T), "Unsupported type!");
    return -1;
}
//DO NOT CHANGE THE FOLLOWING VALUES AS IT WILL BREAK OLD DATASETS AND NETWORK CHECKPOINTS
template<> uint32_t type_hash<uint16_t>() { return 0; }
template<> uint32_t type_hash<uint32_t>() { return 1; }
template<> uint32_t type_hash<int16_t >() { return 2; }
template<> uint32_t type_hash<int32_t >() { return 3; }
template<> uint32_t type_hash<half    >() { return 4; }
template<> uint32_t type_hash<float   >() { return 5; }
template<> uint32_t type_hash<double  >() { return 6; }

#define AI_VERSION 0u

//===============================================
//==================|RANDOM|=====================
//===============================================
#include <x86intrin.h>
#include <random>

namespace Random {
    struct Key {
        __m256i part1;
        __m256i part2;
    } random_key;

    /* used by xorshift128plus_jump_onkeys */
    static void xorshift128plus_onkeys(uint64_t* ps0, uint64_t* ps1) {
        uint64_t s1 = *ps0;
        const uint64_t s0 = *ps1;
        *ps0 = s0;
        s1 ^= s1 << 23; // a
        *ps1 = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5); // b, c
    }

    /* used by avx_xorshift128plus_init */
    static void xorshift128plus_jump_onkeys(uint64_t in1, uint64_t in2, uint64_t* output1, uint64_t* output2) {
        static const uint64_t JUMP[] = { 0x8a5cd789635d2dff, 0x121fd2155c472f96 };
        uint64_t s0 = 0;
        uint64_t s1 = 0;
        for (unsigned int i = 0; i < sizeof(JUMP) / sizeof(*JUMP); i++)
            for (int b = 0; b < 64; b++) {
                if (JUMP[i] & 1ULL << b) {
                    s0 ^= in1;
                    s1 ^= in2;
                }
                xorshift128plus_onkeys(&in1, &in2);
            }
        output1[0] = s0;
        output2[0] = s1;
    }

    void avx_xorshift128plus_init(uint64_t key1, uint64_t key2, Key& key) {
        uint64_t S0[4];
        uint64_t S1[4];
        S0[0] = key1;
        S1[0] = key2;
        xorshift128plus_jump_onkeys(*S0, *S1, S0 + 1, S1 + 1);
        xorshift128plus_jump_onkeys(*(S0 + 1), *(S1 + 1), S0 + 2, S1 + 2);
        xorshift128plus_jump_onkeys(*(S0 + 2), *(S1 + 2), S0 + 3, S1 + 3);
        key.part1 = _mm256_loadu_si256((const __m256i*) S0);
        key.part2 = _mm256_loadu_si256((const __m256i*) S1);
    }

    /* Return a 256-bit random "number" */
    __m256i avx_xorshift128plus(Key& key) {
        __m256i s1 = key.part1;
        const __m256i s0 = key.part2;
        key.part1 = key.part2;
        s1 = _mm256_xor_si256(key.part2, _mm256_slli_epi64(key.part2, 23));
        key.part2 = _mm256_xor_si256(
            _mm256_xor_si256(_mm256_xor_si256(s1, s0),
                _mm256_srli_epi64(s1, 18)), _mm256_srli_epi64(s0, 5));
        return _mm256_add_epi64(key.part2, s0);
    }

    //--------------------|Normal Distribution|-------------------
    //TODO: FAST-NORM
    std::default_random_engine __gen__;
    std::normal_distribution<float> __normal_distr__(0.f, 1.f);
    
    uint32_t rand_uint(uint32_t m) { //Output in ]0,m]
        return __gen__() * ((float)m / (float)(__gen__.max() + 1));
    }

    float rand_float(float m) { //Output in ]0,m]
        return __gen__() * (m / (float)__gen__.max());
    }

    bool rand_prob(float prob) {
        return __gen__() < prob * __gen__.max();
    }

    float rand_normal(float dev) {
        return __normal_distr__(__gen__) * dev;
    }
    
    //------------------------------------------------------------

    void init_rand() {
        avx_xorshift128plus_init(13075623786050785612ull, 8923524891246523895ull, random_key);  //Deterministic
    }
}

//==============================================
//==================|IMAGE|=====================
//==============================================

#include <inttypes.h>
#include <assert.h>
#define cimg_use_jpeg
#define cimg_use_png
#include "CImg.h"

using namespace cimg_library;

template<typename T>
struct Offset2D {
    T x;
    T y;

    Offset2D() {}
  
    Offset2D(T x_, T y_){
        x = x_;
        y = y_;
    }
  
    template<typename Ty>
    void operator*=(Offset2D<Ty> m) {
        x *= m.x;
        y *= m.y;
    }
    template<typename Ty>
    Offset2D<T> operator*(Offset2D<Ty> m) {
        Offset2D<T> off( x * m.x, y * m.y);
        return off;
    }
    Offset2D<T> operator-(Offset2D<T> m) {
        Offset2D<T> off(x - m.x, y - y.x);
        return off;
    }
};

struct Image_Shape {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    Image_Shape() = default;
  
    Image_Shape(uint32_t x_, uint32_t y_, uint32_t z_)
        :x(x_), y(y_), z(z_)
    {}

    Image_Shape(uint32_t len, float aspect, uint32_t channels) {
        x = sqrt((float)len / (float)(channels * aspect));
        y = sqrt((float)(len * aspect) / (float)channels);
        z = channels;

        assert(x * y * z == len);
    }

    template<typename T = uint32_t>
    Offset2D<T> getOffset2D() {
        Offset2D<T> o{ x, y };
        return o;
    }

    template<typename T>
    void setOffset2D(Offset2D<T> off) {
        x = off.x;
        y = off.y;
    }

    template<typename T>
    void operator*=(Offset2D<T> m) {
        x *= m.x;
        y *= m.y;
    }

    template<typename T>
    Offset2D<T> operator*(Offset2D<T> m) {
        Offset2D<uint32_t> o{ x * m.x, y * m.y };
        return o;
    }
};

template<typename T> T max(T a, T b) {return (a>b)?a:b;}
template<typename T> T min(T a, T b) {return (a<b)?a:b;}

namespace Image {
/*
  It makes more sense to store Images channel-first as for cnn's channels are seen in a more general context an can be complety unrelated and are thus not stored interleaved
  Furthermore, channel-first makes cuDnn computations faster. The only reasons, why channel channels-last is supported, are compatability and because it is faster for the cpu, e.g. the augmentation.
  For saturation and brightness, all channels of a pixel are needed and thus a interleaved storage is more cache-coherent. 
*/
    //DO NOT CHANGE ANY OF THESE VALUES OR TYPES!
    enum CHANNEL_ORDER : int8_t {CHANNELS_FIRST = 0, CHANNELS_LAST = 1};
    enum CHANNELS : uint16_t {RGB = 3, GRAY = 1};
    enum PADDING : uint8_t {ZERO_PADDING_NORMAL = 0, ZERO_PADDING_RENORMALIZE = 1, EXTENSION = 2};
    enum DISTRIBUTION : uint8_t {UNIFORM = 0, NORMALIZED = 1};
    struct DATA_FORMAT {
        DISTRIBUTION distribution;
        float range; //If normalized, this is standard deviation and thus has to be >0. If uniform, this is maximum. Minimum is 0 if >0 and else -maximum

        DATA_FORMAT(DISTRIBUTION d, int32_t r)
          :distribution(d), range(r)
        {}

        bool operator==(DATA_FORMAT f){
            return f.distribution == distribution && f.range == range;
        }
        bool operator!=(DATA_FORMAT f){
            return !(f.distribution == distribution && f.range == range);
        }
    } __attribute__((packed));

    template<typename T>
    void remap_format(T* dat, Image_Shape shape, DATA_FORMAT old_format, DATA_FORMAT new_format, CHANNEL_ORDER order = CHANNEL_ORDER::CHANNELS_FIRST){
        assert(shape.z == CHANNELS::RGB || shape.z == CHANNELS::GRAY);

        if(shape.z == 3) {
            float mul[3], shift[3]; // (in-shift)*mul
            if (old_format.distribution == DISTRIBUTION::UNIFORM && new_format.distribution == DISTRIBUTION::UNIFORM){
                mul[0] = mul[1] = mul[2] = (new_format.range * ((new_format.range < 0.f) ? 2.f : 1.f)) / (old_format.range * ((old_format.range < 0.f) ? 2.f: 1.f));
                shift[0] = shift[1] = shift[2] = (old_format.range < 0.f && new_format.range > 0.f) ? old_format.range: \
                  ((old_format.range > 0.f && new_format.range < 0.f) ? (float)old_format.range / 2.f : 0.f);
            }
            if(old_format.distribution == DISTRIBUTION::NORMALIZED && new_format.distribution == DISTRIBUTION::NORMALIZED){
                mul[0] = mul[1] = mul[2] = new_format.range / old_format.range;
                shift[0] = shift[1] = shift[2] = 0.f;
            }
            if(old_format.distribution == DISTRIBUTION::UNIFORM && new_format.distribution == DISTRIBUTION::NORMALIZED){
                //Values taken from ImageNet Dataset

                mul[0] = new_format.range / 0.485;
                mul[1] = new_format.range / 0.456;
                mul[2] = new_format.range / 0.406;

                shift[0] = ((old_format.range < 0.f) ? 1. - 0.229 : 0.229) * old_format.range;
                shift[1] = ((old_format.range < 0.f) ? 1. - 0.224 : 0.224) * old_format.range;
                shift[2] = ((old_format.range < 0.f) ? 1. - 0.225 : 0.225) * old_format.range;
            }
            if(old_format.distribution == DISTRIBUTION::NORMALIZED && new_format.distribution == DISTRIBUTION::UNIFORM){
                //Values taken from ImageNet Dataset
                //Inverse of last transofrmation: in*mul-shift = mul*(in - shift/mul)
                mul[0] = 0.485 / old_format.range;
                mul[1] = 0.456 / old_format.range;
                mul[2] = 0.406 / old_format.range;

                shift[0] = -((new_format.range < 0.f) ? 1. - 0.229 : 0.229) * new_format.range / mul[0];
                shift[1] = -((new_format.range < 0.f) ? 1. - 0.224 : 0.224) * new_format.range / mul[1];
                shift[2] = -((new_format.range < 0.f) ? 1. - 0.225 : 0.225) * new_format.range / mul[2];
            }
            
            uint32_t m1, m2;
            if (order == CHANNEL_ORDER::CHANNELS_LAST){
                m1 = 1;
                m2 = shape.z;
            } else {
                m1 = shape.x * shape.y;
                m2 = 1;
            }

            for(uint32_t channel = 0; channel != shape.z; channel++){
                for(uint32_t ind = 0; ind != shape.x * shape.y; ind++){
                    T* p = dat + channel * m1 + ind * m2;
                    *p = (*p - shift[channel]) * mul[channel];
                }
            }
        } else {
            float mul, shift; // (in-shift)*mul
            if(old_format.distribution == DISTRIBUTION::UNIFORM && new_format.distribution == DISTRIBUTION::UNIFORM){
                mul = (new_format.range * ((new_format.range < 0.f) ? 2.f : 1.f)) / (old_format.range * ((old_format.range < 0.f) ? 2.f: 1.f));
                shift = (old_format.range < 0.f && new_format.range > 0.f) ? old_format.range: \
                  ((old_format.range > 0.f && new_format.range < 0.f) ? (float)old_format.range / 2.f : 0.f);
            }
            if(old_format.distribution == DISTRIBUTION::NORMALIZED && new_format.distribution == DISTRIBUTION::NORMALIZED){
                mul = new_format.range / old_format.range;
                shift = 0.f;
            }
            if(old_format.distribution == DISTRIBUTION::UNIFORM && new_format.distribution == DISTRIBUTION::NORMALIZED){
                mul = new_format.range / 0.5;
                shift = ((old_format.range < 0.f) ? 1. - 0.225 : 0.225) * old_format.range;
            }
            if(old_format.distribution == DISTRIBUTION::NORMALIZED && new_format.distribution == DISTRIBUTION::UNIFORM){
                //Inverse of last transofrmation: in*mul-shift = mul*(in - shift/mul)
                mul = 0.5 / old_format.range;
                shift = -((new_format.range < 0.f) ? 1. - 0.225 : 0.225) * new_format.range / mul;
            }
            
            for(uint32_t ind = 0; ind != shape.x * shape.y; ind++){
                T* p = dat + ind;
                *p = (*p - shift) * mul;
            }
        }
    }
  
    //Return pixel array and its shape for an image specified using its path. CHANNEL_ORDER and CHANNELS can be specified.
    template<typename T, CHANNEL_ORDER o = CHANNEL_ORDER::CHANNELS_FIRST, CHANNELS channels = CHANNELS::RGB, DISTRIBUTION distr = DISTRIBUTION::UNIFORM, int32_t range = 256>
    void getPixels(char* path, T*& dat, Image_Shape& shape) {
        //1.: Open image
        CImg<T> img = CImg<T>(path);
        assert((img.spectrum() == 3 || img.spectrum() == 4) && img.depth() == 1);      //Input has to be single RGB(A) image
        if(img.spectrum() == 4)
            img = img.get_shared_channels(0, 2);
        
        //2.: Convert to the right amount of channels
        if constexpr (channels == CHANNELS::RGB) {
            shape.z = 3;
        }
        else if constexpr (channels == CHANNELS::GRAY) {
            img.sRGBtoLab();
            img.channel(0);
            shape.z = 1;
        }

        //3.: Store data dimensions
        shape.x = img.width();
        shape.y = img.height();
        
        //4.: Convert to  the right CHANNEL_ORDER
        if constexpr (o == CHANNEL_ORDER::CHANNELS_LAST)
            img.permute_axes("cxyz");
                       
        //5.: Store data (img will be deconstructed, save the data)
        dat = (T*)malloc(5 * sizeof(T) * shape.x * shape.y * shape.z);
        memcpy(dat, img.data(), sizeof(T) * shape.x * shape.y * shape.z);

        //6.: Remap
        if constexpr(distr != DISTRIBUTION::UNIFORM || range != 256) {
            DATA_FORMAT old_format{DISTRIBUTION::UNIFORM, 256};
            DATA_FORMAT new_format{distr, range};
            remap_format<T, o>(dat, shape, old_format, new_format);
        }
    }

    template<typename T, CHANNEL_ORDER o = CHANNEL_ORDER::CHANNELS_FIRST>
    void show(T* dat, Image_Shape shape){
        CImg<T> img(dat, shape.x, shape.y, 1, shape.z);

        if constexpr(o == CHANNEL_ORDER::CHANNELS_LAST) {
            img = CImg<T>(dat, shape.z, shape.x, shape.y, 1);
            img.permute_axes("yzcx");
        } else {
            img = CImg<T>(dat, shape.x, shape.y, 1, shape.z);
        }

        CImgDisplay disp(img, "", 0);

        //while(!disp.is_closed()) {}
        getchar();

        if constexpr(0 == CHANNEL_ORDER::CHANNELS_LAST)
            img.permute_axes("yzcx");

    }

  template<typename T, PADDING padding = PADDING::ZERO_PADDING_RENORMALIZE, CHANNEL_ORDER order = CHANNEL_ORDER::CHANNELS_FIRST>
    void boxblur(T* dat, Image_Shape shape, uint32_t w, uint32_t h) {
      //0.: Check parameters
      assert(w % 2 == 1 && h % 2 == 1 && w >= 3 && h >= 3);           //Width and height have to be odd, so the sliding windows has an whole number as radius. 1 makes no sense
      assert(shape.x >= w && shape.y >= h);                           //Image should be bigger than filter radius

      //1.: Compute radii and allocate storage
      uint32_t r_w = (w - 1u) >> 1;
      uint32_t r_h = (h - 1u) >> 1;
      T* buf = (T*)malloc(sizeof(T) * (1 + max(r_w, r_h)));          //Stores last r_w or r_h original image values (for each channel) to subtract from the end of the sliding window when it passes

      uint32_t mul;
      if constexpr(order == CHANNEL_ORDER::CHANNELS_FIRST)
          mul = 1;
      else
          mul = shape.z;

      for(uint32_t chn = 0; chn != shape.z; chn++) {
          //2.: Horizontal pass
          float norm = 1.f / (float)(2 * r_w + 1);
          for(uint32_t line = 0; line != shape.y; line++) {
              uint32_t cur_ind;                  //Index of currently modified component
              if constexpr(order == CHANNELS_FIRST)
                  cur_ind = line * shape.x + chn * shape.x * shape.y;
              else
                  cur_ind = line * shape.x * shape.z + chn;
              uint32_t li = 0;                   //Left index of sliding window, in buf
              uint32_t ri = cur_ind + r_w * mul;

              T firstValue = dat[cur_ind];
              T  lastValue = dat[cur_ind + (shape.x - 1) * mul];
              T accu = 0;
              
              if constexpr (padding == PADDING::EXTENSION)
                  accu = (r_w + 1) * firstValue;                          //Current sliding window (start of with r_w real values and r_w+1 times the first value, this is the window before that)
              
              for(uint32_t j = 0; j !=  r_w; j++)                         //Finish initialization for accumulator
                  accu += dat[cur_ind + j * mul];                         //Add first r_w-1 real values. Do not increment cur_ind
              
              for(uint32_t j = 0; j <= r_w ; j++) {                       //First r+1 values: The left side of sliding window is always the extension or firstValue
                  accu += dat[ri];
                  if constexpr (padding == PADDING::EXTENSION)
                      accu -= firstValue;                                 //Update accmulator and shift right edge of sliding window
                  buf[li] = dat[cur_ind];                                 //Fill the buffer with old values
                  if constexpr(padding == PADDING::ZERO_PADDING_RENORMALIZE)
                      dat[cur_ind] = accu / (r_h + 1 + j);
                  else
                      dat[cur_ind] = accu * norm;

                  ri += mul;
                  cur_ind += mul;
                  if(++li == r_w + 1)
                      li = 0;
              }

              for(uint32_t j = r_w + 1; j != shape.x-r_w; j++) {
                  accu += dat[ri] - buf[li];
                  buf[li] = dat[cur_ind];
                  dat[cur_ind] = accu * norm;

                  ri += mul;
                  cur_ind += mul;
                  if(++li == r_w + 1)
                      li = 0;
              }
              
              for(uint32_t j = shape.x - r_w; j < shape.x  ; j++) {
                  accu -= buf[li];
                  if constexpr ( padding == PADDING::EXTENSION)
                      accu += lastValue;
                  if constexpr(padding == PADDING::ZERO_PADDING_RENORMALIZE)
                      dat[cur_ind] = accu / (r_h + shape.x - j);
                  else
                      dat[cur_ind] = accu * norm;

                  cur_ind += mul;
                  if(++li == r_w + 1)
                      li = 0;
              }
          }
          
          //3.: Vertical pass
          norm = 1.f / (float)(2 * r_h + 1);
          for(uint32_t col = 0; col !=  shape.x; col++) {
              uint32_t cur_ind;
              if constexpr(order == CHANNEL_ORDER::CHANNELS_FIRST)
                  cur_ind = col + chn * shape.x * shape.y;
              else
                cur_ind = col * shape.z + chn; 
              uint32_t li      = 0;                                            //Left index. Not of image, but of buf
              uint32_t ri      = cur_ind + r_h * shape.x * mul;                //Right index
              T firstValue     = dat[cur_ind];                                 //Extends first value over the edge of the image
              T  lastValue     = dat[cur_ind + (shape.y - 1) * shape.x * mul]; //Extends last value over the edge of the image
              T accu = 0;
              if constexpr (padding == PADDING::EXTENSION)
                  accu = (r_h + 1) * firstValue;                          //Current sliding window (start of with r_w real values and r_w+1 times the first value, this is the window before that)
              
              for(uint32_t j = 0; j !=  r_h; j++)                         //Finish initialization for accumulator
                  accu += dat[cur_ind + j * shape.x * mul];               //Add first r_w real values. Do not increment cur_ind
          
              for(uint32_t j = 0; j <= r_h ; j++) {                       //First r+1 values: The left side of sliding window is always the extension or firstValue
                  accu += dat[ri];                                        //Update accmulator and shift right edge of sliding window
                  if constexpr (padding == PADDING::EXTENSION)
                      accu -= firstValue;                                 //Update accmulator and shift right edge of sliding window
                  buf[li] = dat[cur_ind];                                 //Fill the buffer with old values

                  if constexpr(padding == PADDING::ZERO_PADDING_RENORMALIZE)
                      dat[cur_ind] = accu / (r_h + 1 + j);
                  else
                      dat[cur_ind] = accu * norm;

                  cur_ind += shape.x * mul;
                  ri += shape.x * mul;
                  if(++li == r_h + 1)
                      li = 0;
              }
              
              for(uint32_t j = r_h + 1; j < shape.y-r_h; j++) {
                  accu += dat[ri] - buf[li];
                  buf[li] = dat[cur_ind];
                  dat[cur_ind] = accu * norm;
                  
                  cur_ind += shape.x * mul;
                  ri += shape.x * mul;
                  if(++li == r_h + 1)
                      li = 0;
              }
              for(uint32_t j = shape.y - r_h; j <  shape.y  ; j++) {
                  accu -= buf[li];
                  if constexpr ( padding == PADDING::EXTENSION)
                      accu += lastValue;
                  if constexpr(padding == PADDING::ZERO_PADDING_RENORMALIZE)
                      dat[cur_ind] = accu / (r_h + shape.y - j);
                  else
                      dat[cur_ind] = accu * norm;
                  
                  cur_ind += shape.x * mul;
                  if(++li == r_h + 1)
                      li = 0;
              }
          }
      }
    }
    
    /*
        http://blog.ivank.net/fastest-gaussian-blur.html
        https://web.stanford.edu/class/cs448f/lectures/2.2/Fast%20Filtering.pdf
    */
    template<typename T, PADDING padding = PADDING::ZERO_PADDING_RENORMALIZE>
    void pseudo_gausblur(T* dat, Image_Shape shape, float stdev, uint32_t n = 3, CHANNEL_ORDER order = CHANNEL_ORDER::CHANNELS_FIRST) { 
        float wIdeal = sqrt((12.f * stdev * stdev / (float)n) + 1.f);  // Ideal averaging filter width 
        uint32_t wl = wIdeal;  if (wl % 2 == 0) wl--;
        uint32_t wu = wl + 2u;

        float mIdeal = (12.f * stdev * stdev - (float)(n * wl * wl + 4 * n * wl + 3 * n)) / (float)(-4 * wl - 4);
        uint32_t m = mIdeal + 0.5f;
        //printf("Actual sigma: %d", sqrt((float)(m * wl * wl + (n - m) * wu * wu - n) / 12.f ));
        
        for (uint32_t i = 0; i != n; i++) {
            if (i < m)
                boxblur<T, padding, order>(dat, shape, wl, wl);
            else
                boxblur<T, padding, order>(dat, shape, wu, wu);
        }
    }

    template<typename T>
    void flip(T* dat, Image_Shape shape, CHANNEL_ORDER order = CHANNEL_ORDER::CHANNELS_FIRST) {
        uint32_t m1, m2;
        if(order == CHANNEL_ORDER::CHANNELS_LAST){
            m1 = 1;
            m2 = shape.z;
        } else {
            m1 = shape.x * shape.y;
            m2 = 1;
        }

        T a, b;
        for(uint32_t channel = 0; channel != shape.z; channel++){
            for(uint32_t line = 0; line != shape.y; line++){
                uint32_t sta = channel * m1 + line * shape.x * m2;
                uint32_t sto = sta +  (shape.x - 1) * m2;

                while(sta < sto){
                    a = dat[sta];
                    b = dat[sto];

                    dat[sta] = b;
                    dat[sto] = a;

                    sta += m2;
                    sto -= m2;
                }
            }
        }
    }

    template<typename T>
    void random_noise(T* dat, Image_Shape shape, float stdev) {
        for(uint32_t ind = 0; ind != shape.x * shape.y * shape.z; ind++) {
            dat[ind] += Random::rand_normal(stdev);
        }
    }

  template<typename T>
  void random_dropout(T* dat, Image_Shape shape, float prob, CHANNEL_ORDER order = CHANNEL_ORDER::CHANNELS_FIRST) {
        uint32_t m1, m2;
        if(order == CHANNEL_ORDER::CHANNELS_LAST){
            m1 = 1;
            m2 = shape.z;
        } else {
            m1 = shape.x * shape.y;
            m2 = 1;
        }

        for(uint32_t ind = 0; ind != shape.x * shape.y; ind++){
            if(Random::rand_prob(prob)) {
                for(uint32_t channel = 0; channel != shape.z; channel++){
                    dat[ind * m2 + channel * m1] = 0;
                }
            }
        }
    }

    //TODO: Inproper conversion
    template<typename T>
    void mul_saturation(T* dat, Image_Shape shape, float satur_mul, DATA_FORMAT format , CHANNEL_ORDER order = CHANNEL_ORDER::CHANNELS_FIRST) {
        //0.: Check parameters
        assert(shape.z == CHANNELS::RGB);
        assert(format.distribution == DISTRIBUTION::UNIFORM);

        //0.5.: Scalars for the right indexing scheme depending on order
        uint32_t m1, m2;
        if(order == CHANNEL_ORDER::CHANNELS_LAST){
            m1 = 1;
            m2 = shape.z;
        } else {
            m1 = shape.x * shape.y;
            m2 = 1;
        }
        
        //1.:Computation
        T *r, *g, *b;
        for(uint32_t ind = 0; ind != shape.x * shape.y; ind ++){
            r = dat + ind * m2;
            g = r + m1;
            b = g + m1;

            if (format.distribution == DISTRIBUTION::UNIFORM && format.range < 0) {
                *r += format.range;
                *g += format.range;
                *b += format.range;
            }
            
            float P = sqrt((*r) * (*r) * .299f + (*g) * (*g) * .587f + (*b) * (*b) * .114f);

            *r = P + ((*r) - P) * satur_mul;
            *g = P + ((*g) - P) * satur_mul;
            *b = P + ((*b) - P) * satur_mul;

            if (format.distribution == DISTRIBUTION::UNIFORM) {
                if (format.range < 0) {
                    *r -= format.range;
                    *g -= format.range;
                    *b -= format.range;

                    *r = max((T)format.range, *r);
                    *g = max((T)format.range, *g);
                    *b = max((T)format.range, *b);

                    *r = min((T)-format.range, *r);
                    *g = min((T)-format.range, *g);
                    *b = min((T)-format.range, *b);
                } else {
                    *r = min((T)format.range, *r);
                    *g = min((T)format.range, *g);
                    *b = min((T)format.range, *b);
                }
            }
        }
    }

    //TODO: Inproper conversion
    template<typename T>
    void mul_brightness(T* dat, Image_Shape shape, float bright_mul, DATA_FORMAT format, CHANNEL_ORDER order = CHANNEL_ORDER::CHANNELS_FIRST) {
        //0.: Check parameters
        assert(shape.z == CHANNELS::RGB);
        assert(format.distribution == DISTRIBUTION::UNIFORM);

        //0.5.: Scalrs for the right indexing scheme depending on order
        uint32_t m1, m2;
        if(order == CHANNEL_ORDER::CHANNELS_LAST){
            m1 = 1;
            m2 = shape.z;
        } else {
            m1 = shape.x * shape.y;
            m2 = 1;
        }
        
        //1.:Computation
        T *r, *g, *b;
        for(uint32_t ind = 0; ind != shape.x * shape.y; ind ++){
            r = dat + ind + 0 * shape.x * shape.y;
            g = dat + ind + 1 * shape.x * shape.y;
            b = dat + ind + 2 * shape.x * shape.y;

            *r = (*r) * bright_mul;
            *g = (*g) * bright_mul;
            *b = (*b) * bright_mul;

            if (format.distribution == DISTRIBUTION::UNIFORM) {
                if (format.range < 0) {
                    *r = max((T)format.range, *r);
                    *g = max((T)format.range, *g);
                    *b = max((T)format.range, *b);

                    *r = min((T)-format.range, *r);
                    *g = min((T)-format.range, *g);
                    *b = min((T)-format.range, *b);
                } else {
                    *r = min((T)format.range, *r);
                    *g = min((T)format.range, *g);
                    *b = min((T)format.range, *b);
                }
            }
        }
    }

    template<typename T>
    void rotate(T* dat, Image_Shape shape, float deg) { //TODO: not inplemented
        assert(0 == 1);
    }

    template<typename T>
    void crop(T* dat, Image_Shape shape, Offset2D<uint32_t> off, Offset2D<uint32_t> new_size, CHANNEL_ORDER order = CHANNEL_ORDER::CHANNELS_FIRST) {
        assert(off.x + new_size.x <= shape.x && off.y + new_size.y <= shape.y);

        T* i;
        T* o = dat;

        if(order == CHANNEL_ORDER::CHANNELS_FIRST){
            for(uint32_t channel = 0; channel != shape.z; channel++) {
                for(uint32_t l = off.y; l != off.y + new_size.y; l++){
                    i = dat + off.x + l * shape.x + channel * shape.x * shape.y;

                    memcpy(o, i, sizeof(T) * new_size.x);

                    o += new_size.x;
                }
            }
        } else {
            for(uint32_t l = off.y; l != off.y + new_size.y; l++){
                i = dat + (off.x + l * shape.x) * shape.z;

                memcpy(o, i, sizeof(T) * new_size.x * shape.z);

                o += new_size.x * shape.z;
            }      
        }
    }
  
    template<typename T, CHANNEL_ORDER order_in = CHANNEL_ORDER::CHANNELS_FIRST, CHANNEL_ORDER order_out = order_in>
    void resize(T* dat, Image_Shape old_shape, Offset2D<uint32_t> new_size, PADDING padding = PADDING::ZERO_PADDING_RENORMALIZE) {
        assert((old_shape.x <= new_size.x && old_shape.y <= new_size.y) || (old_shape.x >= new_size.x && old_shape.y >= new_size.y));

        uint32_t m1_i, m2_i, m1_o, m2_o;
        if constexpr(order_in == CHANNEL_ORDER::CHANNELS_LAST){
            m1_i = 1;
            m2_i = old_shape.z;
        } else {
            m1_i = old_shape.x * old_shape.y;
            m2_i = 1;
        }
        if constexpr(order_out == CHANNEL_ORDER::CHANNELS_LAST){
            m1_o = 1;
            m2_o = old_shape.z;
        } else {
            m1_o = new_size.x * new_size.y;
            m2_o = 1;
        }
        
#define __resize__INTERNAL1();                                                         \
      float old_y = (y + 0.5f) * ((float)old_shape.y / (float)new_size.y) - 0.5f;      \
      uint32_t old_y1 = old_y;                                                         \
                                                                                       \
      float frac_y1 = old_y - old_y1;                                                  \
      float frac_y2 = 1.f - frac_y1;                                                   \
      T* in1_ = dat + old_y1 * old_shape.x * m2_i + channel * m1_i;                    \
      T* in3_ = in1_ + old_shape.x * m2_i;                                             \
                                                                                       \
      if(old_y < 0 || old_y > old_shape.y - 1){                                        \
          switch(padding){                                                             \
          case PADDING::ZERO_PADDING_NORMAL:                                           \
              frac_y1 = 0.f; /*Now, set in3 to arbitrary, safe to dereference pointer*/\
          case PADDING::ZERO_PADDING_RENORMALIZE:                                      \
          case PADDING::EXTENSION:                                                     \
              in3_ = in1_;        /*old_y1 is 0 because of rounding*/                  \
              break;                                                                   \
          }                                                                            \
      }
#define __resize__INTERNAL2();                                                         \
      float old_x = (x + 0.5f) * ((float)old_shape.x / (float)new_size.x) - 0.5f;      \
      uint32_t old_x1 = old_x;                                                         \
                                                                                       \
      float frac_x1 = old_x - old_x1;                                                  \
      float frac_x2 = 1.f - frac_x1;                                                   \
      T* in1 = in1_ + old_x1 * m2_i;                                                   \
      T* in2 = in1 + m2_i;                                                             \
      T* in3 = in3_ + old_x1 * m2_i;                                                   \
      T* in4 = in3 + m2_i;                                                             \
                                                                                       \
      if(old_x < 0 || old_x > old_shape.x - 1){                                        \
          switch(padding){                                                             \
          case PADDING::ZERO_PADDING_NORMAL:                                           \
              frac_x1 = 0.f; /*Now, set in2 to arbitrary, safe to dereference pointer*/\
          case PADDING::ZERO_PADDING_RENORMALIZE:                                      \
          case PADDING::EXTENSION:                                                     \
              in2 = in1;          /*old_x1 is 0 because of rounding*/                  \
              in4 = in3;                                                               \
              break;                                                                   \
          }                                                                            \
      }                                                                                \
      T newVal = frac_x2*frac_y2*(*in1)+frac_x1*frac_y2*(*in2)+frac_x2*frac_y1*(*in3)+frac_x1*frac_y1*(*in4); \
      dat[channel * m1_o + (y * new_size.x + x) * m2_o] = newVal;

//Real code starts here
        if(old_shape.x >= new_size.x && old_shape.y >= new_size.y) {
            for(uint32_t channel = 0; channel != old_shape.z; channel++) {
                for(uint32_t y = 0; y != new_size.y; y++){
                    __resize__INTERNAL1();
                    for(uint32_t x = 0; x != new_size.x; x++) {
                        __resize__INTERNAL2();
                    }
                }
            }
        } else {
            for(int32_t channel = old_shape.z - 1; channel >= 0; channel--) {
                for(int32_t y = new_size.y - 1; y >= 0; y--){
                    __resize__INTERNAL1();
                    for(int32_t x = new_size.x - 1; x >= 0; x--) {
                      __resize__INTERNAL2();
                    }
                }
            }
        }
#undef __resize__INTERNAL1
#undef __resize__INTERNAL2
    }

    //Note: Do not use this, when using knowledge distilation via teacher-student networks.
    //This may make temparature calibration of softmax obsolete
    template<typename T>
    void label_smoothing(T* dat, Image_Shape shape, float factor = 0.1f){
        for(uint32_t ind = 0; ind != shape.x * shape.y * shape.z; ind++) {
          dat[ind] *=  (1.f - factor);
          dat[ind] += factor / shape.z;
        }
    }
}

//============================================
//==================|CSV|=====================
//============================================

#include <string>
#include <vector>
#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

namespace CSV {
    template<typename T>
    T destringify(std::string in){
        static_assert(typeid(T) != typeid(T), "stringify has currently no implementation for the type u supplied");
    }
    template<> uint32_t destringify<uint32_t>(std::string in){ return std::stoul(in);}
    template<>  int32_t destringify< int32_t>(std::string in){ return std::stoi(in); }
    template<> uint16_t destringify<uint16_t>(std::string in){ return std::stoul(in);}
    template<>  int16_t destringify< int16_t>(std::string in){ return std::stoi(in); }
    template<> uint8_t  destringify<uint8_t >(std::string in){ return std::stoul(in);}
    template<>  int8_t  destringify< int8_t >(std::string in){ return std::stoi(in); }
    template<> float    destringify<float   >(std::string in){ return std::stof(in); }
    template<> double   destringify<double  >(std::string in){ return std::stod(in); }
  
  
    template<typename T>
    T* loadCSV(char* path, char delim, uint32_t& n){
        //Read file into ram
        int fid  = open(path, O_RDONLY);
        struct stat st;
        fstat(fid, &st);
        char* mem = (char*)malloc(st.st_size);

        std::vector<T> out;
        for(char *sta=mem, *sto=mem; sto != mem + st.st_size; sto++){
            if(*sto == delim){ //Region sta->sto contains new element
                std::string el_str = std::string(sta, sto);
                out.push_back(destringify<T>(el_str));

                sta = sto + 1;
                n++;
            }
        }

        n = out.size();
        return out.data();
    }

    template<typename T_in, typename T_out, Image::CHANNEL_ORDER order = Image::CHANNEL_ORDER::CHANNELS_FIRST>
    T_out* vec_to_img(T_in* dat, uint32_t len, uint32_t channels){
        uint32_t m1, m2;
        if constexpr(order == Image::CHANNEL_ORDER::CHANNELS_LAST){
            m1 = 1;
            m2 = channels;
        } else {
            m1 = len;
            m2 = 1;
        }

        T_out* out = (T_out*)malloc(sizeof(T_out) * len * channels);

        for(uint32_t ind = 0; ind != len; ind++){
            T_in val = dat[ind];
            for(uint32_t channel = 0; channel != channels; channel++){
              out[channel * m1 + ind * m2] = (T_out)(channel == val);
            }
        }
    }
}

//===========================================================
//==================|DATASET GENERATING|=====================
//===========================================================
#define EXPERIMENTAL_FILESYSTEM

#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef EXPERIMENTAL_FILESYSTEM
#include <experimental/filesystem>
#define __FILESYSTEM__ std::experimental::filesystem
#else
#include <filesystem>
#define __FILESYSTEM__ std::filesystem
#endif

#ifndef static_warning
#warning static_warning was not defined
#define static_warning(a,b)
#endif

struct HEADER{
    uint32_t type;                //Type of data
    uint32_t x;                   //Width per sample
    uint32_t y;                   //Height per sample
    uint32_t z;                   //Depth per sample
    Image::CHANNEL_ORDER order;   //Channel order
    Image::DATA_FORMAT format;    //Format of data
    uint32_t bytes;               //Offset until real data starts
};
struct HEADER_V1{
    uint32_t type;                //Type of data:        uint32_t  |
    uint32_t x;                   //Width per sample:    uint32_t  |
    uint32_t y;                   //Height per sample:   uint32_t  |
    uint32_t z;                   //Depth per sample:    uint32_t  |
    Image::CHANNEL_ORDER order;   //Channel order:       uint8_t   |
    Image::DISTRIBUTION distr;    //Distribution:        uint8_t   |
    float range;                  //Range of data:       uint32_t
  
    HEADER toHEADER(uint32_t bytes){
        Image::DATA_FORMAT format(distr, range);
        HEADER h{type, x, y, z, order, format, bytes};
        return h;
    }
} __attribute__((packed));

//HEADER: |SIGNATURE(6)|VERSION(2)|HEADER_LENGHT(2)|TYPE(4)|SIZE.X(4)|SIZE.Y(4)|SIZE.Z(4)|CHANNEL_ORDER(1)|DISTRIBUTION(1)|RANGE(4)|------------DATA-------------|
namespace DatasetAssemble {
    template<typename T, Image::CHANNELS channels = Image::CHANNELS::RGB, Image::CHANNEL_ORDER order = Image::CHANNEL_ORDER::CHANNELS_FIRST, Image::DISTRIBUTION distr = Image::DISTRIBUTION::UNIFORM, int32_t range = 1>
    void generateDatasetFile_Image(std::string dir, std::string file_out, Offset2D<uint32_t> size) {
        static_warning(distr == Image::DISTRIBUTION::UNIFORM, "A dataset should contain uniform data to enable easier augmentation!");

        int fd_out = open(file_out.c_str(), O_WRONLY);

        printf("[INFO] Writing header\n");
        char signature[] = {"JVDATA"};
        uint16_t version = AI_VERSION;
        uint16_t lenght;
        Image::DATA_FORMAT format(distr, range);
        HEADER_V1 header{type_hash<T>(), size.x, size.y, channels, order, format.distribution, format.range};
        lenght = sizeof(signature) + sizeof(version) + sizeof(version) + sizeof(lenght);

        write(fd_out, +signature, sizeof(signature)); //Signature:         6*uint8_t
        write(fd_out, &version  , sizeof(version)  ); //Version:             uint16_t
        write(fd_out, &lenght   , sizeof(lenght)   ); //Header lenght:       uint16_t
        write(fd_out, &header   , sizeof(header)   ); //Header data:         HEADER

        printf("[INFO] Reading in filenames\n");
        std::vector<__FILESYSTEM__::path> paths;
        for (const auto& entry : __FILESYSTEM__::directory_iterator(dir)) {
            if (__FILESYSTEM__::is_regular_file(entry.path())) {
              paths.push_back(entry.path());

            }
        }
        printf("[INFO] Sorting filenames\n");
        std::sort(paths.begin(), paths.end(), [](const auto& lhs, const auto& rhs){return lhs.string() < rhs.string();});

        for(const __FILESYSTEM__::path& in_file_path : paths){
            printf("[INFO] Handling file %s\n", in_file_path.c_str());

            //File IO
            Image_Shape shape_;
            T* dat;

            Image::getPixels<T, order, channels, distr, range>(in_file_path.c_str(), dat, shape_);
            Image::resize<T, order, order>(dat, shape_, size.getOffset2D(), Image::PADDING::ZERO_PADDING_RENORMALIZE);
            write(fd_out, dat, sizeof(T) * size.x * size.y * channels);
        }

        close(fd_out);
    }
    
    template<typename T, Image::CHANNEL_ORDER order = Image::CHANNEL_ORDER::CHANNELS_FIRST, Image::DISTRIBUTION distr = Image::DISTRIBUTION::UNIFORM, int32_t range = 1>
    void generateDatasetFile_Segmentation(std::string dir, std::string file_out, Image_Shape shape, Image::DATA_FORMAT old_format) {
        static_warning(distr == Image::DISTRIBUTION::UNIFORM, "A dataset should contain uniform data to enable easier augmentation!");

        int fd_out = open(file_out.c_str(), O_WRONLY);

        printf("[INFO] Writing header\n");
        char signature[] = {"JVDATA"};
        uint16_t version = AI_VERSION;
        uint16_t lenght;
        Image::DATA_FORMAT format(distr, range);
        HEADER_V1 header{type_hash<T>(), shape.x, shape.y, shape.z, order, format};
        lenght = sizeof(signature) + sizeof(version) + sizeof(version) + sizeof(lenght);

        write(fd_out, +signature, sizeof(signature)); //Signature:         6*uint8_t
        write(fd_out, &version  , sizeof(version)  ); //Version:             uint16_t
        write(fd_out, &lenght   , sizeof(lenght)   ); //Header lenght:       uint16_t
        write(fd_out, &header   , sizeof(header)   ); //Header data:         HEADER

        printf("[INFO] Reading in filenames\n");
        std::vector<__FILESYSTEM__::path> paths;
        for (const auto& entry : __FILESYSTEM__::directory_iterator(dir)) {
            if (__FILESYSTEM__::is_regular_file(entry.path())) {
              paths.push_back(entry.path());

            }
        }
        printf("[INFO] Sorting filenames\n");
        std::sort(paths.begin(), paths.end(), [](const auto& lhs, const auto& rhs){return lhs.string() < rhs.string();});

        for(const __FILESYSTEM__::path& in_file_path : paths){
            printf("[INFO] Handling file %s\n", in_file_path.c_str());

            //File IO
            Image::DATA_FORMAT format(distr, range);
            
            Offset2D<T> old_size;
            T* dat = CSV::loadCSV<T>(in_file_path.c_str(), ',', old_size);
            dat = CSV::vec_to_img<T, T, order>(dat, old_size.x * old_size.y, shape.z);
            Image_Shape old_shape = shape;
            old_shape.setOffset2D(old_size);

            Image::resize<T, order, order>(dat, old_shape, shape.getOffset2D(), Image::PADDING::ZERO_PADDING_RENORMALIZE); 
            Image::remap_format<T>(dat, shape, old_format, format, order);
            
            write(fd_out, dat, sizeof(T) * shape.x * shape.y * shape.z);
        }

        close(fd_out);
    }
}

#if 0
using namespace Image;
#define TYPE float
#include <cstdlib>
int main(){
  TYPE* dat;
  Image_Shape shape = Image_Shape();

  Image::getPixels<TYPE, CHANNEL_ORDER::CHANNELS_LAST, CHANNELS::RGB>("/home/julian/test.png", dat, shape);

  Image::show<TYPE, CHANNEL_ORDER::CHANNELS_LAST>(dat, shape);
  
  Offset2D<uint32_t> off(shape.x / 2, shape.y / 2);
  //pseudo_gausblur<TYPE, PADDING::EXTENSION, CHANNEL_ORDER::CHANNELS_LAST>(dat, shape, 2);
  //resize<TYPE, CHANNEL_ORDER::CHANNELS_LAST, CHANNEL_ORDER::CHANNELS_LAST>(dat, shape, off, PADDING::ZERO_PADDING_RENORMALIZE);
  //crop<TYPE, CHANNEL_ORDER::CHANNELS_LAST>(dat, shape, Offset2D<uint32_t>(shape.x / 3, shape.y / 3), off);
  //mul_brightness(dat, shape, 2.5, DATA_FORMAT(DISTRIBUTION::UNIFORM, 255));
  //mul_saturation(dat, shape, 2.2, DATA_FORMAT(DISTRIBUTION::UNIFORM, 255));
  //random_noise(dat, shape, 5);
  //random_dropout(dat, shape, 0.2);
  //flip<TYPE, CHANNEL_ORDER::CHANNELS_LAST>(dat, shape);
  shape.setOffset2D(off);
  
  Image::show<TYPE, CHANNEL_ORDER::CHANNELS_LAST>(dat, shape);
  
  return 0;
}
#endif



//sudo g++ /home/julian/cuda-workspace/AI-master/AI/util.cpp -o /home/julian/cuda-workspace/AI-master/AI/util.exe -I"/home/julian/Downloads/CImg-2.9.2_pre070420" -I"/usr/local/cuda-10.0/include" -O3 -march=native -std=c++17 -lpthread -lz -ldl -lpng -ljpeg -lX11 -g
