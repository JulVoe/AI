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
    std::default_random_engine gen;
    std::normal_distribution<float> normal_distr(0.f, 1.f);
    
    uint32_t rand_uint(uint32_t m) { //Output in ]0,m]
        return gen() * ((float)m / (float)(gen.max() + 1));
    }

    float rand_float(float m) { //Output in ]0,m]
        return gen() * (m / (float)gen.max());
    }

    bool rand_prob(float prob) {
        return gen() < prob * gen.max();
    }

    float rand_normal(float dev) {
        return normal_distr(gen) * dev;
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
#define cimg_use_magick
#include "Magick++.h"
#include "CImg.h"

template<typename T>
struct Offset2D {
    T x;
    T y;

    template<typename Ty>
    void operator*=(Offset2D<Ty> m) {
        x *= m.x;
        y *= m.y;
    }
};

struct Image_Shape {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    Image_Shape(uint32_t x_, uint32_t y_, uint32_t z_)
        :x(x_), y(y_), z(z_)
    {}

    Image_Shape(uint32_t len, float aspect, uint32_t channels) {
        x = sqrt((float)len / (float)(channels * aspect));
        y = sqrt((float)(len * aspect) / (float)channels);
        z = channels;

        assert(x * y * z == len);
    }

    Offset2D<uint32_t> getOffset2D() {
        Offset2D<uint32_t> o{ x, y };
        return o;
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

namespace Image {
    template<typename T>
    void getRGB(char* path, T*& dat, uint32_t& len) {
        CImg<T> img(path);
        dat = img.data();
        len = img.size() * 3 *sizeof(T); //Each pixel consist of 3 channels of type T
    }

    
    function boxBlur_4 (scl, tcl, w, h, r) {
        for(var i=0; i<scl.length; i++) tcl[i] = scl[i];
        boxBlurH_4(tcl, scl, w, h, r);
        boxBlurT_4(scl, tcl, w, h, r);
    }
    function boxBlurH_4 (scl, tcl, w, h, r) {
        var iarr = 1 / (r+r+1);
        for(var i=0; i<h; i++) {
            var ti = i*w, li = ti, ri = ti+r;
            var fv = scl[ti], lv = scl[ti+w-1], val = (r+1)*fv;
            for(var j=0; j<r; j++) val += scl[ti+j];
            for(var j=0  ; j<=r ; j++) { val += scl[ri++] - fv       ;   tcl[ti++] = Math.round(val*iarr); }
            for(var j=r+1; j<w-r; j++) { val += scl[ri++] - scl[li++];   tcl[ti++] = Math.round(val*iarr); }
            for(var j=w-r; j<w  ; j++) { val += lv        - scl[li++];   tcl[ti++] = Math.round(val*iarr); }
        }
    }
    function boxBlurT_4 (scl, tcl, w, h, r) {
        var iarr = 1 / (r+r+1);
        for(var i=0; i<w; i++) {
            var ti = i, li = ti, ri = ti+r*w;
            var fv = scl[ti], lv = scl[ti+w*(h-1)], val = (r+1)*fv;
            for(var j=0; j<r; j++) val += scl[ti+j*w];
            for(var j=0  ; j<=r ; j++) { val += scl[ri] - fv     ;  tcl[ti] = Math.round(val*iarr);  ri+=w; ti+=w; }
            for(var j=r+1; j<h-r; j++) { val += scl[ri] - scl[li];  tcl[ti] = Math.round(val*iarr);  li+=w; ri+=w; ti+=w; }
            for(var j=h-r; j<h  ; j++) { val += lv      - scl[li];  tcl[ti] = Math.round(val*iarr);  li+=w; ti+=w; }
        }
    }

    template<typename T>
    void boxblur(T* dat, Image_Shape shape, uint32_t range) {

    }
    
    /*
        http://blog.ivank.net/fastest-gaussian-blur.html
        https://web.stanford.edu/class/cs448f/lectures/2.2/Fast%20Filtering.pdf
    */
    template<typename T>
    void psuedo_gauﬂblur(T* dat, Image_Shape shape, float stdev, uint32_t n = 3) { 
        float wIdeal = sqrt((12.f * stdev * stdev / (float)n) + 1.f);  // Ideal averaging filter width 
        uint32_t wl = wIdeal;  if (wl % 2 == 0) wl--;
        uint32_t wu = wl + 2u;

        float mIdeal = (12.f * stdev * stdev - (float)(n * wl * wl + 4 * n * wl + 3 * n)) / (float)(-4 * wl - 4);
        uint32_t m = mIdeal + 0.5f;
        //printf("Actual sigma: %d", sqrt((float)(m * wl * wl + (n - m) * wu * wu - n) / 12.f ));

        for (var i = 0; i < n; i++) {
            if (i < m)
                boxblur<T>(dat, shape, (wl - 1u) >> 1);  //Width to radius
            else
                boxblur<T>(dat, shape, (wu - 1u) >> 1);   //Width to radius
        }
    }

    template<typename T>
    void flip(T* dat, Image_Shape shape) {

    }

    template<typename T>
    void random_noise(T* dat, Image_Shape shape, float stdev) {

    }

    template<typename T>
    void random_dropout(T* dat, Image_Shape shape, float stdev) {

    }

    template<typename T>
    void random_saturation(T* dat, Image_Shape shape, float satur_mul) {

    }

    template<typename T>
    void random_brightness(T* dat, Image_Shape shape, float bright_mul) {

    }

    template<typename T>
    void rotate(T* dat, Image_Shape shape, float deg) { //TODO

    }

    template<typename T>
    void crop(T* dat, Image_Shape shape, Offset2D<uint32_t> off, Offset2D<uint32_t> new_size) {

    }

    template<typename T>
    void resize(T* dat, Image_Shape old_shape, Offset2D<uint32_t> new_size) {

    }                                                            
}

//===========================================================
//==================|DATASET GENERATING|=====================
//===========================================================

#include <string>
#include <filesystem>

//TODO: HEADER AND MAKE SURE THAT ALL RGB ARRAYS HAVE SAME SIZE
//TODO: NORMALIZATION
namespace DatasetAssemble { //Generates in- and out-file used by DatasetHandler from seperate files for each input and output
    template<typename T>
    void generate(std::string dir_in, std::string dir_exp, std::string file_out1, std::string file_out2) {
        int fd_out1 = open(file_out1.c_str(), O_RDONLY);
        int fd_out2 = open(file_out2.c_str(), O_RDONLY);
        int fd_in, fd_exp;

        std::filesystem::path exp_file_dir(dir_exp);
        struct stat sb;

        for (const auto& entry : std::filesystem::directory_iterator(dir_in)) {
            if (entry.is_regular_file()) {
                //Get paths
                std::filesystem::path in_file_path = entry.path();
                std::filesystem::path exp_file_path = exp_file_dir / in_file_path.filename();

                //Console output
                if (!std::filesystem::exists(exp_file_path))
                    fprintf(stderr, "Input file %s has no file for expected output (%s is missing!)", in_file_path.c_str(), exp_file_path.c_str());
                else
                    printf("Handling files %s and %s", in_file_path.c_str(), exp_file_path.c_str());

                //File IO
                uint32_t len;
                T* dat;

                Image::getRGB(in_file_path.c_str(), dat, len);
                write(fd_out1, dat, len);

                Image::getRGB(exp_file_path.c_str(), dat, len);
                write(fd_out2, dat, len);
            }
        }

        close(fd_out1);
        close(fd_out2);
    }
}
