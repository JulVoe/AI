#pragma once
//TODO: RANDOM and IMAGE namespace algorithms are slow
#define AI_VERSION 1u

#undef min
#undef max
template<typename T> constexpr T max(T a, T b) { return (a > b) ? a : b; }
template<typename T> constexpr T min(T a, T b) { return (a < b) ? a : b; }
template<typename T> constexpr T abs(T x) { return (x >= (T)0) ? x : -x; }
template<typename T> constexpr int32_t sgn(T val) { return (T(0) < val) - (val < T(0)); }
template<typename T> constexpr T bound(T x, T low, T up) { if (x <= low) return low; if (x >= up) return up; return x; }
#include <inttypes.h>
uint32_t reverse_bits_uint(uint32_t in) {
    //https://graphics.stanford.edu/~seander/bithacks.html#BitReverseObvious
    uint32_t ret;

    constexpr uint8_t BitReverseTable256[256] =
    {
#define R2(n)     n,     n + 2*64,     n + 1*64,     n + 3*64
#define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
#define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
        R6(0), R6(2), R6(1), R6(3)
#undef R2
#undef R4
#undef R6
    };

    uint8_t* p = (uint8_t*)&in;
    uint8_t* q = (uint8_t*)&ret;
    q[3] = BitReverseTable256[p[0]];
    q[2] = BitReverseTable256[p[1]];
    q[1] = BitReverseTable256[p[2]];
    q[0] = BitReverseTable256[p[3]];

    return ret;
}

#if defined(__GNUC__) or defined(__clang__)
#include <byteswap.h>
#endif
constexpr uint32_t reverse_bytes_uint(uint32_t in) {
#if defined(__GNUC__) or defined(__clang__)
    return __bswap_32(in);
#else
    return ((in >> 24) & 0xff) | ((in << 8) & 0xff0000) | ((in >> 8) & 0xff00) | ((in << 24) & 0xff000000);
#endif
}

uint32_t gcd(uint32_t a, uint32_t b)
{
    while (b) b ^= a ^= b ^= a %= b;
    return a;
}

/*union {
        float f;
        int32_t i;
    } b = { .f = x };
    b.i &= ~(1 << 31);

    return b.f;*/

//==============================================
//==================|Types|=====================
//==============================================

#include <stdexcept>
#include <cassert>
#include <typeinfo>
#include <inttypes.h>
#include <cuda_fp16.h>

//DO NOT CHANGE THE FOLLOWING VALUES AS IT WILL BREAK OLD DATASETS AND NETWORK CHECKPOINTS
enum TYPE : uint32_t { TYPE_UINT8 = 0, TYPE_INT8 = 1, TYPE_UINT16 = 2, TYPE_INT16 = 3, TYPE_UINT32 = 4, TYPE_INT32 = 5, TYPE_HALF = 6, TYPE_FLOAT = 7, TYPE_DOUBLE = 8 }; 
template<typename T> TYPE type_hash() {
    static_assert(typeid(T) != typeid(T), "Unsupported type!");
    return (TYPE)-1;
}
template<> TYPE type_hash<uint8_t >() { return TYPE::TYPE_UINT8 ; }
template<> TYPE type_hash<int8_t  >() { return TYPE::TYPE_INT8  ; }
template<> TYPE type_hash<uint16_t>() { return TYPE::TYPE_UINT16; }
template<> TYPE type_hash<int16_t >() { return TYPE::TYPE_INT16 ; }
template<> TYPE type_hash<uint32_t>() { return TYPE::TYPE_UINT32; }
template<> TYPE type_hash<int32_t >() { return TYPE::TYPE_INT32 ; }
template<> TYPE type_hash<half    >() { return TYPE::TYPE_HALF  ; }
template<> TYPE type_hash<float   >() { return TYPE::TYPE_FLOAT ; }
template<> TYPE type_hash<double  >() { return TYPE::TYPE_DOUBLE; }

template<TYPE T>
struct TypeById { using type = void; };
template<> struct TypeById<TYPE::TYPE_UINT8 > { using type = uint8_t ; };
template<> struct TypeById<TYPE::TYPE_INT8  > { using type = int8_t  ; };
template<> struct TypeById<TYPE::TYPE_UINT16> { using type = uint16_t; };
template<> struct TypeById<TYPE::TYPE_INT16 > { using type = int16_t ; };
template<> struct TypeById<TYPE::TYPE_UINT32> { using type = uint32_t; };
template<> struct TypeById<TYPE::TYPE_INT32 > { using type = int32_t ; };
template<> struct TypeById<TYPE::TYPE_HALF  > { using type = half    ; };
template<> struct TypeById<TYPE::TYPE_FLOAT > { using type = float   ; };
template<> struct TypeById<TYPE::TYPE_DOUBLE> { using type = double  ; };


constexpr uint16_t sizeOfType(TYPE typeId) {
    switch (typeId) {
    case TYPE_UINT8:
        return sizeof(uint8_t);
    case TYPE_INT8:
        return sizeof(int8_t);
    case TYPE_UINT16:
        return sizeof(uint16_t);
    case TYPE_INT16:
        return sizeof(int16_t);
    case TYPE_UINT32:
        return sizeof(uint32_t);
    case TYPE_INT32:
        return sizeof(int32_t);
    case TYPE_HALF:
        return sizeof(half);
    case TYPE_FLOAT:
        return sizeof(float);
    case TYPE_DOUBLE:
        return sizeof(double);
    default:
        assert(0 == 1);
    }
}

template<typename T>
void convertType(T in, void* out, TYPE typeOut) {
    switch (typeOut) {
    case TYPE::TYPE_UINT8 : *((uint8_t*)out) = (uint8_t)(int)in; break;
    case TYPE::TYPE_INT8  : *((uint8_t*)out) = (uint8_t)(int)in; break;
    case TYPE::TYPE_UINT16: *((uint8_t*)out) = (uint8_t)(int)in; break;
    case TYPE::TYPE_INT16 : *((uint8_t*)out) = (uint8_t)(int)in; break;
    case TYPE::TYPE_UINT32: *((uint8_t*)out) = (uint8_t)(int)in; break;
    case TYPE::TYPE_INT32 : *((uint8_t*)out) = (uint8_t)(int)in; break;
    case TYPE::TYPE_HALF  : *((uint8_t*)out) = (uint8_t)(int)in; break;
    case TYPE::TYPE_FLOAT : *((uint8_t*)out) = (uint8_t)(int)in; break;
    case TYPE::TYPE_DOUBLE: *((uint8_t*)out) = (uint8_t)(int)in; break;
    default: throw new std::runtime_error("[ERROR] Unkown type to convert to\n");
    }
}

void convertType(void* in, void* out, TYPE typeIn, TYPE typeOut) {
    switch (typeIn) {
    case TYPE::TYPE_UINT8 : convertType(*((uint8_t *)in), out, typeOut); break;
    case TYPE::TYPE_INT8  : convertType(*((int8_t  *)in), out, typeOut); break;
    case TYPE::TYPE_UINT16: convertType(*((uint16_t*)in), out, typeOut); break;
    case TYPE::TYPE_INT16 : convertType(*((int16_t *)in), out, typeOut); break;
    case TYPE::TYPE_UINT32: convertType(*((uint32_t*)in), out, typeOut); break;
    case TYPE::TYPE_INT32 : convertType(*((int32_t *)in), out, typeOut); break;
    case TYPE::TYPE_HALF  : convertType(*((half    *)in), out, typeOut); break;
    case TYPE::TYPE_FLOAT : convertType(*((float   *)in), out, typeOut); break;
    case TYPE::TYPE_DOUBLE: convertType(*((double  *)in), out, typeOut); break;
    //default: throw new std::runtime_error("[ERROR] Unkown type to convert from\n");
    }
}


#include "cudnn_ops_infer.h"
template<typename T>
cudnnDataType_t cudnnTypeOf() {
    static_assert(typeid(T) != typeid(T), "Unsupported type!");
    std::exit(-1);
}

template<> cudnnDataType_t cudnnTypeOf<uint8_t>() { return CUDNN_DATA_UINT8 ; };
template<> cudnnDataType_t cudnnTypeOf<int8_t >() { return CUDNN_DATA_INT8  ; };
template<> cudnnDataType_t cudnnTypeOf<int32_t>() { return CUDNN_DATA_INT32 ; };
template<> cudnnDataType_t cudnnTypeOf<half   >() { return CUDNN_DATA_HALF  ; };
template<> cudnnDataType_t cudnnTypeOf<float  >() { return CUDNN_DATA_FLOAT ; };
template<> cudnnDataType_t cudnnTypeOf<double >() { return CUDNN_DATA_DOUBLE; };
//=======================================================
//==================|ERROR CHECKING|=====================
//=======================================================

//C++
#if defined(__GNUC__) or defined(__clang__)
#include <x86intrin.h>
#else
#include <immintrin.h>
#endif

#define CONC_(x,y) x##y
#define CONC(x,y) CONC_(x,y)
#define BUGP(x) printf(x);fflush(stdout);
#define PADDR(x) printf("|%p|\n",&(x));fflush(stdout);
#define STALL(); while(true){}
void YMM_PRINT(__m256  x) { float v[8]; _mm256_storeu_ps((float*)+v, x); printf("%f %f %f %f %f %f %f %f\n", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]); }
void YMM_PRINT(__m256i x) { int   v[8]; _mm256_storeu_si256((__m256i*) + v, x); printf("%d %d %d %d %d %d %d %d\n", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]); }                                                                                                                                            
template<typename T> char* PRINTF_FLAG(T i) { if (typeid(T) == typeid(char)) return "%c"; if (typeid(T)==typeid(char*)) return "%s"; if (typeid(T) == typeid(uint8_t) || typeid(T) == typeid(uint16_t) || typeid(T) == typeid(uint32_t) || typeid(T) == typeid(uint64_t) || typeid(T) == typeid(bool))return (char*)"%llu"; if (typeid(T) == typeid(int8_t) || typeid(T) == typeid(int16_t) || typeid(T) == typeid(int32_t) || typeid(T) == typeid(int64_t))return (char*)"%lld"; if (typeid(T) == typeid(float) || typeid(T) == typeid(double))return (char*)"%.18g"; if (std::is_pointer<T>::value)return (char*)"%p"; assert(0 == 1); return (char*)"[ERROR] Unknown supplied to \"PRINT_VAR\"";/*Unknown type*/ }
template<typename T> void  PRINT_VAR(T i) { printf(PRINTF_FLAG(i), i); fflush(stdout); }
template<typename T> void  ARR_PRINT(T* arr, uint32_t x, uint32_t y) { printf("----------------\n");for(uint32_t y_=0;y_!=y;y_++){for(uint32_t x_=0;x_!=x;x_++){PRINT_VAR(arr[x_+y_*x]);printf("\t");}printf("\n");}printf("----------------\n");}
template<typename T> void  ARR_PRINT_COLMAJ(T* arr, uint32_t x, uint32_t y) { printf("----------------\n");for(uint32_t y_=0;y_!=y;y_++){for(uint32_t x_=0;x_!=x;x_++){PRINT_VAR(arr[x_*y+y_]);printf("\t");}printf("\n");}printf("----------------\n");}

template<typename T> bool CHECK_MEM(T* mem, uint32_t len) { for (uint32_t i = 0; i != len; i++) { if (!isfinite(mem[i])) { return false; } } return true; }

#ifndef static_warning
#if defined(__GNUC__) or defined(__clang__)
#warning static_warning was not defined
#else
#pragma message("static_warning was not defined")
#endif

#include <cstdio>
#define static_warning(a,b) do{if(!(a)){printf(b"\n");}}while(0);
#endif

#ifdef __unix__
#include <cstdio>
#elif defined(WIN32) || defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif
void clear_console() {
#ifdef __unix__
    printf("\x1B[2J\x1B[H");
#elif defined(WIN32) || defined(_WIN32) || defined(_WIN64)
    static HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
    constexpr COORD topLeft = { 0, 0 };
    CONSOLE_SCREEN_BUFFER_INFO screen;
    DWORD written;

    GetConsoleScreenBufferInfo(console, &screen);
    FillConsoleOutputCharacterA(
        console, ' ', screen.dwSize.X * screen.dwSize.Y, topLeft, &written
    );
    FillConsoleOutputAttribute(
        console, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE,
        screen.dwSize.X * screen.dwSize.Y, topLeft, &written
    );
    SetConsoleCursorPosition(console, topLeft);
#else
    static_warning(0==1, "[WARNING] Unsupported operating system for clearing console.");
#endif
}

#include <string>
std::string demangle(const char* name);
#if defined(__GNUC__) or defined(__clang__)
#include <memory>
#include <cxxabi.h>

std::string demangle(const char* name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void(*)(void*)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };

    return (status == 0) ? res.get() : name;
}
#else
#include <Windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

std::string demangle(const char* name) {
    char undecorated_name[1024];
    UnDecorateSymbolName(name, +undecorated_name, 1024, 0x0000);
    return std::string(+undecorated_name);
}
#endif



//Cuda + Cublas + Cudnn
#ifdef DEBUG
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "Cuda assertion triggered: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) std::exit(-code);
    }
}
#define CHECK_CUDA_ERROR();\
    do{\
        auto error = cudaGetLastError(); \
        if (error != cudaSuccess) {\
            /* print the CUDA error message and exit*/\
            printf("CUDA error: %s\n", cudaGetErrorString(error)); \
        }\
    } while (0);


#define CUBLAS_ERROR(e); \
    if((e)!=CUBLAS_STATUS_SUCCESS){\
        printf("[ERROR] Cublas Error %d was encoutered in line %d", e, __LINE__);\
    }

#include "cudnn_ops_infer.h"
#define CUDNN_ERROR(e);\
    if((e)!=CUDNN_STATUS_SUCCESS){\
        printf("[ERROR] Cudnn error %s was encountered in line %d", cudnnGetErrorString(e), __LINE__);\
        std::exit(-1);\
    }
#else
#define gpuErrchk(ans)
#define CHECK_CUDA_ERROR()
#define CUBLAS_ERROR(e)
#define CUDNN_ERROR(e)
#endif

//OpenGl
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#ifdef WIN32
#include <C:\Program Files (x86)\Windows Kits\10\Include\10.0.18362.0\um\gl\GLU.h>
#endif
void showGLerror()
{
    GLenum err;

    while ((err = glGetError()) != GL_NO_ERROR)
        fprintf(stderr, "[Error] OpenGL Error-Code: %d. This is an %s.\n", err, gluErrorString(err));
}

const char* ETB_GL_DEBUG_SOURCE_STR(GLenum source)
{
    static const char* sources[] = {
      "API",   "Window System", "Shader Compiler", "Third Party", "Application",
      "Other", "Unknown"
    };

    int str_idx =
        min<int>(source - GL_DEBUG_SOURCE_API,
            sizeof(sources) / sizeof(const char*));

    return sources[str_idx];
}

const char* ETB_GL_DEBUG_TYPE_STR(GLenum type)
{
    static const char* types[] = {
      "Error",       "Deprecated Behavior", "Undefined Behavior", "Portability",
      "Performance", "Other",               "Unknown"
    };

    int str_idx =
        min<int>(type - GL_DEBUG_TYPE_ERROR,
            sizeof(types) / sizeof(const char*));

    return types[str_idx];
}

const char* ETB_GL_DEBUG_SEVERITY_STR(GLenum severity)
{
    static const char* severities[] = {
      "High", "Medium", "Low", "Unknown"
    };

    int str_idx =
        min<int>(severity - GL_DEBUG_SEVERITY_HIGH,
            sizeof(severities) / sizeof(const char*));

    return severities[str_idx];
}

uint32_t ETB_GL_DEBUG_SEVERITY_COLOR(GLenum severity)
{
    static uint32_t severities[] = {
      0xff0000ff, // High (Red)
      0xff00ffff, // Med  (Yellow)
      0xff00ff00, // Low  (Green)
      0xffffffff  // ???  (White)
    };

    int col_idx =
        min<int>(severity - GL_DEBUG_SEVERITY_HIGH,
            sizeof(severities) / sizeof(uint32_t));

    return severities[col_idx];
}

#define eTB_ColorPrintf printf
#define eTB_FlushConsole(); std::fflush(stdout);
void ETB_GL_ERROR_CALLBACK(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, GLvoid* userParam)
{
    eTB_ColorPrintf(/*0xff00ffff,*/ "OpenGL Error:\n");
    eTB_ColorPrintf(/*0xff808080,*/ "=============\n");
    /*			 */
    eTB_ColorPrintf(/*0xff6060ff,*/ " Object ID: ");
    eTB_ColorPrintf(/*0xff0080ff,*/ "%d\n", id);
    /*			 */
    eTB_ColorPrintf(/*0xff60ff60,*/ " Severity:  ");
    eTB_ColorPrintf(/*ETB_GL_DEBUG_SEVERITY_COLOR(severity),*/
        "%s\n",
        ETB_GL_DEBUG_SEVERITY_STR(severity));

    eTB_ColorPrintf(/*0xffddff80,*/ " Type:      ");
    eTB_ColorPrintf(/*0xffccaa80,*/ "%s\n", ETB_GL_DEBUG_TYPE_STR(type));
    /*			 */
    eTB_ColorPrintf(/*0xffddff80,*/ " Source:    ");
    eTB_ColorPrintf(/*0xffccaa80,*/ "%s\n", ETB_GL_DEBUG_SOURCE_STR(source));
    /*			 */
    eTB_ColorPrintf(/*0xffff6060,*/ " Message:   ");
    eTB_ColorPrintf(/*0xff0000ff,*/ "%s\n\n", message);

    // Force the console to flush its contents before executing a breakpoint
    eTB_FlushConsole();
}

void CheckDebugLog()
{
    constexpr unsigned int count = 10; // max. num. of messages that will be read from the log
    constexpr unsigned int bufsize = 2048;

    unsigned int* sources = new unsigned int[count];
    unsigned int* types = new unsigned int[count];
    unsigned int* ids = new unsigned int[count];
    unsigned int* severities = new unsigned int[count];
    int* lengths = new          int[count];

    char* messageLog = new char[bufsize];
    unsigned int retVal = glGetDebugMessageLogARB(count, bufsize, sources, types, ids, severities, lengths, messageLog);

    if (retVal > 0)
    {
        unsigned int pos = 0;
        for (unsigned int i = 0; i < retVal; i++)
        {
            printf("Source:%s\tType:%s\tID:%d\tSeverity:%s\tMessage:%s\n", ETB_GL_DEBUG_SOURCE_STR(sources[i]), ETB_GL_DEBUG_TYPE_STR(types[i]), ids[i], ETB_GL_DEBUG_SEVERITY_STR(severities[i]), &messageLog[pos]);
            pos += lengths[i];
        }
    }

    delete[] sources;
    delete[] types;
    delete[] ids;
    delete[] severities;
    delete[] lengths;
    delete[] messageLog;
}

#define ALL_GL_ERRORS(); CheckDebugLog(); showGLerror();


//=====================================================
//==================|Library Setup|=====================
//=====================================================
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

cublasHandle_t cublas_handle;
struct CublasConstants {
    double* d[2];
    float * f[2];
    half  * h[2];
} cublasConst;

/*
    Initializes the global cublas variables (see above).

    @param cublasWorkspaceSize: The number of bytes to use for cublas' workspace. Will be alloced internally.
    @param stream             : The stream to use for cublas operations
    @param logging            : If true, turns on cublas logging to stderr 
*/
void cublasSetup(cudaStream_t stream, size_t workspaceSize, void* workspace = nullptr, bool logging = false) {
    //1.: Create handle
    CUBLAS_ERROR(cublasCreate(&cublas_handle));

    //Maybe have to change indexing mode to start with 0 instead of 1 using #define IDX2C(i,j,ld) (((j)*(ld))+(i))

    //2.: Configure Options
    cublasSetAtomicsMode(cublas_handle, CUBLAS_ATOMICS_ALLOWED);
    cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    
    cublasSetStream(cublas_handle, stream);

    //3.: Set workspace
    if(workspace == nullptr)
        cudaMalloc(&workspace, workspaceSize);
    cublasSetWorkspace(cublas_handle, workspace, workspaceSize);

    //4.: Set up constants
    double* d;
    float * f;
    half  * h;
    cudaMalloc(&d, sizeof(double) * 2);
    cudaMalloc(&f, sizeof(float ) * 2);
    cudaMalloc(&h, sizeof(half  ) * 2);

    double d_host[2] = { (double)0., (double)1. };
    float  f_host[2] = { (float )0., (float )1. };
    half   h_host[2] = { (half  )0., (half  )1. };

    cudaMemcpyAsync(d, +d_host, sizeof(double) * 2, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(f, +f_host, sizeof(float ) * 2, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(h, +h_host, sizeof(half  ) * 2, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    cublasConst.d[0] = d;
    cublasConst.d[1] = d + 1;
    cublasConst.f[0] = f;
    cublasConst.f[1] = f + 1;
    cublasConst.h[0] = h;
    cublasConst.h[1] = h + 1;

    //4.: Logging
    if(logging)
        cublasLoggerConfigure(true, false, true, nullptr);   //Turn on logging to stderr
}


#include "cudnn.h"
cudnnHandle_t cudnn_handle;
void* cudnn_workspace;
uint32_t cudnn_workspace_size;

void cudnnLoggingCallback(cudnnSeverity_t sev, void* udata, const cudnnDebug_t* dbg, const char* msg) {
    fprintf(stderr, "[CUDNN] %d: %s\n", (int)sev, msg); 
}

void cudnnSetup(cudaStream_t stream, size_t workspaceSize, void* workspace = nullptr, bool logging = false) {
    //1.: Create handle
    CUDNN_ERROR(cudnnCreate(&cudnn_handle));
    CUDNN_ERROR(cudnnOpsInferVersionCheck()); //Load in all kernels as to make sure the loading is not happening during stream capture and ends up in operation graph

    //2.: Configure Options
    cudnnSetStream(cudnn_handle, stream);

    //3.: Set workspace
    if (workspace == nullptr)
        cudaMalloc(&cudnn_workspace, workspaceSize);
    else
        cudnn_workspace = workspace;
    cudnn_workspace_size = workspaceSize;

    //4.: Logging
    if (logging) {
        printf("[INFO] Enabling cudnn logging\n");
        cudnnSetCallback(CUDNN_SEV_INFO_EN, NULL, &cudnnLoggingCallback);   //Turn on default logging
    }
}

//==========================================================
//==================|Memory management|=====================
//==========================================================
#include <inttypes.h>
#include <cstdio>
#include <memory>
#include "cuda_runtime.h"

template<typename T>
constexpr T roundUpMult(T numToRound, T multiple)               //Returns first number >=numberToRound divvisible by multiple. multiple has to be positive
{
    assert(multiple);
    return ((numToRound + multiple - 1) / multiple);
}

template<typename T>
constexpr T roundUpMultPow2(T numToRound, T multiple)       //Returns first number >=numberToRound divvisible by multiple. multiple has to be a power of two
{
    assert(multiple && ((multiple & (multiple - 1)) == 0));
    return (numToRound + multiple - 1) & -multiple;
}

inline bool is_aligned(const void* ptr, uint64_t alignment) noexcept {
    static_assert(sizeof(uintmax_t) >= sizeof(void*), "[ERROR] No suitable integer type for conversion from pointer type");
    return !(reinterpret_cast<std::uintptr_t>(ptr) % alignment);
}
template<typename T>
inline T* align_pointer_unsafe(T* ptr, uint64_t alignment) {
    //See https://github.com/KabukiStarship/KabukiToolkit/wiki/Fastest-Method-to-Align-Pointers
    return reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(ptr) + alignment - 1u) & -(alignment));
}

struct MemoryRequirement {
public:
    uint64_t num_bytes;   //Lenght of memory in bytes
    uint32_t alignment;   //The minimum number of bytes the pointer needs to be aligned to. Has to be power of 2

    constexpr MemoryRequirement(uint64_t num_bytes = 0ull, uint32_t alignment = 1u) :
        num_bytes(num_bytes),
        alignment(alignment)
    {
        assert((alignment & (alignment - 1)) == 0);
    }

    constexpr MemoryRequirement operator+(MemoryRequirement mr2) {
        return MemoryRequirement(
            roundUpMultPow2<uint64_t>(num_bytes, mr2.alignment) + mr2.num_bytes, //num_bytes
            max(alignment, mr2.alignment)                                        //Alignment
        );
    }

    constexpr void operator+=(MemoryRequirement mr2) {
        MemoryRequirement sum = operator+(mr2);

        num_bytes = sum.num_bytes;
        alignment = sum.alignment;
    }

    void print() {
        printf("\nmr.num_bytes = %llu\nmr.alignment = %u\n", num_bytes, alignment);
    }

    void serialize(FILE* file) {
        fwrite(&num_bytes, sizeof(num_bytes), 1, file);
        fwrite(&alignment, sizeof(alignment), 1, file);
    }
    static void deserialize(FILE* file, MemoryRequirement* out) {
        fread(&out->num_bytes, sizeof(out->num_bytes), 1, file);
        fread(&out->alignment, sizeof(out->alignment), 1, file);
    }
};

struct MemoryRequirementLifetime {
public:
    uint64_t num_bytes;   //Lenght of memory in bytes
    uint32_t alignment;   //The minimum number of bytes the pointer needs to be aligned to. Has to be power of 2
    bool     batchsize_dependend;

    constexpr MemoryRequirementLifetime(uint64_t num_bytes = 0ull, uint32_t alignment = 1u, bool batchsize_dependend = false) :
        num_bytes(num_bytes),
        alignment(alignment),
        batchsize_dependend(batchsize_dependend)
    {
        assert((alignment & (alignment - 1)) == 0);
    }

    constexpr MemoryRequirementLifetime(MemoryRequirement mr, bool batchsize_dependend) :
        MemoryRequirementLifetime(mr.num_bytes, mr.alignment, batchsize_dependend)
    {}

    constexpr MemoryRequirement getMemoryRequirements() {
        return MemoryRequirement(num_bytes, alignment);
    }

    constexpr MemoryRequirementLifetime operator+(MemoryRequirementLifetime mr2) {
        assert(batchsize_dependend == mr2.batchsize_dependend);  //Should not add MemoryRequirements of different lifetime

        return MemoryRequirementLifetime(
            roundUpMultPow2<uint64_t>(num_bytes, mr2.alignment) + mr2.num_bytes, //num_bytes
            max(alignment, mr2.alignment),                                       //Alignment
            batchsize_dependend
        );
    }

    constexpr void operator+=(MemoryRequirementLifetime mr2) {
        MemoryRequirementLifetime sum = operator+(mr2);

        num_bytes = sum.num_bytes;
        alignment = sum.alignment;
    }

    void print() {
        printf("\nmr.num_bytes = %llu\nmr.alignment = %u\nbatch size dependend = %u", num_bytes, alignment, (uint32_t)batchsize_dependend);
    }

    void serialize(FILE* file) {
        fwrite(&num_bytes, sizeof(num_bytes), 1, file);
        fwrite(&alignment, sizeof(alignment), 1, file);
        fwrite(&batchsize_dependend, sizeof(batchsize_dependend), 1, file);
    }
    static void deserialize(FILE* file, MemoryRequirementLifetime* out) {
        fread(&out->num_bytes, sizeof(out->num_bytes), 1, file);
        fread(&out->alignment, sizeof(out->alignment), 1, file);
        fread(&out->batchsize_dependend, sizeof(out->batchsize_dependend), 1, file);
    }
};

template<> constexpr MemoryRequirement max<MemoryRequirement>(MemoryRequirement a, MemoryRequirement b) { 
    return MemoryRequirement(max(a.num_bytes, b.num_bytes), max(a.alignment, b.alignment));  
}

template<> constexpr MemoryRequirementLifetime max<MemoryRequirementLifetime>(MemoryRequirementLifetime a, MemoryRequirementLifetime b) {
    assert(a.batchsize_dependend == b.batchsize_dependend);

    return MemoryRequirementLifetime(max(a.num_bytes, b.num_bytes), max(a.alignment, b.alignment));
}

MemoryRequirement max(std::vector<MemoryRequirementLifetime> requirements) {
    MemoryRequirement ret;
    
    for (uint32_t ind = 0; ind != requirements.size(); ind++)
        ret = max(ret, requirements[ind].getMemoryRequirements());

    return ret;
}

cudaError_t cudaMallocAligned(void** out, MemoryRequirement mr) {
    //Accrding to https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses 5.3.2.1.1, cudaMalloc has alignemt of 256bytes

    if (mr.alignment <= 256) {
        return cudaMalloc(out, mr.num_bytes);
    }
    else {
        fprintf(stderr, "[WARNING] Very high alignment of %u bytes requested!", mr.alignment);
        cudaError_t ret = cudaMalloc(out, mr.num_bytes + mr.alignment - 1);

        //std::align(mr.alignment, mr.num_bytes, out, mr.num_bytes + mr.alignment - 1);
        *out = align_pointer_unsafe(*out, mr.alignment);

        return ret;
    }
}

/*
class GpuMemBlock {
    const uint32_t id;

    const bool growable;

    uint64_t size;
    uint64_t capacity;
    void* mem;
};

class CudaAllocator {
    void* memory;
    uint64_t size;


};
*/

//=======================================================================
//==================|Memory management - Deprecated|=====================
//=======================================================================

struct Dependencies {
    /*
        An object of this class guards a memory region. Each operation that is perfomed on this memory region has to register how it uses the memory region and this class in turn computes the dependencies of this operation
    */

    //If both vectors contain elements, the last operation was a read and the writes are only stored because each new read depends on them
private:
    std::vector<cudaGraphNode_t> unblocked_reads;
    std::vector<cudaGraphNode_t> unblocked_write;           //Could be multiple if using atomics. If there are also read dependencies, all write dependencies are blocked but need to be stored as each new read needs to depend on them

public:
    Dependencies() :
        unblocked_reads(), unblocked_write()
    {}

    /*
        Applies dependecies to a node that performs a operation on the memory region guarded by this

        @param write: True, if "node" writes to the memory segment guarded by this. False, if it just reads it
        @param node : The node that either reads or writes to the memory segment guarded by this
    */
    template<bool write, bool atomic = false>
    void apply(cudaGraph_t graph, cudaGraphNode_t node) {
        static_assert(!atomic, "[ERROR] Dependencie::apply error: Atomic operations not implemented yet!");
        static_assert(write || !atomic, "[ERROR] Dependencies::apply error: A read cannot be atomic (or rather is always atomic... Anyhow, just leave second template parameter empty, ok?)!");

        if constexpr (write) { //node depends on read and write dependencies of "dep"    
            if (unblocked_reads.size()) {                      //There are unblocked reads. Thus, all writes are blocked and we only need to depend on every read
                //1.: Apply dependencies
                for(cudaGraphNode_t& n : unblocked_reads)
                    cudaGraphAddDependencies(graph, &n, &node, 1);

                //2.: Update dependencies
                unblocked_reads.clear();
                unblocked_write.clear();
                unblocked_write.push_back(node);
            }
            else {                                             //There are no unblocked reads. Thus, the node is only dependend on the last writes
                //1.: Apply dependencies
                for (cudaGraphNode_t& n : unblocked_write)
                    cudaGraphAddDependencies(graph, &n, &node, 1);

                //2.: Update dependencies
                unblocked_write.clear();
                unblocked_write.push_back(node);
            }
        }
        else {       //node only interferes with writes
            //1.: Applies dependencies
            for (cudaGraphNode_t& n : unblocked_write)
                cudaGraphAddDependencies(graph, &n, &node, 1);

            //2.: Update dependencies
            unblocked_reads.push_back(node);
        }
    }

    /*
        Resets this dependency guard to its starting state
    */
    void clear() {
        unblocked_reads.clear();
        unblocked_write.clear();
    }
};

//===============================================
//==================|RANDOM|=====================
//===============================================
#if defined(__GNUC__) or defined(__clang__)
#include <x86intrin.h>
#else
#include <immintrin.h>
#endif
#include <random>

namespace Random {
    struct Key {
        __m256i part1;
        __m256i part2;
    } random_key;

    //--------------------|Normal Distribution|-------------------
    //TODO: FAST-NORM
    std::default_random_engine __gen__(1337);
    std::normal_distribution<float> __normal_distr__(0.f, 1.f);
    std::uniform_int_distribution<uint32_t> __uniform_distr__(0u, (uint32_t)-1);

    uint32_t rand_uint(uint32_t m) { //Output in ]0,m]
        return __uniform_distr__(__gen__);
    }

    float rand_float(float m) { //Output in ]0,m]
        return __gen__() * (m / (float)__gen__.max());
    }

    bool rand_prob(float prob) {
        return __gen__() < prob * (float)__gen__.max();
    }

    float rand_normal(float dev) {
        float ret;
        do {
            ret = __normal_distr__(__gen__) * dev;
        } while (!isfinite(ret));
        return ret;
    }

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
#if 0
        __m256i s1 = key.part1;
        const __m256i s0 = key.part2;
        key.part1 = key.part2;
        s1 = _mm256_xor_si256(key.part2, _mm256_slli_epi64(key.part2, 23));
        key.part2 = _mm256_xor_si256(
            _mm256_xor_si256(_mm256_xor_si256(s1, s0),
                _mm256_srli_epi64(s1, 18)), _mm256_srli_epi64(s0, 5));
        return _mm256_add_epi64(key.part2, s0);
#else
        uint32_t arr[8];
        for (uint32_t u = 0; u != 8; u++)
            arr[u] = rand_uint((uint32_t)-1);
        return _mm256_loadu_si256((__m256i*)(+arr));
#endif
    }
    
    //------------------------------------------------------------

    void init_rand() {
        avx_xorshift128plus_init(13075623786050785612ull, 8923524891246523895ull, random_key);  //Deterministic
        __gen__ = std::default_random_engine(679232737264u);
    }
}

//==============================================
//==================|IMAGE|=====================
//==============================================

#include <inttypes.h>
#include <assert.h>
#include <type_traits>
#define cimg_use_jpeg 1
#define cimg_use_png 1
#include "CImg.h"

using namespace cimg_library;

template<typename T>
struct Offset2D {
    T x;
    T y;

    //Constructors
    Offset2D() {}
  
    Offset2D(T x_, T y_){
        x = x_;
        y = y_;
    }

    //Utility
    template<typename Ty> Offset2D<Ty> convert() {
        return Offset2D<Ty>(x, y);
    }
  
    //Arithmetic
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
        Offset2D<T> off(x - m.x, y - m.y);
        return off;
    }

    //Serialization
    void serialize(FILE* file) {
        fwrite(&x, sizeof(x), 1, file);
        fwrite(&y, sizeof(y), 1, file);
    }

    static Offset2D<T> deserialize(FILE* file) {
        Offset2D<T> ret{};
        fread(&ret.x, sizeof(ret.x), 1, file);
        fread(&ret.y, sizeof(ret.y), 1, file);
        return ret;
    }
};

struct Image_Shape {
    uint32_t x;
    uint32_t y;
    uint32_t z;

    //Constructor
    Image_Shape() = default;
  
    constexpr Image_Shape(uint32_t x, uint32_t y, uint32_t z)
        :x(x), y(y), z(z)
    {}

    Image_Shape(uint32_t len, float aspect, uint32_t channels):
        x(sqrt((float)len / (float)(channels * aspect))),
        y(sqrt((float)(len* aspect) / (float)channels)),
        z(channels)
    {
        assert(x * y * z == len);
    }

    //Utility
    constexpr uint32_t prod() const{
        return x * y * z;
    }

    //Setter and Getter
    template<typename T = uint32_t>
    constexpr Offset2D<T> getOffset2D() const  {
        return Offset2D<T>((T)x, (T)y);
    }

    template<typename T>
    constexpr void setOffset2D(Offset2D<T> off) {
        x = off.x;
        y = off.y;
    }

    //Arithmetic
    template<typename T>
    constexpr void operator*=(Offset2D<T> m) {
        x *= m.x;
        y *= m.y;
    }

    template<typename T>
    constexpr Offset2D<T> operator*(Offset2D<T> m) const {
        Offset2D<uint32_t> o{ x * m.x, y * m.y };
        return o;
    }

    //Comparison
    constexpr inline bool operator==(Image_Shape is2) const  {
        return (x == is2.x && y == is2.y && z == is2.z);
    }
    constexpr inline bool operator!=(Image_Shape is2) const  {
        return !operator==(is2);
    }

    //Serialization
    void serialize(FILE* file) const {
        fwrite(&x, sizeof(x), 1, file);
        fwrite(&y, sizeof(y), 1, file);
        fwrite(&z, sizeof(z), 1, file);
    }

    static void deserialize(FILE* file, Image_Shape* out) {
        fread(&out->x, sizeof(out->x), 1, file);
        fread(&out->y, sizeof(out->y), 1, file);
        fread(&out->z, sizeof(out->z), 1, file);
    }
};

namespace Image {
/*
  It makes more sense to store Images channel-first as for cnn's channels are seen in a more general context an can be complety unrelated and are thus not stored interleaved
  Furthermore, channel-first makes cuDnn computations faster. The only reasons, why channel channels-last is supported, are compatability and because it is faster for the cpu, e.g. the augmentation.
  For saturation and brightness, all channels of a pixel are needed and thus a interleaved storage is more cache-coherent. 
*/
    //DO NOT CHANGE ANY OF THESE VALUES OR TYPES!
    enum CHANNEL_ORDER : int8_t   {CHANNELS_FIRST = 0, CHANNELS_LAST = 1};
    enum CHANNELS      : uint16_t {RGBA = 4, RGB = 3, GRAY = 1};
    enum PADDING       : uint8_t  {ZERO_PADDING_NORMAL = 0, ZERO_PADDING_RENORMALIZE = 1, EXTENSION = 2};
    enum DISTRIBUTION  : uint8_t  {UNIFORM = 0, NORMALIZED = 1};  //UNIFORM=uniform distribution in range [0;x]. NORMALIZED=normal distribution in range [-inf, inf]
    struct DATA_FORMAT {
        DISTRIBUTION distribution;
        float range; //If normalized, this is standard deviation and thus has to be >0. If uniform, this is maximum. Minimum is 0 if >0 and else -maximum

        //Constructor
        DATA_FORMAT() = default;

        DATA_FORMAT(DISTRIBUTION d, float r)
            :distribution(d), range(r)
        {}

        DATA_FORMAT(const DATA_FORMAT& d)
            :distribution(d.distribution), range(d.range)
        {}

        //Comparison
        bool operator==(DATA_FORMAT f) {
            return f.distribution == distribution && f.range == range;
        }
        bool operator!=(DATA_FORMAT f) {
            return !operator==(f);
        }

        //Serialization
        void serialize(FILE* file) {
            fwrite(&distribution, sizeof(distribution), 1, file);
            fwrite(&range       , sizeof(range)       , 1, file);
        }

        static DATA_FORMAT deserialize(FILE* file) {
            DATA_FORMAT ret{};
            fread(&ret.distribution, sizeof(distribution), 1, file);
            fread(&ret.range       , sizeof(range)       , 1, file);
            return ret;
        }
    };

    template<typename T> //TODO: Conversion to uniform is wrong
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
        } 
        else {
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

    /*
        Return pixel array and its shape for an image specified using its path. CHANNEL_ORDER and CHANNELS can be specified.
    */
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
            remap_format<T>(dat, shape, old_format, new_format, o);
        }
    }

    /*
        Displays an image above black background

        @param T: The type of the image data
        @param renormalize: Whether to renormalize the image data
        @param dat: The image data
        @param shape: The shape of the image to display
        @param o: Channel order of the data
    */
    template<typename T, bool renormalize = false>
    void show(T* dat, Image_Shape shape, CHANNEL_ORDER o = CHANNEL_ORDER::CHANNELS_FIRST){
        assert(shape.z <= 4);

        CImg<T> img(dat, shape.x, shape.y, 1, shape.z);

        if (o == CHANNEL_ORDER::CHANNELS_LAST) {
            img = CImg<T>(dat, shape.z, shape.x, shape.y, 1);
            img.permute_axes("yzcx");
        } else {
            img = CImg<T>(dat, shape.x, shape.y, 1, shape.z);
        }

        if (shape.z == 4) {
            assert(shape.z != 4 || (std::is_same<T, uint8_t>::value));// "[Error] Currently, only uint8_t RGBA-Images can be displayed");
            constexpr uint8_t background_color = 0;                                                            //Black backgound color
            CImg<T> render(shape.x, shape.y, 1, 3, background_color);
            render.draw_image(0, 0, 0, 0, img, img.get_channel(3), 1, 255);
            

            CImgDisplay disp(render, "", renormalize);
            disp.move(0, 0);

            while (!disp.is_closed()) { if (disp.key(cimg::keyESC)) break; disp.wait(); }
            //getchar();
        }
        else {
            CImgDisplay disp(img, "", renormalize);
            disp.move(0, 0);

            while (!disp.is_closed()) { if (disp.key(cimg::keyESC)) break; disp.wait();}
            //getchar();
        }

        if constexpr(0 == CHANNEL_ORDER::CHANNELS_LAST)
            img.permute_axes("yzcx");
    }

    template<typename T, PADDING padding = PADDING::ZERO_PADDING_RENORMALIZE, typename ACCU_T = double>
    void boxblur(T* dat, Image_Shape shape, uint32_t w, uint32_t h, CHANNEL_ORDER order = CHANNEL_ORDER::CHANNELS_FIRST) {
        //0.: Check parameters
        assert(w % 2 == 1 && h % 2 == 1 && w >= 3 && h >= 3);           //Width and height have to be odd, so the sliding windows has an whole number as radius. 1 makes no sense
        assert(shape.x >= w && shape.y >= h);                           //Image should be bigger than filter radius
        
        //1.: Compute radii and allocate storage
        uint32_t r_w = (w - 1u) >> 1;
        uint32_t r_h = (h - 1u) >> 1;
        T* buf = (T*)malloc(sizeof(T) * (1 + max(r_w, r_h)));          //Stores last r_w or r_h original image values (for each channel) to subtract from the end of the sliding window when it passes
        
        uint32_t mul;
        if (order == CHANNEL_ORDER::CHANNELS_FIRST)
            mul = 1;
        else
            mul = shape.z;
        
        for(uint32_t chn = 0; chn != shape.z; chn++) {
            //2.: Horizontal pass
            float norm = 1.f / (float)(2 * r_w + 1);
            for(uint32_t line = 0; line != shape.y; line++) {
                uint32_t cur_ind;                  //Index of currently modified component
                if (order == CHANNELS_FIRST)
                    cur_ind = line * shape.x + chn * shape.x * shape.y;
                else
                    cur_ind = line * shape.x * shape.z + chn;
                uint32_t li = 0;                   //Left index of sliding window, in buf
                uint32_t ri = cur_ind + r_w * mul;
        
                T firstValue = dat[cur_ind];
                T  lastValue = dat[cur_ind + (shape.x - 1) * mul];
                ACCU_T accu = 0;
                
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
                        dat[cur_ind] = accu / (r_w + 1 + j);
                    else
                        dat[cur_ind] = accu * norm;
        
                    ri += mul;
                    cur_ind += mul;
                    if(++li == r_w + 1)
                        li = 0;
                }
        
                for(uint32_t j = r_w + 1; j != shape.x-r_w; j++) {
                    accu += (ACCU_T)dat[ri] - (ACCU_T)buf[li];
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
                if (order == CHANNEL_ORDER::CHANNELS_FIRST)
                    cur_ind = col + chn * shape.x * shape.y;
                else
                    cur_ind = col * shape.z + chn; 
                uint32_t li      = 0;                                            //Left index. Not of image, but of buf
                uint32_t ri      = cur_ind + r_h * shape.x * mul;                //Right index
                T firstValue     = dat[cur_ind];                                 //Extends first value over the edge of the image
                T  lastValue     = dat[cur_ind + (shape.y - 1) * shape.x * mul]; //Extends last  value over the edge of the image
                ACCU_T accu = 0;
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
                    accu += (ACCU_T)dat[ri] - (ACCU_T)buf[li];
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

        float mIdeal = (12.f * stdev * stdev - (float)(n * wl * wl + 4.f * n * wl + 3.f * n)) / (float)(-4.f * wl - 4.f);
        uint32_t m = mIdeal + 0.5f;
        //printf("Actual sigma: %d", sqrt((float)(m * wl * wl + (n - m) * wu * wu - n) / 12.f ));
        
        for (uint32_t i = 0; i != n; i++) {
            if (i < m)
                if(wl != 1)                                           //Compiler will optimize out loop independent condition
                    boxblur<T, padding>(dat, shape, wl, wl, order);
            else
                boxblur<T, padding>(dat, shape, wu, wu, order);
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
            dat[ind] = bound(dat[ind] + Random::rand_normal(stdev), (T)0, (T)1);
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
        
        //1.: Computation
        T *r, *g, *b;
        for(uint32_t ind = 0; ind != shape.x * shape.y; ind ++){
            r = dat + m2 * ind + 0 * m1;
            g = dat + m2 * ind + 1 * m1;
            b = dat + m2 * ind + 2 * m1;

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
                *r = bound<T>(*r, 0, format.range);
                *g = bound<T>(*g, 0, format.range);
                *b = bound<T>(*b, 0, format.range);
                
                if (format.range < 0) {
                    *r -= format.range;
                    *g -= format.range;
                    *b -= format.range;
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

        //0.5.: Scalars for the right indexing scheme depending on order
        uint32_t m1, m2;
        if(order == CHANNEL_ORDER::CHANNELS_LAST){
            m1 = 1;
            m2 = shape.z;
        } else {
            m1 = shape.x * shape.y;
            m2 = 1;
        }
        
        //1.: Computation
        T *r, *g, *b;
        for(uint32_t ind = 0; ind != shape.x * shape.y; ind ++){
            r = dat + m2 * ind + 0 * m1;
            g = dat + m2 * ind + 1 * m1;
            b = dat + m2 * ind + 2 * m1;

            *r = (*r) * bright_mul;
            *g = (*g) * bright_mul;
            *b = (*b) * bright_mul;

            if (format.distribution == DISTRIBUTION::UNIFORM) {
                *r = bound<T>(*r, 0, format.range);
                *g = bound<T>(*g, 0, format.range);
                *b = bound<T>(*b, 0, format.range);

                if (format.range < 0) {
                    *r = min((T)-format.range, *r);
                    *g = min((T)-format.range, *g);
                    *b = min((T)-format.range, *b);
                }
            }
        }
    }

    //TODO: not implemented
    template<typename T>
    void rotate(T* dat, Image_Shape shape, float deg) {
        assert(0 == 1);
    }

    template<typename T>
    void translate(T* dat, Image_Shape shape, Offset2D<int32_t> off, CHANNEL_ORDER order = CHANNEL_ORDER::CHANNELS_FIRST) {
        //0.: Scalars for the right indexing scheme depending on order
        uint32_t m1, m2;
        if (order == CHANNEL_ORDER::CHANNELS_LAST) {
            m1 = 1;
            m2 = shape.z;
        }
        else {
            m1 = shape.x * shape.y;
            m2 = 1;
        }
        
        if (off.x > 0) {//To the right
            if (off.y > 0) {//Down
                for (int32_t new_x = shape.x - 1; new_x >= 0; new_x--) {
                    for (int32_t new_y = shape.y - 1; new_y >= 0; new_y--) {
                        int32_t old_x = new_x - off.x;
                        int32_t old_y = new_y - off.y;

                        for (uint32_t channel = 0; channel != shape.z; channel++) {
                            T* new_p = dat + m2 * (new_x + new_y * shape.x) + channel * m1;
                            T* old_p = dat + m2 * (old_x + old_y * shape.x) + channel * m1;

                            if (old_x < 0 || old_x >= shape.x || old_y < 0 || old_y >= shape.y)
                                *new_p = (T)0;
                            else
                                *new_p = *old_p;
                        }
                    }
                }
            }
            else {//Up
                for (int32_t new_x = shape.x - 1; new_x >= 0; new_x--) {
                    for (int32_t new_y = 0; new_y != shape.y; new_y++) {
                        int32_t old_x = new_x - off.x;
                        int32_t old_y = new_y - off.y;

                        for (uint32_t channel = 0; channel != shape.z; channel++) {
                            T* new_p = dat + m2 * (new_x + new_y * shape.x) + channel * m1;
                            T* old_p = dat + m2 * (old_x + old_y * shape.x) + channel * m1;

                            if (old_x < 0 || old_x >= shape.x || old_y < 0 || old_y >= shape.y)
                                *new_p = (T)0;
                            else
                                *new_p = *old_p;
                        }
                    }
                }
            }
        }
        else {//To the left
            if (off.y > 0) {//Down
                for (int32_t new_x = 0; new_x != shape.x; new_x++) {
                    for (int32_t new_y = shape.y - 1; new_y >= 0; new_y--) {
                        int32_t old_x = new_x - off.x;
                        int32_t old_y = new_y - off.y;

                        for (uint32_t channel = 0; channel != shape.z; channel++) {
                            T* new_p = dat + m2 * (new_x + new_y * shape.x) + channel * m1;
                            T* old_p = dat + m2 * (old_x + old_y * shape.x) + channel * m1;

                            if (old_x < 0 || old_x >= shape.x || old_y < 0 || old_y >= shape.y)
                                *new_p = (T)0;
                            else
                                *new_p = *old_p;
                        }
                    }
                }
            }
            else {//Up
                for (int32_t new_x = 0; new_x != shape.x; new_x++) {
                    for (int32_t new_y = 0; new_y != shape.y; new_y++) {
                        int32_t old_x = new_x - off.x;
                        int32_t old_y = new_y - off.y;

                        for (uint32_t channel = 0; channel != shape.z; channel++) {
                            T* new_p = dat + m2 * (new_x + new_y * shape.x) + channel * m1;
                            T* old_p = dat + m2 * (old_x + old_y * shape.x) + channel * m1;

                            if (old_x < 0 || old_x >= shape.x || old_y < 0 || old_y >= shape.y)
                                *new_p = (T)0;
                            else
                                *new_p = *old_p;
                        }
                    }
                }
            }
        }
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
  
    template<typename T, CHANNEL_ORDER order = CHANNEL_ORDER::CHANNELS_FIRST>
    void resize(T* dat, Image_Shape old_shape, Offset2D<uint32_t> new_size, PADDING padding = PADDING::ZERO_PADDING_RENORMALIZE) {
        assert((old_shape.x <= new_size.x && old_shape.y <= new_size.y) || (old_shape.x >= new_size.x && old_shape.y >= new_size.y));

        //printf("\n%p %u %u %u %u %u %u\n", dat, old_shape.x, old_shape.y, old_shape.z, new_size.x, new_size.y, (unsigned)padding);

        uint32_t m1_i, m2_i, m1_o, m2_o;
        if constexpr (order == CHANNEL_ORDER::CHANNELS_LAST) {
            m1_i = 1;
            m2_i = old_shape.z;

            m1_o = 1;
            m2_o = old_shape.z;
        }
        else {
            m1_i = old_shape.x * old_shape.y;
            m2_i = 1;

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
      T newVal = frac_x2*frac_y2*(float)(*in1)+frac_x1*frac_y2*(float)(*in2)+frac_x2*frac_y1*(float)(*in3)+frac_x1*frac_y1*(float)(*in4); \
      dat[channel * m1_o + (y * new_size.x + x) * m2_o] = newVal;

        //Real code starts here
        if (old_shape.x >= new_size.x && old_shape.y >= new_size.y) {
            for (uint32_t channel = 0; channel != old_shape.z; channel++) {
                for (uint32_t y = 0; y != new_size.y; y++) {
                    __resize__INTERNAL1();
                    for (uint32_t x = 0; x != new_size.x; x++) {
                        __resize__INTERNAL2();
                    }
                }
            }
        }
        else {
            for (int32_t channel = old_shape.z - 1; channel >= 0; channel--) {
                for (int32_t y = new_size.y - 1; y >= 0; y--) {
                    __resize__INTERNAL1();
                    for (int32_t x = new_size.x - 1; x >= 0; x--) {
                        __resize__INTERNAL2();
                    }
                }
            }
        }
#undef __resize__INTERNAL1
#undef __resize__INTERNAL2
    }

    //No aliasing allowed
    template<typename T, CHANNEL_ORDER order_in = CHANNEL_ORDER::CHANNELS_FIRST, CHANNEL_ORDER order_out = order_in>
    void resize(T* dat, T* out , Image_Shape old_shape, Offset2D<uint32_t> new_size, PADDING padding = PADDING::ZERO_PADDING_RENORMALIZE) {
        assert((old_shape.x <= new_size.x && old_shape.y <= new_size.y) || (old_shape.x >= new_size.x && old_shape.y >= new_size.y));

        //printf("\n%p %u %u %u %u %u %u\n", dat, old_shape.x, old_shape.y, old_shape.z, new_size.x, new_size.y, (unsigned)padding);

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
      T newVal = frac_x2*frac_y2*(float)(*in1)+frac_x1*frac_y2*(float)(*in2)+frac_x2*frac_y1*(float)(*in3)+frac_x1*frac_y1*(float)(*in4); \
      out[channel * m1_o + (y * new_size.x + x) * m2_o] = newVal;

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
    //This may make temperature calibration of softmax obsolete
    template<typename T>
    void label_smoothing(T* dat, Image_Shape shape, float factor = 0.1f){
        for(uint32_t ind = 0; ind != shape.prod(); ind++) {
            dat[ind] *=  (1.f - factor);
            dat[ind] += factor / shape.prod();
        }
    }


    /*
        Converts float channel first images with data in a uniform distribution over [0,1] to uint8_t channel last images with uniform distibution over [0,255]
    */
    template<typename T>
    void shrinkToGL(T* dat, Image_Shape shape, uint8_t* out) {
#define NEW_IND(x,y,c) (*(out + 3 * x + c + y * (3 * shape.x)))
#define OLD_IND(x,y,c) (*(dat + x + y * shape.x + c * shape.x * shape.y * (shape.z!=1)))
        for (uint32_t y = 0; y != shape.y; y++) {
            for (uint32_t x = 0; x != shape.x; x++) {
                NEW_IND(x, y, 0) = OLD_IND(x, y, 0) * 256.f;
                NEW_IND(x, y, 1) = OLD_IND(x, y, 1) * 256.f;
                NEW_IND(x, y, 2) = OLD_IND(x, y, 2) * 256.f;
            }
        }
    }

    /*
        Converts single channel to 3 channels

        @param in:  Input  data. Has to be lenght   len
        @param out: Output data. Has to be lenght 3*len
        @param len: Number of elements of type T that "in" holds
    */
    template<typename T_in, typename T_out, CHANNEL_ORDER ord>
    void grayToRGB(T_in* in, T_out* out, uint32_t len) {
        if constexpr (ord == CHANNEL_ORDER::CHANNELS_FIRST) {
            T_in* i  = in;
            T_out* o  = out;
            T_out* o1 = o  + len;
            T_out* o2 = o1 + len;
            T_out* o3 = o2 + len;

            while (o != o1)
                *o++ = (T_out)*i++;
            i = in;
            while (o != o2)
                *o++ = (T_out)*i++;
            i = in;
            while (o != o3)
                *o++ = (T_out)*i++;
        }
        else {
            T_in*  i  = in;
            T_in* end = i + len;
            T_out* o  = out;
            while (i != end) {
                *o++ = (T_out)*i;
                *o++ = (T_out)*i;
                *o++ = (T_out)*i;
                i++;
            }
        }
    }

    /*
        Sets each red value to r, each green value to g and each blue value to b. Uses input data as alph channel

        @param in:  Input  data. Has to be lenght   len. Gets copied to alpha channel
        @param out: Output data. Has to be lenght 4*len
        @param len: Number of elements of type T that "in" holds
        @param r: Red   channel gets set to this value
        @param g: Green channel gets set to this value
        @param b: Blue  channel gets set to this value
    */
    template<typename T_in, typename T_out, CHANNEL_ORDER ord> //TODO: Not working according to "show"
    void grayToRGBA(T_in* in, T_out* out, uint32_t len, T_out r, T_out g, T_out b) {
        if constexpr (ord == CHANNEL_ORDER::CHANNELS_FIRST) {
            T_in * i = in;
            T_out* o = out;
            T_out* o1 = o + len;
            T_out* o2 = o1 + len;
            T_out* o3 = o2 + len;
            T_out* o4 = o3 + len;

            while (o != o1)
                *o++ = r;
            while (o != o2)
                *o++ = g;
            while (o != o3)
                *o++ = b;
            while (o != o4)
                *o++ = *i++;
        }
        else {
            T_in*  i = in;
            T_in*  end = i + len;
            T_out* o = out;
            while (i != end) {
                *o++ = r;
                *o++ = g;
                *o++ = b;
                *o++ = (T_out)*i++;
            }
        }
    }
}

//============================================
//==================|CSV|=====================
//============================================

#include <string>
#include <vector>
#include <cstdio>

namespace CSV {
    template<typename T>
    T destringify(std::string in){
        static_assert(typeid(T) != typeid(T), "stringify has currently no implementation for the type u supplied");
    }
    template<> uint32_t destringify<uint32_t>(std::string in){ return std::stoul(in);}
    template<> int32_t  destringify< int32_t>(std::string in){ return std::stoi(in); }
    template<> uint16_t destringify<uint16_t>(std::string in){ return std::stoul(in);}
    template<> int16_t  destringify< int16_t>(std::string in){ return std::stoi(in); }
    template<> uint8_t  destringify<uint8_t >(std::string in){ return std::stoul(in);}
    template<> int8_t   destringify< int8_t >(std::string in){ return std::stoi(in); }
    template<> float    destringify<float   >(std::string in){ return std::stof(in); }
    template<> double   destringify<double  >(std::string in){ return std::stod(in); }
  
  
    /*
        Loads a file of comma-seperated values.

        Values are seperated either by delim or a newline. A newline is only allowed to occur after a delimiter.
        After the last value, no delimiter or newline shall be used. Each line has to have the same number of values.

        @param T: Type of values
        @param path: Path to file
        @param delim: Delimiter
        @param size: Output. Number of lines and number of elements per line.
    */
    template<typename T>
    T* loadCSV(const char* path, Offset2D<uint32_t>& size, char delim = ','){
        //Read file into ram
        FILE* fid  = fopen(path, "r");
        fseek(fid, 0, SEEK_END);
        uint32_t num_bytes = ftell(fid);
        rewind(fid);
        char* mem = (char*)malloc(num_bytes);
        fread(mem, sizeof(char), num_bytes, fid);

        uint32_t num_elements = 0;
        uint32_t line_counter = 0;

        T* out = (T*)malloc(sizeof(T)* ((num_bytes+1) >> 1));             //At most half of the characters are individual numbers
        for(char *sta=mem, *sto=mem; sto <= mem + num_bytes; sto++){
            if (*sto == '\n') {
                line_counter++;
                assert(sta == sto);                                       //Newline only right after delimiter
                sta++;
            }
            else if(*sto == delim || sto == mem + num_bytes){             //Region sta->sto contains new element
                std::string el_str = std::string(sta, sto);
                out[num_elements++] = destringify<T>(el_str);

                sta = sto + 1;
            }
        }
        line_counter++;                                                   //Also count last line

        assert(num_elements % line_counter == 0);                         //Each line has same ammount of numbers
        size = Offset2D<uint32_t>(num_elements / line_counter, line_counter);
        return out;
    }

    /*
        Takes a vector "dat" of "len" values in the range ]0, channels-1[.
        Expands this to an image, as for each value is replaced by "channel" values - channels. The original value is used an an index. The channel of
        thius index is set to 1, the other channels of this values are set to 0.
    */
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

        return out;
    }
}

//===========================================================
//==================|DATASET GENERATING|=====================
//===========================================================
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <fstream>

#ifdef EXPERIMENTAL_FILESYSTEM
#include <experimental/filesystem>
#define __FILESYSTEM__ std::experimental::filesystem
#else
#include <filesystem>
#define __FILESYSTEM__ std::filesystem
#endif

/*
    Each Dataset must start with a header. It consist of (in this order)
     - A 6 byte signature ("JVDATA")
     - 2 bytes that store the library version that was used to generate the dataset
     - A HEADER_V* object according to the version previously specified
     
     After that, the data follows.

     In newer versions of this library, the file format is allowed to change. In that case, one might want to use a different file header which stores
     more or different infomation. In that case, implement a new HEADER_V* class. Whereever a HEADER_V* was used before, see what HEADER_V* version is
     needed an implement the correct conversion in your new class and call this conversion at those places. Thus, only new conversions need to be implemented
     and the code will still work.
*/

struct HEADER_V1 {
    TYPE type;                    //Type of data:        uint32_t  
    uint32_t x;                   //Width per sample:    uint32_t  
    uint32_t y;                   //Height per sample:   uint32_t  
    uint32_t z;                   //Depth per sample:    uint32_t  
    Image::CHANNEL_ORDER order;   //Channel order:       Image::CHANNEL_ORDER   
    Image::DATA_FORMAT format;    //Format of data:      Image::DATA:FORMAT

    //Constructor
    HEADER_V1() = default;

    HEADER_V1(TYPE type, uint32_t x, uint32_t y, uint32_t z, Image::CHANNEL_ORDER order, Image::DATA_FORMAT format) :
        type(type),
        x(x),
        y(y),
        z(z),
        order(order),
        format(format)
    {}

    HEADER_V1(const HEADER_V1& h) :
        type(h.type), x(h.x), y(h.y), z(h.z), order(h.order), format(h.format)
    {}

    //Conversion to different versions


    //Serialization
    void serialize(FILE* file) {
        fwrite(&type, sizeof(type), 1, file);
        fwrite(&x, sizeof(x), 1, file);
        fwrite(&y, sizeof(y), 1, file);
        fwrite(&z, sizeof(z), 1, file);
        fwrite(&order, sizeof(order), 1, file);
        format.serialize(file);
    }

    static HEADER_V1 deserialize(FILE* file) {
        HEADER_V1 ret{};

        fread(&ret.type, sizeof(type), 1, file);
        fread(&ret.x, sizeof(x), 1, file);
        fread(&ret.y, sizeof(y), 1, file);
        fread(&ret.z, sizeof(z), 1, file);
        fread(&ret.order, sizeof(order), 1, file);
        ret.format = Image::DATA_FORMAT::deserialize(file);
        return ret;
    }
};

//HEADER: |SIGNATURE(6)|VERSION(2)|HEADER_LENGHT(2)|TYPE(4)|SIZE.X(4)|SIZE.Y(4)|SIZE.Z(4)|CHANNEL_ORDER(1)|DISTRIBUTION(1)|RANGE(4)|------------DATA-------------|
namespace DatasetAssemble {
    //TODO: Memory leaks
    template<typename T, Image::CHANNELS channels = Image::CHANNELS::RGB, Image::CHANNEL_ORDER order = Image::CHANNEL_ORDER::CHANNELS_FIRST, Image::DISTRIBUTION distr = Image::DISTRIBUTION::UNIFORM, int32_t range = 1>
    void generateDatasetFile_Image(std::string dir, std::string file_out, Offset2D<uint32_t> size) {
        //0.: Check arguments
        static_warning(distr == Image::DISTRIBUTION::UNIFORM, "A dataset should contain uniform data to enable easier augmentation!");

        //1.: Open file
        FILE* file = fopen(file_out.c_str(), "wb");

        //2.: Write file header
        printf("[INFO] Writing header\n");
        char signature[] = {"JVDATA"};
        uint16_t version = AI_VERSION;
        HEADER_V1 header(
            type_hash<T>(), 
            size.x, size.y, channels, 
            order, 
            Image::DATA_FORMAT(distr, range)
        );
        
        fwrite(+signature, sizeof(signature) - 1, 1, file); //Signature:         6*uint8_t
        fwrite(&version  , sizeof(version)      , 1, file); //Version:             uint16_t
        header.serialize(file);

        //3.: Read and sort filenames
        printf("[INFO] Reading in filenames\n");
        std::vector<std::string> paths;
        for (const auto& entry : __FILESYSTEM__::directory_iterator(dir)) {
            if (__FILESYSTEM__::is_regular_file(entry.path())) {
                paths.push_back(entry.path().string());
            }
        }
        printf("[INFO] Sorting filenames\n"); //Filenames are sorted, so that input and output files are matched correctly (same name, different extension)
        std::sort(paths.begin(), paths.end(), [](const auto& lhs, const auto& rhs){return lhs < rhs;}); 

        //4.: Handle files one by one
        for(const std::string& in_file_path : paths){
            printf("[INFO] Handling file %s\n", in_file_path.c_str());

            //File IO
            Image_Shape shape_;
            T* dat;

            Image::getPixels<T, order, channels, distr, range>((char*)in_file_path.c_str(), dat, shape_);
            Image::resize<T, order>(dat, shape_, size, Image::PADDING::ZERO_PADDING_RENORMALIZE);

            fwrite(dat, sizeof(T), size.x * size.y * channels, file);
        }
        fclose(file);
    }
    
    /*
        Propability for each class seperated by ","
    */
    template<typename T, Image::CHANNEL_ORDER order = Image::CHANNEL_ORDER::CHANNELS_FIRST, Image::DISTRIBUTION distr = Image::DISTRIBUTION::UNIFORM, int32_t range = 1>
    void generateDatasetFile_Raw(std::string dir, std::string file_out) {
        //0.: Check arguments
        static_warning(distr == Image::DISTRIBUTION::UNIFORM, "A dataset should contain uniform data to enable easier augmentation!");

        //1.: Open file
        FILE* file = fopen(file_out.c_str(), "wb");

        //2.: Write file header
        printf("[INFO] Writing header\n");
        char signature[] = {"JVDATA"};
        uint16_t version = AI_VERSION;
        HEADER_V1 header(
            type_hash<T>(), 
            0, 0, 1, 
            order, 
            Image::DATA_FORMAT(distr, range)
        );
        
        fwrite(+signature, sizeof(signature) - 1, 1, file); //Signature:         6*uint8_t
        fwrite(&version  , sizeof(version)      , 1, file); //Version:             uint16_t

        //3.: Read and sort filenames
        printf("[INFO] Reading in filenames\n");
        std::vector<std::string> paths;
        for (const auto& entry : __FILESYSTEM__::directory_iterator(dir)) {
            if (__FILESYSTEM__::is_regular_file(entry.path())) {
                paths.push_back(entry.path().string());
            }
        }
        printf("[INFO] Sorting filenames\n");               //Sort filenames to corretly match input and output files (same name, different extension)
        std::sort(paths.begin(), paths.end(), [](const auto& lhs, const auto& rhs){return lhs < rhs;});

        //4.: Handle file one by on
        for(const std::string& in_file_path : paths){
            printf("[INFO] Handling file %s\n", in_file_path.c_str());

            //File IO
            Offset2D<uint32_t> size;
            T* dat = CSV::loadCSV<T>(in_file_path.c_str(), size);

            if (header.x == 0) {
                header.x = size.x;
                header.y = size.y;
                header.serialize(file);
            }
            else
                assert((header.x == size.x) && (header.y == size.y));
            
            fwrite(dat, sizeof(T), size.x * size.y, file);
        }

        fclose(file);
    }

    /*
        Index of right class
    */
    template<typename T, Image::CHANNEL_ORDER order = Image::CHANNEL_ORDER::CHANNELS_FIRST, Image::DISTRIBUTION distr = Image::DISTRIBUTION::UNIFORM, int32_t range = 1>
    void generateDatasetFile_Classification(std::string dir, std::string file_out, uint32_t num_classes) {
        //0.: Check arguments
        static_warning(distr == Image::DISTRIBUTION::UNIFORM, "A dataset should contain uniform data to enable easier augmentation!");

        //1.: Open file
        FILE* file = fopen(file_out.c_str(), "wb");

        //2.: Write file header
        printf("[INFO] Writing header\n");
        char signature[] = { "JVDATA" };
        uint16_t version = AI_VERSION;
        HEADER_V1 header(
            type_hash<T>(),
            num_classes, 1, 1,
            order,
            Image::DATA_FORMAT(distr, range)
        );

        fwrite(+signature, sizeof(signature) - 1, 1, file); //Signature:         6*uint8_t
        fwrite(&version  , sizeof(version)      , 1, file); //Version:             uint16_t
        header.serialize(file);

        //3.: Read and sort filenames
        printf("[INFO] Reading in filenames\n");
        std::vector<std::string> paths;
        for (const auto& entry : __FILESYSTEM__::directory_iterator(dir)) {
            if (__FILESYSTEM__::is_regular_file(entry.path())) {
                paths.push_back(entry.path().string());
            }
        }
        printf("[INFO] Sorting filenames\n");               //Sort filenames to corretly match input and output files (same name, different extension)
        std::sort(paths.begin(), paths.end(), [](const auto& lhs, const auto& rhs) {return lhs < rhs; });

        //4.: Handle file one by on
        for (const std::string& in_file_path : paths) {
            printf("[INFO] Handling file %s\n", in_file_path.c_str());

            //File IO
            uint32_t class_index;
            std::fstream(in_file_path.c_str(), std::ios_base::in) >> class_index;
            T* dat = CSV::vec_to_img<uint32_t, T, order>(&class_index, 1u, num_classes);

            fwrite(dat, sizeof(T), num_classes, file);
            free(dat);
        }

        fclose(file);
    }

    /*
        Input file must contain the class of each pixel as a number, seperated by ",".
    */
    template<typename T, Image::CHANNEL_ORDER order = Image::CHANNEL_ORDER::CHANNELS_FIRST, Image::DISTRIBUTION distr = Image::DISTRIBUTION::UNIFORM, int32_t range = 1>
    void generateDatasetFile_Segmentation(std::string dir, std::string file_out, Image_Shape shape, Image::DATA_FORMAT old_format) {
        //0.: Check arguments
        static_warning(distr == Image::DISTRIBUTION::UNIFORM, "A dataset should contain uniform data to enable easier augmentation!");

        //1.: Open file
        FILE* file = fopen(file_out.c_str(), "wb");

        //2.: Write file header
        printf("[INFO] Writing header\n");
        char signature[] = { "JVDATA" };
        uint16_t version = AI_VERSION;
        HEADER_V1 header(
            type_hash<T>(), 
            shape.x, shape.y, shape.z, 
            order,
            Image::DATA_FORMAT(distr, range)
        );

        fwrite(+signature, sizeof(signature) - 1, 1, file); //Signature:         6*uint8_t
        fwrite(&version  , sizeof(version)      , 1, file); //Version:             uint16_t
        header.serialize(file);

        //3.: Read and sort filenames
        printf("[INFO] Reading in filenames\n");
        std::vector<std::string> paths;
        for (const auto& entry : __FILESYSTEM__::directory_iterator(dir)) {
            if (__FILESYSTEM__::is_regular_file(entry.path())) {
                paths.push_back(entry.path().string());
            }
        }
        printf("[INFO] Sorting filenames\n");              //Sort filenames to correctly match input and output files (same name, different extension)
        std::sort(paths.begin(), paths.end(), [](const auto& lhs, const auto& rhs) {return lhs < rhs; });

        //4.: Handle files one by one
        for (const std::string& in_file_path : paths) {
            printf("[INFO] Handling file %s\n", in_file_path.c_str());

            //File IO
            Image::DATA_FORMAT format(distr, range);

            Offset2D<uint32_t> old_size;
            T* dat = CSV::loadCSV<T>(in_file_path.c_str(), old_size);
            dat = CSV::vec_to_img<T, T, order>(dat, old_size.x * old_size.y, shape.z);
            Image_Shape old_shape = shape;
            old_shape.setOffset2D(old_size);

            Image::resize<T, order>(dat, old_shape, shape.getOffset2D(), Image::PADDING::ZERO_PADDING_RENORMALIZE);
            Image::remap_format<T>(dat, shape, old_format, format, order);

            fwrite(dat, sizeof(T), shape.x * shape.y * shape.z, file);
        }

        fclose(file);
    }


    template<typename T, Image::CHANNELS channels = Image::CHANNELS::RGB, Image::CHANNEL_ORDER order = Image::CHANNEL_ORDER::CHANNELS_FIRST, Image::DISTRIBUTION distr = Image::DISTRIBUTION::UNIFORM, int32_t range = 1>
    void convertMNIST(std::string path_in, std::string path_out, Offset2D<uint32_t> size) {
        //0.: Check arguments
        static_warning(distr == Image::DISTRIBUTION::UNIFORM, "A dataset should contain uniform data to enable easier augmentation!");

        //1.: Open file
        FILE*  input_w = fopen((path_out + std::string( "/in.jvdata")).c_str(), "wb");
        FILE* output_w = fopen((path_out + std::string("/out.jvdata")).c_str(), "wb");

        //2.: Write file header
        {
            printf("[INFO] Writing input header\n");
            char signature[] = { "JVDATA" };
            uint16_t version = AI_VERSION;
            HEADER_V1 header(
                type_hash<T>(),
                size.x, size.y, (uint32_t)channels,
                order,
                Image::DATA_FORMAT(distr, range)
            );

            fwrite(+signature, sizeof(signature) - 1, 1, input_w); //Signature:         6*uint8_t
            fwrite(&version  , sizeof(version)      , 1, input_w); //Version:             uint16_t
            header.serialize(input_w);
        }

        {
            printf("[INFO] Writing output header\n");
            char signature[] = { "JVDATA" };
            uint16_t version = AI_VERSION;
            HEADER_V1 header(
                type_hash<T>(),
                10, 1, 1,
                order,
                Image::DATA_FORMAT(Image::DISTRIBUTION::UNIFORM, 1.f)
            );

            fwrite(+signature, sizeof(signature) - 1, 1, output_w); //Signature:         6*uint8_t
            fwrite(&version  , sizeof(version)      , 1, output_w); //Version:             uint16_t
            header.serialize(output_w);
        }

        //3.: Open up dataset files
        printf("[INFO] Trying to open MNIST dataset ... ");
        FILE*  input1_r = fopen((path_in + std::string( "/t10k-images.idx3-ubyte")).c_str(), "rb");
        FILE*  input2_r = fopen((path_in + std::string("/train-images.idx3-ubyte")).c_str(), "rb");
        FILE* output1_r = fopen((path_in + std::string( "/t10k-labels.idx1-ubyte")).c_str(), "rb");
        FILE* output2_r = fopen((path_in + std::string("/train-labels.idx1-ubyte")).c_str(), "rb");
        printf("Success\n");

        //4: Parse headers
        //4.1: Signature
        uint32_t tmp;
        fread(&tmp, 4, 1,  input1_r); assert(tmp == 0x3080000);
        fread(&tmp, 4, 1,  input2_r); assert(tmp == 0x3080000);
        fread(&tmp, 4, 1, output1_r); assert(tmp == 0x1080000);
        fread(&tmp, 4, 1, output2_r); assert(tmp == 0x1080000);

        //4.2: #Samples
        uint32_t samples_test, samples_train;
        fread(& samples_test, 4, 1, input1_r);
        fread(&samples_train, 4, 1, input2_r);
        fread(&tmp, 4, 1, output1_r); assert(tmp == samples_test);
        fread(&tmp, 4, 1, output2_r); assert(tmp == samples_train);
         samples_test = reverse_bytes_uint( samples_test);
         samples_train = reverse_bytes_uint(samples_train);

        //4.3: Sample shapes
        uint32_t height, width;
        fread(&height, 4, 1, input1_r);
        fread(& width, 4, 1, input1_r);
        fread(&tmp, 4, 1, input2_r); assert(tmp == height);
        fread(&tmp, 4, 1, input2_r); assert(tmp == width);
        height = reverse_bytes_uint(height);
         width = reverse_bytes_uint( width);

        //5.: Write data
        uint8_t buf1[28 * 28];
        T* buf2 = (T*)malloc(sizeof(T) * std::max<uint32_t>(size.x * size.y * (uint32_t)channels, 10u));
        
        //5.1: Input
        for (uint32_t i = 0; i != samples_test; i++) {
            fread(buf1, sizeof(uint8_t), 28 * 28, input1_r);

            //1.: #Channel, channel order and data_type
            if constexpr (channels == Image::CHANNELS::RGB)
                Image::grayToRGB<uint8_t, T, order>(buf1, buf2, 28 * 28);
            else if constexpr (channels == Image::CHANNELS::RGBA)
                Image::grayToRGBA<uint8_t, T, order>(buf1, buf2, 28 * 28, (T)1, (T)1, (T)1);
            else if constexpr (channels == Image::CHANNELS::GRAY)
                std::copy<uint8_t*, T*>(std::begin(buf1), std::end(buf1), buf2);

            //2.: Format
            Image::remap_format<T>(buf2, Image_Shape(size.x, size.y, (uint32_t)channels), Image::DATA_FORMAT(Image::DISTRIBUTION::UNIFORM, 256.f), Image::DATA_FORMAT(distr, range), order);

            //3.: Resize
            Image::resize<T, order>(buf2, Image_Shape(28u, 28u, (uint32_t)channels), size);

            //4.: Write
            fwrite(buf2, sizeof(T), size.x * size.y * (uint32_t)channels, input_w);
        }

        for (uint32_t i = 0; i != samples_train; i++) {
            fread(buf1, sizeof(uint8_t), 28 * 28, input2_r);

            //1.: #Channel, channel order and data_type
            if constexpr (channels == Image::CHANNELS::RGB)
                Image::grayToRGB<uint8_t, T, order>(buf1, buf2, 28 * 28);
            else if constexpr (channels == Image::CHANNELS::RGBA)
                Image::grayToRGBA<uint8_t, T, order>(buf1, buf2, 28 * 28, (T)1, (T)1, (T)1);
            else if constexpr (channels == Image::CHANNELS::GRAY)
                std::copy<uint8_t*, T*>(std::begin(buf1), std::end(buf1), buf2);

            //2.: Format
            Image::remap_format<T>(buf2, Image_Shape(size.x, size.y, (uint32_t)channels), Image::DATA_FORMAT(Image::DISTRIBUTION::UNIFORM, 256.f), Image::DATA_FORMAT(distr, range), order);

            //3.: Resize
            Image::resize<T, order>(buf2, Image_Shape(28u, 28u, (uint32_t)channels), size);

            //4.: Write
            fwrite(buf2, sizeof(T), size.x * size.y * (uint32_t)channels, input_w);
        }
        fclose(input_w);

        //5.2: Output
        uint8_t label;
        for (uint32_t i = 0; i != samples_test; i++) {
            fread(&label, 1, 1, output1_r);
            T* dat = CSV::vec_to_img<uint8_t, T, Image::CHANNEL_ORDER::CHANNELS_LAST>(&label, 1u, 10u);

            fwrite(dat, sizeof(T), 10, output_w);
            free(dat);
        }

        for (uint32_t i = 0; i != samples_train; i++) {
            fread(&label, 1, 1, output2_r);
            T* dat = CSV::vec_to_img<uint8_t, T, Image::CHANNEL_ORDER::CHANNELS_LAST>(&label, 1u, 10u);

            fwrite(dat, sizeof(T), 10, output_w);
            free(dat);
        }
        fclose(output_w);
    }
}


//=======================================================
//==================|RENDERING|=====================
//=======================================================
#include <ft2build.h>
#include FT_FREETYPE_H

//See  https://learnopengl.com/In-Practice/Text-Rendering
namespace TextRenderer {
    struct Character {
        uint32_t           TextureID;  // ID handle of the glyph texture
        Offset2D<uint32_t> Size;       // Size of glyph
        Offset2D<uint32_t> Bearing;    // Offset from baseline to left/top of glyph
        uint32_t           Advance;    // Offset to advance to next glyph
    }; 
    Character characters[128];

    /*
        Initializes the FreeType library used to render text. This has to be called before the "RenderText" function is called!
    */
    void initFreeType(char* font, uint32_t size) {
        //0.: Initialize library
        FT_Library ft;
        if (FT_Init_FreeType(&ft))
        {
            fprintf(stderr, "[ERROR] FREETYPE: Could not init FreeType Library\n");
            return;
        }

        //1.: Load font
        FT_Face face;
        if (FT_New_Face(ft, font, 0, &face))
        {
            fprintf(stderr, "[ERROR] FREETYPE: Failed to load font\n");
            return;
        }

        //2.: Set size
        FT_Set_Pixel_Sizes(face, 0, size);

        //3.: Generate textures
        uint8_t* buf = nullptr;                                                //Buffer used to convert grayscale to RGB
        uint32_t allocated = 0;                                                //Number of uint8_t that buf can hold

        GLuint texture[128];
        glGenTextures(128, +texture);

        for (uint8_t c = 0; c < 128; c++)
        {
            //Load character glyph 
            if (FT_Load_Char(face, c, FT_LOAD_RENDER))
            {
                fprintf(stderr, "[ERROR] FREETYTPE: Failed to load Glyph\n");
                continue;
            }

            //Convert to RGB
            uint32_t x = face->glyph->bitmap.width;
            uint32_t y = face->glyph->bitmap.rows;
            if (x * y * 4u > allocated) {                                      //Too little space was allocated. 
                free(buf);                                                     //Deallocate last buffer and ...
                buf = (uint8_t*)malloc(sizeof(uint8_t) * x * y * 4u);          //Allocate a new buffer, that is big enough
                allocated = x * y * 4u;
            }

            Image::grayToRGBA<uint8_t, uint8_t, Image::CHANNEL_ORDER::CHANNELS_LAST>(face->glyph->bitmap.buffer, buf, x * y, 255, 255, 255);
            
                //if (c > 64) {
                //    BUGP("K");
                //    ARR_PRINT<uint8_t>(face->glyph->bitmap.buffer, x, y);
                //    Image::show<uint8_t, false>(face->glyph->bitmap.buffer, Image_Shape(x, y, 1u), Image::CHANNEL_ORDER::CHANNELS_LAST);
                //    Image::show<uint8_t, false>(buf, Image_Shape(x, y, 4u), Image::CHANNEL_ORDER::CHANNELS_LAST);
                //}

            //Load into texture
            glBindTexture(GL_TEXTURE_2D, texture[(uint32_t)c]);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA8,
                x,
                y,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                buf
            );

            // set texture options
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            // now store character for later use
            Character character = {
                texture[(uint32_t)c],
                Offset2D<uint32_t>(x, y),
                Offset2D<uint32_t>(face->glyph->bitmap_left, face->glyph->bitmap_top),
                (uint32_t)face->glyph->advance.x
            };
            characters[(uint32_t)c] =  character;
        }
    }

    /*
        Render a text with specific scale to specific position. Does unbind GL_TEXTURE_2D and the vertex array

        @param text: Pointer to the text to render
        @param vbo: Vertex buffer of window  
        @param x: x-Coordinate of where to render the text
        @param y: y-Coordinate of where to render the text
        @param scale: The scale of the text
    */
    void renderText(char* text, float x, float y, float scale = 1.f)
    {
        // iterate through all characters
        char c;
        while ((c = *text++) != '\0') {
            Character ch = characters[(uint32_t)c];

            //PRINT_VAR(ch.Bearing.y); BUGP("\n");

            float xpos = x + ch.Bearing.x * scale;
            float ypos = y - ch.Bearing.y * scale;

            float w = ch.Size.x * scale;
            float h = ch.Size.y * scale;

            // render glyph texture over quad
            glBindTexture(GL_TEXTURE_2D, ch.TextureID);
            glBegin(GL_QUADS);
            glTexCoord2f(0.f, 0.f); glVertex2f(xpos    , ypos);
            glTexCoord2f(0.f, 1.f); glVertex2f(xpos    , ypos + h);
            glTexCoord2f(1.f, 1.f); glVertex2f(xpos + w, ypos + h);
            glTexCoord2f(1.f, 0.f); glVertex2f(xpos + w, ypos);
            glEnd();

            // now advance cursors for next glyph (note that advance is number of 1/64 pixels)
            x += (ch.Advance >> 6) * scale; // bitshift by 6 to get value in pixels (2^6 = 64)
        }
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

#include <numeric>
#include <cmath>
namespace Plotter {
    /*
        Plots the graphs in "data". It contains multiple datalines which each contain the y-values for the x-values 1,2,3,...
        Unbind GL_TEXTURE_2D.

        @param data         : Vector of lines
        @param x_min        : Smallest x-Coordinate of plot on screen
        @param x_max        : Biggest  x-Coordinate of plot on screen
        @param y_min        : Smallest y-Coordinate of plot on screen
        @param y_max        : Biggest  y-Coordinate of plot on screen
        @param padding      : The number of pixels on the screen to extend axis
        @param label        : If true, label the axis
        @param data_min     : The minimum value of the data (the lowest  data value obtainable)
        @param data_max     : The maximum value of the data (the highest data value obtainable)
        @param n_data_points: The maximum number of data points in one data line
    */
    template<typename T, bool logaritmic = false>
    void renderPlot(const std::vector<std::vector<T>> data, float x_min, float x_max, float y_min, float y_max, float padding, bool label, T data_min, T data_max, uint32_t n_data_points) {
        //0.: Check arguments
        for (uint32_t ind = 0; ind != data.size(); ind++)
            assert(data[ind].size() <= n_data_points);
        //printf("%g %g %g %g %g %g\n", x_min, x_max, y_min, y_max, data_min, data_max); fflush(stdout);
        assert((x_min <x_max) && (y_min < y_max) && (data_min <= data_max));
        assert(padding < y_max - y_min);
        assert(!label);                                      //Labeling not implemented yet

        if (data_max - data_min <= std::numeric_limits<T>::epsilon()) { //Ensure that min and max are not the same
            data_min -= (T)1;
            data_max += (T)1;
        }

        //1.: Plot axis
        glBindTexture(GL_TEXTURE_2D, 0); //Unbind texture
        glColor3f(0.0f, 0.0f, 0.0f);     //Color black
      
        glBegin(GL_LINES);
        glVertex2i(x_min, y_max - padding);  //x-Axis
        glVertex2i(x_max, y_max - padding);  //x-Axis
        glVertex2i(x_max - padding, y_max - 2 * padding);  //Arrow part 1
        glVertex2i(x_max          , y_max -     padding);  //Arrow part 1
        glVertex2i(x_max - padding, y_max              );  //Arrow part 2
        glVertex2i(x_max          , y_max -     padding);  //Arrow part 2

        glVertex2i(x_min + padding, y_min);  //y-Axis
        glVertex2i(x_min + padding, y_max);  //y-Axis
        glVertex2i(x_min              , y_min + padding);  //Arrow part 1
        glVertex2i(x_min +     padding, y_min          );  //Arrow part 1
        glVertex2i(x_min + 2 * padding, y_min + padding);  //Arrow part 2
        glVertex2i(x_min +     padding, y_min          );  //Arrow part 2
        glEnd();

        //2.: Plot data lines
        uint32_t n_data_lines = data.size();
        for (uint32_t data_line_ind = 0; data_line_ind != n_data_lines; data_line_ind++) {
            //Set color using hsv to rgb approximation
            const     double h = 6. * (float)data_line_ind / (float)n_data_lines;
            constexpr double s = 0.9;
            constexpr double v = 0.9;

            uint32_t region = h;
            double   remainder = h - region;

            uint8_t p = v * (1. - (s                    ));
            uint8_t q = v * (1. - (s * (      remainder)));
            uint8_t t = v * (1. - (s * (1.0 - remainder)));

            switch (region) {
            case 0:
                glColor3f(v, t, p);
                break;  
            case 1:     
                glColor3f(q, v, p);
                break;  
            case 2:     
                glColor3f(p, v, t);
                break;  
            case 3:     
                glColor3f(p, q, v);
                break;  
            case 4:     
                glColor3f(t, p, v);
                break;  
            default:    
                glColor3f(v, p, q);
                break;
            }

            //Get data line            
            const std::vector<T> data_line = data[data_line_ind];
            
            //Draw data line
            float y; //in [0,1]
            glBegin(GL_LINE_STRIP);
            for (uint32_t data_ind = 0; data_ind != data_line.size(); data_ind++) {
                if constexpr (logaritmic)
                    y = (std::log((float)data_line[data_ind]) - std::log((float)data_min)) / (std::log((float)data_max) - std::log((float)data_min));
                else
                    y = (float)(data_line[data_ind] - data_min) / (float)(data_max - data_min);

                glVertex2f(x_min + padding + (x_max - x_min - padding) * (float)data_ind / (float)n_data_points, y_max - padding - (y_max - y_min - padding) * y);
            }
            glEnd();
        }
    }
}



#if 0
//=============================================
//==================|Main|=====================
//=============================================

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
