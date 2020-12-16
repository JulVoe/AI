#pragma once
#include <inttypes.h>
#include <cstdio>

#include <memory>
#include <typeinfo>

#include "util.cpp"

#include "cuda_runtime.h"



template<typename T>
T roundUpMult(T numToRound, T multiple)               //Returns first number >=numberToRound divvisible by multiple. multiple has to be positive
{
    assert(multiple);
    return ((numToRound + multiple - 1) / multiple);
}

template<typename T>
T roundUpMultPow2(T numToRound, T multiple)       //Returns first number >=numberToRound divvisible by multiple. multiple has to be a power of two
{
    assert(multiple && ((multiple & (multiple - 1)) == 0));
    return (numToRound + multiple - 1) & -multiple;
}

struct MemoryRequirement {
public:
    uint64_t num_bytes;   //Lenght of memory in bytes
    uint32_t alignment;   //The minimum number of bytes the pointer needs to be aligned to. Has to be power of two

    MemoryRequirement(uint64_t num_bytes, uint32_t alignment) :
        num_bytes(num_bytes),
        alignment(alignment)
    {
        assert((alignment & (alignment - 1)) == 0);
    }

    MemoryRequirement operator+(MemoryRequirement mr2) {
        MemoryRequirement ret(max<uint64_t>(alignment, mr2.alignment), 0u);

        ret.num_bytes = roundUpMultPow2<uint64_t>(num_bytes, mr2.alignment) + mr2.num_bytes;

        return ret;
    }

    void operator+=(MemoryRequirement mr2) {
        MemoryRequirement sum = operator+(mr2);

        num_bytes = sum.num_bytes;
        alignment = sum.alignment;
    }

    void serialize(FILE* file) {
        fwrite(&num_bytes, sizeof(num_bytes), 1, file);
        fwrite(&alignment, sizeof(alignment), 1, file);
    }
    static MemoryRequirement deserialize(FILE* file) {
        MemoryRequirement mr;
        fread(&mr.num_bytes, sizeof(mr.num_bytes), 1, file);
        fread(&mr.alignment, sizeof(mr.alignment), 1, file);
    }
};

inline bool is_aligned(const void* ptr, uint32_t alignment) noexcept {
    static_assert(sizeof(uintmax_t) >= sizeof(void*), "[ERROR] No suitable integer type for conversion from pointer type");
    return !(reinterpret_cast<std::uintptr_t>(ptr) % alignment);
}
inline void* align_pointer_unsafe(void* ptr, uint32_t alignment) {
    //See https://github.com/KabukiStarship/KabukiToolkit/wiki/Fastest-Method-to-Align-Pointers
    return (reinterpret_cast<uintptr_t>(ptr) - 1u + alignment) & -alignment;
}

void cudaMallocAligned(void* out, MemoryRequirement mr) {
    //Accrding to https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses 5.3.2.1.1, cudaMalloc has alignemt of 256bytes

    if (mr.alignment <= 256) {
        cudaMalloc(&out, mr.num_bytes);
    }
    else {
        fprintf(stderr, "[WARNING] Very high alignment of %u bytes requested!", mr.alignment);
        cudaMalloc(&out, mr.num_bytes + mr.alignment - 1);
        
        void* align(mr.alignment, mr.num_bytes, out, mr.num_bytes + mr.alignment - 1);
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