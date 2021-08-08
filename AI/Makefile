#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Compilations
#nvcc --gpu-architecture=sm_50 --device-c a.cu b.cu
#nvcc --gpu-architecture=sm_50 --device-link a.o b.o --output-file link.o
#nvcc --lib --output-file libgpu.a a.o b.o link.o
#g++ host.o --library=gpu --library-path=<path> \
#    --library=cudadevrt --library=cudart

#Old
#IN_CU   := "C:/Users/julia/source/repos/AI/AI/kernel.cu"
#IN_CXX  := "C:/Users/julia/source/repos/AI/AI/kernel.cpp"
#DIR_PATH = "C:/Users/julia/source/repos/AI/AI/"
#OUT      = x64/Release/AI_make.exe
#
#DISABLED_WARNINGS = -Wno-disabled-macro-expansion -Wno-reserved-id-macro -Wno-c++98-compat-pedantic -Wno-c++98-compat -Wno-old-style-cast -Wno-documentation -Wno-documentation-unknown-command -Wno-zero-as-null-pointer-constant -Wno-nested-anon-types -Wno-gnu-anonymous-struct
#MAKROS   = -DWIN32 -DWIN64 -DNDEBUG -D_CONSOLE -D_MBCS -DCUSTOM -DHELL_NO_GET_OUT_MSVC_YOU_DUMB_LITTLE_PIECE_OF_SHIT
#
#normal:
#	$(NVCC) $(CUDAFLAGS) -Xcompiler "$(MVSCFLAGS)" $(MAKROS) $(INCDIRS) $(MSVCDEP) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -Xptxas "$(ASSEMBLERFLAGS)" --link -o $(OUT) $(IN_CXX) $(IN_CU)
#
#clang_backend:
#	$(NVCC) $(CUDAFLAGS) -ccbin $(CLANGCL) -Xcompiler "$(CLANGCLFLAGS) $(DISABLED_WARNINGS) $(MAKROS)" $(MAKROS) $(INCDIRS) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -Xptxas "$(ASSEMBLERFLAGS)" --link -o $(OUT) $(IN_CXX) $(IN_CU)
#
#gcc_backend:
#	$(NVCC) $(CUDAFLAGS) -ccbin $(GCC) -Xcompiler "$(GCCFLAGS)" $(MAKROS) $(INCDIRS) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -Xptxas "$(ASSEMBLERFLAGS)" --link -o $(OUT) $(IN_CXX) $(IN_CU)
#
#clang_frontend:
#	$(CLANG) -x cuda $(CLANGFLAGS) $(CLANG_CUDAFLAGS) $(MAKROS) -Xcuda-ptxas "$(ASSEMBLERFLAGS)" $(INCDIRS) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -o $(OUT) $(IN_CXX) $(IN_CU)
#
#device_only:
#	$(NVCC) $(IN_CU) -o dev_code_.obj $(CUDAFLAGS) -Xcompiler "$(MVSCFLAGS)" $(MAKROS) $(INCDIRS) $(MSVCDEP) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -Xptxas "$(ASSEMBLERFLAGS)" --device-c
#	$(NVCC) dev_code_.obj -o dev_code.obj $(CUDAFLAGS) -Xcompiler "$(MVSCFLAGS)" $(MAKROS) $(INCDIRS) $(MSVCDEP) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -Xptxas "$(ASSEMBLERFLAGS)" -dlink
#
#clang_hybrid: dev_code.obj
#	$(CLANG) $(CLANGFLAGS) $(MAKROS) $(INCDIRS) -c -o host_code.obj $(IN_CXX)
#	$(CLANG) host_code.obj dev_code.obj dev_code_.obj $(LIBDIRS) $(CXXLIBS) $(CULIBS) -o $(OUT)
#
#gcc_hybrid: dev_code.obj
#	$(GCC) $(GCCFLAGS) $(MAKROS) $(INCDIRS) -c -o host_code.obj $(IN_CXX)
#	$(GCC) host_code.obj dev_code.obj dev_code_.obj $(LIBDIRS) $(CXXLIBS) $(CULIBS) -o $(OUT)

#Future																												 
#FUTURE = (-run) (-pg + line info) (-v) (-noprof) (-default-stream=per-thread) (-flto)
#MISSING = --ptxas-path + openmp lib + pthread
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Global
IN_MAIN := "C:/Users/julia/source/repos/AI/AI/Inference.cpp"
OUT     := x64/Release/AI_make.exe

#NVCC
NVCC           := "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin/nvcc.exe"
NVCC_FLAGS      = -gencode=arch=compute_75,code=\"sm_75,compute_75\" --use-local-env --keep-dir x64/Release --machine 64 -cudart static -O3 --std=c++17 -expt-relaxed-constexpr -expt-extended-lambda
ASSEMBLER_FLAGS = -opt-fp-atomics

#MVSC
MVSCDEP  := -I"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\include" -I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.18362.0\um" -I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.18362.0\ucrt" -I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.18362.0\shared" -L"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64" -L"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\lib\x64" -L"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64"
MVSCFLAGS = /EHsc /nologo /FS /Zi /MT /openmp:experimental /std:c++17 /O2

#CLANG-CL
CLANGCL       := "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/Llvm/bin/clang-cl.exe"
CLANGCLFLAGS   = /openmp /std:c++17 /W3 -march=native /Ox -Wall -ferror-limit=100 /MD                                  #TODO: Compare to MVSCFLAGS

#CLANG (Host)
CLANG      := "C:/Program Files/LLVM/bin/clang++.exe"
CLANGFLAGS  = -fopenmp -Wall -std=c++17 -march=native -ferror-limit=100 -O3

#CLANG (Device)
CLANG_CUDAFLAGS = --cuda-gpu-arch=sm_75 --cuda-path="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"

#GCC
GCC      = "C:/msys64/usr/bin/g++.exe"
GCCFLAGS = -fopenmp -Wall -Wextra -std=c++17 -march=native -fmax-errors=100 -O3

#DEPENDNECIES
MAKROS   = -DDEBUG #-DEXPERIMENTAL_FILESYSTEM
CULIBS   = -lcudart_static -lcublas -lcudnn
CXXLIBS  = -lzlib -llibpng16 -ljpeg -lkernel32 -luser32 -lgdi32 -lopengl32 -lglu32 -lglew32 -lglfw3dll -lpsapi -lwinspool -lcomdlg32 -ladvapi32 -lshell32 -lole32 -loleaut32 -luuid -lodbc32 -lodbccp32 -lfreetype
LIBDIRS := -L"D:\Librarys\GLFW\lib" -L"D:\Librarys\glew-2.1.0\lib\Release\x64" -L"D:\Librarys\VS-NuGet\lib" -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64" -L"D:\Librarys\freetype-2.10.3\objs" -L"D:/Librarys/cudnn-11.2-windows-x64-v8.1.1.33/lib/x64"
INCDIRS := -I"D:\Librarys\CImg-2.9.2_pre070420" -I"D:\Librarys\VS-NuGet\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include" -I"D:\Librarys\GLFW\include" -I"D:\Librarys\glew-2.1.0\include" -I"D:\Librarys\freetype-2.10.3\include" -I"D:/Librarys/cudnn-11.2-windows-x64-v8.1.1.33/include"



nvcc_clang:
	$(NVCC) $(NVCC_FLAGS) $(INCDIRS) $(LIBDIRS) $(CXXLIBS) $(CULIBS) $(MAKROS) -Xptxas "$(ASSEMBLERFLAGS)" -ccbin $(CLANG) -Xcompiler "TODO" -x cu -o $(OUT) $(IN_MAIN)

	   

nvcc_clangcl:
	$(NVCC) $(NVCC_FLAGS) $(INCDIRS) $(LIBDIRS) $(CXXLIBS) $(CULIBS) $(MAKROS) -Xptxas "$(ASSEMBLERFLAGS)" -ccbin $(CLANGCL) -Xcompiler "TODO" -x cu -o $(OUT) $(IN_MAIN)

nvcc_gcc:


nvcc_vc:
	$(NVCC) $(NVCC_FLAGS) $(INCDIRS) $(MVSCDEP) $(LIBDIRS) $(CXXLIBS) $(CULIBS) $(MAKROS) -Xptxas "$(ASSEMBLERFLAGS)" -Xcompiler "$(MVSCFLAGS)" -G -x cu -o $(OUT) $(IN_MAIN)

only_device:


gcc_host:


clang_host:


clangcl_host:


vc_host:


clang_unified: