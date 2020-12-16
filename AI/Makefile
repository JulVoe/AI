IN_CU   := "C:/Users/julia/source/repos/AI/AI/kernel.cu"
IN_CXX  := "C:/Users/julia/source/repos/AI/AI/kernel.cpp"
DIR_PATH = "C:/Users/julia/source/repos/AI/AI/"
OUT      = x64/Release/AI_make.exe

CLANG            := "C:/Program Files/LLVM/bin/clang++.exe"
CLANGFLAGS        = -fopenmp -Wall -std=c++17 -march=native -ferror-limit=100 -O3
CLANGCL          := "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/Llvm/bin/clang-cl.exe"
CLANGCLFLAGS      = /openmp /std:c++17 /W3 -march=native /Ox -Wall -ferror-limit=100 /MD
DISABLED_WARNINGS = -Wno-disabled-macro-expansion -Wno-reserved-id-macro -Wno-c++98-compat-pedantic -Wno-c++98-compat -Wno-old-style-cast -Wno-documentation -Wno-documentation-unknown-command -Wno-zero-as-null-pointer-constant -Wno-nested-anon-types -Wno-gnu-anonymous-struct
CLANG_CUDAFLAGS   = --cuda-gpu-arch=sm_75 --cuda-path="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2"

GCC      = "C:/msys64/usr/bin/g++.exe"
GCCFLAGS = -fopenmp -Wall -Wextra -std=c++17 -march=native -fmax-errors=100 -O3

MVSCFLAGS = /EHsc /nologo /Ox /FS /Zi /MT /openmp:experimental /std:c++14

NVCC          := "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin/nvcc.exe"
CUDAFLAGS      = -gencode=arch=compute_75,code=\"sm_75,compute_75\" --use-local-env --keep-dir x64/Release --machine 64 -cudart static -O3 --std=c++14 -expt-relaxed-constexpr -expt-extended-lambda
ASSEMBLERFLAGS = -opt-fp-atomics

MAKROS   = -DWIN32 -DWIN64 -DNDEBUG -D_CONSOLE -D_MBCS -DCUSTOM -DHELL_NO_GET_OUT_MSVC_YOU_DUMB_LITTLE_PIECE_OF_SHIT
CULIBS   = -lcublas -lcudart_static
CXXLIBS  = -lkernel32 -luser32 -lgdi32 -lwinspool -lcomdlg32 -ladvapi32 -lshell32 -lole32 -loleaut32 -luuid -lodbc32 -lodbccp32 -llibomp -lMagick++ -lMagickWand -lMagickCore -lz -ldl -lpng -ljpeg -lX11  # -lpthread -llibiomp5md
LIBDIRS := -L"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/bin" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64" -L"C:\Program Files\LLVM\lib" -L"C:\Program Files\ImageMagick-7.0.10-Q16-HDRI\lib"
INCDIRS := -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2/include" -I"C:\Program Files\ImageMagick-7.0.10-Q16-HDRI\include"
MSVCDEP := -I"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\include" -I"C:\Program Files (x86)\Windows Kits\10\Include\10.0.18362.0\ucrt" -L"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64" -L"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.26.28801\lib\x64" -L"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64"
																												 
#FUTURE = (-run) (-pg + line info) (-v) (-noprof) (-default-stream=per-thread) (-flto)
#MISSING = --ptxas-path + openmp lib + pthread

normal:
	$(NVCC) $(CUDAFLAGS) -Xcompiler "$(MVSCFLAGS)" $(MAKROS) $(INCDIRS) $(MSVCDEP) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -Xptxas "$(ASSEMBLERFLAGS)" --link -o $(OUT) $(IN_CXX) $(IN_CU)

clang_backend:
	$(NVCC) $(CUDAFLAGS) -ccbin $(CLANGCL) -Xcompiler "$(CLANGCLFLAGS) $(DISABLED_WARNINGS) $(MAKROS)" $(MAKROS) $(INCDIRS) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -Xptxas "$(ASSEMBLERFLAGS)" --link -o $(OUT) $(IN_CXX) $(IN_CU)

gcc_backend:
	$(NVCC) $(CUDAFLAGS) -ccbin $(GCC) -Xcompiler "$(GCCFLAGS)" $(MAKROS) $(INCDIRS) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -Xptxas "$(ASSEMBLERFLAGS)" --link -o $(OUT) $(IN_CXX) $(IN_CU)

clang_frontend:
	$(CLANG) -x cuda $(CLANGFLAGS) $(CLANG_CUDAFLAGS) $(MAKROS) -Xcuda-ptxas "$(ASSEMBLERFLAGS)" $(INCDIRS) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -o $(OUT) $(IN_CXX) $(IN_CU)

device_only:
	$(NVCC) $(IN_CU) -o dev_code_.obj $(CUDAFLAGS) -Xcompiler "$(MVSCFLAGS)" $(MAKROS) $(INCDIRS) $(MSVCDEP) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -Xptxas "$(ASSEMBLERFLAGS)" --device-c
	$(NVCC) dev_code_.obj -o dev_code.obj $(CUDAFLAGS) -Xcompiler "$(MVSCFLAGS)" $(MAKROS) $(INCDIRS) $(MSVCDEP) $(LIBDIRS) $(CXXLIBS) $(CULIBS) -Xptxas "$(ASSEMBLERFLAGS)" -dlink

clang_hybrid: dev_code.obj
	$(CLANG) $(CLANGFLAGS) $(MAKROS) $(INCDIRS) -c -o host_code.obj $(IN_CXX)
	$(CLANG) host_code.obj dev_code.obj dev_code_.obj $(LIBDIRS) $(CXXLIBS) $(CULIBS) -o $(OUT)

gcc_hybrid: dev_code.obj
	$(GCC) $(GCCFLAGS) $(MAKROS) $(INCDIRS) -c -o host_code.obj $(IN_CXX)
	$(GCC) host_code.obj dev_code.obj dev_code_.obj $(LIBDIRS) $(CXXLIBS) $(CULIBS) -o $(OUT)