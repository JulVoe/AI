#include <cstdio>
#include <cstring>

#include <type_traits>
#include <thread>
#include <memory>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "util.cpp"

/*Bugs:
 - DatasetHandler deadlock?
 - CSV read sta==sto assert error
 - Can't load 6-9.png and 7-6.png
 - Wrong "out" on gpu with two 1's
 - Softmax with batchSize>1 fails




 - Batch size 8 error. Why? Scheduler does not have it. Scheduler seems to be working with higher batch size while this just produces trash
  - Network alown seems to work with higher batchsize without a problem (checked internals). Scheduler exibits plausible behaviour (tested output accuracy). Inference just returns trash, no convergence at all.
*/


namespace Drawer_internal {
    template<typename T>
    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        if (action == GLFW_PRESS) {
            void** user_data = (void**)glfwGetWindowUserPointer(window);
            Image_Shape shape = *((Image_Shape*)user_data[2]);
            int32_t* brush_size = (int32_t*)user_data[3];
            float* opacity = (float*)user_data[4];

            switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case 93: //+
                *brush_size += 2 * (*brush_size >= 0) - 1;
                break;
            case 47: //-
                *brush_size -= 2 * (*brush_size >= 0) - 1;
                break;
            case GLFW_KEY_I:
                *brush_size = -*brush_size;
                break;
            case GLFW_KEY_O:
                *opacity = bound(*opacity + 0.1f, 0.f, 1.f);
                break;
            case GLFW_KEY_P:
                *opacity = bound(*opacity - 0.1f, 0.f, 1.f);
                break;
            }
        }
    }

    template<typename T>
    void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
        void** user_data = (void**)glfwGetWindowUserPointer(window);
        T* channel_first = (T*)user_data[0];
        T* channel_last = (T*)user_data[1];
        Image_Shape shape = *((Image_Shape*)user_data[2]);
        int32_t* brush_size = (int32_t*)user_data[3];
        float* opacity = (float*)user_data[4];
        bool* mouse_presses = (bool*)user_data[5];
        float* scale = (float*)user_data[6];

        if (*mouse_presses) {
            for (uint32_t c = 0; c != shape.z; c++) {
                for (uint32_t x = 0; x != shape.x; x++) {
                    for (uint32_t y = 0; y != shape.y; y++) {
                        double dq = (x - (xpos / *scale)) * (x - (xpos / *scale)) + (y - (ypos / *scale)) * (y - (ypos / *scale));
                        dq /= (double)(*brush_size * *brush_size);
                        dq = (1. - dq) * (1. - dq) * (1. - dq);
                        if (dq > 0) {
                            channel_first[c * shape.x * shape.y + y * shape.x + x] *= (1.f - *opacity * dq);
                            channel_first[c * shape.x * shape.y + y * shape.x + x] += *opacity * dq * sgn(*brush_size);

                            channel_last[c + y * shape.x * shape.z + x * shape.z] *= (1.f - *opacity * dq);
                            channel_last[c + y * shape.x * shape.z + x * shape.z] += *opacity * dq * sgn(*brush_size);
                        }
                    }
                }
            }
        }
    }

    void click_callback(GLFWwindow* window, int button, int action, int mods) {
        void** user_data = (void**)glfwGetWindowUserPointer(window);
        bool* mouse_pressed = (bool*)user_data[5];


        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
            *mouse_pressed = true;
        if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
            *mouse_pressed = false;
    }
}

template<typename T>
class ImageDrawer {
private:
    T* channel_first;
    T* channel_last; 
    Image_Shape shape;

    int32_t brush_size;
    float opacity;

    bool mouse_pressed;

    bool stop_signal;

#ifdef WIN32
    static constexpr char* FONT_PATH = "C:/Windows/Fonts/Arial.ttf";
#else
    static constexpr char* FONT_PATH = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf";
#endif
    void window_thread(float scale) {
        //0.: Compute window size
        Offset2D<uint32_t> windows_size(shape.x * scale, shape.y * scale);

        //1.: Create Window and initialize GLFW
        GLFWwindow* window;
        if (!glfwInit())
            return;
        window = glfwCreateWindow(windows_size.x, windows_size.y, "Drawer", NULL, NULL);
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
        void* user_data[] = { (void*)channel_first, (void*)channel_last, (void*)&shape, (void*)&brush_size, (void*)&opacity, (void*)&mouse_pressed, (void*)&scale};
        glfwSetWindowUserPointer(window, (void*)user_data);
        glfwSetKeyCallback(window, Drawer_internal::key_callback<T>);
        glfwSetMouseButtonCallback(window, Drawer_internal::click_callback);
        glfwSetCursorPosCallback(window, Drawer_internal::mouse_callback<T>);

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
        gluOrtho2D(0, windows_size.x, windows_size.y, 0);
        glEnable(GL_TEXTURE_2D);

        //7.: Setting up textures
        GLenum channels;
        GLenum type;
        GLenum format;
        switch (shape.z) {
        case 1:
            channels = GL_RED;
            if constexpr (std::is_same<T, uint8_t>::value) {
                type = GL_UNSIGNED_BYTE;
                format = GL_R8UI;
            }
            if constexpr (std::is_same<T, int8_t>::value) {
                type = GL_BYTE;
                format = GL_R8I;
            }
            if constexpr (std::is_same<T, uint16_t>::value) {
                type = GL_UNSIGNED_SHORT;
                format = GL_R16UI;
            }
            if constexpr (std::is_same<T, int16_t>::value) {
                type = GL_SHORT;
                format = GL_R16I;
            }
            if constexpr (std::is_same<T, uint32_t>::value) {
                type = GL_UNSIGNED_INT;
                format = GL_R32UI;
            }
            if constexpr (std::is_same<T, int32_t>::value) {
                type = GL_INT;
                format = GL_R32I;
            }
            if constexpr (std::is_same<T, float>::value) {
                type = GL_FLOAT;
                format = GL_R32F;
            }
            break;
        case 3:
            channels = GL_RGB;
            if constexpr (std::is_same<T, uint8_t>::value) {
                type = GL_UNSIGNED_BYTE;
                format = GL_RGB8UI;
            }
            if constexpr (std::is_same<T, int8_t>::value) {
                type = GL_BYTE;
                format = GL_RGB8I;
            }
            if constexpr (std::is_same<T, uint16_t>::value) {
                type = GL_UNSIGNED_SHORT;
                format = GL_RGB16UI;
            }
            if constexpr (std::is_same<T, int16_t>::value) {
                type = GL_SHORT;
                format = GL_RGB16I;
            }
            if constexpr (std::is_same<T, uint32_t>::value) {
                type = GL_UNSIGNED_INT;
                format = GL_RGB32UI;
            }
            if constexpr (std::is_same<T, int32_t>::value) {
                type = GL_INT;
                format = GL_RGB32I;
            }
            if constexpr (std::is_same<T, float>::value) {
                type = GL_FLOAT;
                format = GL_RGB32F;
            }
            break;
        case 4:
            channels = GL_RGBA;
            if constexpr (std::is_same<T, uint8_t>::value) {
                type = GL_UNSIGNED_BYTE;
                format = GL_RGBA8UI;
            }
            if constexpr (std::is_same<T, int8_t>::value) {
                type = GL_BYTE;
                format = GL_RGBA8I;
            }
            if constexpr (std::is_same<T, uint16_t>::value) {
                type = GL_UNSIGNED_SHORT;
                format = GL_RGBA16UI;
            }
            if constexpr (std::is_same<T, int16_t>::value) {
                type = GL_SHORT;
                format = GL_RGBA16I;
            }
            if constexpr (std::is_same<T, uint32_t>::value) {
                type = GL_UNSIGNED_INT;
                format = GL_RGBA32UI;
            }
            if constexpr (std::is_same<T, int32_t>::value) {
                type = GL_INT;
                format = GL_RGBA32I;
            }
            if constexpr (std::is_same<T, float>::value) {
                type = GL_FLOAT;
                format = GL_RGBA32F;
            }
            break;
        default:
            fprintf(stderr, "[ERROR] ImageDrawer only supperts images with 1,3, or 4 channels. You supplied %u!", shape.z);
        }

        GLuint tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexStorage2D(GL_TEXTURE_2D, 1, format, shape.x, shape.y);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

        //8.: Display-Loop
        while (!stop_signal && !glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            glBindTexture(GL_TEXTURE_2D, tex);

            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, shape.x, shape.y, channels, type, (void*)channel_last);
            glBegin(GL_QUADS);
            glTexCoord2f(0.f, 0.f); glVertex2i(0, 0);
            glTexCoord2f(0.f, 1.f); glVertex2i(0, windows_size.y);
            glTexCoord2f(1.f, 1.f); glVertex2i(windows_size.x, windows_size.y);
            glTexCoord2f(1.f, 0.f); glVertex2i(windows_size.x, 0);
            glEnd();

            //Print text
            TextRenderer::renderText("Usage: Press \"ESC\" to exit. \"+\" and \"-\" controll the brush size. \"o\" and \"p\" controll the opacity. \"i\" controlls color", 1000, 1280, 0.3f);
            glBindTexture(GL_TEXTURE_2D, tex);

            glfwSwapBuffers(window);
            glfwPollEvents();
        }


        glfwDestroyWindow(window);
        glfwTerminate();
    }

public:
    ImageDrawer(Image_Shape shape) :
        shape(shape),
        brush_size(2),
        opacity(1.f),
        mouse_pressed(false),
        stop_signal(false)
    {
        cudaMallocHost((void**)&channel_first, shape.prod() * sizeof(T));
        cudaMallocHost((void**)&channel_last , shape.prod() * sizeof(T));

        std::memset(channel_first, 0, shape.prod() * sizeof(T));
        std::memset(channel_last , 0, shape.prod() * sizeof(T));
    }

    void start(float scale = 1.f) {
        stop_signal = false;
        std::thread tracker = std::thread(&ImageDrawer<T>::window_thread, this, scale);
        tracker.detach();
    }

    void stop() {
        stop_signal = true;
    }

    template<bool channelFirst>
    void setData(T* data) {
        std::memcpy(channel_first, data, sizeof(T) * shape.prod());
        std::memcpy(channel_last , data, sizeof(T) * shape.prod());
        if constexpr (channelFirst)
            Image::resize<T, Image::CHANNEL_ORDER::CHANNELS_FIRST, Image::CHANNEL_ORDER::CHANNELS_LAST>(data, channel_last, shape, shape.getOffset2D());
        else
            Image::resize<T, Image::CHANNEL_ORDER::CHANNELS_LAST, Image::CHANNEL_ORDER::CHANNELS_FIRST>(data, channel_first, shape, shape.getOffset2D());
    }

    template<bool channelFirst>
    T* getData() {
        if constexpr(channelFirst)
            return channel_first;
        else
            return channel_last;
    }
};





#include "Scheduler.cpp"
#include <string>
#include "util.cpp"

#define P_IN    "C:/Users/julia/source/repos/AI/Datasets/1/Raw/In"
#define P_OUT   "C:/Users/julia/source/repos/AI/Datasets/1/Raw/Out"
#define P_D_IN  "C:/Users/julia/source/repos/AI/Datasets/2/in.jvdata"
#define P_D_OUT "C:/Users/julia/source/repos/AI/Datasets/2/out.jvdata"
#define CKP_DIR "C:/Users/julia/source/repos/AI/CKP"
int main()
{
    if (!glfwInit())
        return;

    cudaStream_t str;
    cudaStreamCreateWithFlags(&str, cudaStreamNonBlocking);

    //8Mb workspace (default for cublas according to debugging with nsight compute, even though documentation says 4mb)
    void* workspace;
    uint64_t workspaceBytes = 8ull * 1024ull * 1024ull;
    cudaMallocAligned(&workspace, MemoryRequirement(workspaceBytes, 256u));
    cudnnSetup (str, workspaceBytes, workspace, true);
    cublasSetup(str, workspaceBytes, workspace, false);

    Random::init_rand();

    //Constants
    using T = float;
    using L = T;
    constexpr uint32_t N = 256;  //Layer Size
    constexpr uint32_t S = 32;   //Input Size
    constexpr uint32_t B  = 8;   //Dataset batch size
    constexpr uint32_t B2 = 8;   //Network batch size
    constexpr float    lRateScale = 1.3;

    Image_Shape in_shape(S, S, 1u);

    //Network
    std::vector<Layer<T>*> layers;
    layers.push_back(new Input_Layer<T>(in_shape));
    layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::RELU, N));
    layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::RELU, N));
    layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::RELU, N));
    layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::RELU, N));
    layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::RELU, N));
    layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::RELU, N));
    //layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::SOFTPLUS, N));
    //layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::SOFTPLUS, Image_Shape(64u, 64u, 3u)));
    //layers.push_back(new Convolution_Layer<T>(ACTIVATION_TYPE::RELU, 10, 3));
    layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::SOFTMAX, 10u));
    
    //NetworkBuilder<T, L> network(layers, 16, B2);
    //network.allocate();
    //network.initialize();
        NetworkBuilder<T, L> network(CKP_DIR"/199.jvcheck", 16, B2);
    
    Layer<T, L>* firstLayer, * lastLayer;
    network.getFirstAndLastLayer(firstLayer, lastLayer);

    //Optimizer
    Optimizer<T>* opt = Optimizer<T>::getOptimizerOfType(OPTIMIZER_TYPE::SGD);
    opt->setNumOptimizables(network.getNumOptimizables());
    opt->allocate();
    opt->initMem();

    //Loss
    Loss<T>* loss = new MSE_Loss<T>();
    loss->setParameters((T*)lastLayer->output, lastLayer->outputShape, B2, (T*)network.getDeltaMem());

    //Dataset
    Offset2D<uint32_t> size(32, 32);
    //generateDatasetFile_Image<float>(P_IN, P_D_IN, size);
    //generateDatasetFile_Raw<float>(P_OUT, P_D_OUT);
        //generateDatasetFile_Raw<float>(P_IN, P_D_IN);
        //generateDatasetFile_Classification<float>(P_OUT, P_D_OUT, 4);
            //convertMNIST<T, Image::CHANNELS::GRAY>("C:/Users/julia/source/repos/AI/Datasets/2/Raw", "C:/Users/julia/source/repos/AI/Datasets/2", size);

    AugmentationInfo2D_IN  agi_in (DATA_FORMAT(DISTRIBUTION::UNIFORM, 1.f), 0, false, 0.0f, 0.1f, 0.1f, 0.f, 0.f, Offset2D<float>(.95f, .95f), in_shape.getOffset2D<int32_t>());              //(DATA_FORMAT(DISTRIBUTION::UNIFORM, 1.f), 2, false, 0.f, 0.05f, 0.2f, 0.2f, 0.2f, Offset2D<float>(0.95f, 0.95f), Offset2D<int32_t>(S, S));
    AugmentationInfo2D_OUT agi_out(DATA_FORMAT(DISTRIBUTION::UNIFORM, 1.f), Offset2D<int32_t>(-1, -1), 0.0f, false);                                                                 //(DATA_FORMAT(DISTRIBUTION::UNIFORM, 1.f), Offset2D<int32_t>(-1, -1), 0.01f, false);

    DatasetHandler<T> handler(P_D_IN, P_D_OUT, 0.95f, 8192);
    handler.set_augmentation(agi_in, agi_out);
    handler.set_batching(B, 8, 8);

    //Scheduler
    Scheduler<T> scheduler(&network, opt, loss, &handler, str);
    scheduler.setNumRuns(0, 1000);
    scheduler.setLossGoal((T)-100000);
    scheduler.setPlateau(0, 0);
    scheduler.setLRSchedule(
        LRScheduleComplete<T>(new Linear_LRSchedule<L>(0.01 * lRateScale, 0.02 * lRateScale), new Cosine_LRSchedule<L>(0.02 * lRateScale, 0.002 * lRateScale), 20),
        LRScheduleComplete<T>(new  Const_LRSchedule<L>(0.01 * lRateScale), new Const_LRSchedule<L>(0.01 * lRateScale), 0),
        LRScheduleComplete<T>(new  Const_LRSchedule<L>(0.01 * lRateScale), new Const_LRSchedule<L>(0.01 * lRateScale), 0)
    );

    //Train
    getchar();

    handler.start_debugWindow();
    scheduler.launch(8, 4, true, true, false, CKP_DIR, 10u);

    //Infere
    getchar();

    ImageDrawer<float> drawer(in_shape);
    drawer.start(10.f);

    cudaGraph_t forward = network.getForwardGraph(str);
    cudaGraphExec_t inference;
    char errorBuf[512];
    cudaGraphNode_t errNode;
    cudaGraphInstantiate(&inference, forward, &errNode, +errorBuf, 512);
    if (errorBuf[0]) {
        fprintf(stderr, "[ERROR] The following error arose during the instantiation of the inference graph: %s", +errorBuf);
        exit(-1);
    }

    T* host_pointer = drawer.getData<true>();
    T* device_mem;
    cudaMalloc((void**)&device_mem, in_shape.prod() * sizeof(T));
    T* output;
    cudaMallocHost((void**)&output, sizeof(T) * lastLayer->outputShape.prod());

    ((Input_Layer<T>*)firstLayer)->setInputPointer(&device_mem, str);





    T* dat = nullptr;
    Image_Shape shape;
    for (uint32_t inf = 0; inf != 100; inf++) {
        char c = getchar();

        if ('1' <= c && c <= '7') {
            std::string file = P_IN "/";
            file += c;
            file += "-1.png";
            char* complete = (char*)file.c_str();

            Image::getPixels<T, Image::CHANNEL_ORDER::CHANNELS_FIRST, Image::CHANNELS::RGB, Image::DISTRIBUTION::UNIFORM, 1>(complete, dat, shape);
            //Image::show<T, true>(dat, shape, Image::CHANNEL_ORDER::CHANNELS_FIRST);
            Image::resize<T, Image::CHANNEL_ORDER::CHANNELS_FIRST>(dat, shape, in_shape.getOffset2D());
            //Image::show<T, true>(dat, in_shape, Image::CHANNEL_ORDER::CHANNELS_FIRST);
            drawer.setData<true>(dat);
        }

        if (c == 'c') {
            Image::show<T, true>(host_pointer, in_shape, Image::CHANNEL_ORDER::CHANNELS_FIRST);
        }


        if (c == 'x') {
            T* in;
            T* out;
            handler.advance<false>(&in, &out);
            dat = (T*)malloc(in_shape.prod() * sizeof(T));
            cudaMemcpyAsync(output, out, sizeof(T) * 10             , cudaMemcpyDeviceToHost, str);
            cudaMemcpyAsync(dat   , in , sizeof(T) * in_shape.prod(), cudaMemcpyDeviceToHost, str);

            cudaStreamSynchronize(str);
            BUGP("Expected:")
            ARR_PRINT(output, lastLayer->outputShape.prod(), 1);
            drawer.setData<true>(dat);
        }
        if (c == 'b')
            break;



        cudaMemcpyAsync(device_mem, host_pointer, in_shape.prod() * sizeof(T), cudaMemcpyHostToDevice, str);
        cudaGraphLaunch(inference, str);
        cudaMemcpyAsync(output, lastLayer->output, sizeof(T) * lastLayer->outputShape.prod(), cudaMemcpyDeviceToHost, str);
        cudaStreamSynchronize(str);

        uint32_t maxInd = 0;
        float maxConf = output[0];
        for (uint32_t i = 1; i != lastLayer->outputShape.prod(); i++) {
            if (output[i] > maxConf) {
                maxConf = output[i];
                maxInd = i;
            }
        }
        printf("Prediction: %d (%f)\n", maxInd, maxConf);
    }


    BUGP("\n\nDone");
    getchar();


    CHECK_CUDA_ERROR();
    CUBLAS_ERROR(cublasDestroy(cublas_handle));
    glfwTerminate();

    return 0;
}

int main2() {
    Image_Shape shape(256u, 256u, 3u);
    ImageDrawer<float> drawer(shape);
    drawer.start();

    getchar();

    drawer.stop();

    getchar();

    return 0;
}