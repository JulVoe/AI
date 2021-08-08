#if defined(_MSC_VER) 
#define __builtin_unreachable() __assume(0)
#endif

#define DEBUG

#ifdef __NVCC__
#pragma warning( disable : 4514)
#pragma warning( disable : 4711)
#pragma warning( disable : 4710)
#pragma warning( disable : 5039)
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
//#include <mma.h>
//#include <cublasXt.h>

#include <cmath>
#include <cstdio>
#include <cinttypes>
#include <functional>
#include <algorithm>
#include <numeric>

#include "Dataset.cpp"
#include "Network.cpp"

//==================================================
//==================|Scheduler|=====================
//==================================================
#define PI 3.141592653589793238462643383279502884197
enum LRSCHEDULE_TYPE : uint32_t { CONSTANT=0, LINEAR=1, COSINE=2, DECAY=3, EXPONENTIAL=4, DEMON=5 };
template<typename L>
struct LRSchedule {
    L lr_sta, lr_sto;
    uint32_t warm_restarts;  //Warm restart each "warm_restart" epochs. If 0, this feature is deactivated

    LRSchedule(L learning_rate_start, L learning_rate_end, uint32_t warm_restarts = 0) :
        lr_sta(learning_rate_start), lr_sto(learning_rate_end), warm_restarts(warm_restarts)
    {}

    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) {
        fprintf(stderr, "[ERROR] You called \"getLrEpoch\" on the LRSchedule base class. You should only use the derived classes!");
        std::exit(-1);
    }
    virtual LRSCHEDULE_TYPE getType(){
        fprintf(stderr, "[ERROR] You called \"getType\" on the LRSchedule base class. You should only use the derived classes!");
        std::exit(-1);
    }
};

template<typename L>
struct Const_LRSchedule : public LRSchedule<L> {
    Const_LRSchedule(L learning_rate) :
        LRSchedule<L>(learning_rate, learning_rate)
    {}

    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) override {
        return lr_sta;
    }
    virtual LRSCHEDULE_TYPE getType() override { return LRSCHEDULE_TYPE::CONSTANT; }
};

template<typename L>
struct Linear_LRSchedule : public LRSchedule<L> {
    Linear_LRSchedule(L learning_rate_start, L learning_rate_end, uint32_t warm_restarts = 0) :
        LRSchedule<L>(learning_rate_start, learning_rate_end, warm_restarts)
    {}
    
    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) override {
        double round;
        L frac = modf((float)epoch / (float)(warm_restarts ? warm_restarts : num_epochs), &round);

        return lr_sta - (lr_sta - lr_sto) * frac;
    }
    virtual LRSCHEDULE_TYPE getType() override { return LRSCHEDULE_TYPE::LINEAR; }
};

template<typename L>
struct Cosine_LRSchedule : public LRSchedule<L> {
    Cosine_LRSchedule(L learning_rate_start, L learning_rate_end, uint32_t warm_restarts = 0) :
        LRSchedule<L>(learning_rate_start, learning_rate_end, warm_restarts)
    {}
    
    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) override {
        double round;
        L frac = modf((float)epoch / (float)(warm_restarts ? warm_restarts : num_epochs), &round);

        return lr_sto + (lr_sta - lr_sto) * ((L)1 + (L)cos(PI * frac)) / (L)2;
    }
    virtual LRSCHEDULE_TYPE getType() override { return LRSCHEDULE_TYPE::COSINE; }
};

template<typename L>
struct Decay_LRSchedule : public LRSchedule<L> {
    Decay_LRSchedule(L learning_rate_start, L learning_rate_end, uint32_t warm_restarts = 0) :
        LRSchedule<L>(learning_rate_start, learning_rate_end, warm_restarts)
    {}

    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) override {
        double round;
        L frac = modf((float)epoch / (float)(warm_restarts ? warm_restarts : num_epochs), &round);

        return lr_sta * (L)1 / ((L)1 + ((lr_sta / lr_sto) - (L)1) * frac);
    }
    virtual LRSCHEDULE_TYPE getType() override { return LRSCHEDULE_TYPE::DECAY; }
};

template<typename L>
struct Exponential_LRSchedule : public LRSchedule<L> {
    Exponential_LRSchedule(L learning_rate_start, L learning_rate_end, uint32_t warm_restarts = 0) :
        LRSchedule<L>(learning_rate_start, learning_rate_end, warm_restarts)
    {}
    
    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) override {
        double round;
        L frac = modf((float)epoch / (float)(warm_restarts ? warm_restarts : num_epochs), &round);

        return lr_sta * pow(lr_sto / lr_sta, frac);
    }
    virtual LRSCHEDULE_TYPE getType() override  { return LRSCHEDULE_TYPE::EXPONENTIAL; }
};

template<typename L>
struct Demon_LRSchedule : public LRSchedule<L> {
    Demon_LRSchedule(L learning_rate_start, L learning_rate_end, uint32_t warm_restarts = 0) :
        LRSchedule<L>(learning_rate_start, learning_rate_end, warm_restarts)
    {}
    
    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) override {
        double round;
        L frac = modf((float)epoch / (float)(warm_restarts ? warm_restarts : num_epochs), &round);

        return (((L)1 - frac) * lr_sta) / ((L)1 - frac * lr_sta);
    }
    virtual LRSCHEDULE_TYPE getType() override { return LRSCHEDULE_TYPE::DEMON; }
};

template<typename L>
struct LRScheduleComplete {
    LRSchedule<L> *warmup, *regular;
    uint32_t warmup_length;

    /*
        Initializes object in a way, that "getLrEpoch" will always return 0.
    */
    LRScheduleComplete():
        warmup (new Const_LRSchedule<L>(0)),
        regular(new Const_LRSchedule<L>(0)),
        warmup_length(0)
    {}

    LRScheduleComplete(LRSchedule<L>* warmup, LRSchedule<L>* regular, uint32_t warmup_length) :  //takes ownership
        warmup(warmup), regular(regular), warmup_length(warmup_length)
    {}

    L getLrEpoch(uint32_t epoch, uint32_t num_epochs) {
        if (epoch < warmup_length)
            return warmup->getLrEpoch(epoch, warmup_length);
        else
            return regular->getLrEpoch(epoch - warmup_length, num_epochs - warmup_length);
    }
};



template<typename T, typename L = T>
class Scheduler {
    //Components
    NetworkBuilder<T, L>* network;
    Optimizer<T, L>*      opt;
    Loss<T>*              loss;
    DatasetHandler<T>*    dataset;

    //Hyperparameters
    uint32_t num_epochs;
    uint32_t steps_per_epoch;

    LRScheduleComplete<L> alpha_schedule, beta1_schedule, beta2_schedule;

    uint32_t plateau_start;
    T        plateau_threshold;
    uint32_t patienceLRChange, patienceEarlyStopping;   //0 to disable
    L        lrPlateauFactor;
    L        lrAccumulatedFactor;

    T        loss_goal;             //Stop early, when validation loss is under this goal

    //Execution stuff
    cudaStream_t execStream;

    //Monitoring
    std::vector<T> alpha_history, beta1_history, beta2_history;
    std::vector<T> val_loss_history, train_loss_history;


    // /+================+\
    // ||Opengl functions||
    // \+================+/

#ifdef WIN32
    static constexpr char* FONT_PATH = "C:/Windows/Fonts/Arial.ttf";
    static constexpr uint32_t YSTART = 0;
#else
    static constexpr char* FONT_PATH = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf";
    static constexpr uint32_t YSTART = 300;
#endif
    static constexpr uint32_t XRES = 700;
    static constexpr uint32_t YRES = 400;
    static constexpr uint32_t XIMG = 300;
    static constexpr uint32_t YIMG = 300;
    static constexpr uint32_t PADDING = 15;
    static constexpr uint32_t GAP = 50;
    static constexpr uint32_t PSIZE = 6;
    /*
        Uses opengl to draw a window which always shows the pixel data
    */
    void window_thread() {
        //1.: Create Window and initialize GLFW
        GLFWwindow* window;
            //if (!glfwInit())
            //    return;
        glfwWindowHint(GLFW_SAMPLES, 4);
        window = glfwCreateWindow(XRES, YRES, "Tracking", NULL, NULL);
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
        //glfwSetKeyCallback(window, key_callback);

        //4.: Debugging
        if (glDebugMessageControlARB != NULL) {
            printf("[INFO] Setting up Opengl-Debugging\n");
            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
            glDebugMessageControlARB(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
            glDebugMessageCallbackARB((GLDEBUGPROCARB)ETB_GL_ERROR_CALLBACK, NULL);
        }

        //4.5: FreeType initialization and blending
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
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
        glLineWidth(1.5f);

        //7.: Display-Loop
        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            constexpr uint32_t textHeight = 30;
            uint32_t x_off = PADDING;
            uint32_t y_off = PADDING;

            //Loss
            glColor3f(1.f, 1.f, 1.f);
            TextRenderer::renderText("Loss", x_off + 2 * PADDING, y_off + PADDING + textHeight, 0.3f);
            
            if (val_loss_history.size()) {
                const auto& [min_val_loss, max_val_loss] = std::minmax_element(val_loss_history.begin(), val_loss_history.end());
                if (train_loss_history.size()) {
                    const auto& [min_train_loss, max_train_loss] = std::minmax_element(train_loss_history.begin(), train_loss_history.end());
                    Plotter::renderPlot<T, true>(std::vector<std::vector<T>>{val_loss_history, train_loss_history}, x_off, x_off + XIMG, y_off, y_off + YIMG, PADDING, false, min(*min_val_loss, *min_train_loss), max(*max_val_loss, *max_train_loss), num_epochs);
                }
                else
                    Plotter::renderPlot<T, true>(std::vector<std::vector<T>>{val_loss_history}, x_off, x_off + XIMG, y_off, y_off + YIMG, PADDING, false, *min_val_loss,*max_val_loss, num_epochs);
            }
            else
                Plotter::renderPlot<T, true>(std::vector<std::vector<T>>{}, x_off, x_off + XIMG, y_off, y_off + YIMG, PADDING, false, 0, 1, num_epochs);


            x_off += PADDING + XIMG;


            //Learning rates
            glColor3f(1.f, 1.f, 1.f);
            TextRenderer::renderText("Learning rates", x_off + 2 * PADDING, y_off + PADDING + textHeight, 0.3f);

            if (alpha_history.size()) {
                const auto& [min_alpha, max_alpha] = std::minmax_element(std::begin(alpha_history), std::end(alpha_history));
                const auto& [min_beta1, max_beta1] = std::minmax_element(std::begin(beta1_history), std::end(beta1_history));
                const auto& [min_beta2, max_beta2] = std::minmax_element(std::begin(beta2_history), std::end(beta2_history));
                Plotter::renderPlot<T, false>(std::vector<std::vector<T>>{ alpha_history, beta1_history, beta2_history }, x_off, x_off + XIMG, y_off, y_off + YIMG, PADDING, false, std::min({ *min_alpha, *min_beta1, *min_beta2 }), std::max({ *max_alpha, *max_beta1, *max_beta2 }), num_epochs);
            }
            else
                Plotter::renderPlot<T, false>(std::vector<std::vector<T>>{}, x_off, x_off + XIMG, y_off, y_off + YIMG, PADDING, false, 0, 1, num_epochs);

            x_off += PADDING + XIMG;


            glfwSwapBuffers(window);
            glfwPollEvents();
        }


        glfwDestroyWindow(window);
    }

public:
    Scheduler(NetworkBuilder<T, L>* network, Optimizer<T, L>* opt, Loss<T>* loss, DatasetHandler<T>* dataset, cudaStream_t execStream):
        network(network), opt(opt), loss(loss), dataset(dataset),
        num_epochs(0), steps_per_epoch(0),
        alpha_schedule(), beta1_schedule(), beta2_schedule(),
        plateau_start(0), plateau_threshold((T)0), patienceLRChange(0), patienceEarlyStopping(0), lrPlateauFactor((L)1), lrAccumulatedFactor((L)1),
        loss_goal((T)0),
        execStream(execStream),
        alpha_history(), beta1_history(), beta2_history(), val_loss_history(), train_loss_history()
    {}


    void setNumRuns(uint32_t num_epochs_, uint32_t steps_per_epoch_) {
        num_epochs = num_epochs_;
        steps_per_epoch = steps_per_epoch_;
    }

    void setLRSchedule(LRScheduleComplete<L> alpha_schedule_, LRScheduleComplete<L> beta1_schedule_, LRScheduleComplete<L> beta2_schedule_) {
        alpha_schedule = alpha_schedule_;
        beta1_schedule = beta1_schedule_;
        beta2_schedule = beta2_schedule_;
    }

    void setPlateau(uint32_t start, T threshold, uint32_t patienceLRChange_ = 0, L lrPlateauFactor_ = (L)0.1, uint32_t patienceEarlyStopping_ = 0) {
        if (0 < patienceEarlyStopping && patienceEarlyStopping < patienceLRChange)
            fprintf(stderr, "[WARNING] Early stopping will occur before the learning rate change. Thus, a learning rate change resulting of a plateau will never occur!\n");

        plateau_start = start;
        plateau_threshold = threshold;
        patienceLRChange = patienceLRChange_;
        lrPlateauFactor = lrPlateauFactor_;
        patienceEarlyStopping = patienceEarlyStopping_;
    }

    void setLossGoal(T goal) {
        loss_goal = goal;
    }

    void launch(uint32_t dataset_workers, uint32_t dataset_streams, bool debug_window, bool train_loss, bool check = false, char* ckp_dir_ = "", uint32_t save_every = 0) {
        //-1.: Create variables
        std::string ckp_dir(ckp_dir_);
        ckp_dir += "/";
        
        //0.: Test whether components match
        Image_Shape in_shape, out_shape;
        uint32_t training_samples, validation_samples;
        dataset->getAugmentedShapes(in_shape, out_shape);
        dataset->getNumSamples(training_samples, validation_samples);

        Layer<T, L>* firstLayer_, * lastLayer;
        network->getFirstAndLastLayer(firstLayer_, lastLayer);

        if (firstLayer_->getLayerType() != LAYER_TYPE::INPT) {
            fprintf(stderr, "[ERROR] First layer of network has to be an input layer, yet it has type %u", firstLayer_->getLayerType());
            exit(-1);
        }

        Input_Layer<T, L>* firstLayer = (Input_Layer<T, L>*)firstLayer_;

        if (firstLayer->outputShape != in_shape) {
            if (firstLayer->outputShape.prod() == in_shape.prod()) {
                fprintf(stderr, "[WARNING] The input layer of the network has the right size, yet its shape(%u x %u x %u) is not the same as the sample from the dataset(%u x %u x %u)\n",
                    firstLayer->outputShape.x, firstLayer->outputShape.y, firstLayer->outputShape.z,
                    in_shape.x, in_shape.y, in_shape.z
                );
            }
            else {
                fprintf(stderr, "[ERROR] The shape of the input layer(%u x %u x %u) does not match the shape of samples from the dataset(%u x %u x %u)",
                    firstLayer->outputShape.x, firstLayer->outputShape.y, firstLayer->outputShape.z,
                    in_shape.x, in_shape.y, in_shape.z
                );
                std::exit(-1);
            }
        }
        if (lastLayer->outputShape != out_shape) {
            if (lastLayer->outputShape.prod() == out_shape.prod()) {
                fprintf(stderr, "[WARNING] The last layer of the network has the right size, yet its shape(%u x %u x %u) is not the same as the sample from the dataset(%u x %u x %u)\n",
                    lastLayer->outputShape.x, lastLayer->outputShape.y, lastLayer->outputShape.z,
                    out_shape.x, out_shape.y, out_shape.z
                );
            }
            else {
                fprintf(stderr, "[ERROR] The shape of the last layer(%u x %u x %u) does not match the shape of samples from the dataset(%u x %u x %u)\n",
                    lastLayer->outputShape.x, lastLayer->outputShape.y, lastLayer->outputShape.z,
                    out_shape.x, out_shape.y, out_shape.z
                );
                std::exit(-1);
            }
        }

        //Loss and Optimizer remains unchecked
        //loss.setParameters(lastLayer->state, lastLayer->outStateShape, lastLayer->batch_size);

        //1.: Build graphs
        cudaGraph_t trainStep, validationStep;
        cudaGraphCreate(&trainStep, 0);
        cudaGraphCreate(&validationStep, 0);
        cudaGraphNode_t node, depNode;

        //1.1.: trainStep
        cudaGraph_t networkForward = network->getForwardGraph(execStream);
        cudaGraph_t networkBackward = network->getBackwardsGraph(opt, execStream);
        cudaGraph_t lossDelta = loss->getDeltaGraph(execStream);
        cudaGraph_t lossLoss = loss->getLossGraph(execStream);

        cudaGraphAddChildGraphNode(&node, trainStep, nullptr, 0, networkForward);
        depNode = node;
        if (train_loss) {
            cudaGraphAddChildGraphNode(&node, trainStep, &depNode, 1, lossLoss);
            depNode = node;
        }
        cudaGraphAddChildGraphNode(&node, trainStep, &depNode, 1, lossDelta);
        depNode = node;
        cudaGraphAddChildGraphNode(&node, trainStep, &depNode, 1, networkBackward);

        //1.2.: validationStep
        cudaGraphAddChildGraphNode(&node, validationStep, nullptr, 0, networkForward);
        cudaGraphAddChildGraphNode(&node, validationStep, &node, 1, lossLoss);

        //2.: Instantiate graphs
        cudaGraphExec_t trainExec, validationExec;
        char errorBuf[512];
        cudaGraphNode_t errNode;

        cudaGraphInstantiate(&trainExec, trainStep, &errNode, +errorBuf, 512);
        if (errorBuf[0]) {
            fprintf(stderr, "[ERROR] The following error arose during the instantiation of the training graph: %s\n", +errorBuf);
            exit(-1);
        }

        cudaGraphInstantiate(&validationExec, validationStep, &errNode, +errorBuf, 512);
        if (errorBuf[0]) {
            fprintf(stderr, "[ERROR] The following error arose during the instantiation of the validation graph: %s\n", +errorBuf);
            exit(-1);
        }


        //3.: Start tracking window
        if (debug_window) {
            std::thread tracker = std::thread(&Scheduler<T, L>::window_thread, this);
            tracker.detach();
        }


        //4.: Start dataset workers
        //Pointer to current gpu tile
        T* loss_buf;
        L* alpha_buf, * beta1_buf, * beta2_buf;
        cudaMallocHost((void**)&loss_buf , sizeof(T ));
        cudaMallocHost((void**)&alpha_buf, sizeof(L ));
        cudaMallocHost((void**)&beta1_buf, sizeof(L ));
        cudaMallocHost((void**)&beta2_buf, sizeof(L ));

        //Start dataset workers
        printf("[INFO] Starting dataset workers\n");
        dataset->start_workers(dataset_workers, dataset_streams, WORKER_STATUS::TRAINING);

        //5.: Main loop
        printf("[INFO] Starting training loop\n");

            T* in_data_host = (T*)malloc(sizeof(T) * 128 * 128 *  64 * 3 * 9);
            T* gradients = ((Debug_Optimizer<T>*)opt)->getOptBuf();
            opt->initMem();
            uint64_t num_opt = network->getNumOptimizables();
            cudaDeviceSynchronize();


        T** in, ** out;
        cudaMallocHost((void**)&in , sizeof(T*));
        cudaMallocHost((void**)&out, sizeof(T*));
        for (uint32_t epoch = 0; epoch != num_epochs; epoch++) {
            //Compute LRs
            *alpha_buf = alpha_schedule.getLrEpoch(epoch, num_epochs) * lrAccumulatedFactor;
            *beta1_buf = beta1_schedule.getLrEpoch(epoch, num_epochs) * lrAccumulatedFactor;
            *beta2_buf = beta2_schedule.getLrEpoch(epoch, num_epochs) * lrAccumulatedFactor;

            alpha_history.push_back(*alpha_buf);
            beta1_history.push_back(*beta1_buf);
            beta2_history.push_back(*beta2_buf);

            //Set LRs
            opt->setLR(alpha_buf, beta1_buf, beta2_buf, execStream);

            //Training
            for (uint32_t step = 0; step != steps_per_epoch; step++) {
                    //clear_console();
                //Set input and output
                dataset->template advance<true>(in, out);

                firstLayer->setInputPointer(in, execStream);
                loss->setTarget(out, execStream);

                cudaGraphLaunch(trainExec, execStream);
                cudaStreamSynchronize(execStream);
                    //CHECK_CUDA_ERROR();
                    //BUGP("-------------------------\n");
                    //
                    //std::this_thread::sleep_for(std::chrono::duration<double>(0.000001));
                    //cudaDeviceSynchronize();
                    //printf("%u %u\n", epoch, step);
                    //printf("\n\Input\n");
                    //gpuErrchk(cudaMemcpy(in_data_host, *in, sizeof(T) * network->layers[0]->outputShape.prod() * /*B*/1, cudaMemcpyDeviceToHost));
                    //cudaDeviceSynchronize();
                    //ARR_PRINT(in_data_host, network->layers[0]->outputShape.prod(), /*B*/1);
                    //for (uint32_t l_ind = 1; l_ind != network->layers.size(); l_ind++) {
                    //    printf("\n\Bias Layer %u:\n", l_ind);
                    //    gpuErrchk(cudaMemcpy(in_data_host, network->layers[l_ind]->intern[0], sizeof(T) * network->layers[l_ind]->outputShape.prod(), cudaMemcpyDeviceToHost));
                    //    cudaDeviceSynchronize();
                    //    ARR_PRINT_COLMAJ(in_data_host, network->layers[l_ind]->outputShape.prod(), 1);
                    //    
                    //    //printf("\n\Weights Layer %u:\n", l_ind);
                    //    //gpuErrchk(cudaMemcpy(in_data_host, network->layers[l_ind]->intern[1], sizeof(T) * network->layers[l_ind]->outputShape.prod() * network->layers[l_ind-1]->outputShape.prod(), cudaMemcpyDeviceToHost));
                    //    //cudaDeviceSynchronize();
                    //    //ARR_PRINT_COLMAJ(in_data_host, network->layers[l_ind]->outputShape.prod(), network->layers[l_ind-1]->outputShape.prod());
                    //    //
                    //    //No way to recover output: Memory was already filled with deltas
                    //    //printf("\n\Output Layer %u:\n", l_ind);
                    //    //gpuErrchk(cudaMemcpy(in_data_host, network->layers[l_ind]->output, sizeof(T) * network->layers[l_ind]->outputShape.prod(), cudaMemcpyDeviceToHost));
                    //    //cudaDeviceSynchronize();
                    //    //ARR_PRINT(in_data_host, network->layers[l_ind]->outputShape.prod(), 1);
                    //}
                    //
                    //
                    //
                    //for (uint32_t l = 1; l < network->layers.size(); l++) {
                    //    printf("\n\Delta Layer %u:\n", l);
                    //    gpuErrchk(cudaMemcpy(in_data_host, network->layers[l]->output, sizeof(T) * network->layers[l]->outputShape.prod() * /*B*/8, cudaMemcpyDeviceToHost));
                    //    cudaDeviceSynchronize();
                    //    ARR_PRINT(in_data_host, network->layers[l]->outputShape.prod(), /*B*/8);
                    //}
                    //
                    //printf("\n\Output\n");
                    //gpuErrchk(cudaMemcpy(in_data_host, *out, sizeof(T) * network->layers.back()->outputShape.prod() * /*B*/8, cudaMemcpyDeviceToHost));
                    //cudaDeviceSynchronize();
                    //ARR_PRINT(in_data_host, network->layers.back()->outputShape.prod(), /*B*/8);
                    //
                    //
                    //printf("\n\Gradients\n");
                    //gpuErrchk(cudaMemcpy(in_data_host, gradients, sizeof(T) * num_opt * /*B*/1, cudaMemcpyDeviceToHost));
                    //cudaDeviceSynchronize();
                    //ARR_PRINT_COLMAJ(in_data_host + 0, 4, 4); //Weights layer 1
                    //ARR_PRINT(in_data_host + 4, 4, 1); //Bias   layer 1
                    //opt->initMem();
                    //cudaDeviceSynchronize();
                    //
                    //getchar();


                if (check) {
                    T test;
                    loss->getAccumulator(&test, execStream);
                    cudaStreamSynchronize(execStream);
                    if (!CHECK_MEM(&test, 1)) {
                        //CHECK_CUDA_ERROR();
                        //BUGP("-------------------------\n");
                        //std::this_thread::sleep_for(std::chrono::duration<double>(0.000001));
                        //cudaDeviceSynchronize();
                        //printf("%u %u\n", epoch, step);
                        printf("\n\Input\n");
                        gpuErrchk(cudaMemcpy(in_data_host, *in, sizeof(T) * network->layers[0]->outputShape.prod() * /*B*/8, cudaMemcpyDeviceToHost));
                        cudaDeviceSynchronize();
                        ARR_PRINT(in_data_host, network->layers[0]->outputShape.prod(), /*B*/8);
                        for (uint32_t l_ind = 1; l_ind != network->layers.size(); l_ind++) {
                            printf("\n\Bias Layer %u:\n", l_ind);
                            gpuErrchk(cudaMemcpy(in_data_host, network->layers[l_ind]->intern[0], sizeof(T) * network->layers[l_ind]->outputShape.prod(), cudaMemcpyDeviceToHost));
                            cudaDeviceSynchronize();
                            ARR_PRINT_COLMAJ(in_data_host, network->layers[l_ind]->outputShape.prod(), 1);

                            //printf("\n\Weights Layer %u:\n", l_ind);
                            //gpuErrchk(cudaMemcpy(in_data_host, network->layers[l_ind]->intern[1], sizeof(T) * network->layers[l_ind]->outputShape.prod() * network->layers[l_ind-1]->outputShape.prod(), cudaM
                            //cudaDeviceSynchronize();
                            //ARR_PRINT_COLMAJ(in_data_host, network->layers[l_ind]->outputShape.prod(), network->layers[l_ind-1]->outputShape.prod());
                            //
                            //No way to recover output: Memory was already filled with deltas
                            //printf("\n\Output Layer %u:\n", l_ind);
                            //gpuErrchk(cudaMemcpy(in_data_host, network->layers[l_ind]->output, sizeof(T) * network->layers[l_ind]->outputShape.prod(), cudaMemcpyDeviceToHost));
                            //cudaDeviceSynchronize();
                            //ARR_PRINT(in_data_host, network->layers[l_ind]->outputShape.prod(), 1);
                        }

                        for (uint32_t l = 1; l < network->layers.size(); l++) {
                            printf("\n\Delta Layer %u:\n", l);
                            gpuErrchk(cudaMemcpy(in_data_host, network->layers[l]->output, sizeof(T) * network->layers[l]->outputShape.prod() * /*B*/8, cudaMemcpyDeviceToHost));
                            cudaDeviceSynchronize();
                            ARR_PRINT(in_data_host, network->layers[l]->outputShape.prod(), /*B*/8);
                        }

                        printf("\n\Output\n");
                        gpuErrchk(cudaMemcpy(in_data_host, *out, sizeof(T) * network->layers.back()->outputShape.prod() * /*B*/8, cudaMemcpyDeviceToHost));
                        cudaDeviceSynchronize();
                        ARR_PRINT(in_data_host, network->layers.back()->outputShape.prod(), /*B*/8);

                        printf("\n\Gradients\n");
                        gpuErrchk(cudaMemcpy(in_data_host, gradients, sizeof(T) * num_opt * /*B*/1, cudaMemcpyDeviceToHost));
                        cudaDeviceSynchronize();
                        ARR_PRINT_COLMAJ(in_data_host + 0, 4, 4); //Weights layer 1
                        ARR_PRINT(in_data_host + 4, 4, 1); //Bias   layer 1
                        opt->initMem();
                        cudaDeviceSynchronize();

                        getchar();
                    }
                }
            }

            if (train_loss) {
                loss->getAccumulator(loss_buf, execStream);
                cudaStreamSynchronize(execStream);
                loss->clearAccumulator(execStream);
                train_loss_history.push_back(*loss_buf / (T)(steps_per_epoch * network->getBatchSize()));
            }

            //Validation
            for (uint32_t step = 0; step != validation_samples; step++) {
                //Set input and output
                dataset->template advance<false>(in, out);

                firstLayer->setInputPointer(in, execStream);
                loss->setTarget(out, execStream);

                //cudaStreamSynchronize(execStream);
                cudaGraphLaunch(validationExec, execStream);
                cudaStreamSynchronize(execStream);
            }

            loss->getAccumulator(loss_buf, execStream);
            cudaStreamSynchronize(execStream);
            loss->clearAccumulator(execStream);
            val_loss_history.push_back(*loss_buf / (T)(validation_samples * network->getBatchSize()));

            //Debugging
            if(train_loss)
                printf("Epoch %u/%u | alpha %f | Loss %f | Train Loss %f\n", epoch+1, num_epochs, alpha_history.back(), val_loss_history.back(), train_loss_history.back());
            else
                printf("Epoch %u/%u | alpha %f | Loss %f |\n", epoch+1, num_epochs, alpha_history.back(), val_loss_history.back());

            //Save
            if (save_every != 0 && ((epoch+1) % save_every == 0))
                network->compress((ckp_dir+std::to_string(epoch / save_every)+".jvcheck").c_str());

            //Inspect loss
            if (val_loss_history.back() < loss_goal) {
                printf("[INFO] Loss goal reached!\n");
                return;
            }

            if (patienceLRChange != 0 && patienceLRChange + plateau_start < epoch) {
                bool plateau = true;
                for (uint32_t pat = 1; pat <= patienceLRChange; pat++) {
                    if (val_loss_history.back() < plateau_threshold * val_loss_history[val_loss_history.size() - pat]) {
                        plateau = false;
                        break;
                    }
                }
                if (plateau) {
                    printf("[INFO] Plateau detected. Adapting learning rate\n");
                    lrAccumulatedFactor *= lrPlateauFactor;
                }
            }
            if (patienceEarlyStopping != 0 && patienceEarlyStopping + plateau_start < epoch) {
                bool plateau = true;
                for (uint32_t pat = 1; pat <= patienceEarlyStopping; pat++) {
                    if (val_loss_history.back() < plateau_threshold * val_loss_history[val_loss_history.size() - pat]) {
                        plateau = false;
                        break;
                    }
                }
                if (plateau) {
                    printf("[INFO] Early stopping triggered!\n");
                    return;
                }

            }
        }
    }

    //https://arxiv.org/pdf/1812.01187.pdf
    //https://arxiv.org/pdf/1608.03983.pdf
};


#if 0
#define P_IN    "C:/Users/julia/source/repos/AI/Datasets/1/Raw/In"
#define P_OUT   "C:/Users/julia/source/repos/AI/Datasets/1/Raw/Out"
#define P_D_IN  "C:/Users/julia/source/repos/AI/Datasets/1/in.jvdata"
#define P_D_OUT "C:/Users/julia/source/repos/AI/Datasets/1/out.jvdata"
int main()
{
    cudaStream_t str;
    cudaStreamCreateWithFlags(&str, cudaStreamNonBlocking);

    cublasSetup(8ull * 1024ull * 1024ull, str); //8Mb workspace (default according to debugging with nsight compute, even though documentation says 4mb)

    Random::init_rand();

    //Constants
    using T = float;
    using L = T;
    constexpr uint32_t N  = 64;
    constexpr uint32_t S  = 64;
    constexpr uint32_t B  = 8;  //Dataset batch size
    constexpr uint32_t B2 = 8;  //Network batch size

    //Network
    std::vector<Layer<T>*> layers;
    layers.push_back(new Input_Layer<T>(Image_Shape(S, S, 3u)));
    //layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::SOFTPLUS, N));
    //layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::SOFTPLUS, N));
    layers.push_back(new FullyConnected_Layer<T>(ACTIVATION_TYPE::SOFTMAX , 9u));

    NetworkBuilder<T, L> network(layers, 16, B2);
    network.allocate();
    network.initialize();

    //Optimizer
    Optimizer<T>* opt = new Debug_Optimizer<T>(B); //Optimizer<T>::getOptimizerOfType(OPTIMIZER_TYPE::SGD);
    opt->setNumOptimizables(network.getNumOptimizables());
    opt->allocate();
    opt->initMem();

    //Loss
    Loss<T>* loss = new MSE_Loss<T>();
    loss->setParameters((T*)layers.back()->output, layers.back()->outputShape, B2, (T*)network.getDeltaMem());

    //Dataset
    Offset2D<uint32_t> size(100, 100); 
    //generateDatasetFile_Image<float>(P_IN, P_D_IN, size);
    //generateDatasetFile_Raw<float>(P_OUT, P_D_OUT);
        //generateDatasetFile_Raw<float>(P_IN, P_D_IN);
        //generateDatasetFile_Classification<float>(P_OUT, P_D_OUT, 4);

    AugmentationInfo2D_IN  agi_in (DATA_FORMAT(DISTRIBUTION::UNIFORM, 1.f), 0, false, 0.f, 0.0f, 0.f, 0.f, 0.f, Offset2D<float>(-1.f, -1.f), Offset2D<int32_t>(S, S));              //(DATA_FORMAT(DISTRIBUTION::UNIFORM, 1.f), 2, false, 0.f, 0.05f, 0.2f, 0.2f, 0.2f, Offset2D<float>(0.95f, 0.95f), Offset2D<int32_t>(S, S));
    AugmentationInfo2D_OUT agi_out(DATA_FORMAT(DISTRIBUTION::UNIFORM, 1.f), Offset2D<int32_t>(-1, -1), 0.f, false);                                                                 //(DATA_FORMAT(DISTRIBUTION::UNIFORM, 1.f), Offset2D<int32_t>(-1, -1), 0.01f, false);

    DatasetHandler<T> handler(P_D_IN, P_D_OUT, 0.81f, 8192);
    handler.set_augmentation(agi_in, agi_out);
    handler.set_batching(B, 8, 8);
    //handler.start_debugWindow();

    //Scheduler
    Scheduler<T> scheduler(&network, opt, loss, &handler, str);
    scheduler.setNumRuns(500, 100);
    scheduler.setLossGoal((T)0);
    scheduler.setPlateau(0, 0);
    scheduler.setLRSchedule(
        LRScheduleComplete<T>(new Linear_LRSchedule<L>(0.005 / (float)B2, 0.01 / (float)B2), new Decay_LRSchedule<L>(0.01 / (float)B2, 0.001 / (float)B2), 10),
        LRScheduleComplete<T>(new  Const_LRSchedule<L>(0.01       ), new Const_LRSchedule<L>(0.01       ),  0), 
        LRScheduleComplete<T>(new  Const_LRSchedule<L>(0.01       ), new Const_LRSchedule<L>(0.01       ),  0)
    );
    
    //Launch
    getchar();
    
    scheduler.launch(3, 2, true, true);

    getchar();


    BUGP("\n\nDone");

    CHECK_CUDA_ERROR();

    CUBLAS_ERROR(cublasDestroy(cublas_handle));

    return 0;
}
#endif


//TODO: For some reason, loss is depending on wait period between passes