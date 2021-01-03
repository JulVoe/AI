#define THE_VERSION_JULIAN_DID_NOT_SCREW_WITH
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

#include "Dataset.cpp"
#include "Network.cpp"

//==================================================
//==================|Scheduler|=====================
//==================================================
#define PI 3.141592653589793238462643383279502884197
enum LRSCHEDULE_TYPE : uint32_t { LINEAR=0, COSINE=1, DECAY=2, EXPONENTIAL=3, DEMON=4 };
template<typename L>
struct LRSchedule {
    L lr_sta, lr_sto;
    uint32_t warm_restarts;  //Warm restart each "warm_restart" epochs. If 0, this feature is deactivated

    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs);
    virtual LRSCHEDULE_TYPE getType();
};

template<typename L>
struct Linear_LRSchedule : public LRSchedule<L> {
    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) override {
        double round;
        L frac = modf(epoch / (warm_restarts ? warm_restarts : num_epochs), &nullptr);

        return lr_sta - (lr_sta - lr_sto) * frac;
    }
    virtual LRSCHEDULE_TYPE getType() { return LRSCHEDULE_TYPE::LINEAR; }
};

template<typename L>
struct Cosine_LRSchedule : public LRSchedule<L> {
    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) override {
        double round;
        L frac = modf(LRSchedule / (warm_restarts ? warm_restarts : num_epochs), &nullptr);

        return lr_sto + (lr_sta - lr_sto) * ((L)1 + (L)cos(PI * frac)) / (L)2;
    }
    virtual LRSCHEDULE_TYPE getType() { return LRSCHEDULE_TYPE::COSINE; }
};

template<typename L>
struct Decay_LRSchedule : public LRSchedule<L> {
    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) override {
        double round;
        L frac = modf(epoch / (warm_restarts ? warm_restarts : num_epochs), &nullptr);

        return lr_sta * (L)1 / ((L)1 + ((lr_sta / lr_sto) - (L)1) * frac);
    }
    virtual LRSCHEDULE_TYPE getType() { return LRSCHEDULE_TYPE::DECAY; }
};

template<typename L>
struct Exponential_LRSchedule : public LRSchedule<L> {
    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) override {
        double round;
        L frac = modf(epoch / (warm_restarts ? warm_restarts : num_epochs), &nullptr);

        return lr_sta * pow(lr_sto / lr_sta, frac);
    }
    virtual LRSCHEDULE_TYPE getType() { return LRSCHEDULE_TYPE::EXPONENTIAL; }
};

template<typename L>
struct Demon_LRSchedule : public LRSchedule<L> {
    virtual L getLrEpoch(uint32_t epoch, uint32_t num_epochs) override {
        double round;
        L frac = modf(epoch / (warm_restarts ? warm_restarts : num_epochs), &nullptr);

        return (((L)1 - frac) * lr_sta) / ((L)1 - frac * lr_sta);
    }
    virtual LRSCHEDULE_TYPE getType() { return LRSCHEDULE_TYPE::DEMON; }
};

template<typename L>
struct LRScheduleComplete {
    LRSchedule<L> warmup, regular;
    uint32_t warmup_length;

    L getLrEpoch(uint32_t epoch, uint32_t num_epochs) {
        if (epoch < warmup_length)
            return warmup.getLrEpoch(epoch, warmup_length);
        else
            return regular.getLrEpoch(epoch - warmup_length, num_epochs - warmup_length);
    }
};

template<typename T, typename L = T>
class Scheduler {
    //Components
    NetworkBuilder<T, L>* network; //Also optimizer
    Loss<T>* loss;
    DatasetHandler<T>* dataset;

    //Hyperparameters
    uint32_t num_epochs;
    uint32_t steps_per_epoch;

    LRScheduleComplete<L> alpha_schedule, beta1_schedule, beta2_schedule;

    uint32_t plataue_start;
    T        plateau_threshold;
    uint32_t patienceLRChange, patienceEarlyStopping;   //0 to disable
    L        lrPlateauFactor;
    L        lrAccumulatedFactor;

    T        loss_goal;             //Stop early, when validation loss is under this goal

    //Execution stuff
    cudaStream_t execStream;

    //Monitoring
    std::vector<T> alpha_history, beta1_history, beta2_history;
    std::vector<L> loss_history;

public:
    Scheduler() = default;
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
        plateau_thresholde = threshold;
        patienceLRChange = patienceLRChange_;
        lrPlateauFactor = lrPlateauFactor_;
        patienceEarlyStopping = patienceEarlyStopping_;
    }

    void setLossGoal(T goal) {
        loss_goal = goal;
    }

    void launch(uint32_t dataset_workers, uint32_t dataset_streams, bool debug_window) {
        //0.: Test whether components match
        Image_Shape in_shape, out_shape;
        uint32_t training_samples, validation_samples;
        dataset->getAugmentedShapes(in_shape, out_shape);
        dataset->getNumSamples(training_samples, validation_samples);

        Layer<T, L>* firstLayer_, * lastLayer;
        network->getFirstAndLastLayer(firstLayer, lastLayer);

        if (firstLayer->getLayerType() != LAYER_TYPE::INPT) {
            fprintf(stderr, "[ERROR] First layer of network has to be an input layer, yet it has type %u", firstLayer->getLayerType());
            exit(-1);
        }

        Input_Layer<T, L>* firstLayer = firstLayer_;

        if (firstLayer->outStateShape != in_shape) {
            if (firstLayer->outStateShape.prod() != in_shape.prod) {
                fprintf(stderr, "[WARNING] The input layer of the network has the right size, yet its shape(%u x %u x %u) is not the same as the sample from the dataset(%u x %u x %u)  ßn",
                    firstLayer->outStateShape.x, firstLayer->outStateShape.y, firstLayer->outStateShape.z,
                    in_shape.x, in_shape.y, in_shape.z
                );
            }
            else {
                fprintf(stderr, "[ERROR] The shape of the input layer(%u x %u x %u) does not match the shape of samples from the dataset(%u x %u x %u)",
                    firstLayer->outStateShape.x, firstLayer->outStateShape.y, firstLayer->outStateShape.z,
                    in_shape.x, in_shape.y, in_shape.z
                );
                exit(-1);
            }
        }
        if (lastLayer->outStateShape != out_shape) {
            if (lastLayer->outStateShape.prod() != out_shape.prod) {
                fprintf(stderr, "[WARNING] The last layer of the network has the right size, yet its shape(%u x %u x %u) is not the same as the sample from the dataset(%u x %u x %u)           ßn",
                    lastLayer->outStateShape.x, lastLayer->outStateShape.y, lastLayer->outStateShape.z,
                    out_shape.x, out_shape.y, out_shape.z
                );
            }
            else {
                fprintf(stderr, "[ERROR] The shape of the last layer(%u x %u x %u) does not match the shape of samples from the dataset(%u x %u x %u)",
                    lastLayer->outStateShape.x, lastLayer->outStateShape.y, lastLayer->outStateShape.z,
                    out_shape.x, out_shape.y, out_shape.z
                );
                exit(-1);
            }
        }

        loss.setParameters(lastLayer->state, lastLayer->outStateShape, lastLayer->batch_size);


        //1.: Build graphs
        cudaStreamCreateWithFlags(&execStream, cudaStreamNonBlocking);

        cudaGraph_t trainStep, validationStep;
        cudaGraphCreate(&trainStep, 0);
        cudaGraphCreate(&validationStep, 0);
        cudaGraphNode_t node, depNode;

        //1.1.: trainStep
        //Forward propagation
        cudaGraphAddChildGraphNode(&node, trainStep, nullptr, 0, network->getForwardGraph(execStream));
        depNode = node;

        //Last Deltas
        cudaGraphAddChildGraphNode(&node, trainStep, depNode, 1, loss->getDeltaGraph(execStream));
        depNode = node;

        //Backward propagation
        cudaGraphAddChildGraphNode(&node, trainStep, depNode, 1, network->getBackwardsGraph(execStream));
        depNode = node;

        //1.2.: validationStep
        //Forward propagation
        cudaGraphAddChildGraphNode(&node, validationStep, nullptr, 0, network->getForwardGraph(execStream));
        depNode = node;

        //Compute loss
        cudaGraphAddChildGraphNode(&node, validationStep, depNode, 1, loss->getLossGraph(execStream));
        depNode = node;


        //2.: Instantiate graphs
        cudaGraphExec_t trainExec, validationExec;
        char errorBuf[512] = ",";
        cudaGraphNode_t errNode;

        cudaGraphInstantiate(&trainExec, trainStep, &errNode, +errorBuf, 512);
        if (errorBuf[0] != ',') {
            fprintf(stderr, "[ERROR] The following error arose during the instantiation of the training graph: %s", +errorBuf);
            exit(-1);
        }

        cudaGraphInstantiate(&validationExec, validationStep, &errNode, +errorBuf, 512);
        if (errorBuf[0] != ',') {
            fprintf(stderr, "[ERROR] The following error arose during the instantiation of the validation graph: %s", +errorBuf);
            exit(-1);
        }

        //3.: Run
        //Pointer to current gpu tile
        T** in, ** out, * loss_buf;
        L* alpha_buf, * beta1_buf, * beta2_buf;
        cudaMallocHost((void**)&in, sizeof(T*));
        cudaMallocHost((void**)&out, sizeof(T*));
        cudaMallocHost(&loss_buff, sizeof(T));
        cudaMallocHost(&alpha_buf, sizeof(L));
        cudaMallocHost(&beta1_buf, sizeof(L));
        cudaMallocHost(&beta2_buf, sizeof(L));

        //Start dataset workers
        printf("[INFO] Starting dataset workers\n");
        dataset->start_workers(dataset_workers, dataset_streams, WORKER_STATUS::TRAINING);

        printf("[INFO] Starting training loop\n");
        for (uint32_t epoch = 0; epoch != num_epochs; epoch++) {
            //Compute LRs
            *alpha_buf = alpha_schedule.getLrEpoch(epoch, num_epochs) * lrAccumulatedFactor;
            *beta1_buf = beta1_schedule.getLrEpoch(epoch, num_epochs) * lrAccumulatedFactor;
            *beta2_buf = beta2_schedule.getLrEpoch(epoch, num_epochs) * lrAccumulatedFactor;

            alpha_history.append(*alpha_buf);
            beta1_history.append(*beta1_buf);
            beta2_history.append(*beta2_buf);

            //Set LRs
            network->opt.setLR(alpha_buf, beta1_buf, beta2_buf, execStream);

            //Training
            for (uint32_t step = 0; step != steps_per_epoch; step++) {
                //Set input and output
                T* in_, * out_;
                dataset->advance<true>(in_, out_);
                *in = in_;
                *out = out_;

                firstLayer->setInputPointer(in, execStream);
                loss->setTarget(out, execStream);

                cudaGraphLaunch(trainExec, execStream);
            }

            //Validation
            for (uint32_t step = 0; step != validation_samples; step++) {
                //Set input and output
                T* in_, * out_;
                dataset->advance<false>(in_, out_);
                *in = in_;
                *out = out_;

                firstLayer->setInputPointer(in, execStream);
                loss->setTarget(out, execStream);

                cudaGraphLaunch(validationExec, execStream);
            }

            loss->getAccumulator(loss_buff, execStream);
            loss->clearAccumulator(execStream);
            loss_history.push_back(*loss_buff / (T)validation_samples);

            //Debugging
            if (debug) {
                printf("Epoch %u/%u | Loss %d |\n", epoch, num_epochs, loss_history.back());
            }

            //Inspect loss
            if (loss_history.back() < loss_goal) {
                printf("[INFO] Loss goal reached!\n");
                return;
            }

            if (patienceLRChange != 0 && patienceLRChange + plataue_start < epoch) {
                bool plateau = true;
                for (uint32_t pat = 1; pat <= patienceLRChange; pat++) {
                    if (loss_history.back() < plateau_threshold * loss_history[loss_history.size() - pat]) {
                        plateau = false;
                        break;
                    }
                }
                if (plateau) {
                    printf("[INFO] Plateau detected. Adapting learning rate\n");
                    lrAccumulatedFactor *= lrPlateauFactor;
                }
            }
            if (patienceEarlyStopping != 0 && patienceEarlyStopping + plataue_start < epoch) {
                bool plateau = true;
                for (uint32_t pat = 1; pat <= patienceEarlyStopping; pat++) {
                    if (loss_history.back() < plateau_threshold * loss_history[loss_history.size() - pat]) {
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
