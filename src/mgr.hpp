#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/render_mgr.hpp>

#include "sim_flags.hpp"

namespace madEscape {

// The Manager class encapsulates the linkage between the outside training
// code and the internal simulation state (src/sim.hpp / src/sim.cpp)
//
// Manager is responsible for initializing the simulator, loading physics
// and rendering assets off disk, and mapping ECS components to tensors
// for learning
class Manager {
public:
    struct Config {
        madrona::ExecMode execMode; // CPU or CUDA
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        bool autoReset; // Immediately generate new world on episode end
        SimFlags simFlags;
        RewardMode rewardMode;
        bool enableBatchRenderer;
        uint32_t batchRenderViewWidth = 64;
        uint32_t batchRenderViewHeight = 64;
        madrona::render::APIBackend *extRenderAPI = nullptr;
        madrona::render::GPUDevice *extRenderDev = nullptr;
        float buttonWidth;
        float doorWidth;
        float rewardPerDist;
        float slackReward;
    };

    Manager(const Config &cfg);
    ~Manager();

    void step();

#ifdef MADRONA_CUDA_SUPPORT
    void gpuRolloutStep(cudaStream_t strm, void **rollout_buffers);
#endif

    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    madrona::py::Tensor checkpointResetTensor() const;
    madrona::py::Tensor checkpointTensor() const;
    madrona::py::Tensor resetTensor() const;
    madrona::py::Tensor actionTensor() const;
    madrona::py::Tensor rewardTensor() const;
    madrona::py::Tensor doneTensor() const;
    madrona::py::Tensor selfObservationTensor() const;
    madrona::py::Tensor partnerObservationsTensor() const;
    madrona::py::Tensor roomEntityObservationsTensor() const;
    madrona::py::Tensor roomDoorObservationsTensor() const;
    madrona::py::Tensor lidarTensor() const;
    madrona::py::Tensor stepsRemainingTensor() const;
    madrona::py::Tensor agentIDTensor() const;
    madrona::py::TrainInterface trainInterface() const;

    // These functions are used by the viewer to control the simulation
    // with keyboard inputs in place of DNN policy actions
    void triggerReset(int32_t world_idx);
    void setSaveCheckpoint(int32_t world_idx, int32_t value);
    void triggerLoadCheckpoint(int32_t world_idx);
    void setAction(int32_t world_idx,
                   int32_t agent_idx,
                   int32_t move_amount,
                   int32_t move_angle,
                   int32_t rotate,
                   int32_t interact);
    madrona::py::Tensor rgbTensor() const;
    madrona::py::Tensor depthTensor() const;

    madrona::render::RenderManager & getRenderManager();

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
