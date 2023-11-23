#pragma once
#ifdef madrona_3d_example_mgr_EXPORTS
#define MGR_EXPORT MADRONA_EXPORT
#else
#define MGR_EXPORT MADRONA_IMPORT
#endif

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/mw.hpp>
#include <madrona/viz/system.hpp>

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
    };

    MGR_EXPORT Manager(
        const Config &cfg,
        const madrona::viz::VizECSBridge *viz_bridge = nullptr);
    MGR_EXPORT ~Manager();

    MGR_EXPORT void step();

#ifdef MADRONA_CUDA_SUPPORT
    MGR_EXPORT void gpuRolloutStep(cudaStream_t strm, void **rollout_buffers);
#endif

    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    MGR_EXPORT madrona::py::Tensor checkpointResetTensor() const;
    MGR_EXPORT madrona::py::Tensor checkpointTensor() const;
    MGR_EXPORT madrona::py::Tensor resetTensor() const;
    MGR_EXPORT madrona::py::Tensor actionTensor() const;
    MGR_EXPORT madrona::py::Tensor rewardTensor() const;
    MGR_EXPORT madrona::py::Tensor doneTensor() const;
    MGR_EXPORT madrona::py::Tensor selfObservationTensor() const;
    MGR_EXPORT madrona::py::Tensor partnerObservationsTensor() const;
    MGR_EXPORT madrona::py::Tensor roomEntityObservationsTensor() const;
    MGR_EXPORT madrona::py::Tensor doorObservationTensor() const;
    MGR_EXPORT madrona::py::Tensor lidarTensor() const;
    MGR_EXPORT madrona::py::Tensor stepsRemainingTensor() const;
    MGR_EXPORT madrona::py::Tensor agentIDTensor() const;
    MGR_EXPORT madrona::py::TrainInterface trainInterface() const;

    // These functions are used by the viewer to control the simulation
    // with keyboard inputs in place of DNN policy actions
    MGR_EXPORT void triggerReset(int32_t world_idx);
    MGR_EXPORT void setSaveCheckpoint(int32_t world_idx, int32_t value);
    MGR_EXPORT void triggerLoadCheckpoint(int32_t world_idx);
    MGR_EXPORT void setAction(int32_t world_idx,
                              int32_t agent_idx,
                              int32_t move_amount,
                              int32_t move_angle,
                              int32_t rotate,
                              int32_t grab,
                              int32_t jump);

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
