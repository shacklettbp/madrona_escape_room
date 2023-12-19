#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

namespace nb = nanobind;

using namespace madrona::py;

namespace madEscape {

// This file creates the python bindings used by the learning code.
// Refer to the nanobind documentation for more details on these functions.
NB_MODULE(madrona_escape_room, m) {
    // Each simulator has a madrona submodule that includes base types
    // like Tensor and PyExecMode.
    setupMadronaSubmodule(m);

    nb::enum_<SimFlags>(m, "SimFlags", nb::is_arithmetic())
        .value("Default", SimFlags::Default)
        .value("UseFixedWorld", SimFlags::UseFixedWorld)
        .value("StartInDiscoveredRooms", SimFlags::StartInDiscoveredRooms)
        .value("UseComplexLevel", SimFlags::UseComplexLevel)
    ;

    nb::enum_<RewardMode>(m, "RewardMode")
        .value("OG", RewardMode::OG)
        .value("Dense1", RewardMode::Dense1)
        .value("Dense2", RewardMode::Dense2)
        .value("Dense3", RewardMode::Dense3)
        .value("Sparse1", RewardMode::Sparse1)
        .value("Sparse2", RewardMode::Sparse2)
        .value("Complex", RewardMode::Complex)
    ;

    auto mgr_class = nb::class_<Manager> (m, "SimManager")
        .def("__init__", [](Manager *self,
                            PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            bool auto_reset,
                            uint32_t sim_flags,
                            RewardMode reward_mode,
                            bool enable_batch_renderer) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .autoReset = auto_reset,
                .simFlags = SimFlags(sim_flags),
                .rewardMode = reward_mode,
                .enableBatchRenderer = enable_batch_renderer,
            });
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("auto_reset"),
           nb::arg("sim_flags"),
           nb::arg("reward_mode"),
           nb::arg("enable_batch_renderer") = false)
        .def("step", &Manager::step)
        .def("checkpoint_reset_tensor", &Manager::checkpointResetTensor)
        .def("checkpoint_tensor", &Manager::checkpointTensor)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("self_observation_tensor", &Manager::selfObservationTensor)
        .def("partner_observations_tensor", &Manager::partnerObservationsTensor)
        .def("room_entity_observations_tensor",
             &Manager::roomEntityObservationsTensor)
        .def("room_door_observation_tensor",
             &Manager::roomDoorObservationsTensor)
        .def("lidar_tensor", &Manager::lidarTensor)
        .def("steps_remaining_tensor", &Manager::stepsRemainingTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
    ;
}

}
