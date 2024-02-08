#pragma once

#include <cstdint>

namespace madEscape {

enum class SimFlags : uint32_t {
    Default                = 0,
    UseFixedWorld          = 1 << 0,
    StartInDiscoveredRooms = 1 << 1,
    UseComplexLevel        = 1 << 2,
};

enum class RewardMode : uint32_t {
    OG,
    Dense1,
    Dense2,
    Dense3,
    Sparse1,
    Sparse2,
    Complex,
    Sparse3,
};

inline SimFlags & operator|=(SimFlags &a, SimFlags b);
inline SimFlags operator|(SimFlags a, SimFlags b);
inline SimFlags & operator&=(SimFlags &a, SimFlags b);
inline SimFlags operator&(SimFlags a, SimFlags b);

}

#include "sim_flags.inl"
