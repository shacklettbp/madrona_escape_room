namespace madEscape {

inline SimFlags & operator|=(SimFlags &a, SimFlags b)
{
    a = SimFlags(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

inline SimFlags operator|(SimFlags a, SimFlags b)
{
    a |= b;

    return a;
}

inline SimFlags & operator&=(SimFlags &a, SimFlags b)
{
    a = SimFlags(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    return a;
}

inline SimFlags operator&(SimFlags a, SimFlags b)
{
    a &= b;

    return a;
}

}
