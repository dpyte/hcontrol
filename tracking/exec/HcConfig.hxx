#ifndef HCCONFIG_HXX
#define HCCONFIG_HXX

#ifdef hc_auto
#	undef hc_auto
#endif

#define hc_auto auto const

#ifdef IGNORE_Z
#	undef IGNORE_Z
#endif

#define IGNORE_Z 1

#include <array>
#if IGNORE_Z
using hc_axis_arr = std::array<double, 2>;
#else
using hc_axis_arr = std::array<double, 3>;
#endif

#endif // HCCONFIG_HXX
