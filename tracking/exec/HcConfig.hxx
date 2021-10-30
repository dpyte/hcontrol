#ifndef HCCONFIG_HXX
#define HCCONFIG_HXX



#ifdef hc_auto
#	undef hc_auto
#endif

#define hc_auto auto const

#include <array>
using hc_axis_arr = std::array<double, 3>;

#endif // HCCONFIG_HXX
