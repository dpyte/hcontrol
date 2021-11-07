#ifndef HCCONFIG_HXX
#define HCCONFIG_HXX

#ifdef hc_auto
#	undef hc_auto
#endif

#define hc_auto auto const

#include <array>
using hc_axis_arr = std::array<double, 2>;
using hc_cord_arr = std::array<unsigned int, 2>;

#endif // HCCONFIG_HXX
