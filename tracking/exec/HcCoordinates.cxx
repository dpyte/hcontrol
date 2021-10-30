#include <iostream>

#ifdef __x86_64__
#	include <immintrin.h>
#	define USING_INTRINSICS 1
#endif

#include "HcCoordinates.hxx"
#include "HcConfig.hxx"

namespace HC = HControl::Coordinates;

namespace {

static constexpr hc_auto hc_change_folds = 0.07f;

// !TODO: Implement ARM equivalent for this. Currently this code is being executed on a modern ARM64 processor
// inline this ... and try to squeeze as much performance it can provide
// !TODO: benchmark this
double *hc_change_in_vector(const hc_axis_arr &axis, const hc_axis_arr &old_axis) {
	double *magntd = nullptr;
	auto old_axis_1 = _mm256_set_pd(old_axis[0], old_axis[1], old_axis[2], 0.0f);
	auto old_axis_2 = _mm256_set_pd(old_axis[0], old_axis[1], old_axis[2], 0.0f);
	auto old_axis_sqrd = _mm256_mul_pd(old_axis_1, old_axis_2);
	auto intrin_magnitude = _mm256_sqrt_pd(old_axis_sqrd);
	magntd = (double*)&intrin_magnitude;
	return magntd;
}

} // anon ns

HC::Coordinates::Coordinates(HControl::Coordinates::Point pt) noexcept
	:point(pt) {}

using hc_vec = std::array<std::pair<double, double>, 3>;
// Average Time: 1.95 seconds
// Sleep Time: 1-E6 seconds
void HC::Coordinates::append(const std::array<unsigned int, 2> &coord, const hc_axis_arr &plot_values) {
	// Not sure what use the coordinate vector will be useful for
	// I'll just move on to using the axis values as they more relevant
	// std::cerr << "\nPoint: " << point << '\n'
	// 	<< "axis[0]: " << plot_values[0] << '\n'
	// 	<< "axis[1]: " << plot_values[1] << '\n'
	// 	<< "axis[2]: " << plot_values[2] << '\n';

	coordinates = coord;
	auto can_update = hc_change_in_vector(plot_values, axis);
	update_to_new_position = *can_update >= hc_change_folds;

	// Update only if the calculated difference is >= 7x folds
	// Although this value is developmemnt approximation therefore it can change to something else
	if (update_to_new_position) {
		axis = plot_values;
		change = can_update;
	}
}

hc_axis_arr HC::Coordinates::axis_points() const {
	return axis;
}
