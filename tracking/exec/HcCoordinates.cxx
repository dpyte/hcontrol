#include <iostream>

#ifdef __x86_64__
#	include <immintrin.h>
#	define USING_INTRINSICS 1
#endif

#include <cmath>

#include "HcCoordinates.hxx"
#include "HcConfig.hxx"

namespace HC = HControl::Coordinates;

namespace {

#if IGNORE_Z
constexpr hc_auto hc_change_folds = 0.03f;
#else
constexpr hc_auto hc_change_folds = 0.07f;
#endif

/**
 * Calculate magnitude to check how far it has traveled and (!TODO) check
 * for incremental change in coordinates
 * !TODO: Implement ARM equivalent for this. Currently this code is being executed
 * on a modern ARM64 processor
 * inline this ... and try to squeeze as much performance it can provide
 * !TODO: benchmark this
 *
 * Ignore Z if it is way too far from the lock-in zone
 */

inline double hc_delta_x_y_z(const hc_axis_arr &axis, const hc_axis_arr &old_axis) {
	hc_auto x = std::pow(old_axis[0] - axis[0], 2);
	hc_auto y = std::pow(old_axis[1] - axis[1], 2);
#if IGNORE_Z
	hc_auto vect_mag = std::sqrt(x + y);
#else
	hc_auto z = std::pow(old_axis[2] - axis[2], 2);
	hc_auto vect_mag = std::sqrt(x + y + z);
#endif
	return vect_mag;
}
} // anon ns

HC::Coordinates::Coordinates(HControl::Coordinates::Point pt) noexcept
	:point(pt) {}

using hc_vec = std::array<std::pair<double, double>, 3>;
// Average Time: 1.95 seconds
// Sleep Time: 1-E6 seconds
void HC::Coordinates::append(const std::array<unsigned int, 2> &coord, hc_axis_arr const &plot_values) {
	coordinates = coord;
	auto update_value = hc_delta_x_y_z(plot_values, axis);
	update_to_new_position = update_value >= hc_change_folds;

	// Update only if the calculated difference is >= 7x folds
	// Although this value is development approximation therefore it can change to something else
	if (update_to_new_position) {
		// fprintf(stderr, "[%-3f, %-3f, %-3f] updated to [%-3f, %-3f, %-3f]\n",
		// 		axis[0], axis[1], axis[2], plot_values[0], plot_values[1], plot_values[2]);
		axis = plot_values;
		delta_requires_update = true;
	}
}

bool HC::Coordinates::hc_delta_ready() const {
	return update_to_new_position;
}

hc_axis_arr HC::Coordinates::axis_points() const {
	return axis;
}
