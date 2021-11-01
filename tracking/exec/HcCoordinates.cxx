#include <iostream>

#ifdef __x86_64__
#	include <immintrin.h>
#	define USING_INTRINSICS 1
#endif

#include "HcCoordinates.hxx"
#include "HcConfig.hxx"

namespace HC = HControl::Coordinates;

namespace {

constexpr hc_auto hc_change_folds = 0.07f;

// Calculate magnitude to check how far it has traveled and (!TODO) check for incremental change in coordinates
// !TODO: Implement ARM equivalent for this. Currently this code is being executed on a modern ARM64 processor
// inline this ... and try to squeeze as much performance it can provide
// !TODO: benchmark this
double hc_delta_x_y_z(const hc_axis_arr &axis, const hc_axis_arr &old_axis) {
	auto old_axis_1 = _mm256_set_pd(old_axis[0], old_axis[1], old_axis[2], 0.0f);
	auto old_axis_2 = _mm256_set_pd(old_axis[0], old_axis[1], old_axis[2], 0.0f);
	auto old_axis_sqrd = _mm256_mul_pd(old_axis_1, old_axis_2);
	auto intrin_old_magntd = _mm256_sqrt_pd(old_axis_sqrd);

	std::fprintf(stderr, "Old Coordinates: %-4f %-4f %-4f\n", old_axis[0], old_axis[1], old_axis[2]);
	std::fprintf(stderr, "New Coordinates: %-4f %-4f %-4f\n", axis[0], axis[1], axis[2]);

	auto axis_1 = _mm256_set_pd(axis[0], axis[1], axis[2], 1);
	auto axis_2 = _mm256_set_pd(axis[0], axis[1], axis[2], 1);
	auto axis_sqrd = _mm256_mul_pd(axis_1, axis_2);
	auto intrin_new_magntd = _mm256_sqrt_pd(axis_sqrd);

	hc_auto old_magntd = *((double*)&intrin_old_magntd);
	hc_auto new_magntd = *((double*)&intrin_new_magntd);
	std::fprintf(stderr, "Magnitude: %-4f, %-4f\n", old_magntd, new_magntd);
	return old_magntd - new_magntd;
}

} // anon ns

HC::Coordinates::Coordinates(HControl::Coordinates::Point pt) noexcept
	:point(pt) {}

using hc_vec = std::array<std::pair<double, double>, 3>;
// Average Time: 1.95 seconds
// Sleep Time: 1-E6 seconds
void HC::Coordinates::append(const std::array<unsigned int, 2> &coord, hc_axis_arr const &plot_values) {
	coordinates = coord;
	auto can_update = hc_delta_x_y_z(plot_values, axis);
	update_to_new_position = can_update >= hc_change_folds;

	// Update only if the calculated difference is >= 7x folds
	// Although this value is development approximation therefore it can change to something else
	if (update_to_new_position || !delta_requires_update) {
		axis = plot_values;
		change = can_update;
		delta_requires_update = true;
	}
}

hc_axis_arr HC::Coordinates::axis_points() const {
	return axis;
}
