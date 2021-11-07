#include <cmath>
#include <iostream>

#ifdef __x86_64__
#	include <immintrin.h>
#endif

#include "HcCoordinates.hxx"
#include "HcConfig.hxx"

namespace HC = HControl::Coordinates;

namespace {
constexpr hc_auto hc_change_folds = 2;

template <typename T = unsigned int>
inline T hc_squared(T t) { return static_cast<T>(t * t); }

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
inline unsigned int hc_delta_x_y(const hc_cord_arr &cord, const hc_cord_arr &old_cord) {
	hc_auto x = hc_squared(old_cord[0] - cord[0]);
	hc_auto y = hc_squared(old_cord[1] - cord[1]);
	hc_auto vect_mag = std::sqrt(x + y);
	return vect_mag;
}
} // anon ns

HC::Coordinates::Coordinates(HControl::Coordinates::Point pt) noexcept
	:point(pt) {}

void HC::Coordinates::append(const std::array<unsigned int, 2> &coord, hc_axis_arr const &plot_values) {
	auto update_value = hc_delta_x_y(coord, coordinates);
	update_to_new_position = update_value >= hc_change_folds;
	if (update_to_new_position) {
		// std::fprintf(stderr, "updating [%d, %d] to [%d, %d]\n", coordinates[0], coordinates[1], coord[0], coord[1]);
		coordinates = coord;
		axis = plot_values;
		delta_requires_update = true;
	}
}

hc_axis_arr HC::Coordinates::axis_points() const { return axis; }

hc_cord_arr HC::Coordinates::coordinates_points() const { return coordinates; }

