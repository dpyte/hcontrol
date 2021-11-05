#include <iostream>

#include "HcCoordinates.hxx"
#include "HcConfig.hxx"

namespace HC = HControl::Coordinates;

HC::Coordinates::Coordinates(HControl::Coordinates::Point pt) noexcept
	:point(pt) {}

using hc_vec = std::array<std::pair<float, float>, 3>;
// Average Time: 1.95 seconds
// Sleep Time: 1-E6 seconds
void HC::Coordinates::append(const std::array<unsigned int, 2> &coord, const std::array<float, 3> &plot_values) {
	// Not sure what use the coordinate vector will be useful for
	// I'll just move on to using the axis values as they more relevant
	// std::cerr << "\nPoint: " << point << '\n'
	// 	<< "axis[0]: " << plot_values[0] << '\n'
	// 	<< "axis[1]: " << plot_values[1] << '\n'
	// 	<< "axis[2]: " << plot_values[2] << '\n';

	// coordinates = coord;
	// hc_vec vec = {
	// 	std::make_pair(axis[0], plot_values[0]),
	// 	std::make_pair(axis[1], plot_values[1]),
	// 	std::make_pair(axis[2], plot_values[2]),
	// };

	// axis = plot_values;
	// Update only if the calculated difference is >= 7x folds
}

std::array<float, 3> HC::Coordinates::axis_points() const {
	return axis;
}
