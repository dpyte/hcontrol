
#include <iostream>

#include "HcCoordinates.hxx"


namespace HC = HControl::Coordinates;

HC::Coordinates::Coordinates(HControl::Coordinates::Point pt) noexcept
	:point(pt) {}

// Not sure what use the coordinate vector will be useful for
// I'll just move on to using the axis values as they more relevant
// Average Time: 1.95 seconds
// Sleep Time: 1-E6 seconds
void HC::Coordinates::append(const std::array<unsigned int, 2> &coord, const std::array<float, 3> &plot_values) {
	coordinates = coord;
	axis = plot_values;
	// Update only if the calculated difference is >= 7x folds
}
