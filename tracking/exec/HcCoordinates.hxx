#ifndef HCCOORDINATES_HXX
#define HCCOORDINATES_HXX

#include <array>
#include "HcPoints.hxx"

namespace HControl::Coordinates {
class Coordinates {
private:
	const Point point;
	std::array<unsigned int, 2> coordinates;
	std::array<float, 3> axis;
	bool update_to_new_position = false;

public:
	Coordinates();
	explicit Coordinates(Point pt) noexcept;
	void append(std::array<unsigned int, 2> const &coord, std::array<float, 3> const &plot_values);

	std::array<float, 3> axis_points() const;
};

}

#endif // HCCOORDINATES_HXX
