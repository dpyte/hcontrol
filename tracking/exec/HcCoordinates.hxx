#ifndef HCCOORDINATES_HXX
#define HCCOORDINATES_HXX

#include <array>

#include "HcPoints.hxx"
#include "HcConfig.hxx"

namespace HControl::Coordinates {

class Coordinates {
private:
	const Point point;
	hc_axis_arr axis {};
	hc_cord_arr coordinates {};

	// To update when it is performing an initial run
	bool delta_requires_update = false;
	bool update_to_new_position = false;

public:
	Coordinates();
	explicit Coordinates(Point pt) noexcept;
	void append(hc_cord_arr const &coord, hc_axis_arr const &plot_values);

	[[nodiscard]] hc_axis_arr axis_points() const;
	[[nodiscard]] hc_cord_arr coordinates_points() const;
};

}

#endif // HCCOORDINATES_HXX
