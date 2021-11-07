#ifndef HCEXECCONTROLLER_HXX
#define HCEXECCONTROLLER_HXX

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include "HcPoints.hxx"
#include "HcCoordinates.hxx"
#include "pybind11/pybind11.h"

namespace pyb = pybind11;
namespace HControl::Coordinates {

class HandLocation {
private:
	std::unordered_map<Point, std::shared_ptr<Coordinates>> coordinates = {
		std::make_pair(Wrist, 			  std::make_shared<Coordinates>(Wrist)),
		std::make_pair(THUMB_CMC,         std::make_shared<Coordinates>(THUMB_CMC)),
		std::make_pair(THUMB_MCP,         std::make_shared<Coordinates>(THUMB_MCP)),
		std::make_pair(INDEX_FINGER_MCP,  std::make_shared<Coordinates>(INDEX_FINGER_MCP)),
		std::make_pair(INDEX_FINGER_PIP,  std::make_shared<Coordinates>(INDEX_FINGER_PIP)),
		std::make_pair(MIDDLE_FINGER_MCP, std::make_shared<Coordinates>(MIDDLE_FINGER_MCP)),
		std::make_pair(MIDDLE_FINGER_PIP, std::make_shared<Coordinates>(MIDDLE_FINGER_PIP)),
	};

	// Declare some variable where a perfect scan can be stored at.
	// This will be used to measure angle of displacement

	hc_cord_arr lock_slope {};
	hc_cord_arr locked_thumb_mcp {};
	hc_cord_arr locked_wrist {};
	bool axis_lock_counter = false;
	bool coordinates_write_out = true;

	float lock_angle = 0.0f;
	float angle = 0.0f;

public:
	HandLocation() = default;
	// Passing too many values into the function will became a major hassle I guess ...
	void update_values(const pyb::list &update_values);
	// Implement some sort of gaurd so that this function is only being called once ...
	float hc_delta_theta() const;
	void enable_coordinates_write_out();
};
}

PYBIND11_MODULE(HandCoordinates, m) {
	pyb::class_<HControl::Coordinates::HandLocation>(m, "HandLocation").def(pyb::init<>())
		.def("update_values", &HControl::Coordinates::HandLocation::update_values)
		.def("hc_delta_theta", &HControl::Coordinates::HandLocation::hc_delta_theta)
		.def("enable_coordinates_write_out", &HControl::Coordinates::HandLocation::enable_coordinates_write_out);
}

#endif // HCEXECCONTROLLER_HXX
