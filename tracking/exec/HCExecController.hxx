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
		std::make_pair(THUMB_IP,          std::make_shared<Coordinates>(THUMB_IP)),
		std::make_pair(THUMB_MCP,         std::make_shared<Coordinates>(THUMB_MCP)),
		std::make_pair(THUMB_TIP,         std::make_shared<Coordinates>(THUMB_TIP)),
		std::make_pair(INDEX_FINGER_MCP,  std::make_shared<Coordinates>(INDEX_FINGER_MCP)),
		std::make_pair(INDEX_FINGER_PIP,  std::make_shared<Coordinates>(INDEX_FINGER_PIP)),
		std::make_pair(INDEX_FINGER_DIP,  std::make_shared<Coordinates>(INDEX_FINGER_DIP)),
		std::make_pair(INDEX_FINGER_TIP,  std::make_shared<Coordinates>(INDEX_FINGER_TIP)),
		std::make_pair(MIDDLE_FINGER_MCP, std::make_shared<Coordinates>(MIDDLE_FINGER_MCP)),
		std::make_pair(MIDDLE_FINGER_PIP, std::make_shared<Coordinates>(MIDDLE_FINGER_PIP)),
		std::make_pair(MIDDLE_FINGER_DIP, std::make_shared<Coordinates>(MIDDLE_FINGER_DIP)),
		std::make_pair(MIDDLE_FINGER_TIP, std::make_shared<Coordinates>(MIDDLE_FINGER_TIP)),
		std::make_pair(RING_FINGER_MCP,   std::make_shared<Coordinates>(RING_FINGER_MCP)),
		std::make_pair(RING_FINGER_PIP,   std::make_shared<Coordinates>(RING_FINGER_PIP)),
		std::make_pair(RING_FINGER_DIP,   std::make_shared<Coordinates>(RING_FINGER_DIP)),
		std::make_pair(RING_FINGER_TIP,   std::make_shared<Coordinates>(RING_FINGER_TIP)),
		std::make_pair(PINKY_MCP,         std::make_shared<Coordinates>(PINKY_MCP)),
		std::make_pair(PINKY_PIP,         std::make_shared<Coordinates>(PINKY_PIP)),
		std::make_pair(PINKY_DIP,         std::make_shared<Coordinates>(PINKY_DIP)),
		std::make_pair(PINKY_TIP,         std::make_shared<Coordinates>(PINKY_TIP)),
	};

	bool recv_terminate_signal = false;
	bool coordinates_write_out = true;
	void async_controller();

public:
	HandLocation() = default;
	// Passing too many values into the function will became a major hassle I guess ...
	void update_values(const pyb::dict &update_values);
	// Implement some sort of guard so that this function is only being called once ...
	void take_action();

	void enable_coordinates_write_out();
};
}

PYBIND11_MODULE(HandCoordinates, m) {
	pyb::class_<HControl::Coordinates::HandLocation>(m, "HandLocation").def(pyb::init<>())
		.def("update_values", &HControl::Coordinates::HandLocation::update_values)
		.def("take_action", &HControl::Coordinates::HandLocation::take_action)
		.def("enable_coordinates_write_out", &HControl::Coordinates::HandLocation::enable_coordinates_write_out);
}

#endif // HCEXECCONTROLLER_HXX
