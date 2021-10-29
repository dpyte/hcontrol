#include <functional>
#include <future>
#include <map>
#include <thread>
#include <fstream>

#include "HcConfig.hxx"
#include "HCExecController.hxx"

namespace HC = HControl::Coordinates;

namespace {
HC::Point map_value(unsigned int value) {
	if (value <= 20 && value >= 0)
		return static_cast<HC::Point>(value);
	return {};
}

std::array<unsigned int, 2> extract_pyb_coordinates(const pyb::tuple &coord) {
	std::array<unsigned int, 2> retval;
	retval[1] = coord[1].cast<unsigned int>();
	retval[0] = coord[0].cast<unsigned int>();
	return retval;
}

std::array<float, 3> extract_pyb_axis(const pyb::list &raw_axis) {
	std::array<float, 3> retval;
	retval[0] = raw_axis[0].cast<float>();
	retval[1] = raw_axis[1].cast<float>();
	retval[2] = raw_axis[2].cast<float>();
	return retval;
}

// Modify this buffer as required
// Right now, the only points it has to track are the listed below.
// Using double tap from the finger as saving buffer
std::array<HC::Point, 13> registered_landmark_pinouts = {
	// Mid -> Required
	map_value(0),
	// Thumb
	map_value(2),
	map_value(1),
	// Mid
	map_value(10),
	map_value(9),
	// Ring
	map_value(14),
	map_value(13),
};

std::map<unsigned int, HC::Point> registered_landmark_points = {
	std::make_pair(0, static_cast<HC::Point>(0)),
	std::make_pair(5, static_cast<HC::Point>(5)),
	std::make_pair(6, static_cast<HC::Point>(6)),
	std::make_pair(9, static_cast<HC::Point>(9)),
	std::make_pair(10, static_cast<HC::Point>(10)),
};

// Undefined behavior
// !FIXME
bool find_within_registered_points(unsigned int pt) {
	hc_auto search = registered_landmark_points.find(pt);
	return search != registered_landmark_points.end();
}

void writef(const HC::Point point, const std::array<unsigned int, 2> &coord, const std::array<float, 3> &arr) {
	std::ofstream file;
	file.open("COORDINATES.txt", std::ofstream::out | std::ofstream::app);
	file << point << ',' << coord[0]
		<< ',' << coord[1]
		<< ',' << arr[0]
		<< ',' << arr[1]
		<< ',' << arr[2] << '\n';
	file.close();
}

} // namespace


void HC::HandLocation::enable_coordinates_write_out() {
	coordinates_write_out = true;
}

void HC::HandLocation::async_controller() {
while (!recv_terminate_signal) {
	for (const auto &registered_landmarks: registered_landmark_pinouts) {

	}
	std::this_thread::sleep_for(std::chrono::microseconds(1));
}
}

void HC::HandLocation::update_values(const pyb::dict &updated_values) {
Point point;
std::array<unsigned int, 2> coordnts;
	std::array<float, 3> axis;
	for (const auto &it: updated_values) {
		auto const key = std::string(pyb::str(it.first));
		if (key == "point") {
			auto const raw_uint_point = std::stoi(std::string(pyb::str(it.second)));
			point = map_value(raw_uint_point);
		} else if (key == "coordinates") {
			auto const raw_und_coord = pyb::cast<pyb::tuple>(it.second);
			coordnts = extract_pyb_coordinates(raw_und_coord);
		} else if (key == "axis") {
			auto const raw_und_axis = pyb::cast<pyb::list>(it.second);
			axis = extract_pyb_axis(raw_und_axis);
		}
		if (coordinates.find(point) != coordinates.end()) {
			auto coorinates_ptr = coordinates[point].get();
			coorinates_ptr->append(coordnts, axis);
			if (coordinates_write_out && find_within_registered_points(point)) writef(point, coordnts, axis);
		}
	}
}

void HC::HandLocation::take_action() {
	std::thread([&](){ async_controller(); }).detach();
}
