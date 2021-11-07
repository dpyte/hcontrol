
#include <cmath>
#include <fstream>
#include <functional>
#include <map>

#include "HCExecController.hxx"
#include "HcConfig.hxx"

namespace HC = HControl::Coordinates;

namespace {

constexpr hc_auto HcPi = 3.14159265;

std::map<unsigned int, HC::Point> registered_landmark_points = {
	std::make_pair(0, static_cast<HC::Point>(0)),
	std::make_pair(1, static_cast<HC::Point>(1)),
	std::make_pair(5, static_cast<HC::Point>(5)),
	std::make_pair(6, static_cast<HC::Point>(6)),
	std::make_pair(9, static_cast<HC::Point>(9)),
	std::make_pair(10, static_cast<HC::Point>(10)),
};

template <typename T = unsigned int>
HC::Point map_value(T value) {
	if (value <= 20 && value >= 0)
		return static_cast<HC::Point>(value);
	return {};
}

std::array<unsigned int, 2> extract_pyb_coordinates(const pyb::tuple &coord) {
	std::array<unsigned int, 2> retval{};
	retval[0] = coord[0].cast<unsigned int>();
	retval[1] = coord[1].cast<unsigned int>();
	return retval;
}

hc_axis_arr extract_pyb_axis(const pyb::list &raw_axis) {
	hc_axis_arr retval{};
	retval[0] = raw_axis[0].cast<float>();
	retval[1] = raw_axis[1].cast<float>();
	retval[2] = raw_axis[2].cast<float>();
	return retval;
}

bool find_within_registered_points(unsigned int pt) {
	hc_auto search = registered_landmark_points.find(pt);
	return search != registered_landmark_points.end();
}

void writef(const HC::Point point, const std::array<unsigned int, 2> &coord,
            const hc_axis_arr &arr) {
	std::ofstream file;
	file.open("COORDINATES.txt", std::ofstream::out | std::ofstream::app);
	file << point << ',' << coord[0] << ',' << coord[1] << ',' << arr[0] << ','
	     << arr[1] << ',' << arr[2] << '\n';
	file.close();
}

inline bool hc_check_for_of(const hc_axis_arr &arr) {
	return !(arr[0] > 1 || arr[1] > 1 || arr[2] > 1);
}
} // namespace

void HC::HandLocation::update_values(const pyb::list &updated_values) {
	hc_auto wrist = coordinates[Wrist].get();
	hc_auto thumb_cmc = coordinates[THUMB_CMC].get();
	hc_auto idx_mcp = coordinates[INDEX_FINGER_MCP].get();
	hc_auto mid_mcp = coordinates[MIDDLE_FINGER_MCP].get();

	unsigned int axis_pass_count = 0;
	for (hc_auto &it : updated_values) {
		Point point;
		std::array<unsigned int, 2> coords;
		hc_axis_arr axis;
		bool update = true;

		hc_auto dict_value = pyb::cast<pyb::dict>(it);
		hc_auto registered_point_value = std::stoi(std::string(pyb::str(dict_value.begin()->second)));

		auto iter = dict_value.begin();
		point = map_value(registered_point_value);
		iter++;
		update &= find_within_registered_points(map_value(point));
		if (update) {
			coords = extract_pyb_coordinates(pyb::cast<pyb::tuple>(iter->second));
			iter++;
			axis = extract_pyb_axis(pyb::cast<pyb::list>(iter->second));
			update &= hc_check_for_of(axis);
		}

		if (update) {
			auto coordinate_point = coordinates[point].get();
			coordinate_point->append(coords, axis);
			axis_pass_count++;
			if (coordinates_write_out)
				writef(point, coords, axis);
		}
		if (point == THUMB_CMC && update && (axis_lock_counter % 2 == 0)) {
			std::cerr << "Locked axis: " << axis[0] << ' ' << axis[1] << '\n';
			locked_thumb_mcm = axis;
			axis_lock_counter++;
		}
	}

	if (axis_pass_count == 6) {
		hc_auto thmcm = thumb_cmc->axis_points();
		// std::fprintf(stderr, "[%0.4f, %0.4f] [%0.4f, %0.4f]\n",
		// 		locked_thumb_mcm[0], locked_thumb_mcm[1], thmcm[0], thmcm[1]);
		hc_auto uv = (locked_thumb_mcm[0] * thmcm[0]) + (locked_thumb_mcm[1] * thmcm[1]);
		hc_auto u_component = std::sqrt(std::pow(locked_thumb_mcm[0], 2) + std::pow(locked_thumb_mcm[1], 2));
		hc_auto v_component = std::sqrt(std::pow(thmcm[0], 2) + std::pow(thmcm[1], 2));
		angle = std::acos(uv / (u_component * v_component));
		std::fprintf(stderr, "Angle: %0.4f\n", angle);
	}
}

float HC::HandLocation::hc_delta_theta() const {
	return angle;
}

void HC::HandLocation::enable_coordinates_write_out() {
	coordinates_write_out = true;
}

