
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

template <typename T = double>
inline T hc_squared(T exp) { return exp * exp; }

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

hc_cord_arr hc_slope(hc_cord_arr const &begin, hc_cord_arr const &end) {
	hc_cord_arr retval { end[0] - begin[0], end[1] - begin[1] };
	return retval;
}

} // namespace

void HC::HandLocation::update_values(const pyb::list &updated_values) {
	unsigned int axis_pass_count = 0;
	hc_auto wrist = coordinates[Wrist].get();
	hc_auto thumb_mcm = coordinates[THUMB_CMC].get();

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
		if (point == THUMB_CMC && update && !axis_lock_counter) locked_thumb_mcm = coords;
		if (point == Wrist && update && !axis_lock_counter) locked_wrist = coords;
	}

	if (!axis_lock_counter) {
		std::fprintf(stderr, "Old Coordinates: [%d, %d] [%d, %d]\n", locked_wrist[0], locked_wrist[1],
				locked_thumb_mcm[0], locked_thumb_mcm[1]);
		hc_auto wrst = wrist->coordinates_points();
		hc_auto thmb = thumb_mcm->coordinates_points();
		lock_slope = hc_slope(wrst, thmb);
		axis_lock_counter = true;
	}

	if (axis_pass_count == 6) {
		hc_auto thmcm = thumb_mcm->coordinates_points();
		hc_auto wrst  = wrist->coordinates_points();
		hc_auto axis_slope = hc_slope(wrst, thmcm);

		hc_auto rise = locked_wrist[1] + axis_slope[1];
		hc_auto run  = rise + axis_slope[0];
		hc_cord_arr const new_coords = {run, rise};

		hc_auto uv = (locked_thumb_mcm[0] * new_coords[0]) + (locked_thumb_mcm[0] * new_coords[0]);
		hc_auto u_component = std::sqrt(std::pow(locked_thumb_mcm[0], 2) + std::pow(locked_thumb_mcm[1], 2));
		hc_auto v_component = std::sqrt(std::pow(new_coords[0], 2) + std::pow(new_coords[1], 2));
		angle = std::acos(uv / (u_component * v_component)) * (180.0 / HcPi) - 5;

		// std::fprintf(stderr, "slope: [%d, %d] [%d, %d] @ Angle [%0.4f]\n",
		// 		lock_slope[0], lock_slope[1], new_coords[0], new_coords[1], angle);
	}
}

float HC::HandLocation::hc_delta_theta() const { return angle; }

void HC::HandLocation::enable_coordinates_write_out() { coordinates_write_out = true; }

