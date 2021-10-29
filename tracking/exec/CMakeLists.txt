# $hydra$:%lerna%
cmake_minimum_required (VERSION 3.16)

add_subdirectory (pybind11)

find_package (PythonLibs 3 REQUIRED)
include_directories (${PYTHON_INCLUDE_DIRS})

set (HCLIB HandCoordinates CACHE STRING "program name")

set (sources
	HCExecController.hxx    HCExecController.cxx
	HcCoordinates.hxx		HcCoordinates.cxx
	HcPoints.hxx
	)

set (lib_srcs ${sources})

message (STATUS "Building python client frontend")
add_library (
	${HCLIB}
	SHARED
	${lib_srcs}
	)

target_link_libraries (
	${HCLIB}
	PRIVATE
	)

target_include_directories (
	${HCLIB}
	PRIVATE
	${pybind11_INCLUDE_DIR}
	)

set_target_properties (${HCLIB} PROPERTIES SUFFIX .so)
set_target_properties (${HCLIB} PROPERTIES PREFIX "")
