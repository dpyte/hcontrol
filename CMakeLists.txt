# $hcontrol$

cmake_minimum_required (VERSION 3.16)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message (STATUS "No build type selected, default to Debug")
	set (CMAKE_BUILD_TYPE "Debug" CACHE
		STRING "Build type default to Debug" FORCE)
	set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
		"Debug"
		"Release"
		"RelWithDebInfo")
endif ()

set (HCONTROL_VERSION_MAJOR   "0" CACHE STRING "Major candidate value")
set (HCONTROL_VERSION_RELEASE "0" CACHE STRING "Release candidate value")
set (HCONTROL_VERSION_PHASE   "0" CACHE STRING "Development phase")

set (HCONTROL_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE PATH "Path to cmake directory")
list (APPEND CMAKE_MODULE_PATH ${HCONTROL_CMAKE_DIR})

project (HCONTROL
	VERSION ${HCONTROL_VERSION_MAJOR}.${HCONTROL_VERSION_RELEASE}.${HCONTROL_VERSION_PHASE}
	LANGUAGES C CXX ASM)

if (MSVC)
	message (FATAL_ERROR "Current platform is not supported, yet")
endif ()


# Use ccache to speed up recompilation
include (ExternalProject)
find_program (CCACHE_PROGRAM ccache)
mark_as_advanced (FORCE CCACHE_PROGRAM)

if (CCACHE_PROGRAM)
	message (STATUS "found ccache: ${CCACHE_PROGRAM}")
	set (CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
	set (CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else ()
	message (STATUS "could not find ccache in the system")
endif ()

## print info
message (STATUS "build info:")
message (STATUS "  Host System: ${CMAKE_HOST_SYSTEM}")
message (STATUS "  Host System Name: ${CMAKE_HOST_SYSTEM_NAME}")
message (STATUS "  Host System Processor: ${CMAKE_HOST_SYSTEM_PROCESSOR}")

option (EXPORT_COMPILE_COMMANDS    "cmake export compile commands" ON)

# set language requirements
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

if (EXPORT_COMPILE_COMMANDS)
	set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif ()

# Enable cache usage when compiling
find_program (CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
	set_property (GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif ()

# Check for SIMD instructions
include (CheckCXXCompilerFlag)
include (CheckIncludeFile)

check_cxx_compiler_flag ("-march=native" HAS_NATIVE_SUPPORT)

check_include_file ("float.h;stdlib.h;string.h" STDC_HEADERS)
check_include_file ("xmmintrin.h" HAVE_XMMINTRIN_H)
check_include_file ("immintrin.h" HAVE_IMMINTRIN_H)


if (HAS_NATIVE_SUPPORT)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()

# Make executable available within the build directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory (tracking/exec)
add_subdirectory (daemon)
