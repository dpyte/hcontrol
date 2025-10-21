cmake_minimum_required (VERSION 3.20)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (${CMAKE_SOURCE_DIR} STREQUAL {CMAKE_BINARY_DIR})
	message(FATAL_ERROR "In-source build is not allowed, please create a separate build folder.")
endif ()

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(FATAL_ERROR "Cannot compile this project on 32-bit architecture")
endif ()

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
set (CMAKE_CXX_STANDARD 20)
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

############################################################
# Set compiler flags

if (MSVC)
	if (WARN_UNUSED)
		add_compile_options(/W4)
	endif ()

	if (CMAKE_LTO)
		add_compile_options(/GL)
		add_link_options(/LTCG)
	endif ()

	if (ENABLE_EXTRA_DEBUG)
		add_compile_definitions(DEBUG)
		add_compile_options(/Od /Zi /W4 /permissive-)
	endif ()

	if (ENABLE_CLANG_TIDY)
		set (CMAKE_CXX_CLANG_TIDY "clang-tidy")
	endif()

	add_compile_options(/EHsc) # Enable standard C++ exceptions
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

	include(CheckCXXCompilerFlag)
	check_cxx_compiler_flag(/openmp SUPPORTS_OPENMP)
	if (SUPPORTS_OPENMP)
		add_compile_options(/openmp)
	endif ()

else ()
	if (${WARN_UNUSED})
		set (CXX_EXTRA "${CXX_EXTRA} -Wunused")
	endif ()

	if (CMAKE_LTO)
		set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
	endif ()

	if (${ENABLE_EXTRA_DEBUG})
		set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -DDEBUG -Wall -Wnarrowing -Wconversion -pedantic ${CXX_EXTRA}")
		set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
	endif ()

	if (ENABLE_CLANG_TIDY)
		set (CMAKE_CXX_CLANG_TIDY "clang-tidy")
	endif()

	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops ${CXX_EXTRA} -fopenmp")

	include(CheckCXXCompilerFlag)
	check_cxx_compiler_flag (-fno-builtin HAVE_FNO_BUILTIN)
	check_cxx_compiler_flag (-fstack-protector-strong HAS_STACK_PROTECTOR_STRONG)
	check_cxx_compiler_flag (-march=native SUPPORTS_MARCH_NATIVE)
	check_cxx_compiler_flag (-mtune=native SUPPORTS_MTUNE_NATIVE)
	check_cxx_compiler_flag (-mavx HAVE_AVX)
	check_cxx_compiler_flag (-mavx512f HAVE_AVX512F)
	#check_cxx_compiler_flag (-fsanitize=address SUPPORTS_ADDRESS_SANS)

	if (HAVE_FNO_BUILTIN)
		add_compile_options (-fno-builtin)
	endif ()

	if (HAS_STACK_PROTECTOR_STRONG)
		add_compile_options (-fstack-protector-strong)
		link_libraries (-fstack-protector-strong)
	endif ()

	if (SUPPORTS_MARCH_NATIVE)
		message (STATUS "Compiler supports compiling for native architecture")
		add_compile_options (-march=native)
	endif ()

	if (SUPPORTS_MTUNE_NATIVE)
		message (STATUS "Compiler supports tuning for native architecture")
		add_compile_options (-mtune=native)
	endif ()
endif ()
#check_cxx_compiler_flag ("-march=native" HAS_NATIVE_SUPPORT)
#
#check_include_file ("float.h;stdlib.h;string.h" STDC_HEADERS)
#check_include_file ("xmmintrin.h" HAVE_XMMINTRIN_H)
#check_include_file ("immintrin.h" HAVE_IMMINTRIN_H)
#
#
#if (HAS_NATIVE_SUPPORT)
#	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
#	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
#endif ()

# Make executable available within the build directory
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory (tracking/exec)
add_subdirectory (daemon)
