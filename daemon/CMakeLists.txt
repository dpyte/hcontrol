# $hcontrold$
cmake_minimum_required (VERSION 3.16)

find_package (Boost COMPONENTS system REQUIRED)
find_package (PythonLibs 3 REQUIRED)

include_directories (${Boost_INCLUDE_DIR})
include_directories (${PYTHON_INCLUDE_DIRS})

set (EXECUTABLED_NAME hcontrold CACHE STRING "program name")
set (HCONTROLD_SOURCES
    hcontrold.cxx
    daemonize.cxx    daemonize.hxx
    )

add_executable (${EXECUTABLED_NAME} ${HCONTROLD_SOURCES})

link_directories (${Boost_LIBRARY_DIR})

target_link_libraries (
    ${EXECUTABLED_NAME}
    ${PYTHON_LIBRARIES}
    )
