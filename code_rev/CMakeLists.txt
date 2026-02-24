cmake_minimum_required(VERSION 3.0)
project(ellipse-detector)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

file(GLOB SOURCES "*.cpp")

set(name "app" CACHE STRING "Name of the executable")

add_executable(${name} ${SOURCES})

target_link_libraries(${name} ${OpenCV_LIBS})