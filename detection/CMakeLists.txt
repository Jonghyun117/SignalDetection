cmake_minimum_required(VERSION 3.12)
project(detection C)

set(CMAKE_C_STANDARD 11)

add_library(detection_lib STATIC
    cfar.c
    wideband.c
    narrowband.c
    scenario.c
)
target_include_directories(detection_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(detection_lib m)

add_executable(detect main_detect.c)
target_link_libraries(detect detection_lib)
