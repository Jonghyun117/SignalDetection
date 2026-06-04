cmake_minimum_required(VERSION 3.20)
project(amc_infer C)
set(CMAKE_C_STANDARD 11)

set(ORT_ROOT "" CACHE PATH "ONNX Runtime installation root (contains include/ and lib/)")
if(NOT ORT_ROOT)
    message(FATAL_ERROR "Specify ORT_ROOT: cmake -DORT_ROOT=/path/to/onnxruntime ..")
endif()

add_library(amc_preprocess STATIC preprocess.c)
target_include_directories(amc_preprocess PUBLIC .)
target_link_libraries(amc_preprocess m)

add_library(amc_classifier STATIC classifier.c)
target_include_directories(amc_classifier PUBLIC . ${ORT_ROOT}/include)
target_link_libraries(amc_classifier ${ORT_ROOT}/lib/libonnxruntime.so)

add_executable(amc_infer main.c)
target_link_libraries(amc_infer amc_preprocess amc_classifier)

# Tests
add_executable(test_preprocess ../tests/test_preprocess.c)
target_link_libraries(test_preprocess amc_preprocess)

add_executable(test_classifier ../tests/test_classifier.c)
target_link_libraries(test_classifier amc_classifier amc_preprocess)

enable_testing()
add_test(NAME preprocess COMMAND test_preprocess)
add_test(NAME classifier COMMAND test_classifier
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..)
