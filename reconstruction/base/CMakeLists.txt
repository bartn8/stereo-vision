cmake_minimum_required(VERSION 2.8)

include_directories(/usr/include/eigen3/)
include_directories(/usr/local/include/opencv4)

file(GLOB SRC_LIST . *.cc)
add_library(recon_base ${SRC_LIST})
