cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(NurbsRT)
set(CMAKE_CXX_STANDARD 17)

add_library(LiteMath STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/external/LiteMath/Image2d.cpp)
target_include_directories(LiteMath PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/external/LiteMath)

add_library(
  ${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nurbs_rt.cpp
)
target_include_directories(
  ${PROJECT_NAME} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(
  ${PROJECT_NAME} PUBLIC 
    LiteMath 
)