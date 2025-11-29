cmake_minimum_required(VERSION 3.8)
project(kidnapped_vehicle)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

include_directories(
  include
)

add_executable(kidnapped_node
  src/kidnapped_node.cpp
  src/particle_filter.cpp
)

target_include_directories(kidnapped_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(kidnapped_node PUBLIC c_std_99 cxx_std_17)

ament_target_dependencies(kidnapped_node
  rclcpp
  visualization_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
)

install(TARGETS kidnapped_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()



