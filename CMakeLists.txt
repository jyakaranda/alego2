cmake_minimum_required(VERSION 3.5)
project(alego2)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(builtin_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(alego2_msgs REQUIRED)
find_package(pcl_conversions 2.0 REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Ceres REQUIRED)

include_directories(include
  ${EIGEN3_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${tf2_INCLUDE_DIRS}
  ${tf2_ros_INCLUDE_DIRS}
)

add_executable(IP src/IP.cpp)
ament_target_dependencies(IP 
  rclcpp
  tf2_ros
  tf2
  tf2_geometry_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_msgs
  alego2_msgs
  std_srvs
  pcl_conversions
)
target_link_libraries(IP ${Boost_SYSTEM_LIBRARY} ${PCL_LIBRARIES})

add_executable(LO src/LO.cpp)
ament_target_dependencies(LO 
  rclcpp
  tf2_ros
  tf2
  tf2_geometry_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_msgs
  alego2_msgs
  std_srvs
  pcl_conversions
)
target_link_libraries(LO ${Boost_SYSTEM_LIBRARY} ${PCL_LIBRARIES} ${CERES_LIBRARIES}
  ${tf2_LIBRARIES}
)

add_executable(LM src/LM.cpp)
ament_target_dependencies(LM 
  rclcpp
  tf2_ros
  tf2
  tf2_geometry_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf2_msgs
  alego2_msgs
  std_srvs
  pcl_conversions
)
target_link_libraries(LM ${Boost_SYSTEM_LIBRARY} ${PCL_LIBRARIES} ${CERES_LIBRARIES} gtsam
  ${tf2_LIBRARIES}
)

install(TARGETS
  IP
  LO
  LM
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # remove the line when a copyright and license is present in all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # remove the line when this package is a git repo
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
