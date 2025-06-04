cmake_minimum_required(VERSION 3.8)
project(custom_msgs)

# Enable additional warnings for GCC and Clang compilers
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
# find_package(vision_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

set(msg_files
  msg/CustomDetection2D.msg
  msg/CustomTrackedObjects.msg
  msg/CustomBoolean.msg
)

# Find rosidl_default_generators
find_package(rosidl_default_generators REQUIRED)

# Generate custom message interfaces

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
   DEPENDENCIES std_msgs builtin_interfaces)
# Run linters if BUILD_TESTING is enabled
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Skip the linter checking for copyrights; add when necessary
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Register the package
ament_package()
