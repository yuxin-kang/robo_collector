cmake_minimum_required(VERSION 3.8)
project(robo_state)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/robo_state_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest QUIET)
  if(ament_cmake_pytest_FOUND)
    ament_add_pytest_test(test_state_builder test/test_state_builder.py
      APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    )
  endif()
endif()

ament_package()
