cmake_minimum_required(VERSION 3.8)
project(robo_collector)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/lerobot_collector_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest QUIET)
  if(ament_cmake_pytest_FOUND)
    ament_add_pytest_test(test_collector_state test/test_collector_state.py
      APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    )
    ament_add_pytest_test(test_camera_cache test/test_camera_cache.py
      APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    )
    ament_add_pytest_test(test_field_config test/test_field_config.py
      APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    )
    ament_add_pytest_test(test_lerobot_dataset test/test_lerobot_dataset.py
      APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    )
  endif()
endif()

ament_package()
