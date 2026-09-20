cmake_minimum_required(VERSION 3.16)
project(vision_day1_hw1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

find_package(ament_cmake REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(vision_day1_hw1
  src/main.cpp
  src/mainwindow.cpp
  src/mainwindow.h
  src/mainwindow.ui
  src/hw1.cpp
  src/hw1.h
)
target_include_directories(vision_day1_hw1 PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_day1_hw1 Qt6::Widgets ${OpenCV_LIBS})

install(TARGETS vision_day1_hw1 DESTINATION lib/${PROJECT_NAME})

ament_package()
