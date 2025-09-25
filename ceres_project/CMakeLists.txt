cmake_minimum_required(VERSION 3.10)

project(ceres_test) 
find_package(OpenCV REQUIRED)


message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

find_package(Ceres REQUIRED)

message(STATUS "Ceres version: ${Ceres_VERSION}")
message(STATUS "Ceres libraries: ${CERES_LIBRARIES}")
message(STATUS "Ceres include dirs: ${CERES_INCLUDE_DIRS}")


add_executable(ceres_app src/main.cpp)


target_include_directories(ceres_app PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})


target_link_libraries(ceres_app PRIVATE ${OpenCV_LIBS} Ceres::ceres)
