cmake_minimum_required(VERSION 3.16)
project(margent VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_definitions(-march=native)

find_package(GTSAM REQUIRED)
find_package(TBB 4.4 COMPONENTS tbb tbbmalloc)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(aria_viz REQUIRED)

set(DEPENDENCIES
        PUBLIC
        Eigen3::Eigen
        TBB::tbb TBB::tbbmalloc
        gtsam 
        aria_viz
)

add_library(${PROJECT_NAME} SHARED src/subtree_search.cpp)
target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include> # In-place install for headers
  ${INCLUDES}
)
target_link_libraries(${PROJECT_NAME} INTERFACE ${DEPENDENCIES})

add_subdirectory(tests)
