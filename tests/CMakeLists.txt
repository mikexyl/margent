add_executable(test_subtree test_optimal_subtree.cpp)
target_include_directories(test_subtree PRIVATE ${INCLUDES})
target_link_libraries(test_subtree ${PROJECT_NAME})

# copy test data
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tinyGrid3D.g2o DESTINATION ${CMAKE_CURRENT_BINARY_DIR})