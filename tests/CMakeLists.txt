# find boost program options
find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(test_subtree test_optimal_subtree.cpp)
target_include_directories(test_subtree PRIVATE ${INCLUDES})
target_link_libraries(test_subtree ${PROJECT_NAME} Boost::program_options)

# find all *.g2o files in the tests directory
file(GLOB TESTS *.g2o)
# copy them to the build directory
file(COPY ${TESTS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(test_gbp_marg test_gbp_marg.cpp)
target_include_directories(test_gbp_marg PRIVATE ${INCLUDES})
target_link_libraries(test_gbp_marg ${PROJECT_NAME} Boost::program_options)

add_executable(test_CLT test_CLT.cpp)
target_include_directories(test_CLT PRIVATE ${INCLUDES})
target_link_libraries(test_CLT ${PROJECT_NAME})

add_executable(test_marginal_compare test_marginal_compare.cpp)
target_include_directories(test_marginal_compare PRIVATE ${INCLUDES})
target_link_libraries(test_marginal_compare ${PROJECT_NAME})