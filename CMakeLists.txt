cmake_minimum_required(VERSION 3.10...3.31)

project(my_compiler)

file(GLOB COMPILER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
find_program(PYTHON_EXECUTABLE NAMES python3 PATHS $ENV{HOME}/.pyenv/versions/*/bin NO_DEFAULT_PATH)

if(PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch;print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE Torch_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(NOTICE "${Torch_DIR}")

set(CMAKE_PREFIX_PATH "${Torch_DIR}")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} )


add_subdirectory(pybind11)
add_subdirectory(asmjit)

pybind11_add_module(${PROJECT_NAME} SHARED ${COMPILER_SRCS})
target_link_libraries(${PROJECT_NAME} PUBLIC torch pybind11 asmjit)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" pybind11 asmjit)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_include_directories(${PROJECT_NAME} PUBLIC
    ${Torch_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYBIND11_INCLUDE_DIR}
    asmjit/src
)