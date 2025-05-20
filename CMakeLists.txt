cmake_minimum_required(VERSION 3.10...3.31)

project(my_compiler)

file(GLOB COMPILER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
find_program(PYTHON3_EXECUTABLE NAMES python3 PATHS $ENV{HOME}/.pyenv/versions/*/bin NO_DEFAULT_PATH)

if(PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message("${_Python}")
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch;print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE Torch_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_PREFIX_PATH "${Torch_DIR}")

# The set bellow is important. For some reason it thinks that i have python2
set(_Python Python3)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} )

add_subdirectory(asmjit)

pybind11_add_module(${PROJECT_NAME} SHARED ${COMPILER_SRCS})
target_link_libraries(${PROJECT_NAME} PUBLIC torch pybind11 asmjit)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" pybind11 asmjit)

# The pytorch tutorial uses C++17 but we're also using pybind11 
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
target_include_directories(${PROJECT_NAME} PUBLIC
    ${Torch_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYBIND11_INCLUDE_DIR}
    asmjit/src
)