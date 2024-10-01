cmake_minimum_required (VERSION 3.10.0)
project(
	"MyTorch"
	VERSION 1.0.0
	LANGUAGES CXX CUDA
)

find_package(CUDAToolkit 11.4 REQUIRED)

# Make all executables goto bin directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(src)