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

# set(CMAKE_BUILD_TYPE DEBUG)

if (CMAKE_BUILD_TYPE MATCHES DEBUG)
	message("Building in DEBUG mode")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g -O1 ")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -DDEBUG")
else()
	message("Building in RELEASE mode")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Ofast -march=native -DNDEBUG")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -DRELEASE -lineinfo --use_fast_math -arch=compute_75")
endif()

add_subdirectory(src)