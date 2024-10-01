# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build

# Include any dependencies generated for this target.
include src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/compiler_depend.make

# Include the progress variables for this target.
include src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/flags.make

src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/relu.cc.o: src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/flags.make
src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/relu.cc.o: ../src/backend/cpu/relu.cc
src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/relu.cc.o: src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/relu.cc.o"
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/relu.cc.o -MF CMakeFiles/mytorch_backend_cpu.dir/relu.cc.o.d -o CMakeFiles/mytorch_backend_cpu.dir/relu.cc.o -c /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/src/backend/cpu/relu.cc

src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/relu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mytorch_backend_cpu.dir/relu.cc.i"
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/src/backend/cpu/relu.cc > CMakeFiles/mytorch_backend_cpu.dir/relu.cc.i

src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/relu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mytorch_backend_cpu.dir/relu.cc.s"
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/src/backend/cpu/relu.cc -o CMakeFiles/mytorch_backend_cpu.dir/relu.cc.s

src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.o: src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/flags.make
src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.o: ../src/backend/cpu/sigmoid.cc
src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.o: src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.o"
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.o -MF CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.o.d -o CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.o -c /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/src/backend/cpu/sigmoid.cc

src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.i"
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/src/backend/cpu/sigmoid.cc > CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.i

src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.s"
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/src/backend/cpu/sigmoid.cc -o CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.s

src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.o: src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/flags.make
src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.o: ../src/backend/cpu/cmp_tensor.cc
src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.o: src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.o"
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.o -MF CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.o.d -o CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.o -c /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/src/backend/cpu/cmp_tensor.cc

src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.i"
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/src/backend/cpu/cmp_tensor.cc > CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.i

src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.s"
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/src/backend/cpu/cmp_tensor.cc -o CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.s

# Object files for target mytorch_backend_cpu
mytorch_backend_cpu_OBJECTS = \
"CMakeFiles/mytorch_backend_cpu.dir/relu.cc.o" \
"CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.o" \
"CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.o"

# External object files for target mytorch_backend_cpu
mytorch_backend_cpu_EXTERNAL_OBJECTS =

src/backend/cpu/libmytorch_backend_cpu.a: src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/relu.cc.o
src/backend/cpu/libmytorch_backend_cpu.a: src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/sigmoid.cc.o
src/backend/cpu/libmytorch_backend_cpu.a: src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/cmp_tensor.cc.o
src/backend/cpu/libmytorch_backend_cpu.a: src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/build.make
src/backend/cpu/libmytorch_backend_cpu.a: src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libmytorch_backend_cpu.a"
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && $(CMAKE_COMMAND) -P CMakeFiles/mytorch_backend_cpu.dir/cmake_clean_target.cmake
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mytorch_backend_cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/build: src/backend/cpu/libmytorch_backend_cpu.a
.PHONY : src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/build

src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/clean:
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu && $(CMAKE_COMMAND) -P CMakeFiles/mytorch_backend_cpu.dir/cmake_clean.cmake
.PHONY : src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/clean

src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/depend:
	cd /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/src/backend/cpu /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu /mnt/f/pku_files/Fall2024/Programming_in_Artificial_Intelligence/hw/build/src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/backend/cpu/CMakeFiles/mytorch_backend_cpu.dir/depend

