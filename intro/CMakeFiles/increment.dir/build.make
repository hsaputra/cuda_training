# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hsaputra/cuda

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hsaputra/cuda

# Include any dependencies generated for this target.
include intro/CMakeFiles/increment.dir/depend.make

# Include the progress variables for this target.
include intro/CMakeFiles/increment.dir/progress.make

# Include the compile flags for this target's objects.
include intro/CMakeFiles/increment.dir/flags.make

intro/CMakeFiles/increment.dir/increment.cu.o: intro/CMakeFiles/increment.dir/flags.make
intro/CMakeFiles/increment.dir/increment.cu.o: intro/increment.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hsaputra/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object intro/CMakeFiles/increment.dir/increment.cu.o"
	cd /home/hsaputra/cuda/intro && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/hsaputra/cuda/intro/increment.cu -o CMakeFiles/increment.dir/increment.cu.o

intro/CMakeFiles/increment.dir/increment.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/increment.dir/increment.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

intro/CMakeFiles/increment.dir/increment.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/increment.dir/increment.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target increment
increment_OBJECTS = \
"CMakeFiles/increment.dir/increment.cu.o"

# External object files for target increment
increment_EXTERNAL_OBJECTS =

intro/CMakeFiles/increment.dir/cmake_device_link.o: intro/CMakeFiles/increment.dir/increment.cu.o
intro/CMakeFiles/increment.dir/cmake_device_link.o: intro/CMakeFiles/increment.dir/build.make
intro/CMakeFiles/increment.dir/cmake_device_link.o: intro/CMakeFiles/increment.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hsaputra/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/increment.dir/cmake_device_link.o"
	cd /home/hsaputra/cuda/intro && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/increment.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
intro/CMakeFiles/increment.dir/build: intro/CMakeFiles/increment.dir/cmake_device_link.o

.PHONY : intro/CMakeFiles/increment.dir/build

# Object files for target increment
increment_OBJECTS = \
"CMakeFiles/increment.dir/increment.cu.o"

# External object files for target increment
increment_EXTERNAL_OBJECTS =

intro/increment: intro/CMakeFiles/increment.dir/increment.cu.o
intro/increment: intro/CMakeFiles/increment.dir/build.make
intro/increment: intro/CMakeFiles/increment.dir/cmake_device_link.o
intro/increment: intro/CMakeFiles/increment.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hsaputra/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable increment"
	cd /home/hsaputra/cuda/intro && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/increment.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
intro/CMakeFiles/increment.dir/build: intro/increment

.PHONY : intro/CMakeFiles/increment.dir/build

intro/CMakeFiles/increment.dir/clean:
	cd /home/hsaputra/cuda/intro && $(CMAKE_COMMAND) -P CMakeFiles/increment.dir/cmake_clean.cmake
.PHONY : intro/CMakeFiles/increment.dir/clean

intro/CMakeFiles/increment.dir/depend:
	cd /home/hsaputra/cuda && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hsaputra/cuda /home/hsaputra/cuda/intro /home/hsaputra/cuda /home/hsaputra/cuda/intro /home/hsaputra/cuda/intro/CMakeFiles/increment.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : intro/CMakeFiles/increment.dir/depend

