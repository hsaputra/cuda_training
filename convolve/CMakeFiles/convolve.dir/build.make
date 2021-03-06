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
include convolve/CMakeFiles/convolve.dir/depend.make

# Include the progress variables for this target.
include convolve/CMakeFiles/convolve.dir/progress.make

# Include the compile flags for this target's objects.
include convolve/CMakeFiles/convolve.dir/flags.make

convolve/CMakeFiles/convolve.dir/convolve.cu.o: convolve/CMakeFiles/convolve.dir/flags.make
convolve/CMakeFiles/convolve.dir/convolve.cu.o: convolve/convolve.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hsaputra/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object convolve/CMakeFiles/convolve.dir/convolve.cu.o"
	cd /home/hsaputra/cuda/convolve && /usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /home/hsaputra/cuda/convolve/convolve.cu -o CMakeFiles/convolve.dir/convolve.cu.o

convolve/CMakeFiles/convolve.dir/convolve.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/convolve.dir/convolve.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

convolve/CMakeFiles/convolve.dir/convolve.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/convolve.dir/convolve.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target convolve
convolve_OBJECTS = \
"CMakeFiles/convolve.dir/convolve.cu.o"

# External object files for target convolve
convolve_EXTERNAL_OBJECTS =

convolve/CMakeFiles/convolve.dir/cmake_device_link.o: convolve/CMakeFiles/convolve.dir/convolve.cu.o
convolve/CMakeFiles/convolve.dir/cmake_device_link.o: convolve/CMakeFiles/convolve.dir/build.make
convolve/CMakeFiles/convolve.dir/cmake_device_link.o: convolve/CMakeFiles/convolve.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hsaputra/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/convolve.dir/cmake_device_link.o"
	cd /home/hsaputra/cuda/convolve && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convolve.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
convolve/CMakeFiles/convolve.dir/build: convolve/CMakeFiles/convolve.dir/cmake_device_link.o

.PHONY : convolve/CMakeFiles/convolve.dir/build

# Object files for target convolve
convolve_OBJECTS = \
"CMakeFiles/convolve.dir/convolve.cu.o"

# External object files for target convolve
convolve_EXTERNAL_OBJECTS =

convolve/convolve: convolve/CMakeFiles/convolve.dir/convolve.cu.o
convolve/convolve: convolve/CMakeFiles/convolve.dir/build.make
convolve/convolve: convolve/CMakeFiles/convolve.dir/cmake_device_link.o
convolve/convolve: convolve/CMakeFiles/convolve.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hsaputra/cuda/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable convolve"
	cd /home/hsaputra/cuda/convolve && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/convolve.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
convolve/CMakeFiles/convolve.dir/build: convolve/convolve

.PHONY : convolve/CMakeFiles/convolve.dir/build

convolve/CMakeFiles/convolve.dir/clean:
	cd /home/hsaputra/cuda/convolve && $(CMAKE_COMMAND) -P CMakeFiles/convolve.dir/cmake_clean.cmake
.PHONY : convolve/CMakeFiles/convolve.dir/clean

convolve/CMakeFiles/convolve.dir/depend:
	cd /home/hsaputra/cuda && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hsaputra/cuda /home/hsaputra/cuda/convolve /home/hsaputra/cuda /home/hsaputra/cuda/convolve /home/hsaputra/cuda/convolve/CMakeFiles/convolve.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : convolve/CMakeFiles/convolve.dir/depend

