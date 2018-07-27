# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/local/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/local/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/hsaputra/cuda/CMakeFiles /home/hsaputra/cuda/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/hsaputra/cuda/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named boxfilter

# Build rule for target.
boxfilter: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 boxfilter
.PHONY : boxfilter

# fast build rule for target.
boxfilter/fast:
	$(MAKE) -f boxfilter/CMakeFiles/boxfilter.dir/build.make boxfilter/CMakeFiles/boxfilter.dir/build
.PHONY : boxfilter/fast

#=============================================================================
# Target rules for targets named convolve

# Build rule for target.
convolve: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 convolve
.PHONY : convolve

# fast build rule for target.
convolve/fast:
	$(MAKE) -f convolve/CMakeFiles/convolve.dir/build.make convolve/CMakeFiles/convolve.dir/build
.PHONY : convolve/fast

#=============================================================================
# Target rules for targets named cudaDeviceInfo

# Build rule for target.
cudaDeviceInfo: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 cudaDeviceInfo
.PHONY : cudaDeviceInfo

# fast build rule for target.
cudaDeviceInfo/fast:
	$(MAKE) -f cudaDeviceProperties/CMakeFiles/cudaDeviceInfo.dir/build.make cudaDeviceProperties/CMakeFiles/cudaDeviceInfo.dir/build
.PHONY : cudaDeviceInfo/fast

#=============================================================================
# Target rules for targets named hist

# Build rule for target.
hist: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 hist
.PHONY : hist

# fast build rule for target.
hist/fast:
	$(MAKE) -f hist/CMakeFiles/hist.dir/build.make hist/CMakeFiles/hist.dir/build
.PHONY : hist/fast

#=============================================================================
# Target rules for targets named helloworld

# Build rule for target.
helloworld: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 helloworld
.PHONY : helloworld

# fast build rule for target.
helloworld/fast:
	$(MAKE) -f intro/CMakeFiles/helloworld.dir/build.make intro/CMakeFiles/helloworld.dir/build
.PHONY : helloworld/fast

#=============================================================================
# Target rules for targets named increment

# Build rule for target.
increment: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 increment
.PHONY : increment

# fast build rule for target.
increment/fast:
	$(MAKE) -f intro/CMakeFiles/increment.dir/build.make intro/CMakeFiles/increment.dir/build
.PHONY : increment/fast

#=============================================================================
# Target rules for targets named vector_addition

# Build rule for target.
vector_addition: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 vector_addition
.PHONY : vector_addition

# fast build rule for target.
vector_addition/fast:
	$(MAKE) -f intro/CMakeFiles/vector_addition.dir/build.make intro/CMakeFiles/vector_addition.dir/build
.PHONY : vector_addition/fast

#=============================================================================
# Target rules for targets named median2

# Build rule for target.
median2: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 median2
.PHONY : median2

# fast build rule for target.
median2/fast:
	$(MAKE) -f median2/CMakeFiles/median2.dir/build.make median2/CMakeFiles/median2.dir/build
.PHONY : median2/fast

#=============================================================================
# Target rules for targets named pi

# Build rule for target.
pi: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 pi
.PHONY : pi

# fast build rule for target.
pi/fast:
	$(MAKE) -f pi/CMakeFiles/pi.dir/build.make pi/CMakeFiles/pi.dir/build
.PHONY : pi/fast

#=============================================================================
# Target rules for targets named reduction

# Build rule for target.
reduction: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 reduction
.PHONY : reduction

# fast build rule for target.
reduction/fast:
	$(MAKE) -f reduction/CMakeFiles/reduction.dir/build.make reduction/CMakeFiles/reduction.dir/build
.PHONY : reduction/fast

#=============================================================================
# Target rules for targets named texture

# Build rule for target.
texture: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 texture
.PHONY : texture

# fast build rule for target.
texture/fast:
	$(MAKE) -f textures/CMakeFiles/texture.dir/build.make textures/CMakeFiles/texture.dir/build
.PHONY : texture/fast

#=============================================================================
# Target rules for targets named transpose

# Build rule for target.
transpose: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 transpose
.PHONY : transpose

# fast build rule for target.
transpose/fast:
	$(MAKE) -f transpose/CMakeFiles/transpose.dir/build.make transpose/CMakeFiles/transpose.dir/build
.PHONY : transpose/fast

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... boxfilter"
	@echo "... convolve"
	@echo "... cudaDeviceInfo"
	@echo "... hist"
	@echo "... helloworld"
	@echo "... increment"
	@echo "... vector_addition"
	@echo "... median2"
	@echo "... pi"
	@echo "... reduction"
	@echo "... texture"
	@echo "... transpose"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
