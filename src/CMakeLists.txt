##
## This file is part of Co-Section.
##
## Copyright (C) 2020 Embodied Vision Group, Max Planck Institute for Intelligent Systems, Germany.
## Developed by Michael Strecke <mstrecke at tue dot mpg dot de>.
## For more information see <https://cosection.is.tue.mpg.de/>.
## If you use this code, please cite the respective publication as
## listed on the website.
##
## Co-Section is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## Co-Section is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Co-Section.  If not, see <https://www.gnu.org/licenses/>.
##
cmake_minimum_required(VERSION 3.0)

# OpenCV
find_package(OpenCV REQUIRED)

# Boost
find_package(Boost COMPONENTS system filesystem REQUIRED)
message("Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
link_directories(${Boost_LIBRARY_DIRS})

# Threads
find_package(Threads)

## CUDA
find_package(CUDA)

find_package(Sophus)
find_package(Eigen3)

IF(CUDA_FOUND AND Eigen3_DIR)
cuda_add_library(cosection SHARED
    optim/CoSection.cpp
    optim/cuda/CoSection.cu
    optim/OptSDF.cpp
    optim/cuda/OptSDF.cu
    optim/ObjOptSDF.cpp
)

IF (USE_RAYCAST_NORMALS)
    target_compile_definitions(cosection PRIVATE USE_RAYCAST_NORMALS)
ENDIF()

target_include_directories(cosection PUBLIC
    "${CMAKE_PREFIX_PATH}/include"
)

target_link_libraries(cosection
    ${OpenCV_LIBS}
    emfusion
)
ENDIF()
