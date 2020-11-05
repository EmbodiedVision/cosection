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
project(Co-Section)

include_directories("${PROJECT_SOURCE_DIR}/include/")

if (NOT EMFUSION_DIR)
    set(EMFUSION_DIR "${PROJECT_SOURCE_DIR}/external/emfusion")
endif()

add_subdirectory( "${EMFUSION_DIR}" external/emfusion/ )
include_directories( "${EMFUSION_DIR}/include" )

add_subdirectory( src )
add_subdirectory( apps )
