#!/bin/bash
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

EVAL=/is/sg/mstrecke/code/mesh-evaluation/bin/evaluate

mkdir -p "$2"/{tsdf,baseline,hull,inter,full,ref}

export LC_ALL=C # Without this, meshlab is unable to load ply files correctly.

meshlabserver -i "data/truck_ref.ply" -o "$2/ref/0.off"
meshlabserver -i "data/car_ref.ply" -o "$2/ref/1.off"
meshlabserver -i "data/plane_ref.ply" -o "$2/ref/2.off"

meshlabserver -i "$1/car4_full/mesh_2.ply" -o "$2/tsdf/0.off"
meshlabserver -i "$1/car4_full/mesh_4.ply" -o "$2/tsdf/1.off"
meshlabserver -i "$1/car4_full/mesh_1.ply" -o "$2/tsdf/2.off"

for app in {baseline,hull,inter,full}; do
	meshlabserver -i "$1/car4_${app}/optim_meshes/mesh_2.ply" -o "$2/${app}/0.off"
	meshlabserver -i "$1/car4_${app}/optim_meshes/mesh_4.ply" -o "$2/${app}/1.off"
	meshlabserver -i "$1/car4_${app}/optim_meshes/mesh_1.ply" -o "$2/${app}/2.off"
done

for app in {tsdf,baseline,hull,inter,full}; do
	$EVAL --input "$2/${app}" --reference "$2/ref" --output "$2/${app}.txt"
done
