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
RENDERDATA="/is/sg/mstrecke/install/blender-2.82-linux64/blender --background --python apps/renderdata.py --"
#RENDERDATA="blender --background --python apps/renderdata.py --"

# Car transitioning airplane
startframe=150
endframe=300

mkdir -p $2/car4/truck_transition/final/close
$RENDERDATA -d $1/car4 -f $startframe-$endframe -p $2/car4/truck_transition/pose.txt -o $2/car4/truck_transition/final/close

mkdir -p $2/car4/truck_transition/tsdf/close
$RENDERDATA -d $1/car4 -f $startframe-$endframe -t -p $2/car4/truck_transition/pose.txt -o $2/car4/truck_transition/tsdf/close

for app in {nohull,nointer,nohull_nointer}; do
	mkdir -p $2/car4/truck_transition/$app/close;
	$RENDERDATA -d $1/car4_$app -f $startframe-$endframe -p $2/car4/truck_transition/pose.txt -o $2/car4/truck_transition/$app/close
done

mkdir -p $2/car4/truck_transition/final/campose
$RENDERDATA -d $1/car4 -f $startframe-$endframe -o $2/car4/truck_transition/final/campose

mkdir -p $2/car4/truck_transition/tsdf/campose
$RENDERDATA -d $1/car4 -f $startframe-$endframe -t -o $2/car4/truck_transition/tsdf/campose

for app in {nohull,nointer,nohull_nointer}; do
	mkdir -p $2/car4/truck_transition/$app/campose;
	$RENDERDATA -d $1/car4_$app -f $startframe-$endframe -o $2/car4/truck_transition/$app/campose
done

# Placement of dustbin and bottle
startframe=90
endframe=670

mkdir -p $2/place-items/dustbin_bottle/final/close
$RENDERDATA -d $1/place-items -f $startframe-$endframe -p $2/place-items/dustbin_bottle/pose.txt -o $2/place-items/dustbin_bottle/final/close

mkdir -p $2/place-items/dustbin_bottle/tsdf/close
$RENDERDATA -d $1/place-items -f $startframe-$endframe -t -p $2/place-items/dustbin_bottle/pose.txt -o $2/place-items/dustbin_bottle/tsdf/close

for app in {nohull,nointer,nohull_nointer}; do
	mkdir -p $2/place-items/dustbin_bottle/$app/close;
	$RENDERDATA -d $1/place-items_$app -f $startframe-$endframe -p $2/place-items/dustbin_bottle/pose.txt -o $2/place-items/dustbin_bottle/$app/close
done

mkdir -p $2/place-items/dustbin_bottle/final/campose
$RENDERDATA -d $1/place-items -f $startframe-$endframe -o $2/place-items/dustbin_bottle/final/campose

mkdir -p $2/place-items/dustbin_bottle/tsdf/campose
$RENDERDATA -d $1/place-items -f $startframe-$endframe -t -o $2/place-items/dustbin_bottle/tsdf/campose

for app in {nohull,nointer,nohull_nointer}; do
	mkdir -p $2/place-items/dustbin_bottle/$app/campose;
	$RENDERDATA -d $1/place-items_$app -f $startframe-$endframe -o $2/place-items/dustbin_bottle/$app/campose
done

# Placement of teddy
startframe=770
endframe=970

mkdir -p $2/place-items/teddy/final/close
$RENDERDATA -d $1/place-items -f $startframe-$endframe -p $2/place-items/teddy/pose.txt -o $2/place-items/teddy/final/close

mkdir -p $2/place-items/teddy/tsdf/close
$RENDERDATA -d $1/place-items -f $startframe-$endframe -t -p $2/place-items/teddy/pose.txt -o $2/place-items/teddy/tsdf/close

for app in {nohull,nointer,nohull_nointer}; do
	mkdir -p $2/place-items/teddy/$app/close;
	$RENDERDATA -d $1/place-items_$app -f $startframe-$endframe -p $2/place-items/teddy/pose.txt -o $2/place-items/teddy/$app/close
done

mkdir -p $2/place-items/teddy/final/campose
$RENDERDATA -d $1/place-items -f $startframe-$endframe -o $2/place-items/teddy/final/campose

mkdir -p $2/place-items/teddy/tsdf/campose
$RENDERDATA -d $1/place-items -f $startframe-$endframe -t -o $2/place-items/teddy/tsdf/campose

for app in {nohull,nointer,nohull_nointer}; do
	mkdir -p $2/place-items/teddy/$app/campose;
	$RENDERDATA -d $1/place-items_$app -f $startframe-$endframe -o $2/place-items/teddy/$app/campose
done
