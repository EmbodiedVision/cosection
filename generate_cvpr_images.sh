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

#RENDERDATA=./build/renderdata
RENDERDATA="blender --background --python apps/renderdata.py --"

# Car from different poses
frame=142;
for i in {1,2}; do
	mkdir -p $2/car4/truck"$i"_tsdf;
	$RENDERDATA -d $1/car4_full -p data/car4_poses/view_pose"$i".txt -t -f $frame -o $2/car4/truck"$i"_tsdf/;

	for app in {baseline,hull,inter,full}; do
		mkdir -p $2/car4/truck"$i"_"$app";
		$RENDERDATA -d $1/car4_"$app" -p data/car4_poses/view_pose"$i".txt -f $frame -o $2/car4/truck"$i"_"$app"/;
	done
done

#Clock from different poses
frame=347
for i in {1,2}; do
	mkdir -p $2/sliding-clock/clock"$i"_tsdf;
	$RENDERDATA -d $1/sliding-clock_full -p data/sliding-clock_poses/view_pose"$i".txt -t -f $frame -o $2/sliding-clock/clock"$i"_tsdf/;

	for app in {baseline,hull,inter,full}; do
		mkdir -p $2/sliding-clock/clock"$i"_"$app";
		$RENDERDATA -d $1/sliding-clock_"$app" -p data/sliding-clock_poses/view_pose"$i".txt -f $frame -o $2/sliding-clock/clock"$i"_"$app"/;
	done
done

# Teddy placement
for frame in {859,869,870}; do
	mkdir -p $2/place-items/teddy_tsdf;
	$RENDERDATA -d $1/place-items_full -p data/place-items_poses/view_pose_teddy.txt -t -f $frame -o $2/place-items/teddy_tsdf;

	for app in {baseline,hull,inter,full}; do
		mkdir -p $2/place-items/teddy_"$app";
		$RENDERDATA -d $1/place-items_"$app" -p data/place-items_poses/view_pose_teddy.txt -f $frame -o $2/place-items/teddy_"$app";
	done
done

#Bottle/dustbin
frame=600
for i in {1,2}; do
	mkdir -p $2/place-items/bottle"$i"_tsdf;
	$RENDERDATA -d $1/place-items_full -p data/place-items_poses/view_pose"$i".txt -t -f $frame -o $2/place-items/bottle"$i"_tsdf/;

	for app in {baseline,hull,inter,full}; do
		mkdir -p $2/place-items/bottle"$i"_"$app";
		$RENDERDATA -d $1/place-items_"$app" -p data/place-items_poses/view_pose"$i".txt -f $frame -o $2/place-items/bottle"$i"_"$app"/;
	done
done

