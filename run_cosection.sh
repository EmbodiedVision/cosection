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

cd build
for app in {baseline,hull,inter,full}; do
	for ds in {place-items,sliding-clock}; do
		./Co-Section -d "$1/$ds" -c ../config/cosection_${app}.cfg --background -e "$2/${ds}_${app}"
	done
	./Co-Section -d "$1/car4-full" --depthdir depth_noise -c ../config/cosection_${app}.cfg --background -e "$2/car4_${app}"
	cp "$1/car4-full/calibration.txt" "$2/car4_${app}/calibration.txt"
done
