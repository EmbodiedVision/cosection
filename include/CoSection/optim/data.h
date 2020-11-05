/*
 * This file is part of Co-Section.
 *
 * Copyright (C) 2020 Embodied Vision Group, Max Planck Institute for Intelligent Systems, Germany.
 * Developed by Michael Strecke <mstrecke at tue dot mpg dot de>.
 * For more information see <https://cosection.is.tue.mpg.de/>.
 * If you use this code, please cite the respective publication as
 * listed on the website.
 *
 * Co-Section is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Co-Section is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Co-Section.  If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

namespace cosection {

/**
 * Class for storing optimization parameters.
 */
class OptParams {
public:
    /**
     * Default parameters used in paper experiments.
     */
    OptParams() :
        wsigma ( 1.f ),
        sdfalpha ( 0.005f ),
        sdfbetaHull ( 0.001f ),
        sdfbetaInter ( 0.001f ),
        FJcyclelength ( 20 ) {}

    /**
     * Sigma for weight computation from points. Given as a factor of
     * voxel size.
     */
    float wsigma;
    /** Smoothing parameter for SDF optimization. */
    float sdfalpha;
    /** Weighting parameter for hull/intersection constraint. */
    float sdfbetaHull;
    float sdfbetaInter;
    /** Cycle length for fast jacobi method. */
    int FJcyclelength;
};

/**
 * Class for storing Co-Section parameters.
 */
class Params {
public:
    Params () :
        keyFrameFreq ( 10 ) {}

    /** Framerate for optimization keyframes. */
    int keyFrameFreq;

    /** Optimization parameters for SDFs. */
    OptParams optParams;
};

}
