/*
 * This file is part of Co-Section.
 *
 * Copyright (C) 2020 Max-Planck-Gesellschaft.
 * Developed by Michael Strecke <mstrecke at tue dot mpg dot de>.
 * For more information see <https://cosection.is.tue.mpg.de/>.
 * If you use this code, please cite the respective publication as
 * listed on the website.
 */
#pragma once

#include "EMFusion/core/cuda/common.cuh"

namespace cosection {
namespace cuda {
namespace CoSection {

/**
 * Computes normals from pointcloud in camera coordinate system.
 *
 * @param points input pointcloud
 * @param normals output normal map
 */
void computeNormals ( const cv::cuda::GpuMat& points,
                      cv::cuda::GpuMat& normals );

}
}
}
