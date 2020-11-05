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
#include "EMFusion/core/cuda/TSDF.cuh"

namespace cosection {
namespace cuda {
namespace OptSDF {

/**
 * Compute the SDF dataterm from an oriented input pointcloud.
 *
 * @param points input point cloud
 * @param normals the corresponding normals
 * @param assocW association weights from EM-Fusion
 * @param w output weightings (confidence of the measurements)
 * @param c weighted distances output
 * @param wcount counts of how many measurements were entered in the voxels (for
 *               normalization
 * @param d hull constraint distances (to avoid overriding free measured space)
 * @param sigma width parameter for the gaussian weight falloff away from the
 *              surface
 * @param rot_WO relative rotation from world to volume coordinates
 * @param trans_WO relative translation from world to volume coordinates
 * @param volumeRes the volume resolution
 * @param voxelsize the voxel size
 */
void compSDFWeights ( const cv::cuda::GpuMat& points,
                      const cv::cuda::GpuMat& normals,
                      const cv::cuda::GpuMat& assocW, cv::cuda::GpuMat& w,
                      cv::cuda::GpuMat& c, cv::cuda::GpuMat& wcount,
                      cv::cuda::GpuMat& d, float sigma,
                      const cv::Matx33f& rot_WO, const cv::Vec3f& trans_WO,
                      const cv::Vec3i& volumeRes, const float voxelSize );

/**
 * Delete measurements from the given foreground mask.
 *
 * @param probs foreground mask of other object to remove measurements from
 * @param w weights volume
 * @param c weighted distances volume
 * @param wcount counting volume for normalization
 * @param rel_rot relative rotation of the two volumes
 * @param rel_trans relative translation of the two volumes
 * @param thisVoxelSize voxel size of the current volume
 * @param otherVoxelSize voxel size of the other volume
 * @param thisVolumeRes volume resolution of the current volume
 * @param otherVolumeRes volume resolution of the other volume
 */
void delObjFg ( const cv::cuda::GpuMat& probs, cv::cuda::GpuMat& w,
                cv::cuda::GpuMat& c, cv::cuda::GpuMat& wcount,
                const cv::Matx33f& rel_rot, const cv::Vec3f& rel_trans,
                const float thisVoxelSize, const float otherVoxelSize,
                const cv::Vec3i& thisVolumeRes,
                const cv::Vec3i& otherVolumeRes );

/**
 * Optimize the sdf volume.
 *
 * @param sdf the volume to optimize
 * @param w,c,dHull,dInter optimization dataterms
 * @param buf1,buf1 Cached buffer variables
 * @param volumeRes the volume resolution
 * @param voxelSize the voxel size
 * @param alpha smoothing parameter for the hessian norm regularizer
 * @param betaHull influence parameter for hull constraint
 * @param betaInter incluence parameter for intersection constraint
 * @param cycleLength the cycle length for the fast jacobi algorithm
 */
void optimizeSDF ( cv::cuda::GpuMat& sdf, const cv::cuda::GpuMat& w,
                   const cv::cuda::GpuMat& c, const cv::cuda::GpuMat& dHull,
                   const cv::cuda::GpuMat& dInter, cv::cuda::GpuMat& buf1,
                   cv::cuda::GpuMat& buf2, const cv::Vec3i& volumeRes,
                   const float voxelSize, const float alpha,
                   const float betaHull, const float betaInter,
                   const int cycleLength );

/**
 * Compute the intersection constraint.
 *
 * @param otherC the sdf values in the other volume (from point measurements or
 *               optimized).
 * @param d output intersection constraint volume
 * @param rel_rot,rel_trans relative pose of the objects
 * @param thisVoxelSize,otherVoxelSize voxel sizes of the volumes
 * @param thisVolumeRes,otherVolumeRes resolutions of the volumes
 */
void compIntersec ( const cv::cuda::GpuMat& otherC, cv::cuda::GpuMat& d,
                    const cv::Matx33f& rel_rot, const cv::Vec3f& rel_trans,
                    const float thisVoxelSize, const float otherVoxelSize,
                    const cv::Vec3i& thisVolumeRes,
                    const cv::Vec3i& otherVolumeRes );

/**
 * Compute the hull constraint.
 *
 * @param depthMap input depth map
 * @param d output d_hull constraint volume
 * @param w,c,wcount dataterm volumes that are deleted if the space is measured
 *                   free
 * @param rel_rot_OC,rel_trans_OC relative pose of camera and volume
 * @param intr camera intrinsics
 * @param sigma the weight sigma for point measurements
 * @param volumeRes the volume resolution
 * @param voxelSize the voxel size
 */
void compHull ( const cv::cuda::GpuMat& depthMap, cv::cuda::GpuMat& d,
                cv::cuda::GpuMat& w, cv::cuda::GpuMat& c,
                cv::cuda::GpuMat& wcount, const cv::Matx33f& rel_rot_OC,
                const cv::Vec3f& rel_trans_OC, const cv::Matx33f& intr,
                const float sigma, const cv::Vec3i& volumeRes,
                const float voxelSize );

/**
 * Upsample volume by duplicating voxel values.
 *
 * @param lowRes input low resolution volume
 * @param highRes output high resolution volume (should have twice the
 *                resolution in each dimension).
 * @param lowVolumeRes,highVolumeRes the volume resolutions.
 */
void upSample ( const cv::cuda::GpuMat& lowRes, cv::cuda::GpuMat& highRes,
                const cv::Vec3i& lowVolumeRes, const cv::Vec3i& highVolumeRes );

}
}
}
