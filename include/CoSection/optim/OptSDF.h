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

#include "EMFusion/core/TSDF.h"
#include "EMFusion/core/ObjTSDF.h"

#include "CoSection/optim/data.h"

namespace cosection {

/**
 * Class for storing optimized signed distance fields including functions for
 * adding measurements and performing global optimization.
 */
class OptSDF {
    friend class ObjOptSDF;
public:
    /**
     * Constructor for optimization volume.
     *
     * @param params optimization parameters.
     * @param tsdf backend TSDF volume (voxel size, resolution, and pose are
     *             copied from it.
     */
    OptSDF ( const OptParams& params, const emf::TSDF& tsdf );

    /**
     * Reset the volume to the given pose and clear/initialize all data in the
     * volumes.
     *
     * @param pose pose to set for the current volume.
     */
    virtual void reset ( const cv::Affine3f& pose );

    /**
     * Get corners in object coordinates relative to volume center.
     *
     * @param low low corner
     * @param high high corner
     */
    void getCorners ( cv::Vec3f& low, cv::Vec3f& high ) const;

    /**
     * Get metric volume size.
     *
     * @return the metric size of the volume in all three dimensions.
     */
    cv::Vec3f getVolumeSize() const;

    /**
     * Get the volume resolution.
     *
     * @return a vector containing the current volume resolution.
     */
    cv::Vec3i getVolumeRes() const;

    /**
     * Get the size of a single voxel.
     *
     * @return the size of a single voxel.
     */
    float getVoxelSize() const;

    /**
     * Set the pose of the volume (for updating it with the pose from EM-Fusion.
     *
     * @param _pose the pose to set
     */
    void setPose ( const cv::Affine3f& _pose );

    /**
     * Get the pose of the current volume.
     *
     * @return the pose.
     */
    cv::Affine3f getPose () const;

    /**
     * Add new point measurements to the volume.
     *
     * @param points input pointcloud
     * @param normals corresponding normals
     * @param assocW association likelihood for the pointcloud in the current
     *               volume
     */
    void addMeasurements ( const cv::cuda::GpuMat& points,
                           const cv::cuda::GpuMat& normals,
                           const cv::cuda::GpuMat& assocW );

    /**
     * Optimize the global ESDF of the model.
     */
    void optimizeESDF ();

    /**
     * Compute the intersection constraint compared with the other model.
     *
     * @param other the other optimization object to compute the intersection
     *              with
     */
    void computeIntersection ( const OptSDF& other );

    /**
     * Compute the hull constraint (measure free space).
     *
     * @param depthMap input depth map
     * @param cam_pose the current camera pose
     * @param intr camera intrinsics matrix
     */
    void computeHull ( const cv::cuda::GpuMat& depthMap,
                       const cv::Affine3f& cam_pose,
                       const cv::Matx33f& intr );

    /**
     * Clear voxels in the current volume that are foreground for another
     * object.
     *
     * @param obj object to check for foreground measurements
     */
    void clearObjFg ( emf::ObjTSDF& obj );

    /**
     * Extract mesh from ESDF volume using marching cubes.
     *
     * @return the resulting mesh.
     */
    virtual cv::viz::Mesh getMesh ();

    /**
     * Get the volume containing the dataterm (weighted distance from pointcloud
     * along normal direction).
     *
     * @return a volume on CPU containing the data
     */
    virtual cv::Mat getC () const;

    /**
     * Get the volume containing the weights for the dataterm.
     *
     * @return a volume on CPU containing the data
     */
    virtual cv::Mat getW () const;

    /**
     * Get the volume containing the intersection constraint distances (result
     * of eq. 4 from the paper).
     *
     * @return a volume on CPU containing the data
     */
    virtual cv::Mat getDInter () const;

    /**
     * Get the volume containing the hull constraint distances (d_hull variable
     * in the paper).
     *
     * @return a volume on CPU containing the data
     */
    virtual cv::Mat getDHull () const;

    /**
     * Download the ESDF volume from GPU and return the resulting Mat.
     *
     * @return a Mat containing the SDF data.
     */
    virtual cv::Mat getESDF () const;

protected:
    /** Co-Section optimization parameters. */
    OptParams params;

    /** The pose of the current volume. */
    cv::Affine3f pose;

    /** Volume resolution. */
    cv::Vec3i volumeRes;
    /** Voxel size. */
    float voxelSize;

    /** Optimization volume multi-scale pyramid. */
    std::vector<cv::cuda::GpuMat> esdfVol;
    /** Dataterms multi-scale pyramid volumes. */
    std::vector<cv::cuda::GpuMat> w, c, wcount, dHull, dInter;
    /** Resolutions in the multi-scale volumes. */
    std::vector<cv::Vec3i> volRes;
    /** Voxel sizes in the multi-scale volumes. */
    std::vector<float> voxsz;

    /** Caching variable for optimization volume gradients. */
    cv::cuda::GpuMat esdfGrads;

    // Caching variables for Marching cubes
    cv::cuda::GpuMat tsdfVolMask;
    cv::cuda::GpuMat cubeClasses, vertIdxBuffer, triIdxBuffer, vertices;
    cv::cuda::GpuMat normals, triangles;

    // Caching variables for optimization.
    cv::cuda::GpuMat buf1, buf2;

    int countKeyframes = 0;
};

}
