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
#include "CoSection/optim/OptSDF.h"
#include "CoSection/optim/cuda/OptSDF.cuh"
#include "EMFusion/core/cuda/TSDF.cuh"

namespace cosection {

OptSDF::OptSDF ( const OptParams& _params, const emf::TSDF& _tsdf ) :
    params ( _params ),
    volumeRes ( _tsdf.getVolumeRes() ),
    voxelSize ( _tsdf.getVoxelSize() ) {

    esdfGrads = cv::cuda::createContinuous ( volumeRes[1] * volumeRes[2],
                volumeRes[0], CV_32FC3 );

    cubeClasses =
        cv::cuda::createContinuous (
            ( volumeRes[1] - 1 ) * ( volumeRes[2] - 1 ), volumeRes[0] - 1,
            CV_8UC1 );
    vertIdxBuffer =
        cv::cuda::createContinuous (
            ( volumeRes[1] - 1 ) * ( volumeRes[2] - 1 ), volumeRes[0] - 1,
            CV_32SC1 );
    triIdxBuffer =
        cv::cuda::createContinuous (
            ( volumeRes[1] - 1 ) * ( volumeRes[2] - 1 ), volumeRes[0] - 1,
            CV_32SC1 );

    float vox = voxelSize;
    for ( cv::Vec3i res = volumeRes;
            *std::min_element ( res.val, res.val+3 ) >= 32;
            res /= 2, vox *= 2 ) {
        esdfVol.push_back ( cv::cuda::createContinuous (
                                res[1] * res[2], res[0], CV_32FC1 ) );
        w.push_back ( cv::cuda::createContinuous (
                          res[1] * res[2], res[0], CV_32FC1 ) );
        c.push_back ( cv::cuda::createContinuous (
                          res[1] * res[2], res[0], CV_32FC1 ) );
        wcount.push_back ( cv::cuda::createContinuous (
                               res[1] * res[2], res[0], CV_32FC1 ) );
        dHull.push_back ( cv::cuda::createContinuous (
                              res[1] * res[2], res[0], CV_32FC1 ) );
        dInter.push_back ( cv::cuda::createContinuous (
                               res[1] * res[2], res[0], CV_32FC1 ) );
        volRes.push_back ( res );
        voxsz.push_back ( vox );
    }

    reset ( _tsdf.getPose() );
}

void OptSDF::reset ( const cv::Affine3f& _pose ) {
    esdfGrads.setTo ( cv::Scalar::all ( 0 ) );
    for ( int i = 0; i < w.size(); ++i ) {
        esdfVol[i].setTo ( 0 );
        w[i].setTo ( 0 );
        c[i].setTo ( 0 );
        wcount[i].setTo ( 0 );
        dHull[i].setTo ( 0 );
        dInter[i].setTo ( 0 );
    }

    pose = _pose;
}

void OptSDF::getCorners ( cv::Vec3f& low, cv::Vec3f& high ) const {
    const cv::Vec3f corner = ( cv::Vec3f ( volumeRes ) - cv::Vec3f::all ( 1 ) )
                             * voxelSize / 2;
    low = -corner;
    high = corner;
}

cv::Vec3f OptSDF::getVolumeSize() const {
    return cv::Vec3f ( volumeRes ) * voxelSize;
}

cv::Vec3i OptSDF::getVolumeRes() const {
    return volumeRes;
}

float OptSDF::getVoxelSize() const {
    return voxelSize;
}

void OptSDF::setPose ( const cv::Affine3f& _pose ) {
    pose = _pose;
}

cv::Affine3f OptSDF::getPose() const {
    return pose;
}

cv::viz::Mesh OptSDF::getMesh() {
    cv::cuda::createContinuous ( esdfVol[0].size(), CV_8U, tsdfVolMask );
    tsdfVolMask.setTo ( 1 );
    cubeClasses.setTo ( 0 );
    vertIdxBuffer.setTo ( 0 );
    triIdxBuffer.setTo ( 0 );
    vertices.setTo ( 0 );
    triangles.setTo ( 0 );
    emf::cuda::TSDF::marchingCubes ( esdfVol[0], esdfGrads, tsdfVolMask,
                                     volumeRes, voxelSize, cubeClasses,
                                     vertIdxBuffer, triIdxBuffer, vertices,
                                     normals, triangles );

    cv::viz::Mesh mesh;
    vertices.download ( mesh.cloud );
    normals.download ( mesh.normals );
    triangles.download ( mesh.polygons );

    return mesh;
}

cv::Mat OptSDF::getESDF() const {
    cv::Mat cpu_esdf;
    esdfVol[0].download ( cpu_esdf );
    return cpu_esdf;
}

cv::Mat OptSDF::getC() const {
    cv::Mat cpu_c;
    c[0].download ( cpu_c );
    return cpu_c;
}

cv::Mat OptSDF::getW() const {
    cv::Mat cpu_w;
    w[0].download ( cpu_w );
    return cpu_w;
}

cv::Mat OptSDF::getDInter() const {
    cv::Mat cpu_dInter;
    dInter[0].download ( cpu_dInter );
    return cpu_dInter;
}

cv::Mat OptSDF::getDHull() const {
    cv::Mat cpu_dHull;
    dHull[0].download ( cpu_dHull );
    return cpu_dHull;
}

void OptSDF::addMeasurements ( const cv::cuda::GpuMat& points,
                               const cv::cuda::GpuMat& normals,
                               const cv::cuda::GpuMat& assocW ) {
    for ( int i = w.size() - 1; i >= 0; --i ) {
        cuda::OptSDF::compSDFWeights ( points, normals, assocW, w[i], c[i],
                                       wcount[i], dHull[i],
                                       params.wsigma * voxsz[i],
                                       pose.inv().rotation(),
                                       pose.inv().translation(), volRes[i],
                                       voxsz[i] );
    }
}

void OptSDF::optimizeESDF () {
    for ( int i = countKeyframes == 0 ? esdfVol.size() - 1 : 0; i >= 0; --i ) {
        std::cout << "Resolution: " << volRes[i] << std::endl;
        std::cout << "Voxel Size: " << voxsz[i] << std::endl;
        cv::cuda::compare ( wcount[i], 0, tsdfVolMask, cv::CMP_EQ );
        cv::cuda::GpuMat divw, divc;
        cv::cuda::divide ( w[i], wcount[i], divw );
        divw.setTo ( 0, tsdfVolMask );
        cv::cuda::divide ( c[i], wcount[i], divc );
        divc.setTo ( 0, tsdfVolMask );

        cuda::OptSDF::optimizeSDF ( esdfVol[i], divw, divc, dHull[i],
                                    dInter[i], buf1, buf2, volRes[i],
                                    voxsz[i], params.sdfalpha,
                                    params.sdfbetaHull, params.sdfbetaInter,
                                    params.FJcyclelength );

        if ( i > 0 ) {
            cuda::OptSDF::upSample ( esdfVol[i], esdfVol[i-1], volRes[i],
                                     volRes[i-1] );
        }
    }
    countKeyframes++;
    esdfGrads.setTo ( cv::Scalar::all ( 0.f ) );
    emf::cuda::TSDF::computeTSDFGrads ( esdfVol[0], esdfGrads,
                                        volumeRes );
}

void OptSDF::computeHull ( const cv::cuda::GpuMat& depthMap,
                           const cv::Affine3f& cam_pose,
                           const cv::Matx33f& intr ) {
    const cv::Affine3f rel_pose_OC = cam_pose.inv() * getPose();

    cuda::OptSDF::compHull ( depthMap, dHull[0], w[0], c[0], wcount[0],
                             rel_pose_OC.rotation(), rel_pose_OC.translation(),
                             intr, params.wsigma, volumeRes, voxelSize );
}

void OptSDF::computeIntersection ( const OptSDF& other ) {
    cv::Affine3f rel_pose = other.getPose().inv() * getPose();

    cv::cuda::GpuMat divc;
    cv::cuda::divide ( other.c[0], other.wcount[0], divc );
    cuda::OptSDF::compIntersec ( divc, dInter[0], rel_pose.rotation(),
                                 rel_pose.translation(), voxelSize,
                                 other.voxelSize, volumeRes, other.volumeRes );
}

void OptSDF::clearObjFg ( emf::ObjTSDF& obj ) {
    cv::cuda::GpuMat fgVolMask = obj.getFgVolMask();

    cv::Affine3f rel_pose = pose.inv() * obj.getPose();
    for ( int i = 0; i < esdfVol.size(); ++i ) {
        cuda::OptSDF::delObjFg ( fgVolMask, w[i], c[i], wcount[i],
                                 rel_pose.rotation(), rel_pose.translation(),
                                 obj.getVoxelSize(), voxsz[i],
                                 obj.getVolumeRes(), volRes[i] );
    }
}


}
