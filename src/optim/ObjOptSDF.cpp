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
#include "CoSection/optim/ObjOptSDF.h"
#include "CoSection/optim/cuda/OptSDF.cuh"

namespace cosection {

ObjOptSDF::ObjOptSDF ( const OptParams& _params, const emf::ObjTSDF& _tsdf ) :
    OptSDF ( _params, _tsdf ),
    id ( _tsdf.getID() ) {
}

bool ObjOptSDF::operator== ( const ObjOptSDF& other ) const {
    return id == other.id;
}

bool ObjOptSDF::operator!= ( const ObjOptSDF& other ) const {
    return ! ( *this == other );
}

const int ObjOptSDF::getID () const {
    return id;
}

void ObjOptSDF::setClassID ( int id ) {
    classId = id;
}

const int ObjOptSDF::getClassID () const {
    return classId;
}

void ObjOptSDF::resize ( const cv::Vec3i& newRes, const cv::Vec3f& offset ) {
    if ( newRes != volumeRes ) {
        pose = pose.translate ( pose.rotation() * offset );

        cv::Vec3i pixOffset = offset / voxelSize;
        pixOffset -= ( newRes - volumeRes ) / 2.f;

        cv::cuda::GpuMat newESDFVol = cv::cuda::createContinuous (
                                          newRes[2] * newRes[1], newRes[0],
                                          CV_32FC1 );
        cv::cuda::GpuMat newW = cv::cuda::createContinuous (
                                    newRes[2] * newRes[1], newRes[0],
                                    CV_32FC1 );
        cv::cuda::GpuMat newC = cv::cuda::createContinuous (
                                    newRes[2] * newRes[1], newRes[0],
                                    CV_32FC1 );
        cv::cuda::GpuMat newDHull = cv::cuda::createContinuous (
                                        newRes[2] * newRes[1], newRes[0],
                                        CV_32FC1 );
        cv::cuda::GpuMat newDInter = cv::cuda::createContinuous (
                                         newRes[2] * newRes[1], newRes[0],
                                         CV_32FC1 );
        cv::cuda::GpuMat newWcount = cv::cuda::createContinuous (
                                         newRes[2] * newRes[1], newRes[0],
                                         CV_32FC1 );

        newESDFVol.setTo ( voxelSize );
        newW.setTo ( 0.f );
        newC.setTo ( 0.f );
        newDHull.setTo ( 0.f );
        newDInter.setTo ( 0.f );
        newWcount.setTo ( 0.f );

        emf::cuda::TSDF::copyValues ( w[0], newW, pixOffset, volumeRes,
                                      newRes );
        emf::cuda::TSDF::copyValues ( c[0], newC, pixOffset, volumeRes,
                                      newRes );
        emf::cuda::TSDF::copyValues ( dHull[0], newDHull, pixOffset, volumeRes,
                                      newRes );
        emf::cuda::TSDF::copyValues ( dInter[0], newDInter, pixOffset,
                                      volumeRes, newRes );
        emf::cuda::TSDF::copyValues ( wcount[0], newWcount, pixOffset,
                                      volumeRes, newRes );
        emf::cuda::TSDF::copyValues ( esdfVol[0], newESDFVol, pixOffset,
                                      volumeRes, newRes );

        esdfGrads = cv::cuda::createContinuous (
                        newRes[2] * newRes[1], newRes[0], CV_32FC3 );
        esdfGrads.setTo ( cv::Scalar::all ( 0 ) );

        volumeRes = newRes;
        volRes[0] = newRes;
        w[0] = newW;
        c[0] = newC;
        wcount[0] = newWcount;
        dHull[0] = newDHull;
        dInter[0] = newDInter;
        esdfVol[0] = newESDFVol;

        cv::cuda::createContinuous ( ( newRes[1] - 1 ) * ( newRes[2] - 1 ),
                                     newRes[0] - 1, CV_8UC1, cubeClasses );
        cv::cuda::createContinuous ( ( newRes[1] - 1 ) * ( newRes[2] - 1 ),
                                     newRes[0] - 1, CV_32SC1, vertIdxBuffer );
        cv::cuda::createContinuous ( ( newRes[1] - 1 ) * ( newRes[2] - 1 ),
                                     newRes[0] - 1, CV_32SC1, triIdxBuffer );
    }
}

void ObjOptSDF::computeIntersection ( const OptSDF& other ) {
    cv::Affine3f rel_pose = other.getPose().inv() * getPose();

    cuda::OptSDF::compIntersec ( other.esdfVol[0], dInter[0],
                                 rel_pose.rotation(), rel_pose.translation(),
                                 voxelSize, other.voxelSize, volumeRes,
                                 other.volumeRes );
    cv::cuda::GpuMat divc;
    cv::cuda::divide ( other.c[0], other.wcount[0], divc );
    cuda::OptSDF::compIntersec ( divc, dInter[0], rel_pose.rotation(),
                                 rel_pose.translation(), voxelSize,
                                 other.voxelSize, volumeRes, other.volumeRes );
}

void ObjOptSDF::computeIntersection ( const ObjOptSDF& other ) {
    cv::Affine3f rel_pose = other.getPose().inv() * getPose();

    cv::cuda::GpuMat divc;
    cv::cuda::divide ( other.c[0], other.wcount[0], divc );
    cuda::OptSDF::compIntersec ( divc, dInter[0], rel_pose.rotation(),
                                 rel_pose.translation(), voxelSize,
                                 other.voxelSize, volumeRes, other.volumeRes );
}

}
