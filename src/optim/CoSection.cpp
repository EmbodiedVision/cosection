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
#include "CoSection/optim/CoSection.h"
#include "CoSection/optim/cuda/CoSection.cuh"
#include "EMFusion/core/cuda/EMFusion.cuh"

namespace cosection {

CoSection::CoSection ( const emf::Params& _params,
                       const Params& _cosecparams ) :
    EMFusion ( _params ),
    cosecParams ( _cosecparams ),
    bg_opt ( _cosecparams.optParams, background ) {
    normals_in = cv::cuda::createContinuous ( _params.frameSize, CV_32FC3 );
    normals_w = cv::cuda::createContinuous ( _params.frameSize, CV_32FC3 );
}

void CoSection::processFrame ( const RGBD& frame ) {
    new_ids.clear();
    cv::Mat rgb ( frame.getRGB() );
    depth_raw.upload ( frame.getDepth() );

    preprocessDepth ( depth_raw, depth );

    emf::cuda::EMFusion::computePoints ( depth, points, params.intr );
    cuda::CoSection::computeNormals ( points, normals_in );

    if ( frameCount > 0 ) {
        computeAssociationWeights();
        if ( saveOutput ) {
            storeAssocs ( bg_associationWeights, bg_assocWeight_preTrack,
                          associationWeights, obj_assocWeights_preTrack );
        }

        performTracking();

        computeAssociationWeights();
        if ( saveOutput ) {
            storeAssocs ( bg_associationWeights, bg_assocWeight_postTrack,
                          associationWeights, obj_assocWeights_postTrack );
        }

        // Raycast to get reprojected object masks
        raycast();

#ifdef USE_RAYCAST_NORMALS
        // The results reported in the paper overwrote the input normals with
        // the ones computed by raycasting. This simulates that behaviour.
        normals.copyTo ( normals_in );
#endif // USE_RAYCAST_NORMALS
    }
    storePoses();
    updateOptimPoses ();

    std::map<int, cv::cuda::GpuMat> matches;
    int numInstances = -1;
    if ( frameCount % params.maskRCNNFrames == 0 )
        numInstances = initOrMatchObjs ( rgb, matches );

    integrateDepth ();

    if ( numInstances > 0 )
        integrateMasks ( matches );

    cleanUpObjs ( numInstances, matches );

    for ( auto it = obj_opt.begin(); it != obj_opt.end(); )
        if ( associationWeights.count ( it->first ) == 0 )
            it = obj_opt.erase ( it );
        else
            ++it;

    if ( frameCount % cosecParams.keyFrameFreq == 0 ) {
        computeValidPoints ( normals_in, validPoints );

        transformPoints ( points, pose, points_w );
        transformPoints ( normals_in, cv::Affine3f ( pose.rotation() ),
                          normals_w );

        addMeasurements ();

        computeHull ();

        clearObjFg ();
    }

    compBgInter ();

    if ( frameCount % cosecParams.keyFrameFreq == 0 )
        optimBg ();

    compObjInter ();

    if ( frameCount % cosecParams.keyFrameFreq == 0 )
        optimObjs ();

    if ( saveOutput ) {
        background.getHuberWeights ( bg_huberWeights[frameCount] );
        background.getTrackingWeights ( bg_trackWeights[frameCount] );

        if ( expFrameMeshes )
            frame_meshes[frameCount] = background.getMesh();
        for ( auto& obj : objects ) {
            obj.getHuberWeights ( obj_huberWeights[obj.getID()][frameCount] );
            obj.getTrackingWeights (
                obj_trackWeights[obj.getID()][frameCount] );
            obj.getFgProbVals ( obj_fgProbs[obj.getID()][frameCount] );
            if ( params.ignore_person &&
                    emf::MaskRCNN::getClassName ( obj.getClassID() )
                    == "person" && expFrameMeshes )
                continue;
            frame_obj_meshes[obj.getID()][frameCount] = obj.getMesh();

            if ( expVols && frameCount % cosecParams.keyFrameFreq == 0 ) {
                frame_tsdfs[obj.getID()][frameCount] = obj.getTSDF();
                frame_fgProbs[obj.getID()][frameCount] = obj.getFgProbVol();
                frame_meta[obj.getID()][frameCount] =
                    std::make_pair ( obj.getVolumeRes(), obj.getVoxelSize() );
            }
        }

        if ( frameCount % cosecParams.keyFrameFreq == 0 ) {
            if ( expFrameMeshes )
                frame_optim_meshes[frameCount] = bg_opt.getMesh();
            for ( auto& obj : obj_opt ) {
                if ( params.ignore_person &&
                        emf::MaskRCNN::getClassName ( obj.second.getClassID() )
                        == "person" ) {
                    continue;
                }
                if ( expFrameMeshes ) {
                    optim_meshes[obj.first] = obj.second.getMesh();
                    frame_obj_optim_meshes[obj.first][frameCount] =
                        optim_meshes[obj.first];
                }
                if ( expVols ) {
                    frame_esdfs[obj.first][frameCount] = obj.second.getESDF();
                    frame_cs[obj.first][frameCount] = obj.second.getC();
                    frame_ws[obj.first][frameCount] = obj.second.getW();
                    frame_dInter[obj.first][frameCount] = obj.second.getDInter();
                    frame_dHull[obj.first][frameCount] = obj.second.getDHull();
                }
            }
        }

    }

    ++frameCount;
}

void CoSection::render ( cv::Mat& rendered, cv::viz::Viz3d* window,
                         bool show_slices ) {
    EMFusion::render ( rendered );

    if ( frameCount > 0 && window ) {
        window->removeAllWidgets();
        cv::viz::WCameraPosition camPos ( params.intr, rendered, 0.1 );
        camPos.applyTransform ( pose );
        window->showWidget ( "CamPos", camPos );
//             window->addLight( Vec3d::all( 0.0 ) );
        for ( auto& obj : obj_opt ) {
            if ( params.ignore_person &&
                    emf::MaskRCNN::getClassName ( obj.second.getClassID() )
                    == "person" ) {
                continue;
            }
            cv::Vec3f low, high;
            obj.second.getCorners ( low, high );
            cv::Affine3f obj_pose = obj.second.getPose();
            cv::viz::WCube obj_cube ( ( cv::Vec3d ( low ) ),
                                      ( cv::Vec3d ( high ) ) );
            obj_cube.applyTransform ( obj_pose );
            obj_cube.setColor ( cv::viz::Color (
                                    colorMap.at<cv::Vec3b> ( 0,
                                            obj.first ) ) );
            cv::viz::Mesh m_obj;
            if ( expFrameMeshes )
                m_obj = frame_obj_optim_meshes[obj.first][
                            ( frameCount-1 ) / cosecParams.keyFrameFreq
                            * cosecParams.keyFrameFreq
                        ];
            else
                m_obj = obj.second.getMesh();
            if ( m_obj.cloud.rows == 1 ) {
                cv::viz::WMesh m_show_obj ( m_obj );
                m_show_obj.setColor ( cv::viz::Color (
                                          colorMap.at<cv::Vec3b> ( 0,
                                                  obj.first ) ) );
                m_show_obj.applyTransform ( obj.second.getPose() );
                std::stringstream mesh_str;
                mesh_str << "Mesh " << obj.first;
                window->showWidget ( mesh_str.str(), m_show_obj );
                window->setRenderingProperty ( mesh_str.str(),
                                               cv::viz::SHADING,
                                               cv::viz::SHADING_PHONG );
            }
            cv::viz::WCoordinateSystem obj_coords ( 0.2 );
            cv::Affine3f coord_pose = obj_pose
                                      * cv::Affine3f().translate ( low );
            obj_coords.applyTransform ( coord_pose );

            std::stringstream mesh_str, cube_str, coords_str;
            cube_str << "Cube " << obj.first;
            coords_str << "Coords " << obj.first;
            window->showWidget ( cube_str.str(), obj_cube );
            window->showWidget ( coords_str.str(), obj_coords );

            if ( show_slices ) {
                std::stringstream cut_x, cut_y, cut_z;
                cv::Mat esdf = obj.second.getESDF();
                normalize ( esdf, esdf, 0, 1, cv::NORM_MINMAX );
                cv::Mat sdf_show = esdf.col ( obj.second.getVolumeRes() [0]/2 );
                sdf_show = sdf_show.clone().reshape (
                               1, obj.second.getVolumeRes() [1] );
                sdf_show.convertTo ( sdf_show, CV_8U, 255 );
                applyColorMap ( sdf_show, sdf_show, cv::COLORMAP_PARULA );
                cv::viz::WImage3D sdfimx (
                    sdf_show,
                    cv::Size2d ( obj.second.getVolumeSize() [1],
                                 obj.second.getVolumeSize() [2] ),
                    obj.second.getPose().translation(),
                    obj.second.getPose().rotation() * cv::Vec3d ( 1, 0, 0 ),
                    obj.second.getPose().rotation() * cv::Vec3d ( 0, 0, 1 ) );
                cut_x << "SDF cut x " << obj.first;
                window->showWidget ( cut_x.str(), sdfimx );
                sdf_show = esdf.rowRange (
                               obj.second.getVolumeRes() [2]/2
                               * obj.second.getVolumeRes() [1],
                               ( obj.second.getVolumeRes() [2]/2 + 1 )
                               * obj.second.getVolumeRes() [1] );
                sdf_show.convertTo ( sdf_show, CV_8U, 255 );
                applyColorMap ( sdf_show, sdf_show, cv::COLORMAP_PARULA );
                cv::viz::WImage3D sdfimz (
                    sdf_show,
                    cv::Size2d ( obj.second.getVolumeSize() [1],
                                 obj.second.getVolumeSize() [0] ),
                    obj.second.getPose().translation(),
                    obj.second.getPose().rotation() * cv::Vec3d ( 0, 0, 1 ),
                    obj.second.getPose().rotation() * cv::Vec3d ( 0, 1, 0 ) );
                cut_z << "SDF cut z " << obj.first;
                window->showWidget ( cut_z.str(), sdfimz );
                sdf_show = cv::Mat();
                for ( int i = 0; i < obj.second.getVolumeRes() [1]; ++i ) {
                    sdf_show.push_back (
                        esdf.row ( i * obj.second.getVolumeRes() [1]
                                   + obj.second.getVolumeRes() [1]/2 ) );
                }
                sdf_show.convertTo ( sdf_show, CV_8U, 255 );
                applyColorMap ( sdf_show, sdf_show, cv::COLORMAP_PARULA );
                cv::viz::WImage3D sdfimy (
                    sdf_show,
                    cv::Size2d ( obj.second.getVolumeSize() [1],
                                 obj.second.getVolumeSize() [0] ),
                    obj.second.getPose().translation(),
                    obj.second.getPose().rotation() * cv::Vec3d ( 0, -1, 0 ),
                    obj.second.getPose().rotation() * cv::Vec3d ( 0, 0, 1 ) );
                cut_y << "SDF cut y " << obj.first;
                window->showWidget ( cut_y.str(), sdfimy );
            }
        }
        cv::viz::Mesh bg_mesh;
        if ( expFrameMeshes )
            bg_mesh = frame_optim_meshes[ ( frameCount-1 )
                                          / cosecParams.keyFrameFreq
                                          * cosecParams.keyFrameFreq ];
        else
            bg_mesh = bg_opt.getMesh();
        cv::Vec3f low, high;
        background.getCorners ( low, high );
        cv::viz::WCube cube ( ( cv::Vec3d ( low ) ), ( cv::Vec3d ( high ) ) );
        cube.applyTransform ( background.getPose() );

        if ( bg_mesh.cloud.rows == 1 ) {
            cv::viz::WMesh viz_mesh ( bg_mesh );
            viz_mesh.applyTransform ( background.getPose() );
            window->showWidget ( "mesh", viz_mesh );
            window->showWidget ( "cube", cube );
            window->setRenderingProperty ( "mesh", cv::viz::SHADING,
                                           cv::viz::SHADING_PHONG );
        }

        if ( show_slices ) {
            cv::Mat esdf = bg_opt.getESDF();
            normalize ( esdf, esdf, 0, 1, cv::NORM_MINMAX );
            cv::Mat sdf_show = esdf.col ( bg_opt.getVolumeRes() [0]/2 );
            sdf_show = sdf_show.clone().reshape ( 1, bg_opt.getVolumeRes() [1] );
            sdf_show.convertTo ( sdf_show, CV_8U, 255 );
            applyColorMap ( sdf_show, sdf_show, cv::COLORMAP_PARULA );
            cv::viz::WImage3D sdfimx (
                sdf_show, cv::Size2d ( bg_opt.getVolumeSize() [1],
                                       bg_opt.getVolumeSize() [2] ),
                bg_opt.getPose().translation(),
                bg_opt.getPose().rotation() * cv::Vec3d ( 1, 0, 0 ),
                bg_opt.getPose().rotation() * cv::Vec3d ( 0, 0, 1 ) );
            window->showWidget ( "SDF cut x", sdfimx );
            sdf_show = esdf.rowRange (
                           bg_opt.getVolumeRes() [2]/2
                           * bg_opt.getVolumeRes() [1],
                           ( bg_opt.getVolumeRes() [2]/2 + 1 )
                           * bg_opt.getVolumeRes() [1] );
            sdf_show.convertTo ( sdf_show, CV_8U, 255 );
            applyColorMap ( sdf_show, sdf_show, cv::COLORMAP_PARULA );
            cv::viz::WImage3D sdfimz (
                sdf_show, cv::Size2d ( bg_opt.getVolumeSize() [1],
                                       bg_opt.getVolumeSize() [0] ),
                bg_opt.getPose().translation(),
                bg_opt.getPose().rotation() * cv::Vec3d ( 0, 0, 1 ),
                bg_opt.getPose().rotation() * cv::Vec3d ( 0, 1, 0 ) );
            window->showWidget ( "SDF cut z", sdfimz );
            sdf_show = cv::Mat();
            for ( int i = 0; i < bg_opt.getVolumeRes() [1]; ++i ) {
                sdf_show.push_back ( esdf.row (
                                         i * bg_opt.getVolumeRes() [1]
                                         + bg_opt.getVolumeRes() [1]/2 ) );
            }
            sdf_show.convertTo ( sdf_show, CV_8U, 255 );
            applyColorMap ( sdf_show, sdf_show, cv::COLORMAP_PARULA );
            cv::viz::WImage3D sdfimy (
                sdf_show, cv::Size2d ( bg_opt.getVolumeSize() [1],
                                       bg_opt.getVolumeSize() [0] ),
                bg_opt.getPose().translation(),
                bg_opt.getPose().rotation() * cv::Vec3d ( 0, -1, 0 ),
                bg_opt.getPose().rotation() * cv::Vec3d ( 0, 0, 1 ) );
            window->showWidget ( "SDF cut y", sdfimy );
        }

        if ( saveOutput ) {
            window->getScreenshot().copyTo ( mesh_vis[frameCount - 1] );
        }
    }
}

void CoSection::writeResults ( const std::string& path ) {
    EMFusion::writeResults ( path );

    boost::filesystem::path p ( path );
    boost::filesystem::create_directories ( p );

    writeOptimMeshes ( p );

    if ( expVols )
        writeOptimVols ( p );
}

void CoSection::writeTimes ( const std::string& path ) {
    boost::filesystem::path p ( path );
    p /= "times";
    boost::filesystem::create_directories ( p );

    boost::filesystem::path bg_path = p / "bg";
    boost::filesystem::create_directories ( bg_path );

    writeTimeArray ( bg_addMeas_runtimes,
                     ( bg_path / "addMeas.csv" ).string() );
    writeTimeArray ( bg_compInter_runtimes,
                     ( bg_path / "compInter.csv" ).string() );
    writeTimeArray ( bg_compHull_runtimes,
                     ( bg_path / "compHull.csv" ).string() );
    writeTimeArray ( bg_optim_runtimes,
                     ( bg_path / "optim.csv" ).string() );

    for ( const auto& objMeas : obj_addMeas_runtimes ) {
        int objId = objMeas.first;

        boost::filesystem::path obj_path = p / std::to_string ( objId );
        boost::filesystem::create_directories ( obj_path );

        writeTimeArray ( obj_addMeas_runtimes[objId],
                         ( obj_path / "addMeas.csv" ).string() );
        writeTimeArray ( obj_compInter_runtimes[objId],
                         ( obj_path / "compInter.csv" ).string() );
        writeTimeArray ( obj_compHull_runtimes[objId],
                         ( obj_path / "compHull.csv" ).string() );
        writeTimeArray ( obj_optim_runtimes[objId],
                         ( obj_path / "optim.csv" ).string() );
    }
}

void CoSection::updateOptimPoses () {
    bg_opt.setPose ( background.getPose() );
    for ( const auto& obj : objects )
        obj_opt.at ( obj.getID() ).setPose ( obj.getPose() );
}

int CoSection::initOrMatchObjs ( const cv::Mat& rgb,
                                 std::map<int, cv::cuda::GpuMat>& matches ) {
    std::vector<cv::Rect> bounding_boxes;
    std::vector<cv::Mat> segmentation;
    std::vector<std::vector<double>> scores;

    int numInstances = -1;
    numInstances = runMaskRCNN ( rgb, bounding_boxes, segmentation, scores );

//     std::map<int, cv::cuda::GpuMat> matches;
    std::map<int, std::vector<double>> score_matches;
    std::set<int> unmatchedMasks;

    // Prepare for matching with existing models: Get mask for valid depth
    // measurements and transform points to world coordinates for
    // initialization.
    if ( numInstances > 0 ) {
        seg_gpus.resize ( numInstances );
        for ( int i = 0; i < segmentation.size(); ++i )
            seg_gpus[i].upload ( segmentation[i] );

        computeValidPoints ( normals_in, validPoints );

        transformPoints ( points, pose, points_w );
        transformPoints ( normals_in, cv::Affine3f ( pose.rotation() ),
                          normals_w );

        matchSegmentation ( seg_gpus, scores, matches, score_matches,
                            unmatchedMasks );

        initObjsFromUnmatched ( seg_gpus, scores, unmatchedMasks, matches,
                                score_matches );
        for ( auto& obj : objects ) {
            if ( matches.count ( obj.getID() ) ) {
                obj_pose_offsets[obj.getID()][frameCount] =
                    updateObj ( obj, points_w, matches[obj.getID()],
                                score_matches[obj.getID()] );
                obj.updateExProb ( true );
                obj_opt.at ( obj.getID() ).resize (
                    obj.getVolumeRes(),
                    obj_pose_offsets[obj.getID()][frameCount]
                );
                obj_opt.at ( obj.getID() ).setClassID ( obj.getClassID() );
            } else {
                // Unmatched object -> lower existence probability
                obj.updateExProb ( false );
            }
        }
    }

    return numInstances;
}

void CoSection::initObjsFromUnmatched (
    std::vector<cv::cuda::GpuMat>& seg_gpus,
    const std::vector<std::vector<double>>& scores,
    const std::set<int>& unmatchedMasks,
    std::map<int, cv::cuda::GpuMat>& matches,
    std::map<int, std::vector<double>>& score_matches ) {
    for ( int i : unmatchedMasks ) {
        for ( const auto& obj : objects ) {
            // Get reprojected mask for current object
            cv::cuda::compare ( modelSegmentation, obj.getID(), obj_mask,
                                cv::CMP_EQ );
            // If mask was matched, join it with the new mask
            if ( matches.count ( obj.getID() ) ) {
                cv::cuda::bitwise_or ( obj_mask, matches[obj.getID()],
                                       obj_mask );
            }

            // Invert mask and do bitwise and to remove existing object mask
            // from new unmatched mask
            cv::cuda::threshold ( obj_mask, obj_mask, 0, 1,
                                  cv::THRESH_BINARY_INV );
            int count_mask_pre = cv::cuda::countNonZero ( seg_gpus[i] );
            cv::cuda::bitwise_and ( seg_gpus[i], obj_mask, seg_gpus[i] );
            // If we removed more than half of the mask in this process, we do
            // not initialize a new volume.
            if ( static_cast<float> ( cv::cuda::countNonZero ( seg_gpus[i] ) )
                    / static_cast<float> ( count_mask_pre ) < .5f ) {
                seg_gpus[i].setTo ( 0 ); // Zero mask will not initialize volume
            }
        }

        cv::cuda::bitwise_and ( validPoints, seg_gpus[i], mask );
        int obj_id = initNewObjVolume ( mask, points_w, normals_w, pose );
        matches.insert ( std::make_pair ( obj_id, seg_gpus[i] ) );
        score_matches.insert ( std::make_pair ( obj_id, scores[i] ) );
    }
}

int CoSection::initNewObjVolume ( const cv::cuda::GpuMat& mask,
                                  const cv::cuda::GpuMat& points,
                                  const cv::cuda::GpuMat& normals,
                                  const cv::Affine3f& pose ) {
    if ( cv::cuda::countNonZero ( mask ) < params.visibilityThresh ) {
        return -1;
    }

    cv::cuda::GpuMat filteredPoints, objFilteredPoints, filteredNormals;
    emf::cuda::EMFusion::filterPoints ( points, mask, filteredPoints );
    emf::cuda::EMFusion::filterPoints ( normals, mask, filteredNormals );

    // Check if overlap with already existing objects is too large
    for ( const auto& obj : objects ) {
        cv::Affine3f obj_pose = obj.getPose().inv();
        cv::cuda::transformPoints ( filteredPoints,
                                    cv::Mat ( obj_pose.rvec().t() ),
                                    cv::Mat ( obj_pose.translation().t() ),
                                    objFilteredPoints );

        // Compute volume boundaries in object coordinates for overlap check
        cv::Vec3f p10, p90;
        emf::cuda::EMFusion::computePercentiles ( objFilteredPoints, p10, p90 );

        // Check if intersection over union > .5
        float iou = volumeIOU ( obj, p10, p90 );
        if ( iou > params.volIOUThresh ) {
            return -1;
        }
    }

    // New object instance is spawned alinged with world coordinate system
    cv::Vec3f p10, p90;
    emf::cuda::EMFusion::computePercentiles ( filteredPoints, p10, p90 );

    cv::Vec3f center = ( p10 + p90 ) / 2;
    center += 0.2f * norm ( p90 - p10 ) * ( center - pose.translation() )
              / norm ( center - pose.translation() );

    // Only initialize volume if center not too far from camera.
    if ( norm ( center - pose.translation() ) > params.distanceThresh ) {
        return -1;
    }

    // Volume size determined by largest axis of percentile difference
    // vector times padding.
    cv::Vec3f dims = p90 - p10;
    float volSize = params.volPad *
                    ( *std::max_element ( dims.val, dims.val + 3 ) );
//     if ( volSize > 1.f ) // Objects > 1m are assumed to be static.
//         return -1;

    // New objects are aligned with the world coordinate system, so only
    // the center is relevant for the pose.
    cv::Affine3f obj_pose ( cv::Matx33f::eye(), center );

    emf::ObjTSDF obj ( params.objVolumeDims, volSize/params.objVolumeDims[0],
                       params.objRelTruncDist * volSize/params.objVolumeDims[0],
                       obj_pose, params.tsdfParams, params.frameSize );

    objects.push_back ( obj );
    vis_objs.insert ( obj.getID() );
    createObj ( obj.getID() );
    new_ids.insert ( obj.getID() );

    obj_poses[obj.getID()][frameCount] = obj.getPose();

    obj_opt.insert ( std::make_pair ( obj.getID(),
                                      ObjOptSDF ( cosecParams.optParams, obj ) )
                   );

    cv::cuda::GpuMat filteredAssoc;
    emf::cuda::EMFusion::filterPoints ( associationWeights[obj.getID()], mask,
                                        filteredAssoc );
    obj_opt.at ( obj.getID() ).addMeasurements ( filteredPoints,
            filteredNormals,
            filteredAssoc );
//     background.removeMeasurements ( filteredPoints );

    std::cout << "Created new Object with ID: " << obj.getID() << std::endl;

    return obj.getID();
}

void CoSection::addMeasurements () {
    cv::cuda::GpuMat ps, ns, as;
    emf::cuda::EMFusion::filterPoints ( points_w, validPoints, ps );
    emf::cuda::EMFusion::filterPoints ( normals_w, validPoints, ns );
    emf::cuda::EMFusion::filterPoints ( bg_associationWeights, validPoints,
                                        as );

    double t = ( double ) cv::getTickCount();
    bg_opt.addMeasurements ( ps, ns, as );
    bg_addMeas_runtimes[frameCount] =
        ( ( double ) cv::getTickCount() - t ) / cv::getTickFrequency();
    for ( auto& obj : obj_opt ) {
        if ( new_ids.find ( obj.first ) != new_ids.end() )
            continue;
        emf::cuda::EMFusion::filterPoints ( associationWeights[obj.first],
                                            validPoints, as );
        t = ( double ) cv::getTickCount();
        obj.second.addMeasurements ( ps, ns, as );
        obj_addMeas_runtimes[obj.first][frameCount] =
            ( ( double ) cv::getTickCount() - t ) / cv::getTickFrequency();
    }
}

void CoSection::computeHull () {
    double t = ( double ) cv::getTickCount();
    bg_opt.computeHull ( depth, pose, params.intr );
    bg_compHull_runtimes[frameCount] = ( ( double ) cv::getTickCount() - t )
                                       / cv::getTickFrequency();

    for ( auto& obj : obj_opt ) {
        if ( params.ignore_person &&
                emf::MaskRCNN::getClassName ( obj.second.getClassID() )
                == "person" ) {
            continue;
        }
        t = ( double ) cv::getTickCount();
        obj.second.computeHull ( depth, pose, params.intr );
        obj_compHull_runtimes[obj.first][frameCount] =
            ( ( double ) cv::getTickCount() - t ) / cv::getTickFrequency();
    }
}

void CoSection::clearObjFg () {
    for ( auto& obj : objects ) {
        bg_opt.clearObjFg ( obj );
        for ( auto& opt : obj_opt ) {
            if ( opt.first != obj.getID() )
                opt.second.clearObjFg ( obj );
        }
    }
}

void CoSection::compBgInter () {
    bg_compInter_runtimes[frameCount] = 0;
    for ( auto& obj : obj_opt ) {
        double t = ( double ) cv::getTickCount();
        bg_opt.computeIntersection ( obj.second );
        bg_compInter_runtimes[frameCount] +=
            ( ( double ) cv::getTickCount() - t ) / cv::getTickFrequency();
    }
}

void CoSection::optimBg () {
    std::cout << "Optimizing background volume..." << std::endl;
    double t = ( double ) cv::getTickCount();
    bg_opt.optimizeESDF();
    bg_optim_runtimes[frameCount] = ( ( double ) cv::getTickCount() - t )
                                    / cv::getTickFrequency();
}

void CoSection::compObjInter () {
    for ( auto& obj : obj_opt ) {
        obj_compInter_runtimes[obj.first][frameCount] = 0;
        double t = ( double ) cv::getTickCount();
        obj.second.computeIntersection ( bg_opt );
        obj_compInter_runtimes[obj.first][frameCount] +=
            ( ( double ) cv::getTickCount() - t ) / cv::getTickFrequency();
        for ( auto& obj2 : obj_opt )
            if ( obj != obj2 ) {
                t = ( double ) cv::getTickCount();
                obj.second.computeIntersection ( obj2.second );
                obj_compInter_runtimes[obj.first][frameCount] +=
                    ( ( double ) cv::getTickCount() - t )
                    / cv::getTickFrequency();
            }
    }
}

void CoSection::optimObjs () {
    for ( auto& obj : obj_opt ) {
        if ( params.ignore_person &&
                emf::MaskRCNN::getClassName ( obj.second.getClassID() )
                == "person" ) {
            continue;
        }
        std::cout << "Optimizing Object " << obj.first << "..." << std::endl;
        double t = ( double ) cv::getTickCount();
        obj.second.optimizeESDF ();
        obj_optim_runtimes[obj.first][frameCount] =
            ( ( double ) cv::getTickCount() - t ) / cv::getTickFrequency();
    }
}

void CoSection::setupOutput ( bool exp_vols ) {
    EMFusion::setupOutput ( true, exp_vols );
}

void CoSection::writeOptimMeshes ( const boost::filesystem::path& p ) {
    boost::filesystem::path optim_meshes_dir = p / "optim_meshes";
    boost::filesystem::create_directories ( optim_meshes_dir );

    auto mesh = bg_opt.getMesh ();
    writeMesh ( mesh, ( optim_meshes_dir / "mesh_bg.ply" ).string() );

    for ( const auto& mesh : optim_meshes ) {
        std::stringstream filename;
        filename << "mesh_" << mesh.first << ".ply";
        writeMesh ( mesh.second,
                    ( optim_meshes_dir / filename.str() ).string() );
    }

    for ( const auto& bg_mesh : frame_optim_meshes ) {
        boost::filesystem::path bg_mesh_path = optim_meshes_dir / "bg";
        boost::filesystem::create_directories ( bg_mesh_path );
        std::stringstream filename;
        filename << std::setfill ( '0' ) << std::setw ( 4 ) << bg_mesh.first
                 << ".ply";
        writeMesh ( bg_mesh.second,
                    ( bg_mesh_path / filename.str() ).string() );
    }

    for ( const auto& obj_mesh : frame_obj_optim_meshes ) {
        std::stringstream obj_id;
        obj_id << obj_mesh.first;
        boost::filesystem::path obj_mesh_path = optim_meshes_dir / obj_id.str();
        boost::filesystem::create_directories ( obj_mesh_path );
        for ( const auto& mesh : obj_mesh.second ) {
            std::stringstream filename;
            filename << std::setfill ( '0' ) << std::setw ( 4 ) << mesh.first
                     << ".ply";
            writeMesh ( mesh.second,
                        ( obj_mesh_path / filename.str() ).string() );
        }
    }

}

void CoSection::writeOptimVols ( const boost::filesystem::path& p ) {
    boost::filesystem::path esdfs_dir = p / "esdfs";
    boost::filesystem::create_directories ( esdfs_dir );

    for ( const auto& obj_esdf : frame_esdfs ) {
        std::stringstream obj_id;
        obj_id << obj_esdf.first;
        boost::filesystem::path obj_esdf_path = esdfs_dir / obj_id.str();
        boost::filesystem::create_directories ( obj_esdf_path );
        for ( const auto& esdf : obj_esdf.second ) {
            std::stringstream filename;
            filename << "esdf" << std::setfill ( '0' ) << std::setw ( 4 )
                     << esdf.first << ".bin";
            writeVolume ( ( obj_esdf_path / filename.str() ).string(),
                          esdf.second,
                          frame_meta[obj_esdf.first][esdf.first].first,
                          frame_meta[obj_esdf.first][esdf.first].second );

            filename.str ( "" );
            filename << "c" << std::setfill ( '0' ) << std::setw ( 4 )
                     << esdf.first << ".bin";
            writeVolume ( ( obj_esdf_path / filename.str() ).string(),
                          frame_cs[obj_esdf.first][esdf.first],
                          frame_meta[obj_esdf.first][esdf.first].first,
                          frame_meta[obj_esdf.first][esdf.first].second );

            filename.str ( "" );
            filename << "w" << std::setfill ( '0' ) << std::setw ( 4 )
                     << esdf.first << ".bin";
            writeVolume ( ( obj_esdf_path / filename.str() ).string(),
                          frame_ws[obj_esdf.first][esdf.first],
                          frame_meta[obj_esdf.first][esdf.first].first,
                          frame_meta[obj_esdf.first][esdf.first].second );

            filename.str ( "" );
            filename << "dInter" << std::setfill ( '0' ) << std::setw ( 4 )
                     << esdf.first << ".bin";
            writeVolume ( ( obj_esdf_path / filename.str() ).string(),
                          frame_dInter[obj_esdf.first][esdf.first],
                          frame_meta[obj_esdf.first][esdf.first].first,
                          frame_meta[obj_esdf.first][esdf.first].second );

            filename.str ( "" );
            filename << "dHull" << std::setfill ( '0' ) << std::setw ( 4 )
                     << esdf.first << ".bin";
            writeVolume ( ( obj_esdf_path / filename.str() ).string(),
                          frame_dHull[obj_esdf.first][esdf.first],
                          frame_meta[obj_esdf.first][esdf.first].first,
                          frame_meta[obj_esdf.first][esdf.first].second );
        }
    }

    boost::filesystem::path tsdfs_dir = p / "tsdfs";
    boost::filesystem::create_directories ( tsdfs_dir );
    for ( const auto& obj_tsdf : frame_tsdfs ) {
        std::stringstream obj_id;
        obj_id << obj_tsdf.first;
        boost::filesystem::path obj_tsdf_path = tsdfs_dir / obj_id.str();
        boost::filesystem::create_directories ( obj_tsdf_path );
        for ( const auto& tsdf : obj_tsdf.second ) {
            std::stringstream filename;
            filename << "tsdf" << std::setfill ( '0' ) << std::setw ( 4 )
                     << tsdf.first << ".bin";

            writeVolume ( ( obj_tsdf_path / filename.str() ).string(),
                          tsdf.second,
                          frame_meta[obj_tsdf.first][tsdf.first].first,
                          frame_meta[obj_tsdf.first][tsdf.first].second );

            filename.str ( "" );
            filename << "fgProbs" << std::setfill ( '0' ) << std::setw ( 4 )
                     << tsdf.first << ".bin";
            writeVolume ( ( obj_tsdf_path / filename.str() ).string(),
                          frame_fgProbs[obj_tsdf.first][tsdf.first],
                          frame_meta[obj_tsdf.first][tsdf.first].first,
                          frame_meta[obj_tsdf.first][tsdf.first].second );
        }
    }
}

void CoSection::writeTimeArray ( const std::map<int, double>& times,
                                 const std::string& filename ) {
    FILE* file = fopen ( filename.c_str(), "w" );
    if ( !file )
        throw std::runtime_error ( "Could not write computation time file: "
                                   + filename );

    fprintf ( file, "frame;time\n" );
    for ( const auto& time : times )
        fprintf ( file, "%d;%f\n", time.first, time.second );

    fclose ( file );
}

}
