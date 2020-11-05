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
#include <iostream>
#include <fstream>

#include <sstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#ifdef HAVE_OPENCV_VIZ
#include <opencv2/viz.hpp>
#endif

#include <sophus/se3.hpp>
#include <opencv2/core/eigen.hpp>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

cv::viz::Viz3d window;
cv::Mat colorMap;

int frameIncr = 1;

void KeyboardCallback ( const cv::viz::KeyboardEvent &w, void *t ) {
    cv::viz::Viz3d *win = ( cv::viz::Viz3d* ) t;

    if ( w.action ) {
        if ( w.symbol == "Left" ) {
            frameIncr = -1;
        }
        if ( w.symbol == "Right" ) {
            frameIncr = 1;
        }
        if ( w.code == ' ' ) {
            frameIncr = 0;
        }
        if ( w.code == 'I' || w.code == 'i' ) {
            cv::Affine3f viewPose = win->getViewerPose();
            std::cout << "\nViewer pose:\n" << viewPose.matrix << std::endl;
        }
    }
}

cv::Mat randomColors () {
    cv::Mat hsv ( 1, 256, CV_32FC3 );

    for ( int i = 1; i < 256; ++i ) {
        hsv.at<cv::Vec3f> ( i ) = cv::Vec3f ( ( i / 256.f ) * 360.f, 1.f, 1.f );
    }

    cv::Mat rgb;
    cv::cvtColor ( hsv, rgb, cv::COLOR_HSV2RGB );

    rgb.convertTo ( rgb, CV_8U, 255 );

    cv::RNG rng ( 6893 );
    randShuffle ( rgb, 1, &rng );

    rgb.at<cv::Vec3b> ( 0 ) = cv::Vec3b ( 255, 255, 255 );

    return rgb;
}

void show3DVis ( const cv::Affine3f& campose, const cv::Matx33f& intr,
                 const cv::Mat& rendered, const cv::viz::Mesh& bg_mesh,
                 const std::map<int, cv::Affine3f >& poses,
                 const std::map<int, cv::viz::Mesh >& meshes,
                 const bool followcampose ) {
    window.removeAllWidgets();
    window.removeAllLights();
    window.addLight ( cv::Vec3d ( 0,-10,-10 ) );

    if ( followcampose )
        window.setViewerPose ( campose );
    else {
        cv::viz::WCameraPosition camPos ( intr, rendered, 0.1 );
        camPos.applyTransform ( campose );
        window.showWidget ( "CamPos", camPos );
    }
    for ( const auto& mesh : meshes ) {
        cv::Affine3f obj_pose = poses.at ( mesh.first );
        cv::viz::Mesh m_obj = mesh.second;
        if ( m_obj.cloud.rows == 1 ) {
            cv::viz::WMesh m_show_obj ( m_obj );
            m_show_obj.setColor (
                cv::viz::Color ( colorMap.at<cv::Vec3b> ( 0, mesh.first ) ) );
            m_show_obj.applyTransform ( obj_pose );
            std::stringstream mesh_str;
            mesh_str << "Mesh " << mesh.first;
            window.showWidget ( mesh_str.str(), m_show_obj );
            window.setRenderingProperty ( mesh_str.str(), cv::viz::SHADING,
                                          cv::viz::SHADING_PHONG );
        }
    }

    if ( bg_mesh.cloud.rows == 1 ) {
        cv::viz::WMesh viz_mesh ( bg_mesh );
        viz_mesh.applyTransform (
            cv::Affine3f().translate ( cv::Vec3f ( 0, 0, 2.56f ) ) );
        window.showWidget ( "mesh", viz_mesh );
        window.setRenderingProperty ( "mesh", cv::viz::SHADING,
                                      cv::viz::SHADING_PHONG );
    }

    window.spinOnce();
}

void main_loop ( const std::string& dirpath = "", const std::string& opath = "",
                 const int startframe = 0, const int endframe = -1,
                 const bool tsdf_flag = false,
                 const bool followcampose = true ) {
    cv::Size frameSize = cv::Size ( 640, 480 );
    float fx, fy, cx, cy;
    fx = fy = 525.f;
    cx = frameSize.width/2 - .5f;
    cy = frameSize.height/2 - .5f;
    cv::Matx33f intr = cv::Matx33f ( fx,  0, cx,
                                     0, fy, cy,
                                     0,  0,  1 );
    std::ifstream calibstr ( dirpath + "/calibration.txt" );
    if ( calibstr.is_open() ) {
        calibstr >> intr ( 0, 0 )
                 >> intr ( 1, 1 )
                 >> intr ( 0, 2 )
                 >> intr ( 1, 2 );

        calibstr >> frameSize.width
                 >> frameSize.height;
    }
    calibstr.close();
    cv::viz::Camera cam ( intr, frameSize );
    window.setCamera ( cam );

    if ( !opath.empty() )
        window.setOffScreenRendering();

    // read in poses
    std::map<int, cv::Affine3f> campose;

    int lastframe = -1;
    std::ifstream infile ( dirpath + std::string ( "/poses-cam.txt" ) );
    if ( infile.is_open() ) {
        std::string line;
        while ( std::getline ( infile, line ) ) {
            std::istringstream iss ( line );
            int t;
            float tx, ty, tz, qx, qy, qz, qw;
            if ( ! ( iss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw ) ) {
                break;
            }

            Sophus::SE3f se3_pose;
            se3_pose.setQuaternion ( Eigen::Quaternionf ( qw, qx, qy, qz ) );
            se3_pose.translation() = Eigen::Vector3f ( tx, ty, tz );

            cv::Affine3f::Mat4 affine;
            cv::eigen2cv ( se3_pose.matrix(), affine );

            campose[t] = cv::Affine3f ( affine );
            lastframe = t;
        }
    }

    std::map<int, std::map<int, cv::Affine3f >> poses;
    size_t obj = 1;
    while ( true ) {
        std::ifstream infile ( dirpath + std::string ( "/poses-" )
                               + std::to_string ( obj )
                               + std::string ( ".txt" ) );
        if ( infile.is_open() ) {
            poses[obj] = std::map<int, cv::Affine3f>();
            std::string line;
            while ( std::getline ( infile, line ) ) {
                std::istringstream iss ( line );
                int t;
                float tx, ty, tz, qx, qy, qz, qw;
                if ( ! ( iss >> t >> tx >> ty >> tz >> qx >> qy >> qz >> qw ) )
                    break;

                Sophus::SE3f se3_pose;
                se3_pose.setQuaternion (
                    Eigen::Quaternionf ( qw, qx, qy, qz ) );
                se3_pose.translation() = Eigen::Vector3f ( tx, ty, tz );

                cv::Affine3f::Mat4 affine;
                cv::eigen2cv ( se3_pose.matrix(), affine );

                poses[obj][t] = cv::Affine3f ( affine );
            }
        } else
            break;

        obj++;
    }

    int frame = startframe, last_keyframe = -1;
    if ( startframe == endframe && opath.empty() )
        frameIncr = 0;
    cv::viz::Mesh bg_mesh;
    std::map<int, cv::viz::Mesh> meshes;
    do {
        int newFrame = tsdf_flag ? frame : ( frame / 10 ) * 10;
        if ( newFrame != last_keyframe ) {
            last_keyframe = newFrame;
            // read in meshes
            std::stringstream bg_meshfile;
            if ( tsdf_flag )
                bg_meshfile << dirpath << "/frame_meshes/bg/" << std::setw ( 4 )
                            << std::setfill ( '0' ) << last_keyframe << ".ply";
            else
                bg_meshfile << dirpath << "/optim_meshes/bg/" << std::setw ( 4 )
                            << std::setfill ( '0' ) << last_keyframe << ".ply";
            bg_mesh = cv::viz::Mesh::load ( bg_meshfile.str() );
            meshes.clear();

            size_t obj = 1;
            for ( obj = 1; obj <= poses.size(); ++obj ) {
                std::stringstream meshfile;
                if ( tsdf_flag )
                    meshfile << dirpath << "/frame_meshes/" << obj << "/"
                             << std::setw ( 4 ) << std::setfill ( '0' )
                             << last_keyframe << ".ply";
                else
                    meshfile << dirpath << "/optim_meshes/" << obj << "/"
                             << std::setw ( 4 ) << std::setfill ( '0' )
                             << last_keyframe << ".ply";
                std::ifstream infile ( meshfile.str() );
                if ( infile.is_open() ) {
                    infile.close();

                    meshes[obj] = cv::viz::Mesh::load ( meshfile.str() );
                }
            }
        }

        std::stringstream renderfile;
        renderfile << dirpath << "/output/" << std::setw ( 4 )
                   << std::setfill ( '0' ) << frame << ".png";
        cv::Mat rendered = cv::imread ( renderfile.str() );

        std::map<int, cv::Affine3f> obj_poses;
        for ( const auto& mesh : meshes ) {
            if ( poses[mesh.first].count ( frame ) )
                obj_poses[mesh.first] = poses[mesh.first][frame];
        }
        show3DVis ( campose[frame], intr, rendered, bg_mesh, obj_poses, meshes,
                    followcampose );
        std::cout << "\rFrame: " << std::setw ( 4 ) << std::setfill ( '0' )
                  << frame << std::flush;
        if ( !opath.empty() ) {
            std::stringstream filename;
            filename << opath << "/" << std::setw ( 4 ) << std::setfill ( '0' )
                     << frame << ".png";
            if ( !imwrite ( filename.str(), window.getScreenshot() ) ) {
                std::cerr << "\nCould not write " << filename.str() << "!"
                          << " Does the output directory exist?" << std::endl;
                exit ( 1 );
            }
        }
        if ( frame + frameIncr <= lastframe && frame + frameIncr >= 0 )
            frame += frameIncr;
    } while ( ( frame <= endframe || opath.empty() ) && !window.wasStopped() );
    std::cout << std::endl;
}


int main ( int argc, char **argv ) {
    colorMap = randomColors();
    bool followcampose = false;

    po::options_description options ( "Render 3D models created by EM-Fusion or"
                                      " Co-Section" );

    po::options_description required ( "Required input" );
    required.add_options ()
    ( "dir,d", po::value<std::string>(),
      "EM-Fusion or Co-Section output directory" )
    ;

    po::options_description optional ( "Optional flags" );
    optional.add_options ()
    ( "help,h", "Print this help" )
    ( "opath,o", po::value<std::string>(),
      "Output path. Should be a folder. Files will be written as "
      "opath/<framenum>.png" )
    ( "frames,f", po::value<std::string>(),
      "Which frame(s) to render. Either single number n or range n-m" )
    ( "tsdf,t", "Render TSDF models instead of optimized ones" )
    ( "pose,p", po::value<std::string>(),
      "Pose from which to render the 3D models" )
    ( "outpose", po::value<std::string>(),
      "Output file to write final pose to (for reproducing results)" )
    ( "followcam", po::bool_switch ( &followcampose ),
      "Whether to follow the camera viewpoint of the recording" )
    ;

    options.add ( required ).add ( optional );
    po::variables_map result;
    po::store ( po::parse_command_line ( argc, argv, options ), result );
    po::notify ( result );

    if ( result.count ( "help" ) || ! ( result.count ( "dir" ) ) ) {
        std::cout << options << std::endl;
        exit ( 1 );
    }

    window = cv::viz::getWindowByName ( "3D Scene" );
    window.spinOnce();

    if ( result.count ( "dir" ) ) {
        std::string dirpath = result["dir"].as<std::string>();
        std::string opath;
        if ( result.count ( "opath" ) ) {
            opath = result["opath"].as<std::string>();
        } else {
            window.registerKeyboardCallback ( KeyboardCallback, &window );
        }

        int startframe = 0, endframe = -1;
        bool tsdf_flag = false;
        if ( result.count ( "frames" ) ) {
            std::istringstream framestr ( result["frames"].as<std::string>() );
            std::vector<std::string> frames;
            std::string f;
            while ( getline ( framestr, f, '-' ) )
                frames.push_back ( f );

            startframe = stoi ( frames[0] );
            if ( frames.size() > 1 )
                endframe = stoi ( frames[1] );
            else
                endframe = startframe;
        }
        if ( result.count ( "tsdf" ) )
            tsdf_flag = true;

        if ( ! followcampose ) {
            if ( result.count ( "pose" ) ) {
                std::ifstream posefile ( result["pose"].as<std::string>() );
                cv::Affine3d campose;
                for ( int i = 0; i < 16; ++i )
                    posefile >> campose.matrix.val[i];
                posefile.close();
                window.setViewerPose ( campose );
            } else {
                cv::Affine3d campose =
                    cv::Affine3d().translate ( cv::Vec3d ( 0, 0, -1 ) );
                window.setViewerPose ( campose );
            }
        }

        main_loop ( dirpath, opath, startframe, endframe, tsdf_flag,
                    followcampose );
    }

    cv::Affine3d v_pose = window.getViewerPose();
    std::cout << "Viewer pose at finish:" << std::endl;
    std::cout << v_pose.matrix << std::endl;
    if ( result.count ( "outpose" ) ) {
        std::ofstream posematfile ( result["outpose"].as<std::string>() );
        for ( int i = 0; i < 16; ++i ) {
            posematfile << v_pose.matrix.val[i];
            if ( ( i+1 ) % 4 == 0 )
                posematfile << "\n";
            else
                posematfile << " ";
        }
        posematfile.close();
    }
}



