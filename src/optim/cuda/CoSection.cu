/*
 * This file is part of Co-Section.
 *
 * Copyright (C) 2020 Max-Planck-Gesellschaft.
 * Developed by Michael Strecke <mstrecke at tue dot mpg dot de>.
 * For more information see <https://cosection.is.tue.mpg.de/>.
 * If you use this code, please cite the respective publication as
 * listed on the website.
 */
#include "CoSection/optim/cuda/CoSection.cuh"

using emf::cuda::operator-;
using emf::cuda::operator/;
using emf::cuda::operator==;
using emf::cuda::norm;
using emf::cuda::cross;

namespace cosection {
namespace cuda {
namespace CoSection {

__global__
void kernel_computeNormals ( const cv::cuda::PtrStepSz<float3> points,
                             cv::cuda::PtrStep<float3> normals ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x < 1 || x >= points.cols - 1 || y < 1 || y >= points.rows - 1
            || points ( y, x + 1 ).z == 0.f || points ( y, x - 1 ).z == 0
            || points ( y + 1, x ).z == 0.f || points ( y - 1, x ).z == 0
            || points ( y, x ) == 0.f )
        return;

    const float3 dx = ( points ( y, x + 1 ) - points ( y, x - 1 ) );
    const float3 dy = ( points ( y + 1, x ) - points ( y - 1, x ) );
    const float3 dir = cross ( dx, dy );

    const float3 normal = dir / norm ( dir );

    normals ( y, x ) = normal.z > 0 ? -normal : normal;
}

void computeNormals ( const cv::cuda::GpuMat& points,
                      cv::cuda::GpuMat& normals ) {
    // TODO: find good thread/block parameters
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( points.cols + threads.x - 1 ) / threads.x,
                  ( points.rows + threads.y - 1 ) / threads.y );
    normals.setTo ( cv::Scalar::all ( 0.f ) );

    kernel_computeNormals<<<blocks, threads>>> ( points, normals );
    cudaDeviceSynchronize();
}

}
}
}