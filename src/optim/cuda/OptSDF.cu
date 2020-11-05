/*
 * This file is part of Co-Section.
 *
 * Copyright (C) 2020 Max-Planck-Gesellschaft.
 * Developed by Michael Strecke <mstrecke at tue dot mpg dot de>.
 * For more information see <https://cosection.is.tue.mpg.de/>.
 * If you use this code, please cite the respective publication as
 * listed on the website.
 */
#include "CoSection/optim/cuda/OptSDF.cuh"

#include <iomanip>

// using namespace emf::cuda;
using emf::cuda::float33;
using emf::cuda::operator+;
using emf::cuda::operator-;
using emf::cuda::operator/;
using emf::cuda::dot;
using emf::cuda::norm;

namespace cosection {
namespace cuda {
namespace OptSDF {

__global__
void kernel_compSDFWeights ( cv::cuda::PtrStepSz<float3> points,
                             cv::cuda::PtrStep<float3> normals,
                             cv::cuda::PtrStep<float> assocW,
                             cv::cuda::PtrStep<float> ws,
                             cv::cuda::PtrStep<float> cs,
                             cv::cuda::PtrStep<float> wcounts,
                             cv::cuda::PtrStep<float> ds, float sigma,
                             float33 rot, float3 trans, int3 volSize,
                             float voxelSize ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= points.cols )
        return;

    const float3 p = points ( 0, i );
    const float3 n = normals ( 0, i );
    const float assoc = assocW ( 0, i );

    const float3 p_o = rot * p + trans;
    const float3 n_o = rot * n;

    const int3 volIdx = make_int3 (
                            ( p_o.x / voxelSize ) + ( volSize.x - 1 ) / 2.f,
                            ( p_o.y / voxelSize ) + ( volSize.y - 1 ) / 2.f,
                            ( p_o.z / voxelSize ) + ( volSize.z - 1 ) / 2.f
                        );

    for ( int x = max ( ( int ) ( volIdx.x - 3 * sigma/voxelSize ), 0 );
            x < min ( ( int ) ( volIdx.x + 3 * sigma/voxelSize ), volSize.x );
            ++x ) {
        for ( int y = max ( ( int ) ( volIdx.y - 3 * sigma/voxelSize ), 0 );
                y < min ( ( int ) ( volIdx.y + 3 * sigma/voxelSize ),
                          volSize.y );
                ++y ) {
            for ( int z = max ( ( int ) ( volIdx.z - 3 * sigma/voxelSize ), 0 );
                    z < min ( ( int ) ( volIdx.z + 3 * sigma/voxelSize ),
                              volSize.z );
                    ++z ) {
                if ( ds ( z * volSize.y + y, x ) <= 0.f ) {
                    float3 v = make_float3 (
                                   ( x - ( volSize.x - 1 ) / 2.f ) * voxelSize,
                                   ( y - ( volSize.y - 1 ) / 2.f ) * voxelSize,
                                   ( z - ( volSize.z - 1 ) / 2.f ) * voxelSize
                               );
                    float w = exp ( - ( ( p_o.x - v.x ) * ( p_o.x - v.x )
                                        + ( p_o.y - v.y ) * ( p_o.y - v.y )
                                        + ( p_o.z - v.z ) * ( p_o.z - v.z ) )
                                    / ( sigma*sigma ) ) * assoc;
                    float c = w * dot ( v - p_o, n_o );

                    atomicAdd ( &ws ( z * volSize.y + y, x ), w );
                    atomicAdd ( &cs ( z * volSize.y + y, x ), c );
                    atomicAdd ( &wcounts ( z * volSize.y + y, x ), 1.f );
                }
            }
        }
    }
}

void compSDFWeights ( const cv::cuda::GpuMat& points,
                      const cv::cuda::GpuMat& normals,
                      const cv::cuda::GpuMat& assocW, cv::cuda::GpuMat& w,
                      cv::cuda::GpuMat& c, cv::cuda::GpuMat& wcount,
                      cv::cuda::GpuMat& d, float sigma,
                      const cv::Matx33f& rot_WO, const cv::Vec3f& trans_WO,
                      const cv::Vec3i& volumeRes, const float voxelSize ) {
    dim3 threads ( 1024 );
    dim3 blocks ( ( points.cols + threads.x - 1 ) / threads.x );

    const float33 rot = * ( float33 * ) rot_WO.val;
    const float3 trans = * ( float3 * ) trans_WO.val;
    const int3 volSize = * ( int3 * ) volumeRes.val;

    kernel_compSDFWeights<<<blocks, threads>>> (
        points, normals, assocW, w, c, wcount, d, sigma, rot, trans, volSize,
        voxelSize );

    cudaDeviceSynchronize();
}

__global__
void kernel_delObjFg ( const cv::cuda::PtrStepSz<bool> probs,
                       cv::cuda::PtrStep<float> w, cv::cuda::PtrStep<float> c,
                       cv::cuda::PtrStep<float> wcount, const float33 rot,
                       const float3 trans, const float thisVoxelSize,
                       const float otherVoxelSize, const int3 thisRes,
                       const int3 otherRes ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_ = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= probs.cols || y_ >= probs.rows || !probs ( y_, x ) )
        return;

    const int y = y_ % thisRes.y;
    const int z = y_ / thisRes.y;

    const float3 v = make_float3 (
                         ( x - ( thisRes.x - 1 ) / 2.f ) * thisVoxelSize,
                         ( y - ( thisRes.y - 1 ) / 2.f ) * thisVoxelSize,
                         ( z - ( thisRes.z - 1 ) / 2.f ) * thisVoxelSize );
    const float3 v_other = rot * v + trans;
    const float3 idx_other = ( v_other / otherVoxelSize )
                             + ( otherRes - 1 ) / 2.f;
    if ( idx_other.x < 0 || idx_other.x >= otherRes.x - 1
            || idx_other.y < 0 || idx_other.y >= otherRes.y - 1
            || idx_other.z < 0 || idx_other.z >= otherRes.z - 1 )
        return;

    const int3 lowIdx = make_int3 ( static_cast<int> ( idx_other.x ),
                                    static_cast<int> ( idx_other.y ),
                                    static_cast<int> ( idx_other.z ) );

    w ( lowIdx.z * otherRes.y + lowIdx.y, lowIdx.x )                 = 0.f;
    w ( lowIdx.z * otherRes.y + lowIdx.y, lowIdx.x + 1 )             = 0.f;
    w ( lowIdx.z * otherRes.y + lowIdx.y + 1, lowIdx.x )             = 0.f;
    w ( lowIdx.z * otherRes.y + lowIdx.y + 1, lowIdx.x + 1 )         = 0.f;
    w ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y, lowIdx.x )         = 0.f;
    w ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y, lowIdx.x + 1 )     = 0.f;
    w ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y + 1, lowIdx.x )     = 0.f;
    w ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y + 1, lowIdx.x + 1 ) = 0.f;

    c ( lowIdx.z * otherRes.y + lowIdx.y, lowIdx.x )                 = 0.f;
    c ( lowIdx.z * otherRes.y + lowIdx.y, lowIdx.x + 1 )             = 0.f;
    c ( lowIdx.z * otherRes.y + lowIdx.y + 1, lowIdx.x )             = 0.f;
    c ( lowIdx.z * otherRes.y + lowIdx.y + 1, lowIdx.x + 1 )         = 0.f;
    c ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y, lowIdx.x )         = 0.f;
    c ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y, lowIdx.x + 1 )     = 0.f;
    c ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y + 1, lowIdx.x )     = 0.f;
    c ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y + 1, lowIdx.x + 1 ) = 0.f;

    wcount ( lowIdx.z * otherRes.y + lowIdx.y, lowIdx.x )                 = 0.f;
    wcount ( lowIdx.z * otherRes.y + lowIdx.y, lowIdx.x + 1 )             = 0.f;
    wcount ( lowIdx.z * otherRes.y + lowIdx.y + 1, lowIdx.x )             = 0.f;
    wcount ( lowIdx.z * otherRes.y + lowIdx.y + 1, lowIdx.x + 1 )         = 0.f;
    wcount ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y, lowIdx.x )         = 0.f;
    wcount ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y, lowIdx.x + 1 )     = 0.f;
    wcount ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y + 1, lowIdx.x )     = 0.f;
    wcount ( ( lowIdx.z + 1 ) * otherRes.y + lowIdx.y + 1, lowIdx.x + 1 ) = 0.f;
}

void delObjFg ( const cv::cuda::GpuMat& probs, cv::cuda::GpuMat& w,
                cv::cuda::GpuMat& c, cv::cuda::GpuMat& wcount,
                const cv::Matx33f& rel_rot, const cv::Vec3f& rel_trans,
                const float thisVoxelSize, const float otherVoxelSize,
                const cv::Vec3i& thisVolumeRes,
                const cv::Vec3i& otherVolumeRes ) {
    dim3 threads ( 16, 16 );
    dim3 blocks ( ( probs.cols + threads.x - 1 ) / threads.x,
                  ( probs.rows + threads.y - 1 ) / threads.y );

    const float33 rot = * ( float33 * ) rel_rot.val;
    const float3 trans = * ( float3 * ) rel_trans.val;

    const int3 thisRes = * ( int3 * ) thisVolumeRes.val;
    const int3 otherRes = * ( int3 * ) otherVolumeRes.val;

    kernel_delObjFg<<<blocks, threads>>> ( probs, w, c, wcount, rot,
                                           trans, thisVoxelSize, otherVoxelSize,
                                           thisRes, otherRes );
    cudaDeviceSynchronize();
}

inline __device__
float Dxx ( const float* u, const int idx, const int step, const int gidx,
            const int gstep, const int limit ) {
    if ( ( gidx / gstep ) % limit > 0 && ( gidx / gstep ) % limit < limit - 1 )
        return u[idx - step] - 2 * u[idx] + u[idx + step];
    else
        return 0.f;
}

inline __device__
float DxxT ( const float* u, const int idx, const int step, const int gidx,
             const int gstep, const int limit ) {
    return Dxx ( u, idx - step, step, gidx - gstep, gstep, limit )
           - 2 * Dxx ( u, idx, step, gidx, gstep, limit )
           + Dxx ( u, idx + step, step, gidx + gstep, gstep, limit );
}

inline __device__
float Dxy ( const float* u, const int idx, const int xstep,
            const int ystep, const int x, const int y, const int maxX,
            const int maxY ) {
    const int xshiftm = x > 0 ? 1 : 0;
    const int xshiftp = x < maxX - 1 ? 1 : 0;
    const int yshiftm = y > 0 ? 1 : 0;
    const int yshiftp = y < maxY - 1 ? 1 : 0;

    float ret = u[idx - xstep * xshiftm - ystep * yshiftm]
                - u[idx + xstep * xshiftp - ystep * yshiftm]
                - u[idx - xstep * xshiftm + ystep * yshiftp]
                + u[idx + xstep * xshiftp + ystep * yshiftp];
    if ( x > 0 && x < maxX - 1 )
        ret /= 2.f;
    if ( y > 0 && y < maxY - 1 )
        ret /= 2.f;

    return ret;
}

inline __device__
float DxyT ( const float* u, const int idx, const int xstep,
             const int ystep, const int x, const int y, const int maxX,
             const int maxY ) {
    float dxyt = 0.f;
    char signconfig = 0b1001; // 1: plus, 0: minus
    if ( x == 0 )
        signconfig ^= 0b0101;
    else if ( x == maxX - 1 )
        signconfig ^= 0b1010;

    if ( y == 0 )
        signconfig ^= 0b0011;
    else if ( y == maxY - 1 )
        signconfig ^= 0b1100;

    for ( int i = 0; i < 4; ++i ) {
        int xshift = -1 + ( ( i & 1 ) << 1 );
        xshift = x + xshift >= 0 && x + xshift < maxX ? xshift : 0;
        int yshift = -1 + ( i & 2 );
        yshift = y + yshift >= 0 && y + yshift < maxY ? yshift : 0;

        float temp = Dxy ( u, idx + xstep * xshift + ystep * yshift,
                           xstep, ystep, x + xshift, y + yshift, maxX, maxY );
        if ( x + xshift > 0 && x + xshift < maxX - 1 )
            temp /= 2.f;
        if ( y + yshift > 0 && y + yshift < maxY - 1 )
            temp /= 2.f;

        if ( ( signconfig >> i ) & 1 )
            dxyt += temp;
        else
            dxyt -= temp;
    }
    return dxyt;
}

__global__
void kernel_multB ( const cv::cuda::PtrStepSz<float> u,
                    const cv::cuda::PtrStep<float> w,
                    const cv::cuda::PtrStep<float> dHull,
                    const cv::cuda::PtrStep<float> dInter,
                    cv::cuda::PtrStep<float> u_new, const int3 volSize,
                    const float voxelSize, const float alpha,
                    const float betaHull, const float betaInter ) {
    extern __shared__ float data[];

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if ( x < 0 || x >= volSize.x || y < 0 || y >= volSize.y
            || z < 0 || z >= volSize.z )
        return;

    const int bidx = ( ( threadIdx.z + 2 ) * ( blockDim.y + 4 )
                       + threadIdx.y + 2 ) * ( blockDim.x + 4 )
                     + threadIdx.x + 2;
    const int idx = ( z * volSize.y + y ) * volSize.x + x;

    // Load data from global to shared mem
    data[bidx] = u.ptr() [idx];
    if ( threadIdx.x < 2 && x > 2 ) {
        data[bidx - 2] = u.ptr() [idx - 2];

        if ( threadIdx.y < 2 && y > 2 )
            data[bidx - 2 - 2 * ( blockDim.x + 4 )] =
                u.ptr() [idx - 2 - 2 * volSize.x];
        else if ( threadIdx.y >= blockDim.y - 2 && y + 2 < volSize.y )
            data[bidx - 2 + 2 * ( blockDim.x + 4 )] =
                u.ptr() [idx - 2 + 2 * volSize.x];
        if ( threadIdx.z < 2 && z > 2 )
            data[bidx - 2 - 2 * ( blockDim.x + 4 ) * ( blockDim.y + 4 )] =
                u.ptr() [idx - 2 - 2 * volSize.x * volSize.y];
        else if ( threadIdx.z >= blockDim.z - 2 && z + 2 < volSize.z )
            data[bidx - 2 + 2 * ( blockDim.x + 4 ) * ( blockDim.y + 4 )] =
                u.ptr() [idx - 2 + 2 * volSize.x * volSize.y];
    } else if ( threadIdx.x >= blockDim.x - 2 && x + 2 < volSize.x ) {
        data[bidx + 2] = u.ptr() [idx + 2];

        if ( threadIdx.y < 2 && y > 2 )
            data[bidx + 2 - 2 * ( blockDim.x + 4 )] =
                u.ptr() [idx + 2 - 2 * volSize.x];
        else if ( threadIdx.y >= blockDim.y - 2 && y + 2 < volSize.y )
            data[bidx + 2 + 2 * ( blockDim.x + 4 )] =
                u.ptr() [idx + 2 + 2 * volSize.x];
        if ( threadIdx.z < 2 && z > 2 )
            data[bidx + 2 - 2 * ( blockDim.x + 4 ) * ( blockDim.y + 4 )] =
                u.ptr() [idx + 2 - 2 * volSize.x * volSize.y];
        else if ( threadIdx.z >= blockDim.z - 2 && z + 2 < volSize.z )
            data[bidx + 2 + 2 * ( blockDim.x + 4 ) * ( blockDim.y + 4 )] =
                u.ptr() [idx + 2 + 2 * volSize.x * volSize.y];
    }

    if ( threadIdx.y < 2 && y > 2 ) {
        data[bidx - 2 * ( blockDim.x + 4 )] = u.ptr() [idx - 2 * volSize.x];

        if ( threadIdx.z < 2 && z > 2 )
            data[bidx - 2 * ( blockDim.x + 4 )
                      - 2 * ( blockDim.x + 4 ) * ( blockDim.y + 4 )] =
                     u.ptr() [idx - 2 * volSize.x - 2 * volSize.x * volSize.y];
        else if ( threadIdx.z >= blockDim.z - 2 && z + 2 < volSize.z )
            data[bidx - 2 * ( blockDim.x + 4 )
                      + 2 * ( blockDim.x + 4 ) * ( blockDim.y + 4 )] =
                     u.ptr() [idx - 2 * volSize.x + 2 * volSize.x * volSize.y];
    } else if ( threadIdx.y >= blockDim.y - 2 && y + 2 < volSize.y ) {
        data[bidx + 2 * ( blockDim.x + 4 )] = u.ptr() [idx + 2 * volSize.x];

        if ( threadIdx.z < 2 && z > 2 )
            data[bidx + 2 * ( blockDim.x + 4 )
                      - 2 * ( blockDim.x + 4 ) * ( blockDim.y + 4 )] =
                     u.ptr() [idx + 2 * volSize.x - 2 * volSize.x * volSize.y];
        else if ( threadIdx.z >= blockDim.z - 2 && z + 2 < volSize.z )
            data[bidx + 2 * ( blockDim.x + 4 )
                      + 2 * ( blockDim.x + 4 ) * ( blockDim.y + 4 )] =
                     u.ptr() [idx + 2 * volSize.x + 2 * volSize.x * volSize.y];
    }

    if ( threadIdx.z < 2 && z > 2 )
        data[bidx - 2 * ( blockDim.x + 4 ) * ( blockDim.y + 4 )] =
            u.ptr() [idx - 2 * volSize.x * volSize.y];
    else if ( threadIdx.z >= blockDim.z - 2 && z + 2 < volSize.z )
        data[bidx + 2 * ( blockDim.x + 4 ) * ( blockDim.y + 4 )] =
            u.ptr() [idx + 2 * volSize.x * volSize.y];
    __syncthreads();

    const float dxxu = DxxT ( data, bidx, 1, idx, 1, volSize.x );
    const float dyyu = DxxT ( data, bidx, ( blockDim.x + 4 ), idx,
                              volSize.x, volSize.y );
    const float dzzu = DxxT ( data, bidx,
                              ( blockDim.x + 4 ) * ( blockDim.y + 4 ), idx,
                              volSize.x * volSize.y, volSize.z );

    const float dxyu = DxyT ( data, bidx, 1, ( blockDim.x + 4 ), x, y,
                              volSize.x, volSize.y );
    const float dxzu = DxyT ( data, bidx, 1,
                              ( blockDim.x + 4 ) * ( blockDim.y + 4 ), x, z,
                              volSize.x, volSize.z );
    const float dyzu = DxyT ( data, bidx, ( blockDim.x + 4 ),
                              ( blockDim.x + 4 ) * ( blockDim.y + 4 ), y, z,
                              volSize.y, volSize.z );

    const int y_ = idx / u.cols;

    const float hulldist = dHull ( y_, x );
    const float interdist = dInter ( y_, x );
    const float prev_u = u ( y_, x );

    float res = w ( y_, x ) * prev_u
                + alpha // /(voxelSize*voxelSize*voxelSize*voxelSize)
                * ( dxxu + dyyu + dzzu + 2 * ( dxyu + dxzu + dyzu ) );
    if ( hulldist > 0 && hulldist - prev_u > 0 )
        res += betaHull * ( prev_u - hulldist );
    if ( interdist > 0 && interdist - prev_u > 0 )
        res += betaInter * ( prev_u - interdist );

    u_new ( y_, x ) = res;
}

inline __device__
float DxxTDiag ( const int idx, const int limit ) {
    if ( idx > 1 && idx < limit - 2 )
        return 6.f;
    else if ( idx == 1 || idx == limit - 2 )
        return 5.f;
    else
        return 1.f;
}


inline __device__
float DxyTDiag ( const int x, const int y, const int maxX, const int maxY ) {
    if ( x > 1 && x < maxX - 2 && y > 1 && y < maxY - 2 )
        return .25f; // 4 * (1/4)^2 = 1/4 = .25f
    else if ( ( ( x <= 1 || x >= maxX - 2 ) && ( y > 1 && y < maxY - 2 ) )
              || ( ( y <= 1 || y >= maxY - 2 ) && ( x > 1 && x < maxX - 2 ) ) )
        return .625f; // 2 * (1/2)^2 + 2 * (1/4)^2 = 1/2 + 1/8 = 5/8 = .625f
    else
        return 1.5625f; // 1 + 2*(1/2)^2 + (1/4)^2 = 1 + 1/2 + 1/16 = 25/16 = 1.5625f
}

__global__
void kernel_multDinv ( const cv::cuda::PtrStep<float> u_old,
                       cv::cuda::PtrStepSz<float> u,
                       const cv::cuda::PtrStep<float> w,
                       const cv::cuda::PtrStep<float> dHull,
                       const cv::cuda::PtrStep<float> dInter,
                       const int3 volSize, const float voxelSize,
                       const float alpha, const float betaHull,
                       const float betaInter ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_ = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= u.cols || y_ >= u.rows )
        return;

    const int y = y_ % volSize.y;
    const int z = y_ / volSize.y;

    const float dxxt = DxxTDiag ( x, volSize.x );
    const float dyyt = DxxTDiag ( y, volSize.y );
    const float dzzt = DxxTDiag ( z, volSize.z );

    const float dxyt = DxyTDiag ( x, y, volSize.x, volSize.y );
    const float dxzt = DxyTDiag ( x, z, volSize.x, volSize.z );
    const float dyzt = DxyTDiag ( y, z, volSize.y, volSize.z );

    const float hulldist = dHull ( y_, x );
    const float interdist = dInter ( y_, x );
    const float prev_u = u_old ( y_, x );

    float div = w ( y_, x )
                + alpha // /(voxelSize*voxelSize*voxelSize*voxelSize)
                * ( dxxt + dyyt + dzzt + 2 * ( dxyt + dxzt + dyzt ) );
//     if ( hulldist > 0 && hulldist - prev_u > 0 )
//         div += betaHull * ( 1.f - hulldist / ( prev_u + 1e-6 ) );
//     if ( interdist > 0 && interdist - prev_u > 0 )
//         div += betaInter * ( 1.f - interdist / ( prev_u + 1e-6 ) );

    u ( y_, x ) /= div;
}

void optimizeSDF ( cv::cuda::GpuMat& sdf, const cv::cuda::GpuMat& w,
                   const cv::cuda::GpuMat& c, const cv::cuda::GpuMat& dHull,
                   const cv::cuda::GpuMat& dInter, cv::cuda::GpuMat& buf1,
                   cv::cuda::GpuMat& buf2, const cv::Vec3i& volumeRes,
                   const float voxelSize, const float alpha,
                   const float betaHull, const float betaInter,
                   const int cycleLength ) {
    dim3 threads3 ( 8, 8, 8 );
    dim3 blocks3 ( ( volumeRes[0] + threads3.x - 1 ) / threads3.x,
                   ( volumeRes[1] + threads3.y - 1 ) / threads3.y,
                   ( volumeRes[2] + threads3.z - 1 ) / threads3.z );
    dim3 threads ( 16, 32 );
    dim3 blocks ( ( sdf.cols + threads.x - 1 ) / threads.x,
                  ( sdf.rows + threads.y - 1 ) / threads.y );

    int3 volSize = * ( int3 * ) volumeRes.val;

    createContinuous ( sdf.rows, sdf.cols, CV_32FC1, buf1 );
    createContinuous ( sdf.rows, sdf.cols, CV_32FC1, buf2 );

    // This commented code computes the largest eigenvalue mu of the linear
    // problem. With this computation, omega below could be set to 2/mu.
    // However, we found this not to work well in the experiments and
    // empirically set the value as below.
//     auto begItr = GpuMatBeginItr<float>( buf1 );
//     auto endItr = GpuMatEndItr<float>( buf1 );
//     bool converged;
//     float mu;
//     do {
//         thrust::transform ( thrust::make_counting_iterator(0),
//                             thrust::make_counting_iterator(buf1.cols * buf1.rows),
//                             begItr, prg ( -1, 1 ) );
//         cudaDeviceSynchronize();
//
//         double mu_old = cuda::norm ( buf1, NORM_L2 );
//         cuda::divide( buf1, mu_old, buf1 );
//
//         converged = false;
//         int iter;
//         for ( iter = 0, mu = 0; iter < 1000 && !converged; ++iter ) {
//             kernel_multB<<<blocks3, threads3, ( threads3.x + 4 ) * ( threads3.y + 4 ) * ( threads3.z + 4 ) * sizeof(float)>>> ( buf1, w, dHull, dInter, buf2, volSize, alpha, betaHull, betaInter );
//             cudaDeviceSynchronize();
//             kernel_multDinv<<<blocks, threads>>> ( buf1, buf2, w, dHull, dInter, volSize, alpha, betaHull, betaInter );
//             cudaDeviceSynchronize();
//             mu = cuda::norm( buf2, NORM_L2 );
//     //         std::cout << i << ": " << mu_old << ", " << mu << std::endl;
//             if ( abs ( mu - mu_old ) < 1e-6 ) {
//                 converged = true;
//                 break;
//             }
//             cuda::divide( buf2, mu, buf1 );
//             mu_old = mu;
//         }
//         if ( converged )
//             std::cout << "Eigenvalue computation converged after " << iter << " iterations with mu = " << mu << std::endl;
//         else
//             std::cout << "Eigenvalue computation did not converge!" << std::endl;
//     } while ( !converged );


    float omega = 0.3f;
    std::vector<float> omegas ( cycleLength );
    for ( int i = 0; i < cycleLength; ++i ) {
        float cosval = cos ( M_PI * ( 2 * i + 1 ) / ( 4 * cycleLength + 2 ) );
        omegas[i] = omega / ( 2 * cosval * cosval );
    }
    std::vector<int> lejaorder ( cycleLength );
    float maxVal = 0.f;
    for ( int i = 0; i < cycleLength; ++i )
        if ( 1/omegas[i] > maxVal ) {
            lejaorder[0] = i;
            maxVal = 1/omegas[i];
        }

    for ( int i = 1; i < cycleLength; ++i ) {
        maxVal = 0.f;
        for ( int j = 0; j < cycleLength; ++j ) {
            float prod = 1.f;
            for ( int k = 0; k < i; ++k )
                prod *= abs ( 1/omegas[j] - 1/omegas[lejaorder[k]] );
            if ( prod > maxVal ) {
                lejaorder[i] = j;
                maxVal = prod;
            }
        }
    }

    for ( int i = 0; i < 5000; ++i ) {
        buf1.setTo ( 0 );
        buf2.setTo ( 0 );
        for ( int j = 0; j < cycleLength; ++j ) {
            size_t shmem_sz =
                ( threads3.x + 4 ) * ( threads3.y + 4 ) * ( threads3.z + 4 )
                * sizeof ( float );
            kernel_multB<<<blocks3, threads3, shmem_sz>>> (
                sdf, w, dHull, dInter, buf1, volSize, voxelSize, alpha,
                betaHull, betaInter );
            cudaDeviceSynchronize();
            cv::cuda::subtract ( c, buf1, buf1 );
            kernel_multDinv<<<blocks, threads>>> (
                sdf, buf1, w, dHull, dInter, volSize, voxelSize, alpha,
                betaHull, betaInter );
            cudaDeviceSynchronize();
            cv::cuda::multiply ( omegas[lejaorder[j]], buf1, buf1 );
            cv::cuda::add ( sdf, buf1, sdf );

            cv::cuda::add ( buf2, buf1, buf2 );
        }
        float incrNorm = cv::cuda::norm ( buf2, cv::NORM_L2 )
                         / ( cycleLength * volSize.x * volSize.y * volSize.z );
        if ( incrNorm < 1e-10 ) {
            std::cout << std::endl << "Converged in iteration " << i
                      << " with an increment norm of " << incrNorm << std::endl;
            break;
        }

        std::cout << "\r" << std::setfill ( '0' ) << std::setw ( 4 ) << i
                  << ": " << std::scientific << std::setprecision ( 5 )
                  << incrNorm << std::flush;
    }

    std::cout.unsetf ( std::ios_base::floatfield );
    std::cout << std::endl;
}

__global__
void kernel_compIntersec ( const cv::cuda::PtrStep<float> otherC,
                           cv::cuda::PtrStepSz<float> d, const float33 rot,
                           const float3 trans, const float thisVoxelSize,
                           const float otherVoxelSize, const int3 thisRes,
                           const int3 otherRes ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_ = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= d.cols || y_ >= d.rows )
        return;

    const int y = y_ % thisRes.y;
    const int z = y_ / thisRes.y;

    const float3 v = make_float3 (
                         ( x - ( thisRes.x - 1 ) / 2.f ) * thisVoxelSize,
                         ( y - ( thisRes.y - 1 ) / 2.f ) * thisVoxelSize,
                         ( z - ( thisRes.z - 1 ) / 2.f ) * thisVoxelSize );
    const float3 v_other = rot * v + trans;
    const float3 idx_other = ( v_other / otherVoxelSize )
                             + ( otherRes - 1 ) / 2.f;

    if ( idx_other.x < 0 || idx_other.x > otherRes.x - 2
            || idx_other.y < 0 || idx_other.y > otherRes.y - 2
            || idx_other.z < 0 || idx_other.z > otherRes.z - 2 )
        return;

    const float sdf = emf::cuda::TSDF::interpolateTrilinear ( otherC, idx_other,
                                                              otherRes );

    if ( sdf < 0.f ) {
        d ( y_, x ) = fmaxf ( d ( y_, x ), -sdf );
    }
}

void compIntersec ( const cv::cuda::GpuMat& otherC, cv::cuda::GpuMat& d,
                    const cv::Matx33f& rel_rot, const cv::Vec3f& rel_trans,
                    const float thisVoxelSize, const float otherVoxelSize,
                    const cv::Vec3i& thisVolumeRes,
                    const cv::Vec3i& otherVolumeRes ) {
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( d.cols + threads.x - 1 ) / threads.x,
                  ( d.rows + threads.y - 1 ) / threads.y );

    const float33 rot = * ( float33 * ) rel_rot.val;
    const float3 trans = * ( float3 * ) rel_trans.val;

    const int3 thisRes = * ( int3 * ) thisVolumeRes.val;
    const int3 otherRes = * ( int3 * ) otherVolumeRes.val;

    kernel_compIntersec<<<blocks, threads>>> (
        otherC, d, rot, trans, thisVoxelSize, otherVoxelSize, thisRes,
        otherRes );
    cudaDeviceSynchronize();
}

__global__
void kernel_compHull ( const cv::cuda::PtrStepSz<float> depth,
                       cv::cuda::PtrStep<float> d, cv::cuda::PtrStep<float> w,
                       cv::cuda::PtrStep<float> c,
                       cv::cuda::PtrStep<float> wcount, const float33 rot_OC,
                       const float3 trans_OC, const float33 intr,
                       const float sigma, const int3 volSize,
                       const float voxelSize ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_ = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= volSize.x || y_ >= volSize.y * volSize.z )
        return;

    const int y = y_ % volSize.y;
    const int z = y_ / volSize.y;

    const float3 pos_obj = make_float3 (
                               ( x - ( volSize.x - 1 ) / 2.f ) * voxelSize,
                               ( y - ( volSize.y - 1 ) / 2.f ) * voxelSize,
                               ( z - ( volSize.z - 1 ) / 2.f ) * voxelSize );
    const float3 pos_cam = rot_OC * pos_obj + trans_OC;

    if ( pos_cam.z <= 0.f ) {
        return;
    }

    const float3 proj = intr * pos_cam;

    const int2 pix = make_int2 ( __float2int_rn ( proj.x / proj.z ),
                                 __float2int_rn ( proj.y / proj.z ) );

    if ( pix.x < 0 || pix.x >= depth.cols || pix.y < 0 || pix.y >= depth.rows )
        return;

    const float depthVal = depth ( pix.y, pix.x );
    if ( depthVal <= 0.f ) {
        return;
    }

    const float lambda = norm ( make_float3 (
                                    ( pix.x - intr ( 0, 2 ) ) / intr ( 0, 0 ),
                                    ( pix.y - intr ( 1, 2 ) ) / intr ( 1, 1 ),
                                    1.f ) );

    const float sdf = depthVal - ( 1.f / lambda ) * norm ( pos_cam );

    if ( sdf > 3*sigma*voxelSize ) {
        d ( y_, x ) = fmaxf ( d ( y_, x ), voxelSize );
        w ( y_, x ) = 0.f;
        c ( y_, x ) = 0.f;
        wcount ( y_, x ) = 0.f;
    }
}

void compHull ( const cv::cuda::GpuMat& depthMap, cv::cuda::GpuMat& d,
                cv::cuda::GpuMat& w, cv::cuda::GpuMat& c,
                cv::cuda::GpuMat& wcount, const cv::Matx33f& rel_rot_OC,
                const cv::Vec3f& rel_trans_OC, const cv::Matx33f& intr,
                const float sigma, const cv::Vec3i& volumeRes,
                const float voxelSize ) {
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( d.cols + threads.x - 1 ) / threads.x,
                  ( d.rows + threads.y - 1 ) / threads.y );

    const int3 volSize = * ( int3 * ) volumeRes.val;

    const float33 rot = * ( float33 * ) rel_rot_OC.val;
    const float3 trans = * ( float3 * ) rel_trans_OC.val;

    const float33 camIntr = * ( float33 * ) intr.val;

    kernel_compHull<<<blocks, threads>>> (
        depthMap, d, w, c, wcount, rot, trans, camIntr, sigma, volSize,
        voxelSize );
    cudaDeviceSynchronize();
}

__global__
void kernel_upSample ( const cv::cuda::PtrStepSz<float> lowRes,
                       cv::cuda::PtrStep<float> highRes, const int3 lowVolRes,
                       const int3 highVolRes ) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_ = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= lowRes.cols || y_ >= lowRes.rows )
        return;

    const int y = y_ % lowVolRes.y;
    const int z = y_ / lowVolRes.y;

    float val = lowRes ( y_, x );

    highRes ( z * 2 * highVolRes.y + y * 2, x * 2 ) = val;
    highRes ( z * 2 * highVolRes.y + y * 2, x * 2 + 1 ) = val;
    highRes ( z * 2 * highVolRes.y + y * 2 + 1, x * 2 ) = val;
    highRes ( z * 2 * highVolRes.y + y * 2 + 1, x * 2 + 1 ) = val;
    highRes ( ( z * 2 + 1 ) * highVolRes.y + y * 2, x * 2 ) = val;
    highRes ( ( z * 2 + 1 ) * highVolRes.y + y * 2, x * 2 + 1 ) = val;
    highRes ( ( z * 2 + 1 ) * highVolRes.y + y * 2 + 1, x * 2 ) = val;
    highRes ( ( z * 2 + 1 ) * highVolRes.y + y * 2 + 1, x * 2 + 1 ) = val;
}

void upSample ( const cv::cuda::GpuMat& lowRes, cv::cuda::GpuMat& highRes,
                const cv::Vec3i& lowVolumeRes, const cv::Vec3i& highVolumeRes
              ) {
    dim3 threads ( 32, 32 );
    dim3 blocks ( ( lowRes.cols + threads.x - 1 ) / threads.x,
                  ( lowRes.rows + threads.y - 1 ) / threads.y );

    int3 lowVolRes = * ( int3 * ) lowVolumeRes.val;
    int3 highVolRes = * ( int3 * ) highVolumeRes.val;

    kernel_upSample<<<blocks, threads>>> (
        lowRes, highRes, lowVolRes, highVolRes );
    cudaDeviceSynchronize();
}

}
}
}
