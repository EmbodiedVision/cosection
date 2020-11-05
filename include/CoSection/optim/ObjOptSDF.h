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

#include <opencv2/cudaarithm.hpp>

#include "CoSection/optim/OptSDF.h"
#include "EMFusion/core/ObjTSDF.h"

namespace cosection {

/**
 * Optimization volume for a single object.
 */
class ObjOptSDF : public OptSDF {
public:
    /**
     * Create a new object optimization volume.
     * @param params optmization parameters
     * @param tsdf object specific TSDF volume from EM-Fusion.
     */
    ObjOptSDF ( const OptParams& params, const emf::ObjTSDF& tsdf );

    /**
     * Define objects to be equal if their IDs match.
     *
     * @param other the object to compare to.
     *
     * @return true, if objects are the same, false otherwise
     */
    bool operator== ( const ObjOptSDF& other ) const;

    /**
     * The inverse of ==.
     *
     * @param other the object to compare to.
     *
     * @return the inverse of operator==.
     */
    bool operator!= ( const ObjOptSDF& other ) const;

    /**
     * Get object id.
     *
     * @return the ID of this object.
     */
    const int getID() const;

    /**
     * Set the semantic class ID with the current class ID of the object in
     * EM-Fusion.
     * @param id the current class id
     */
    void setClassID ( int id );

    /**
     * Get class ID from current class distribution.
     *
     * @return the ID of the class with maximum probability
     */
    const int getClassID () const;

    /**
     * Resize object to fit percentiles of incoming pointcloud with padding.
     *
     * @param newRes new volume resolution (from ObjTSDF resize)
     * @param offset the shift of the volume
     */
    void resize ( const cv::Vec3i& newRes, const cv::Vec3f& offset );

    /**
     * Compute the intersection constraint with the background volume.
     *
     * @param other the background volume.
     */
    void computeIntersection ( const OptSDF& other );

    /**
     * Compute the intersection constraint with other objects.
     *
     * @param other the other object volume.
     */
    void computeIntersection ( const ObjOptSDF& other );

private:
    /** The object id. */
    const int id;
    /** The class id. */
    int classId;
};

}
