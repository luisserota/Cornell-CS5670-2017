import math
import random

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN

        if (i % 2 == 0):
            A[i] = [a_x, a_y, 1, 0, 0, 0, (-1 * a_x * b_x), (-1 * b_x * a_y), (-1 * b_x)]
        else:
            A[i] = [0, 0, 0, a_x, a_y, 1, (-1 * a_x * b_y), (-1 * a_y * b_y), (-1 * b_y)]

        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN

    H = Vt[8].reshape((3,3))

    #TODO-BLOCK-END
    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslation) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call compute_homography.
    #This function should also call get_inliers and, at the end,
    #least_squares_fit.
    #TODO-BLOCK-BEGIN

    max_inliers = []

    # Determine type of motion model
    if m == eTranslate:
        n_matches = 1
    else:
        n_matches = 4

    # Perform RANSAC
    for i in range(nRANSAC):
        m_indexes = []
        # Find random indexes for matches
        while len(m_indexes) < n_matches:
            random = np.random.randint(0, len(matches) - 1)

            # If this index is not already in found matches, add it
            if random not in m_indexes:
                m_indexes.append(random)

        # Determine actual match values from indexes
        m_ransac = []
        for j in m_indexes:
            m_ransac.append(matches[j])
        motion_model = np.eye(3)

        # If motion model type is translation
        if m == eTranslate:
            (a_x, a_y) = f1[m_ransac[0].queryIdx].pt
            (b_x, b_y) = f2[m_ransac[0].trainIdx].pt
            # Determine differences
            xt = b_x - a_x
            yt = b_y - a_y
            # Create the translation matrix
            motion_model = np.array([
            [1, 0, xt],
            [0, 1, yt],
            [0, 0, 1]
            ])
        else:
            motion_model = computeHomography(f1, f2, m_ransac)

        inliers = getInliers(f1, f2, matches, motion_model, RANSACthresh)

        # If there's more inliers than in the current best, replace with new set
        if len(inliers) > len(max_inliers):
            max_inliers = inliers

    # After finding the max inliers, determine least squares fit
    M = leastSquaresFit(f1, f2, matches, m, max_inliers)

    #TODO-BLOCK-END
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #TODO-BLOCK-BEGIN

        m1i = matches[i].queryIdx
        m2i = matches[i].trainIdx

        # Apply the inter-image transformation matrix
        hom_result = M.dot(np.array([
            f1[m1i].pt[0],
            f1[m2i].pt[1],
            1
            ]).T)

        m2 = np.array([
            f2[m1i].pt[0],
            f2[m2i].pt[1]
        ])

        # First match equals the normalized homography
        m1 = hom_result[:2] / hom_result[2]

        dist = np.linalg.norm(m1 - m2)

        if dist <= RANSACthresh:
            inlier_indices.append(i)

        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.
            #TODO-BLOCK-BEGIN

            (a_x, a_y) = f1[matches[inlier_indices[i]].queryIdx].pt
            (b_x, b_y) = f2[matches[inlier_indices[i]].trainIdx].pt

            u = u + (b_x - a_x)
            v = v + (b_y - a_y)

            #TODO-BLOCK-END
            #END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN

        inlier_matches = []
        for i in inlier_indices:
            inlier_matches.append(matches[i])

        M = computeHomography(f1, f2, inlier_matches)

        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M
