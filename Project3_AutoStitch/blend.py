import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.
       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         minX: int for the maximum X value of a corner
         minY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN

    # Compute homographies for corner points
    # This allows us to not need to compute homography for whole image
    corner1 = np.dot(M, np.array([ # top left
        0.0, 0.0, 1
    ], dtype=float).T)
    corner2 = np.dot(M, np.array([ # bottom left
        0.0, len(img), 1
    ], dtype=float).T)
    corner3 = np.dot(M, np.array([ # top right
        len(img[0]), 0.0, 1
    ], dtype=float).T)
    corner4 = np.dot(M, np.array([ # bottom right
        len(img[0]), len(img), 1
    ], dtype=float).T)

    # Normalize the corner
    corner1 = [np.divide(corner1[0],corner1[2]), np.divide(corner1[1],corner1[2])]
    corner2 = [np.divide(corner2[0],corner2[2]), np.divide(corner2[1],corner2[2])]
    corner3 = [np.divide(corner3[0],corner3[2]), np.divide(corner3[1],corner3[2])]
    corner4 = [np.divide(corner4[0],corner4[2]), np.divide(corner4[1],corner4[2])]

    corners = [corner1, corner2, corner3, corner4]

    # Find min and max values
    sortedXs = sorted(i[0] for i in corners)
    minX = sortedXs[1]
    maxX = sortedXs[2]
    sortedYs = sorted(i[1] for i in corners)
    minY = sortedYs[1]
    maxY = sortedYs[2]
    #TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN

    MInverse = np.linalg.inv(M)
    minX, minY, maxX, maxY = imageBoundingBox(img, M)

    for i in range(minY, maxY):
        for j in range(minX, maxX):
            if j < 0 or i < 0 or j >= acc.shape[1] or i >= acc.shape[0]:
            	continue

            newPt = np.dot(MInverse, np.array([j, i, 1.0]).T)
            newPtX = float(newPt[0])/float(newPt[2])
            newPtY = float(newPt[1])/float(newPt[2])

            if newPtX < 0.0 or newPtY < 0.0 or newPtX > img.shape[1] or newPtY > img.shape[0]:
	    		continue

            # Bilinear interpolation:
            int_x1 = int(np.floor(newPtX))
            int_y1 = int(np.floor(newPtY))
            int_x2 = int(np.floor(newPtX + 1))
            int_y2 = int(np.floor(newPtY + 1))

            if int_x1 < 0:
                int_x1 = 0
            elif int_x1 >= img.shape[1]:
                int_x1 = img.shape[1] - 1

            if int_x2 < 0:
                int_x2 = 0
            elif int_x2 >= img.shape[1]:
                int_x2 = img.shape[1] - 1

            if int_y1 < 0:
                int_y1 = 0
            elif int_y1 >= img.shape[0]:
                int_y1 = img.shape[0] - 1

            if int_y2 < 0:
                int_y2 = 0
            elif int_y2 >= img.shape[0]:
                int_y2 = img.shape[0] - 1


            if all(img[int_y2, int_x2, 0:3] == 0):
            	continue

            if all(img[int_y1, int_x2, 0:3] == 0):
            	continue

            if all(img[int_y2, int_x1, 0:3] == 0):
            	continue

            if all(img[int_y1, int_x1, 0:3] == 0):
            	continue


            Q11 = img[int_y1, int_x1]
            Q12 = img[int_y2, int_x1]
            Q21 = img[int_y1, int_x2]
            Q22 = img[int_y2, int_x2]

            if newPtX == int_x1 or newPtX == int_x2:
                fXY1 = Q21
                fXY2 = Q22

            else:
                fXY1 = np.dot((np.divide((int_x2 - newPtX),(int_x2 - int_x1))),(Q11)) + np.dot((np.divide((newPtX - int_x1),(int_x2 - int_x1))),(Q21))
                fXY2 = np.dot((np.divide((int_x2 - newPtX),(int_x2 - int_x1))),(Q12)) + np.dot((np.divide((newPtX - int_x1),(int_x2 - int_x1))),(Q22))

            if newPtY == int_y1 or newPtY == int_y2:
                newPtRGB = fXY1

            else:
                newPtRGB = np.dot((np.divide((int_y2 - newPtY),(int_y2 - int_y1))),(fXY1)) + np.dot((np.divide((newPtY - int_y1),(int_y2 - int_y1))),(fXY2))

            #Feather blending
            blendValues = []
            for a in range(blendWidth):
            	x = a / float(blendWidth)
            	blendValues.append(x)

            RGBacc = acc[i, j, 0:3]
            if all(RGBacc == np.array([0,0,0])):
            	acc[i, j, 0:3] = newPtRGB[0:3]
            	acc[i, j, 3] = 1.0

            elif not all(newPtRGB[0:3] == np.array([0,0,0])):
            	if ((maxX - j) <= blendWidth):
            		weight = 1 - float(blendValues[blendWidth - (maxX - j)])
            		acc[i, j, 0:3] = (1 - weight) * acc[i,j,3] * RGBacc + weight * newPtRGB[0:3]
            		acc[i, j, 3] = weight + (1 - weight) * acc[i,j,3]

            	elif ((j - minX) < blendWidth):
            		weight = float(blendValues[j-minX])
            		acc[i, j, 0:3] = (1 - weight) * acc[i, j, 3] * acc[i, j, 0:3] + weight * newPtRGB[0:3]
            		acc[i, j, 3] = weight + (1-weight) * acc[i, j , 3]

            	else:
            		acc[i, j, 0:3] = newPtRGB[0:3]
            		acc[i, j, 3] = 1.0

    #TODO-BLOCK-END
    # END TODO

def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    img = np.zeros((acc.shape[0], acc.shape[1], 3), dtype=np.uint8)
    #TODO-BLOCK-BEGIN

    for i in range(len(acc)):
        for j in range(len(acc[0])):
            if acc[i, j, 3] > 0:
                img[i, j, 0] = int(acc[i,j,0]/acc[i, j, 3])
                img[i, j, 1] = int(acc[i,j,1]/acc[i, j, 3])
                img[i, j, 2] = int(acc[i,j,2]/acc[i, j, 3])
            else:
                img[i, j, 0] = int(acc[i, j, 0])
                img[i, j, 1] = int(acc[i, j, 1])
                img[i, j, 2] = int(acc[i, j, 2])

    #TODO-BLOCK-END
    # END TODO

    return img

def getAccSize(ipv):
    # Compute bounding box for the mosaic
    minX = sys.maxint
    minY = sys.maxint
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN

        minXBB, minYBB, maxXBB, maxYBB = imageBoundingBox(img, M)
        minX = np.minimum(minX, minXBB)
        maxX = np.maximum(maxX, maxXBB)
        minY = np.minimum(minY, minYBB)
        maxY = np.maximum(maxY, maxYBB)

        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print 'accWidth, accHeight:', (accWidth, accHeight)
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN

    if is360 == True:
        A = computeDrift(x_init, y_init, x_final, y_final, width)

    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage
