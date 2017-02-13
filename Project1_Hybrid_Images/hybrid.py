import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np
import math # For Guassian calculation


# Executes a cross_correlation on a 2D grayscale array
def perform_cross_correlation_grayscale(img, kernel):

    # Determine dimensions
    kernelR = len(kernel)
    kernelC = len(kernel[0])
    imgR = len(img)
    imgC = len(img[0])

    # Do pre-processing, add 0's on the perimeter to account for edges
    # Add width
    w = kernelC
    h = kernelR
    while w > 1:
        img = np.insert(img, 0, 0, axis=1) # Add column of 0's to the left
        img = np.insert(img, len(img[0]), 0, axis=1) # Add column of 0's to the right
        w = w - 2
    # Add Height
    while h > 1:
        img = np.insert(img, 0, 0, axis=0) # Add row of 0's on top
        img = np.insert(img, len(img), 0, axis=0) # Add row of 0's on bototm
        h = h - 2

    # New image
    newImg = np.copy(img)

    # Perform correlation on pre-processed 2D aray
    # Itereate over 2D array within the boundaries before the added 0's
    for i in range(0 + int(kernelR/2), len(img) - int(kernelR/2)):
        for j in range(0 + int(kernelC/2), len(img) - int(kernelC/2)):

            # Apply kernel to the current pixel
            sum = 0
            for ki in range(0, kernelR):
                for kj in range(0, kernelC):
                    # sum += kernel[ki][kj] * img[i + ki - int(kernelR/2)][j + kj - int(kernelC/2)]
                    sum += kernel[ki][kj] * img[i + ki - int(kernelR/2)][j + kj - int(kernelC/2)]
            newImg[i][j] = sum

    # Remove added 0 rows and columns
    w = kernelC
    h = kernelR
    while w > 1:
        newImg = np.delete(newImg, 0, 1) # remove column from left
        newImg = np.delete(newImg, len(newImg[0])-1, 1)# remove column from right
        w = w - 2
    while h > 1:
        newImg = np.delete(newImg, 0, 0) # remove row from top
        newImg = np.delete(newImg, len(newImg)-1, 0)# remove row from bottom
        h = h - 2

    return newImg

# Executes a cross_correlation on a 2D RGB array
def perform_cross_correlation_RGB(img, kernel):

    # Determine dimensions
    kernelR = len(kernel)
    kernelC = len(kernel[0])
    imgR = len(img)
    imgC = len(img[0])

    # Do pre-processing, add 0's on the perimeter to account for edges
    # Add width
    w = kernelC
    h = kernelR
    while w > 1:
        img = np.insert(img, 0, 0, axis=1) # Add column of 0's to the left
        img = np.insert(img, len(img[0]), 0, axis=1) # Add column of 0's to the right
        w = w - 2
    # Add Height
    while h > 1:
        img = np.insert(img, 0, 0, axis=0) # Add row of 0's on top
        img = np.insert(img, len(img), 0, axis=0) # Add row of 0's on bototm
        h = h - 2

    # Perform correlation on pre-processed 2D RGB array
    # Itereate over 2D array within the boundaries before the added 0's
    for i in range(0 + int(kernelR/2), len(img) - int(kernelR/2)):
        for j in range(0 + int(kernelC/2), len(img) - int(kernelC/2)):

            # Apply kernel to the current pixel
            sumOne = 0
            sumTwo = 0
            sumThree = 0
            for ki in range(0, kernelR):
                for kj in range(0, kernelC):
                    sumOne += kernel[ki][kj][0] * img[i + ki - int(kernelR/2)][j + kj - int(kernelC/2)][0]
                    sumTwo += kernel[ki][kj][1] * img[i + ki - int(kernelR/2)][j + kj - int(kernelC/2)][1]
                    sumThree += kernel[ki][kj][2] * img[i + ki - int(kernelR/2)][j + kj - int(kernelC/2)][2]
            newImg[i][j][0] = sumOne
            newImg[i][j][1] = sumTwo
            newImg[i][j][2] = sumThree

    # Remove added 0 rows and columns
    w = kernelC
    h = kernelR
    while w > 1:
        newImg = np.delete(newImg, 0, 1) # remove column from left
        newImg = np.delete(newImg, len(newImg[0])-1, 1)# remove column from right
        w = w - 2
    while h > 1:
        newImg = np.delete(newImg, 0, 0) # remove row from top
        newImg = np.delete(newImg, len(newImg)-1, 0)# remove row from bottom
        h = h - 2

    return newImg

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # RGB image
    if img.shape == 3:
        return perform_cross_correlation_RBG(img, kernel)

    # GrayScale
    else:
        return perform_cross_correlation_grayscale(img, kernel)

    #raise Exception("TODO in hybrid.py not implemented")

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # Flip the kernel horizontally and vertically
    kernel = np.flipud(np.fliplr(kernel))

    # RGB image
    if img.shape == 3:
        return perform_cross_correlation_RGB(img, kernel)

    # Grayscale image
    else:
        return perform_cross_correlation_grayscale(img, kernel)

    # raise Exception("TODO in hybrid.py not implemented")

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''

    # In a gaussian kernel, width means number of rows, and vice versa for cols
    numRows = width
    numCols = height

    # Initialize empty kernely
    kernel = np.zeros((numRows, numCols))

    # Populate the kernel
    for i in range(-int(numRows/2), int(numRows/2)+1):
        for j in range(-int(numCols/2), int(numCols/2)+1):
            lhs = 1 / (2*np.pi*(sigma*sigma))
            rhs = math.pow(np.e, -((math.pow(i,i) + math.pow(j,j))/(2 * (sigma*sigma))))
            kernel[i + int(numRows/2)][j + int(numCols/2)] = lhs * rhs

    # Normalize the matrix
    kernel = kernel/np.sum(kernel)

    return kernel

    #raise Exception("TODO in hybrid.py not implemented")

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # low_pass = gaussian blur without convolving

    # RGB image
    if img.shape == 3:
        return perform_cross_correlation_RGB(img, gaussian_blur_kernel_2d(sigma, size, size))

    # Grayscale image
    else:
        return perform_cross_correlation_grayscale(img, gaussian_blur_kernel_2d(sigma, size, size))

    # raise Exception("TODO in hybrid.py not implemented")

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''

    # Low pass = original image - low pass image
    return np.subtract(img, low_pass(img, sigma, size))

    # raise Exception("TODO in hybrid.py not implemented")

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
