import cv2

from skimage.morphology import skeletonize as skel_zhang
from skimage.morphology import skeletonize_3d as skel_lee
from skimage.filters import threshold_sauvola
from skimage import transform as tf

import numpy as np
import math

def threshold(img, method='otsu',blur_size=None, window_size=25):
    
    if blur_size:
        blur = cv2.GaussianBlur(img,(blur_size,blur_size),0)
    
    else:
        blur = img

    if method.strip().lower() == 'otsu':
        
        ret, thresh_img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
    if method.strip().lower() == 'sauvola':
        thresh_values = threshold_sauvola(blur, window_size=window_size)
        thresh_img = (blur > thresh_values).astype(np.uint8) * 255

    return thresh_img

def skeletonize(binary_img, method = 'lee'):
    
    # TODO: Add a check to see if the img passed is in-fact binary
    
    inv_binary_word_img = ~binary_img

    word_skeleton_img = None
    
    if method.strip().lower() == 'zhang':
        word_skeleton_img = skel_zhang(inv_binary_word_img / 255).astype(np.uint8) * 255

    elif method.strip().lower() == 'lee':
        word_skeleton_img = skel_lee(inv_binary_word_img / 255).astype(np.uint8)
    
    else:
        #Todo: raise exception with else unknown method
        pass
    
    return word_skeleton_img


def color_connected_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    return labeled_img

def get_tight_crop(bin_image):
    
    # Works for binary images
    
    img_height, img_width = bin_image.shape[:2]

    upper_bound = 0
    for y1 in range(img_height):
        if 0 in bin_image[y1]:
            upper_bound = y1
            break

    lower_bound = img_height - 1
    for y2 in range(img_height-1, -1, -1):
        if 0 in bin_image[y2]:
            lower_bound = y2
            break

    left_bound = 0
    for x1 in range(0,img_width):
        if 0 in bin_image.T[x1]:
            left_bound = x1
            break

    right_bound = img_width-1
    for x2 in range(img_width-1, -1, -1):
        if 0 in bin_image.T[x2]:
            right_bound = x2
            break

    return upper_bound, lower_bound, left_bound, right_bound


def create_shear(img, angle = math.pi/4):

    # TODO: Hardcoded the padding size for now assuming the image to be of lower resoluion
    # Can be calculated based on the input image size...
    padded_img = cv2.copyMakeBorder(img, top = 50, bottom = 150, left = 50, right = 150,
                                    borderType = cv2.BORDER_CONSTANT, value = (255,255,255))

    cv2.imwrite('padded_img.jpg', padded_img)

    tform = tf.AffineTransform(shear = angle)

    sheared_img = (tf.warp(padded_img,tform, cval=1.0) * 255).astype(np.uint8)

    return sheared_img 



def get_word_stroke_width(bin_word_img):
    # get black pixel count
    fg_pixel_sum = np.sum(bin_word_img==0)

    # Skeletonize it
    skel_word_img = skeletonize(bin_word_img, method='lee')
#     cv2.imwrite('skeletonzied_sauvola.jpg', skel_word_img)

    # get black pixel count
    skeleton_fg_pixel_sum = np.sum(skel_word_img==255)

    # get stroke width
    # assert (skeleton_fg_pixel_sum != 0), "No text found in image"
    if skeleton_fg_pixel_sum == 0:
        print("****Warning: No text found in image!!!****")
        return 0

    stroke_width = fg_pixel_sum / skeleton_fg_pixel_sum

    return stroke_width

def draw_hough_lines(original_img, lines, write_output = False):
    #Ensure that the img is color image with 3 channels
    
    # The below for loop runs till r and theta values  
    # are in the range of the 2d array 
    
    img = original_img.copy()


    # if it's a grayscale image convert to color
    if len(img.shape) == 2:
        print('Converting Grayscale image to color')
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    thetas = []

    for rtheta_pack in lines: 

        r, theta = rtheta_pack[0][0], rtheta_pack[0][1] 
        # Stores the value of cos(theta) in a 

        theta_degree = math.degrees(theta)

        # print(theta_degree)
        if (10 < theta_degree < 60) or (120 < theta_degree < 170):
            thetas.append(theta_degree)

            # print(theta_degree)
            a = np.cos(theta) 

            # Stores the value of sin(theta) in b 
            b = np.sin(theta) 

            # x0 stores the value rcos(theta) 
            x0 = a*r 

            # y0 stores the value rsin(theta) 
            y0 = b*r 

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
            x1 = int(x0 + 1000*(-b)) 

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
            y1 = int(y0 + 1000*(a)) 

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
            x2 = int(x0 - 1000*(-b)) 

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
            y2 = int(y0 - 1000*(a)) 

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
            # (0,0,255) denotes the colour of the line to be  
            #drawn. In this case, it is red.  
            cv2.line(img,(x1,y1), (x2,y2), (0,0,255),1)


    # Default value if no slant is detected
    theta_degree = 0
    if len(thetas)!=0:
        theta_degree = np.mean(thetas)

    if write_output:
        print("shape of image:", img.shape)
        cv2.imwrite('hough_lines_output.png', img)

    return theta_degree


def correct_slant(word_img):
    # sample_word  = word_img_files[4]
#     print('Processing img:',word_img_name, end = ', ')
    
#     # load it
#     color_word_img = cv2.imread(str(sample_word))
#     word_img = cv2.cvtColor(color_word_img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite('inp_word.jpg', word_img)

    # binarize it
    bin_word_img = threshold(word_img,method='sauvola', window_size=11)
#     cv2.imwrite('bin_word_sauvola.jpg', bin_word_img)

    stroke_width = get_word_stroke_width(bin_word_img)
#     print("Hadwritten stroke width:", stroke_width)

    ## Label all the connected components
    skel_word_img = skeletonize(bin_word_img, method='lee')
#     cv2.imwrite('skeleton.jpg', skel_word_img)
    ## Connected components not required yet
#     ret, labels = cv2.connectedComponents(skel_word_img)
#     ccl_img = color_connected_components(labels)
#     # cv2.imwrite('ccl.jpg', ccl_img)

    # Calcluate length of L (vertical structuring element)
    # L = stroke_width / tan(theta)
    # Where theata is the maximum slant w.r.t vertical line
    # Maximum value of theta is 45 degrees, so tan(45) = 1, and L = stroke_width

    L = int(stroke_width)
    # Length of vertical structuring element
    print("L:", L, end = ', ')

    ## Remove horizontal strokes, so that the remaining
    # vertical ones can be used to estimate slant angle
    vse  = cv2.getStructuringElement(cv2.MORPH_RECT, (1,L))
    eroded_img = cv2.erode(~bin_word_img, kernel = vse, iterations=2)
#     cv2.imwrite('eroded_img.jpg', eroded_img)
    dilated_img = cv2.dilate(eroded_img, kernel = vse)
#     cv2.imwrite('dilated_img.jpg', dilated_img)
    
    opened_img = dilated_img
#     opened_img = cv2.morphologyEx(~bin_word_img, cv2.MORPH_OPEN, vse, iterations=1)
#     cv2.imwrite('opened_img.jpg', opened_img)

    inverse_open = ~opened_img
#     cv2.imwrite('inverse_open.jpg', inverse_open)

    # Apply edge detection method on the image 
#     edges = skel_word_img
    edges = cv2.Canny(inverse_open,50,150,apertureSize = 3) 
#     cv2.imwrite('canny_edge.jpg', edges)

    # This returns an array of r and theta values 
    lines = cv2.HoughLines(edges, 2, np.pi/180, 50)

    if lines is None:
        print("No lines detected!!!")
#         cv2.imwrite(str(out_dir_unable / word_img_name), word_img)
        return word_img

    else:    
        print("Num of lines detected:", len(lines))

#         theta_degrees = draw_hough_lines(word_img, lines, write_output=True)
        theta_degrees = draw_hough_lines(word_img, lines)
        # Average slant
        print("Theta:", theta_degrees)

        sheared_img = create_shear(word_img, angle = math.radians(theta_degrees))   
#         cv2.imwrite('sheared.jpg', sheared_img)

        bin_sheared = threshold(sheared_img,method='sauvola', window_size=11)
        # cv2.threshold(sheared_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#         cv2.imwrite('bin_sheared.jpg', bin_sheared)

        y1,y2,x1,x2 = get_tight_crop(bin_sheared)
        cropped_word = sheared_img[y1:y2, x1:x2]
#         cv2.imwrite('sheared_cropped.jpg', cropped_word)
        
        return cropped_word


def viz_splits_final(img,totalSplits):
    for split in totalSplits:
        split = int(split)
        # ratio = self.imgSize[0] / self.imgSize[1]
        # split = int(split * ratio)
        cv2.line(img, (split,0), (split, img.shape[0]), (0,0,255), 1)
    
    return img

def viz_splits_all(img,predictedSplits,confidence_threshold):
    predictedSplits = np.array(predictedSplits)
    # Get the index of splits having confidence greater than threshold!
    predictedSplits = np.where(predictedSplits >= confidence_threshold)[0]

    for split in predictedSplits:
        split = int(split)
        # ratio = self.imgSize[0] / self.imgSize[1]
        # split = int(split * ratio)
        cv2.line(img, (split,0), (split, img.shape[0]), (0,0,255), 1)
    
    return img
