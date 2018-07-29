'''
Functions for placing one image inside another
'''
import cv2


def place_image_at(target_image, src_image, roi_mask, x_pos, y_pos):
    '''
    Patches `src_image` into the `target_image` at the location specified 
    by `x_pos` and `y_pos` on the pixels identified by the `roi_mask`

    .. note::

        This function overwrites the roi in target image.
    '''
    rows, cols = roi_mask.shape
    # identify background pixels in source image
    background_mask = cv2.bitwise_not(roi_mask)
    # The region of interest in the target image
    roi = target_image[y_pos:(rows+y_pos), x_pos:(x_pos+cols)]
    # Let's black out the area in the ROI by using the background mask
    roi_background = cv2.bitwise_and(roi, roi, mask=background_mask)
    # Let's prepare the ROI foreground by copying foreground pixels from logo
    roi_foreground = cv2.bitwise_and(src_image, src_image, mask=roi_mask)
    # Let's merge the foreground and background of ROI
    roi_merged = cv2.add(roi_background, roi_foreground)
    # Let's put the ROI with merged logo backinto the image
    target_image[y_pos:(rows+y_pos), x_pos:(x_pos+cols)] = roi_merged
    return target_image
