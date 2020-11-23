'''
Functions for template matching
'''
import numpy as np
import cv2
from cr import vision as iv

def match_template_multscale(image, template):
    '''Matches a template inside an image at multiple scales
    
    Args:
        image (2d-array): Image inside which the template is to be matched 
        Image must be 8 bit gray-level or 32-bit floating point.

        template (2d-array): The template image which is to be searched inside image.

    Returns:
        Cross correlation coefficient, location and scale of matching rectangle
    '''
    # make sure that image is in gray scale
    image = iv.bgr_to_gray(image)
    # make sure that template image is also in gray scale
    template = iv.bgr_to_gray(template)
    # The scales at which we will process the image
    scales = np.linspace(0.2, 1.0, 20)[::-1]
    # Detect edges inside the template image using Canny edge detector
    template_edges = cv2.Canny(template, 50, 200)
    # height and width of original image
    height, width = image.shape[:2]
    # height and width of template
    (template_height, template_width) = template.shape[:2]
    # the location, cross correlation value and scale of best match
    best_match_info = None
    # iterate over scales
    for scale in scales:
        # compute target width for this scale
        target_width = int(width * scale)
        # resize to this target width preserving the aspect ratio
        resized_image = iv.resize_by_width(
            image, target_width=target_width)
        # Measure the actual ratio 
        ratio = float(width) / target_width
        # Height and width of resized image
        resized_height, resized_width = resized_image.shape[:2]
        # Verify that the image is larger than the template
        if resized_height < template_height or resized_width < template_width:
            # The template is bigger than image. It cannot be matched with image
            break
        # Compute the edges in the resized image
        resized_image_edges = cv2.Canny(resized_image, 50, 200)
        # match the edge map of template with the edge map of resized image
        result = cv2.matchTemplate(
            resized_image_edges, template_edges, cv2.TM_CCOEFF)
        # Find the best match
        (_, max_value, _, max_location) = cv2.minMaxLoc(result)
        if best_match_info is None:
            # first best match
            best_match_info = (max_value, max_location, ratio)
        elif max_value > best_match_info[0]:
            # match is better at this scale
            best_match_info = (max_value, max_location, ratio)
    # unpack the details of best match over all scales
    (max_value, max_location, ratio) = best_match_info
    (start_x, start_y) = (
        int(max_location[0] * ratio), int(max_location[1] * ratio))
    (end_x, end_y) = (int((max_location[0] + template_width)
                        * ratio), int((max_location[1] + template_height) * ratio))
    # The best match rectangle
    rectangle = ((start_x, start_y), (end_x, end_y))
    # Return the cross correlation coefficient, location and scale of matching rectangle
    return (max_value, rectangle, 1/ratio)

