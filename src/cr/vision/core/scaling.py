import cv2


# Source https://stackoverflow.com/questions/4195453/how-to-resize-an-image-with-opencv2-0-and-python2-6

def resize_by_height(image, target_height, interpolation=cv2.INTER_LANCZOS4):
    """Resize `image` to `target_height` (preserving aspect ratio)."""
    src_height, src_width = image.shape[0:2]
    if target_height == src_height:
        # There is nothing to do
        return image
    target_width = int(round(target_height * src_width / src_height))
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)

# Source https://stackoverflow.com/questions/4195453/how-to-resize-an-image-with-opencv2-0-and-python2-6


def resize_by_width(image, target_width, interpolation=cv2.INTER_LANCZOS4):
    """Resize `image` to `target_width` (preserving aspect ratio)."""
    src_height, src_width = image.shape[0:2]
    if target_width == src_width:
        # There is nothing to do
        return image
    target_height = int(round(target_width * src_height / src_width))
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)

def resize_by_max_width(image, max_width=600, interpolation=cv2.INTER_LANCZOS4):
    src_height, src_width = image.shape[0:2]
    if src_width <= max_width:
        # There is nothing to do
        return image
    target_width = max_width
    target_height = int(round(target_width * src_height / src_width))
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def resize(image, target_width=None, target_height=None, interpolation=cv2.INTER_LANCZOS4):
    '''Resize `image` to target width or height  or both

    If only `target_width` is specified, then `target_height` is automatically calculated.
    If only `target_height` is specified, then `target_width` is automatically calculated.
    In these two cases, aspect ratio is preserved.

    Otherwise, image is resized as per specified target width and height.
    '''
    if target_height is None and target_width is None:
        # Neither width nor height has been specified. We return the original image
        return image
    if target_width is None:
        # height is specified and width is to be calculated by using aspect ratio
        return resize_by_height(image, target_height, interpolation)
    if target_height is None:
        # width is specified and height is to be calculated using  aspect ratio
        return resize_by_width(image, target_width, interpolation)
    # both width and height are specified
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def resize_fill(image, target_width, target_height):
    return resize(image, target_width, target_height)


def resize_contain(image, target_width, target_height):
    """
    Increases or decreases the size of an image to fill the box while 
    preserving its aspect ratio
    """
    pass

def resize_cover(image, target_width, target_height):
    return resize_crop(image, target_width, target_height)


def resize_inside(image, target_width, target_height):
    """Preserving aspect ratio, resize an image to be as large as
    possible while ensuring its dimensions are less than or
    equal to both those specified
    """
    src_height, src_width = image.shape[0:2]
    h_r = target_height / src_height
    w_r = target_width / src_width
    if h_r < w_r:
        image = resize_by_height(image, target_height=target_height)
    else:
        image = resize_by_width(image, target_width=target_width)
    return image

def resize_outside(image, target_width, target_height):
    """Preserving aspect ratio, resize an image to be as small as
    possible while ensuring its dimensions are greater than or
    equal to both those specified
    """
    src_height, src_width = image.shape[0:2]
    h_r = target_height / src_height
    w_r = target_width / src_width
    if h_r > w_r:
        image = resize_by_height(image, target_height=target_height)
    else:
        image = resize_by_width(image, target_width=target_width)
    return image

def resize_crop(image, target_width, target_height):
    """Crops the image with a centered rectangle of the specified size
    """
    src_height, src_width = image.shape[0:2]
    if src_height < target_height or src_width < target_width:
        h_r = target_height / src_height
        w_r = target_width / src_width
        if h_r > w_r:
            image = resize_by_height(image, target_height=target_height)
        else:
            image = resize_by_width(image, target_width=target_width)
        # update the src paramters
        src_height, src_width = image.shape[0:2]
        #print(image.shape)


    crop_height = src_height - target_height
    assert crop_height >= 0
    half_height = int(crop_height / 2)
    if (crop_height % 2) != 0:
        # uneven cropping
        crop_top, crop_bottom = half_height, half_height + 1
    else:
        # even cropping
        crop_top, crop_bottom = half_height, half_height

    # Width to identify left and right crop
    crop_width = src_width - target_width
    assert crop_width >= 0
    half_width = int(crop_width/2)
    if (crop_width % 2) != 0:
        # uneven cropping
        crop_left, crop_right = half_width, half_width + 1
    else:
        # even cropping
        crop_left, crop_right = half_width, half_width
    result = image[crop_top:src_height-crop_bottom, crop_left:src_width-crop_right, :]
    return result.copy()


