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
