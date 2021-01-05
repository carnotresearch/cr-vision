

from tensorflow.keras import backend


def get_center_crop_location(source, destination):
    """
    Returns the center crop area of source which matches 
    with the destination size

    Returns: 

        ((top_crop, bottom_crop), (left_crop, right_crop))    

    """
    src_shape = backend.int_shape(source)
    dst_shape = backend.int_shape(destination)

    # Height to identify top and bottom crop
    src_height = src_shape[1]
    dst_height = dst_shape[1]
    crop_height = src_height - dst_height
    assert crop_height >= 0
    half_height = int(crop_height / 2)
    if (crop_height % 2) != 0:
        # uneven cropping
        crop_top, crop_bottom = half_height, half_height + 1
    else:
        # even cropping
        crop_top, crop_bottom = half_height, half_height

    # Width to identify left and right crop
    src_width = src_shape[2]
    dst_width = dst_shape[2]
    crop_width = src_width - dst_width
    assert crop_width >= 0
    half_width = int(crop_width/2)
    if (crop_width % 2) != 0:
        # uneven cropping
        crop_left, crop_right = half_width, half_width + 1
    else:
        # even cropping
        crop_left, crop_right = half_width, half_width

    # Return the final cropping rectangle
    return (crop_top, crop_bottom), (crop_left, crop_right)