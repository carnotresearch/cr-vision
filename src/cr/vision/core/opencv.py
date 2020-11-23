'''
Miscellaneous functions for OpenCV library
'''
import cv2

def get_opencv_version():
    '''Returns the version of OpenCV library'''
    return cv2.__version__


def is_opencv2():
    '''Returns if OpenCV version 2 is installed'''
    return get_opencv_version().startswith("2")

def is_opencv3():
    '''Returns if OpenCV version 3 is installed'''
    return get_opencv_version().startswith("3")

def wait_for_key(milliseconds=10):
    '''Waits for user to press esc key'''
    key = cv2.waitKey(milliseconds) & 0xFF
    return key


def wait_for_esc_key(milliseconds=10):
    '''Waits for user to press esc key'''
    key = cv2.waitKey(milliseconds) & 0xFF
    return key == 27


if __name__ == '__main__':
    print(get_opencv_version())
    print(is_opencv2())
    print(is_opencv3())
