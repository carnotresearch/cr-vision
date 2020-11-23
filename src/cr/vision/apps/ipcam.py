'''
'''

import numpy as np
import cv2
import requests



class IPCamera(object):
    '''Wrapper class for capturing frames from IP cameras'''
    
    def __init__(self, url, user=None, password=None):
        '''Constructor'''
        self.url = url
        self.auth = None
        if user is not None:
            self.auth=(user, password)
        if url.endswith('.mjpg'):
            self.cap = cv2.VideoCapture(url)

    def get_frame(self):
        '''Reads a frame from the URL'''
        if self.cap is not None:
            _, frame = self.cap.read()
            return frame
        else:
            # We use requests library to fetch frames
            response = requests.get(self.url, auth=self.auth)
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, 1)
            return frame

    def release(self):
        '''Releases the video capture object if created'''
        if self.cap is not None:
            self.cap.release()
        return



def ipcam_capture(url, title='IP Cam', frame_rate=1, processor=None):
    '''Captures and displays frames from an IP camera.'''
    camera = IPCamera(url)
    frame_period = int(1000 / frame_rate)
    while True:
        frame = camera.get_frame()
        if frame is not None:
            cv2.imshow(title, frame)
            if processor is not None:
                frame = processor(frame)
        k = cv2.waitKey(frame_period) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    camera.release()



def test_ipcamera(url):
    ''' test function for IPCamera class'''
    camera = IPCamera(url)
    frame = camera.get_frame()
    cv2.imshow('frame', frame)
    cv2.waitKey()
    cv2.destroyAllWindows()


class IPCameraDetails:
    '''Details of an IP camera. IGNORE this class for now'''
    place = ''
    ip_address = ''
    country = ''
    protocol = ''
    path = ''

    def __init__(self, place, ip_address, country=''):
        self.place = place
        self.ip_address = ip_address
        self.country = country

    def get_url(self):
        ''' Returns the URL for the camera feed '''
        raise NotImplementedError


if __name__ == '__main__':
    #camera_url = 'http://83.211.71.120:8084/record/current.jpg'
    camera_url = 'http://webcam01.bigskyresort.com/mjpg/video.mjpg'
    #test_ipcamera(camera_url)
    ipcam_capture(camera_url)
