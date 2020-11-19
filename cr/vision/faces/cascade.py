import cv2
from cr import vision



class CascadeDetector:
    def __init__(self, 
        cascade='haarcascade_frontalface_default.xml',
        scale_factor=1.1,
        min_neighbors=5,
        min_size=(60,60)
        ):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        cascade_path = vision.ensure_resource(cascade)
        self.classifier = cv2.CascadeClassifier(str(cascade_path))


    def __call__(self, gray_scale_image):
        cascade = self.classifier
        faces = cascade.detectMultiScale(
            gray_scale_image,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        return faces
