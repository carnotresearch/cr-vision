import cv2
from dirsetup import IMAGES_DIR

from cr import vision


image_path =IMAGES_DIR / 'girl.png'
print (image_path)
image = cv2.imread(str(image_path))

cascade = 'haarcascade_frontalface_default.xml'
cascade_path = vision.ensure_resource(cascade)
faceCascade = cv2.CascadeClassifier(str(cascade_path))


model_path = vision.ensure_resource("lbfmodel.yaml")
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(str(model_path))

# convert image to gray-scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Detect faces in the image
print("Detecting faces")
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60),
    flags = cv2.CASCADE_SCALE_IMAGE
)
print(faces)
print("Detecting landmarks")
ok, landmarks = facemark.fit(gray, faces)
print("landmarks LBF",ok, landmarks)