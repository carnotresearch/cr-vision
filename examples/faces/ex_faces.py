import cv2
from cr import vision
from dirsetup import IMAGES_DIR

image_path =IMAGES_DIR / 'girl.png'

print (image_path)
image = cv2.imread(str(image_path))

cascade = 'haarcascade_frontalface_default.xml'

cascade_path = vision.ensure_resource(cascade)

faceCascade = cv2.CascadeClassifier(str(cascade_path))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print(faces)


modified = image.copy()

for (x, y, w, h) in faces:
    cv2.rectangle(modified, (x, y), (x+w, y+h), (0, 255, 0), 2)
    

dm = vision.DisplayManager(['Image', 'Faces'], gap_x=800)

dm.show_all(image, modified)

key = cv2.waitKey(0) & 0xFF

