import cv2
from cr import vision
from cr.vision import io
from dirsetup import IMAGES_DIR

image_path =IMAGES_DIR / 'girl.png'

print (image_path)
image = cv2.imread(str(image_path))

faceCascade = vision.faces.CascadeDetector()

gray = vision.to_gray(image)

# Detect faces in the image
faces = faceCascade(gray)

print(faces)


modified = image.copy()

for (x, y, w, h) in faces:
    cv2.rectangle(modified, (x, y), (x+w, y+h), (0, 255, 0), 2)
    

dm = io.DisplayManager(['Image', 'Faces'], gap_x=800)

dm.show_all(image, modified)

key = cv2.waitKey(0) & 0xFF

