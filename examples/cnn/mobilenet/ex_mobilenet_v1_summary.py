from tensorflow.keras.applications.mobilenet import MobileNet

from cr.vision.dl.nets.cnn.mobilenet import model_mobilenet

model = model_mobilenet()

print(model.summary())
