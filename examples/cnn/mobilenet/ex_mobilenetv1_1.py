import imageio
from cr import vision
from cr.vision.image import expand_to_batch, preprocess_caffe
from dirsetup import IMAGES_DIR
from tensorflow.keras.applications.mobilenet import decode_predictions,preprocess_input
# keras mobilenet
from tensorflow.keras.applications.mobilenet import MobileNet
# our mobilenet
from cr.vision.dl.nets.cnn.mobilenet import model_mobilenet



image_path =IMAGES_DIR / 'rose.jpg' # artichoke
image_path =IMAGES_DIR / 'lena.jpg'
image_path =IMAGES_DIR / 'military-raptor.jpg' # warplane
image_path =IMAGES_DIR / 'baboon.png' # baboon
image_path =IMAGES_DIR / 'fighter_jet.jpg' # warplane
image_path =IMAGES_DIR / 'watch.png' # stopwatch
image_path =IMAGES_DIR / 'bear.tif' # brown_bear
image_path =IMAGES_DIR / 'wild_flowers.tif' # greenhouse
image_path =IMAGES_DIR / 'elephant.jpg' # tusker


print (image_path)
# read image in RGB format
image = imageio.imread(str(image_path))
print(image.shape)
image = vision.resize(image, 224, 224)
batch = expand_to_batch(image)
batch = preprocess_input(batch)

b_model = model_mobilenet(weights="imagenet")
b_y = b_model.predict(batch)
b_label = decode_predictions(b_y)
b_label = b_label[0][0]
# print the classification
print('OUR: %s (%.2f%%)' % (b_label[1], b_label[2]*100))


a_model = MobileNet(weights="imagenet")
a_y = a_model.predict(batch)
a_label = decode_predictions(a_y)
a_label = a_label[0][0]
# print the classification
print('KERAS: %s (%.2f%%)' % (a_label[1], a_label[2]*100))




