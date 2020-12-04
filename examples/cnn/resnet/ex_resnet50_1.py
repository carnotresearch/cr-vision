import imageio
from cr import vision
from cr.vision.image import expand_to_batch, preprocess_caffe
from cr.vision.dl.nets.cnn import resnet
from dirsetup import IMAGES_DIR
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions,preprocess_input



image_path =IMAGES_DIR / 'rose.jpg' # artichoke
image_path =IMAGES_DIR / 'lena.jpg'
image_path =IMAGES_DIR / 'military-raptor.jpg' # warplane
image_path =IMAGES_DIR / 'baboon.png' # baboon
image_path =IMAGES_DIR / 'fighter_jet.jpg' # warplane
image_path =IMAGES_DIR / 'watch.png' # stopwatch
image_path =IMAGES_DIR / 'bear.tif' # brown_bear
image_path =IMAGES_DIR / 'wild_flowers.tif' # greenhouse


print (image_path)
# read image in RGB format
image = imageio.imread(str(image_path))
print(image.shape)
image = vision.resize(image, 224, 224)
batch = expand_to_batch(image)
batch = preprocess_input(batch)

b_model = resnet.model_resnet50(weights="imagenet")
b_y = b_model.predict(batch)
b_label = decode_predictions(b_y)
b_label = b_label[0][0]
# print the classification
print('OUR: %s (%.2f%%)' % (b_label[1], b_label[2]*100))


a_model = ResNet50(weights="imagenet")
a_y = a_model.predict(batch)
a_label = decode_predictions(a_y)
a_label = a_label[0][0]
# print the classification
print('KERAS: %s (%.2f%%)' % (a_label[1], a_label[2]*100))




