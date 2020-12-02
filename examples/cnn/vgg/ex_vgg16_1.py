
import imageio
from cr.vision.dl.nets.cnn import vgg
from dirsetup import IMAGES_DIR

image_path =IMAGES_DIR / 'girl.png'

print (image_path)
image = imageio.imread(str(image_path))
print(image.shape)


model  = vgg.model_vgg16(weights="imagenet")
print(model.summary())