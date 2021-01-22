
import sda
rootdir  = r'E:\datasets\vision\birds\CUB_200_2011\images'
images = sda.get_dataset(rootdir)
print(images.shape)