
import sda
rootdir  = r'E:\datasets\vision\birds\CUB_200_2011\images'
images = sda.get_dataset(rootdir, samples=2000, force=True)
print(images.shape)