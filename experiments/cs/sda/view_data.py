
import sda
rootdir  = r'E:\datasets\vision\birds\CUB_200_2011\images'
ds = sda.get_dataset(rootdir, size=100, force=True,
    validation=0.1, test=0.1)
training = ds.training_set
print(training.shape)
validation = ds.validation_set
print(validation.shape)
test = ds.test_set
print(test.shape)