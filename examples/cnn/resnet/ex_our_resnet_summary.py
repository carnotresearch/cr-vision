from cr.vision.dl.nets.cnn import resnet

model = resnet.model_resnet50()

print(model.summary())