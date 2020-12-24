from dataclasses import dataclass

@dataclass
class WeightFile:
    root: str
    folder: str
    name: str
    file_hash: str = None

    @property
    def uri(self):
        return '/'.join((self.root, self.folder, self.name))
    


FCHOLLET_ROOT = 'https://github.com/fchollet/deep-learning-models'
FCHOLLET_V_0_1 = 'releases/download/v0.1/'
FCHOLLET_V_0_2 = 'releases/download/v0.2/'
FCHOLLET_V_0_6 = 'releases/download/v0.6/'


FCHOLLET = {
    "VGG16" : {
        "WITH_TOP" : WeightFile(root=FCHOLLET_ROOT, folder=FCHOLLET_V_0_1,
            name="vgg16_weights_tf_dim_ordering_tf_kernels.h5", 
            file_hash="64373286793e3c8b2b4e3219cbf3544b"),  
        "NO_TOP" : WeightFile(root=FCHOLLET_ROOT, folder=FCHOLLET_V_0_1,
            name="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
            file_hash="6d6bbae143d832006294945121d1f1fc")
    },

    "VGG19" : {
        "WITH_TOP" : WeightFile(root=FCHOLLET_ROOT, folder=FCHOLLET_V_0_1,
            name="vgg19_weights_tf_dim_ordering_tf_kernels.h5", 
            file_hash="cbe5617147190e668d6c5d5026f83318"),  
        "NO_TOP" : WeightFile(root=FCHOLLET_ROOT, folder=FCHOLLET_V_0_1,
            name="vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",
            file_hash="253f8cb515780f3b799900260a226db6")
    },
    "RESNET50" : {
        "WITH_TOP" : WeightFile(root=FCHOLLET_ROOT, folder=FCHOLLET_V_0_2,
            name="resnet50_weights_tf_dim_ordering_tf_kernels.h5", 
            file_hash="a7b3fe01876f51b976af0dea6bc144eb"),  
        "NO_TOP" : WeightFile(root=FCHOLLET_ROOT, folder=FCHOLLET_V_0_2,
            name="resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
            file_hash="a268eb855778b3df3c7506639542a6af")
    },
    "MOBILENET_1_0_224" : {
        "WITH_TOP" : WeightFile(root=FCHOLLET_ROOT, folder=FCHOLLET_V_0_6,
            name="mobilenet_1_0_224_tf.h5"),  
        "NO_TOP" : WeightFile(root=FCHOLLET_ROOT, folder=FCHOLLET_V_0_6,
            name="mobilenet_1_0_224_tf_notop.h5")
    }
}

# Model weights hosted by Keras Team
_KT_ROOT = 'https://github.com/keras-team/keras-applications'
_KT_RELEASES = 'releases/download'
_KT_RESNETS = _KT_RELEASES + '/resnet'


KERAS_TEAM = {
    "RESNET50" : {
        "WITH_TOP" : WeightFile(root=_KT_ROOT, folder=_KT_RESNETS,
            name="resnet50_weights_tf_dim_ordering_tf_kernels.h5", 
            file_hash="2cb95161c43110f7111970584f804107"),  
        "NO_TOP" : WeightFile(root=_KT_ROOT, folder=_KT_RESNETS,
            name="resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
            file_hash="4d473c1dd8becc155b73f8504c6f6626")
    },
    
}
