from dataclasses import dataclass

@dataclass
class WeightFile:
    root: str
    folder: str
    name: str
    file_hash: str

    @property
    def uri(self):
        return '/'.join((self.root, self.folder, self.name))
    


FCHOLLET_ROOT = 'https://github.com/fchollet/deep-learning-models'
FCHOLLET_V_0_1 = 'releases/download/v0.1/'


FCHOLLET = {
    "VGG16" : {
        "WITH_TOP" : WeightFile(root=FCHOLLET_ROOT, folder=FCHOLLET_V_0_1,
            name="vgg16_weights_tf_dim_ordering_tf_kernels.h5", 
            file_hash="64373286793e3c8b2b4e3219cbf3544b"),  
        "NO_TOP" : WeightFile(root=FCHOLLET_ROOT, folder=FCHOLLET_V_0_1,
            name="vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
            file_hash="6d6bbae143d832006294945121d1f1fc")
    }
}
