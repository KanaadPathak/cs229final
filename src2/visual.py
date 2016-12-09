from keras.utils.visualize_util import plot

from resnet50 import ResNet50
from vgg16 import VGG16
from vgg19 import VGG19


def select_architecture(architecture):
    if architecture == 'vgg16':
        return VGG16(weights='imagenet', include_top=False)
    elif architecture == 'vgg19':
        return VGG19(weights='imagenet', include_top=False)
    elif architecture == 'resnet50':
        return ResNet50(weights='imagenet', include_top=False)

a = 'vgg16'
model = select_architecture(a)

plot(model, to_file='%s.png' % a)
