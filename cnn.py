import numpy as np
from Affine import*
from Pooling import*
from Convolution import*

conv1 = Conv3x3(3)
pool1 = MaxPool2()
conv2 = Conv3x3(3)
pool2 = MaxPool2()
conv3 = Conv3x3(3)
pool3 = MaxPool2()
layer1 = HideLayer(100)
layer2 = HideLayer(100)
layer3 = HideLayer(100)
softmax = SoftmaxLayer(10)


def cnn(image):
    out = conv1.convolution((image/255)-0.5)
    out = pool1.forward(out)
    out = conv2.convolution(out)
    out = pool2.forward(out)
    out = conv3(out)
    out = pool3
    out = input.flatten(out)
    out = layer1.forward(out)
    out = layer2.forward(out)
    out = layer3.forward(out)
    out = softmax(out)
    print(np.argmax(out))





