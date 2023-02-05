import numpy as np
from commen import*


class Conv3x3:

    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3) / 9  # 随机生成卷积核，归一化防止数据溢出

    #  获取每次要卷积的区域
    def convolution_regions(self, image):
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    #  进行卷积操作
    def convolution(self, input_img):
        h, w = input_img.shape
        output_img = np.zeros((h - 2, w - 2, self.num_filters))
        for im_region, i, j in self.convolution_regions(input_img):
            output_img[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return relu(output_img)





