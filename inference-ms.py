import numpy as np
import time
import cv2
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context

import cv2
import matplotlib.pyplot as plt

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
# context.set_context(mode=context.GRAPH_MODE)

class ResidualBlock(nn.Cell):
    def __init__(self, channels, kernel_size, stride, padding, pad_mode):
        super().__init__()
        self.block = nn.SequentialCell(
            nn.Conv2d(
                channels, channels, kernel_size=kernel_size, stride=stride, pad_mode=pad_mode, padding=padding, has_bias=True
                ),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(
                channels, channels, kernel_size=kernel_size, stride=stride, pad_mode=pad_mode, padding=padding, has_bias=True
                ),
        )

    def construct(self, x):
        #Elementwise Sum (ES)
        return x + self.block(x)

class Generator(nn.Cell):
    def __init__(self, img_channels=3, num_features=32, num_residuals=4, pad_mode="pad"):
        super().__init__()
        self.pad_mode = pad_mode
        self.initial_down = nn.SequentialCell(
            #k7n32s1
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, pad_mode=self.pad_mode, padding=3, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )
        #Down-convolution
        self.down1 = nn.SequentialCell(
            #k3n32s2
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),

            #k3n64s1
            nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )

        self.down2 = nn.SequentialCell(
            #k3n64s2
            nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=2, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),

            #k3n128s1
            nn.Conv2d(num_features*2, num_features*4, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )

        #Bottleneck: 4 residual blocks => 4 times [K3n128s1]
        self.res_blocks = nn.SequentialCell(
            *[ResidualBlock(num_features*4, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1) for _ in range(num_residuals)]
        )

        #Up-convolution
        self.up1 = nn.SequentialCell(
            #k3n128s1 (should be k3n64s1?)
            nn.Conv2d(num_features*4, num_features*2, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )

        self.up2 = nn.SequentialCell(
            #k3n64s1
            nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
            #k3n64s1 (should be k3n32s1?)
            nn.Conv2d(num_features*2, num_features, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )

        self.last = nn.SequentialCell(
            #k3n32s1
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, pad_mode=self.pad_mode, padding=1, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
            #k7n3s1
            nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, pad_mode=self.pad_mode, padding=3, has_bias=True)
        )

    def construct(self, x):
        x1 = self.initial_down(x)
        x2 = self.down1(x1)
        x = self.down2(x2)
        x = self.res_blocks(x)
        x = self.up1(x)


        x = ops.interpolate(x, scale_factor=2.0, mode='area')
        x = self.up2(x + x2)
        x = ops.interpolate(x, scale_factor=2.0, mode='area')
        x = self.last(x + x1)
        #TanH
        return ops.tanh(x)


def camera(network):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()
    
    count = 0
    while True:
        count += 1
        begin = time.time()
        ret, img = cap.read()
        if not ret:
            print("无法读取视频流")
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img[...,::-1] / 255.0 - 0.5) * 2
        img = img.transpose(2, 0, 1)[np.newaxis,:].astype(np.float32)
        inp = mindspore.Tensor(img)
        xg = network(inp)[0]
        xg = (xg + 1) * 0.5
        xg = ops.clamp(xg*255+0.5,0,255)
        xg = xg.permute(1,2,0).asnumpy()[...,::-1]
        # 显示图像
        plt.clf()  # 清除当前图像
        plt.imshow(xg / 255)

        plt.axis('off')
        plt.pause(0.001)  # 暂停一下，以便更新图像
        end = time.time()
        print('process image cost time:{}'.format(end-begin))

def proc_one_img(network, img_path):
    img = cv2.imread(img_path)
    img_h,img_w,_ = img.shape
    print("original image size:",img_h,img_w)
    # n_h, n_w = img_h // 8 * 8, img_w // 8 * 8
    n_h, n_w = 480, 640
    print("resize image to:",n_h,n_w)
    img = cv2.resize(img,(n_w,n_h))
    img = (img[...,::-1] / 255.0 - 0.5) * 2
    img = img.transpose(2, 0, 1)[np.newaxis,:].astype(np.float32)
    begin = time.time()
    inp = mindspore.Tensor(img)
    xg = network(inp)[0]
    xg = (xg + 1) * 0.5
    xg = ops.clamp(xg*255+0.5,0,255)
    xg = xg.permute(1,2,0).asnumpy()[...,::-1]

    end = time.time()
    print('process  image cost time:{}'.format(end-begin))
    cv2.imwrite('output-' + img_path, xg)

if __name__ == "__main__":
    ckpt_path = 'dct-net.ckpt'
    network = Generator(img_channels=3)
    mindspore.load_checkpoint(ckpt_path, network)
    network.set_train(mode=False)
    
    # proc_one_img(network, 'Figure_1.png')

    camera(network)
