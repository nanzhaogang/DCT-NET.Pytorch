import numpy as np
import time
import cv2
import mindspore
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context
from mindspore.amp import auto_mixed_precision

# context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O2"})
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

class Infer:
    def __init__(self,model_path):
        self.net = Generator(img_channels=3)
        mindspore.load_checkpoint(model_path, self.net)
        self.net.set_train(mode=False)

    def run(self,img):
        inp = self.preprocess(img)
        self.net = auto_mixed_precision(self.net, 'O2')
        xg = self.net(inp)
        oup = self.postprocess(xg[0])
        return oup

        
    def preprocess(self,img):
        img = (img[...,::-1] / 255.0 - 0.5) * 2
        img = img.transpose(2,0,1)[np.newaxis,:].astype(np.float32)
        return mindspore.Tensor(img)


    def postprocess(self,img):
        img = (img + 1) * 0.5
        img = ops.clamp(img*255+0.5,0,255)
        return img.permute(1,2,0).asnumpy()[...,::-1]
        

if __name__ == "__main__":
    path = 'dct-net.ckpt'
    model = Infer(path)
    img = cv2.imread('gdg.png')

    img_h,img_w,_ = img.shape
    print("original image size:",img_h,img_w)
    n_h, n_w = img_h // 8 * 8, img_w // 8 * 8
    print("resize image to:",n_h,n_w)
    img = cv2.resize(img,(n_w,n_h))
    begin = time.time()
    oup = model.run(img)
    end = time.time()
    print('process image cost time:',end-begin)
    cv2.imwrite('gdg-out.png',oup)
