# PyTorch
import torch
import torch.nn as nn

m = nn.LeakyReLU(0.2, inplace=True)
inputs = torch.tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]], dtype=float)
output = m(inputs).to(torch.float32).detach().numpy()
print(output)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

x = Tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype('float32')
m = nn.LeakyReLU(0.2)
output = m(x)
print(output)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]
