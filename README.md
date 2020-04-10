## Introduction

![](imagepriortitle.png)

The [Deep Image Prior](https://en.wikipedia.org/wiki/Deep_Image_Prior) is a convolutional neural network (CNN), designed to solve
various inverse problems in computer vision, such as denoising, inpainting and super-resolution. Unlike other CNNs designed for these kinds of tasks, the Deep Image Prior does not need any training data, besides the corrupted input image itself. Generally speaking, the network is trained to reconstruct the corrupted image from noise. However, since the architecture of the Deep Image Prior fits structured (natural) data a lot faster than random noise, one can observe that in many applications recovering the noiseless image can be done by stopping the training process after a predefined number of iterations. The authors of the paper ([Ulyanov et al.](https://arxiv.org/abs/1711.10925)) explain this as follows:

> [...] although in the limit the parametrization can fit un-
structured noise, it does so very reluctantly. In other words,
the parametrization offers high impedance to noise and low
impedance to signal.

This page features an independent reproduction of some of the results published in the original paper, without making use of the already available open-source code. We will describe the design steps that were necessary to get the architecture running and we will explain which ambiguities had to be resolved when interpreting the text material provided by the authors.

## The Network Architecture
The network architecture consists of several convolutional downsampling blocks followed by convolutional upsampling blocks. Furthermore, after each downsampling block a skip connection is added, which links to a corresponding upsampling layer. For a small visualization, see the figure below (taken from the authors [Supplementary Materials](https://box.skoltech.ru/index.php/s/ib52BOoV58ztuPM#pdfviewer)).
![](Data/Visualization/architecture.png)

We reimplemented this in Python 3, making use of the PyTorch framework. To do so, we separately defined one Module Class for each of these blocks. For the full source-code, you can download and experiment with the Jupyter Notebooks attached in the Notebooks directory of this Git repository.

First of all, the downsampling blocks work as follows: (WIP: To be continued)

```python
class Model_Down(nn.Module):
    """
    Convolutional (Downsampling) Blocks.

    nd = Number of Filters
    kd = Kernel size

    """
    def __init__(self,in_channels, nd = 128, kd = 3, padding = 1, stride = 2):
        super(Model_Down,self).__init__()
        self.padder = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = nd, kernel_size = kd, stride = stride)
        self.bn1 = nn.BatchNorm2d(nd)

        self.conv2 = nn.Conv2d(in_channels = nd, out_channels = nd, kernel_size = kd, stride = 1)
        self.bn2 = nn.BatchNorm2d(nd)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.padder(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.padder(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

```

Instead, we defined the upsampling blocks work as follows:
```python
class Model_Up(nn.Module):
    def __init__(self, in_channels = 132, nu = 128, ku = 3):
        super(Model_Up, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.padder = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = nu, kernel_size = ku, stride = 1, padding = 0)
        self.bn2 = nn.BatchNorm2d(nu)

        self.conv2 =  nn.Conv2d(in_channels = nu, out_channels = nu, kernel_size = 1, stride = 1, padding = 0)
        self.bn3 = nn.BatchNorm2d(nu)

        self.relu = nn.LeakyReLU()

    def forward(self,x):
        x = self.bn1(x)
        x = self.padder(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear')
        return x

```

Meanwhile, we defined the skip blocks in the following way:
```python
class Model_Skip(nn.Module):
    def __init__(self,in_channels = 128 ,stride = 1 , ns = 4, ks = 1, padding = 0):
        super(Model_Skip, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = ns, kernel_size = ks, stride = stride, padding = padding)
        self.bn = nn.BatchNorm2d(ns)
        self.relu = nn.LeakyReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
```

