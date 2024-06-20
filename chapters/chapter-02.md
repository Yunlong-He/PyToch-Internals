# 初识PyTorch

## 主要内容
- [PyTorch的设计原则](#PyTorch的设计原则)
- [PyTorch的整体架构](#PyTorch的整体架构)
- [PyTorch的源代码结构](#PyTorch的源代码结构)
- [第三方依赖](#第三方依赖)


## PyTorch的设计原则
PyTorch的设计原则是：简单、易用、易学习、可扩展和可移植。

一、简单

PyTorch是一个用于深度学习的开源机器学习框架，为了简化开发者的任务，它采用了简单的设计原则，使框架的学习和使用更容易。PyTorch的简单性体现在几个方面：

1.能够快速、简单地构建神经网络：PyTorch提供了一个丰富的神经网络构建和训练框架，可以快速和简单地构建神经网络，从而节省大量时间。

2.可以使用Python语言：PyTorch使用Python语言，使得开发者可以轻松和快速地构建神经网络，而不必考虑复杂的编程语言。

3.可以使用Python语言的标准库：PyTorch可以使用Python语言的标准库，使得开发者可以轻松地使用Python语言的标准库，而不必考虑复杂的编程语言。

4.提供实用的工具：PyTorch提供了大量实用的工具，如虚拟机、模型训练、数据加载和可视化等，可以让开发者轻松地构建、调试和训练神经网络。

5.提供一些实用的API：PyTorch提供了一些实用的API，可以让开发者轻松地创建、训练和评估神经网络，从而大大减少开发时间。

二、易用

PyTorch采用了易用的设计原则，使开发者可以轻松地使用它来构建、调试和训练神经网络。PyTorch的易用性体现在几个方面：

1.提供丰富的API：PyTorch提供了丰富的API，可以让开发者轻松地创建、调试和训练神经网络，从而节省大量开发时间。

2.提供友好的用户界面：PyTorch提供了友好的用户界面，可以让开发者轻松地使用它来构建、调试和训练神经网络。

3.提供可视化工具：PyTorch提供了可视化工具，可以让开发者轻松地查看神经网络的结构和训练过程，从而更好地理解神经网络的工作原理。

4.提供大量的教程和文档：PyTorch提供了大量的教程和文档，可以让开发者轻松地学习如何使用PyTorch，从而更好地构建、调试和训练神经网络。

三、易学习

PyTorch采用了易学习的设计原则，使开发者可以快速地学习如何使用它来构建、调试和训练神经网络。PyTorch的易学习性体现在几个方面：

1.提供丰富的教程和文档：PyTorch提供了丰富的教程和文档，可以让开发者快速地学习如何使用PyTorch，从而节省大量学习时间。

2.提供可视化工具：PyTorch提供了可视化工具，可以让开发者轻松地查看神经网络的结构和训练过程，从而更好地理解神经网络的工作原理。

3.提供社区支持：PyTorch提供了大量的社区支持，如GitHub、Stack Overflow等，可以让开发者轻松地获得帮助，从而节省大量学习时间。

四、可扩展

PyTorch采用了可扩展的设计原则，使开发者可以轻松地扩展它的功能，从而使其能够更好地适应新的任务。PyTorch的可扩展性体现在几个方面：

1.支持多种语言：PyTorch支持多种语言，如Python、C++、Java等，可以让开发者轻松地扩展其功能，从而更好地适应新的任务。

2.支持多种深度学习框架：PyTorch支持多种深度学习框架，如TensorFlow、Keras等，可以让开发者轻松地扩展其功能，从而更好地适应新的任务。

3.支持自定义网络：PyTorch支持自定义网络，可以让开发者轻松地扩展其功能，从而更好地适应新的任务。

五、可移植

PyTorch采用了可移植的设计原则，使开发者可以轻松地将它部署到任何目标环境，从而实现高性能的部署。PyTorch的可移植性体现在几个方面：

1.支持多种硬件平台：PyTorch支持多种硬件平台，如CPU、GPU、FPGA等，可以让开发者轻松地将它部署到任何目标环境，从而实现高性能的部署。

2.支持多种操作系统：PyTorch支持多种操作系统，如Linux、macOS、Windows等，可以让开发者轻松地将它部署到任何目标环境，从而实现高性能的部署。

3.支持多种部署方式：PyTorch支持多种部署方式，如Docker、Kubernetes等，可以让开发者轻松地将它部署到任何目标环境，从而实现高性能的部署。

总之，PyTorch的设计原则是简单、易用、易学习、可扩展和可移植，这些设计原则使得PyTorch成为一个功能强大、易于使用和可靠的深度学习框架。

为了让开发者能够充分利用硬件的计算能力，同时保持很好的开发效率，PyTorch提供了功能丰富但设计优雅的Python API，并且将繁重的工作交由C++实现。C++部分是以Python扩展的形式工作的。

PyTorch的算子实现在ATen库中。

为了实现自动微分，在ATen之上，PyTorch增加了AutoGrad框架。

### C++扩展
在Python API层，PyTorch早期有Variable和Tensor两种类型，在v0.4.0之后，合并成了现在的Tensor类。

Python层的Tensor对象，到了C++层面用THPVariable来表示，实际C++层面使用时会先转换成C++的Tensor类型。
THP的含义： TH来自一<font color=red>T</font>orc<font color=red>H</font>，P指的是<font color=red>P</font>ython。

PyTorch使用C++扩展有几个原因：

1.C++是一种更高效的语言，可以提高PyTorch的计算速度。

2.C++拥有更强大的API，可以更好地扩展PyTorch的功能。

3.使用C++可以在PyTorch的框架中添加更多的模型，以满足实际应用的需要。

4.C++可以更好地集成现有的计算代码库，以提高编译速度。

5.C++可以提供更好的性能，并有助于改善PyTorch的性能。
### numpy
考虑到numpy的广泛使用，PyTorch支持Tensor与numpy的互相转换。
PyTorch支持Tensor和numpy的互相转换，是因为它们都是多维数组，可以用来表示向量和矩阵，而且可以方便的进行计算。numpy是一个简单的数学计算库，可以帮助PyTorch处理大量的数据，而Tensor是PyTorch中的多维数组，可以用来存储和处理数据。所以Tensor和numpy之间的转换，可以更好地帮助PyTorch处理大量的数据，并且可以节省大量的时间和空间。

### 零拷贝（zero-copy）

在机器学习和深度学习的场景中，tensor代表数据、权重、以及计算结果等，因此经常会出现带有大量数据的tensor。PyTorch中存在大量的tensor创建、计算、转换等场景，如果每一次都重新拷贝一份数据，会带来巨大的内存浪费和性能损失。，因此PyTorch支持零拷贝技术以减少不必要的消耗。如：
```Python
>>> np_array
   array([[1., 1.],
      [1., 1.]])
>>> torch_array = torch.from_numpy(np_array)
>>> torch_array.add_(1.0)
>>> np_array
   array([[2., 2.],
      [2., 2.]])
```
上面这种tensor和numpy array共用数据的操作，我们称为in-place操作，但有时候in-place操作和普通拷贝数据的操作（standard操作）的界限并不是很清楚，需要仔细甄别。

另外需要说明的是，tensor中的数据由Storage对象进行管理，这样就把tensor的元信息与其真正的数据存储解耦了。

### JIT
PyTorch是基于动态图范式的，开发者可以像写普通程序一样，在网络的执行中加上各种条件分支语句，这样做带来的好处是容易理解，方便调试，但是同时也会有效率的影响，尤其是模型训练好之后用于推理，此时模型基本不需要调试，分支条件基本也固定了，这时候效率反而是需要优先考虑的因素了。另外虽然开发的时候使用Python会提高开发效率，但推理的时候需要支持其他的语言及环境，此时需要把模型与原来的Python代码解耦。

对此，PyTorch在v1.0的版本中引入了torch.jit，支持将PyTorch模型转为可序列化及可优化的格式。作为Python静态类型的子集，TorchScrip也被引入到PyTorch中。


## PyTorch的整体架构
PyTorch的整体架构主要由以下几部分组成：

1. 张量：PyTorch使用张量（tensor）作为其基本数据结构，可以被用作深度学习中的变量。

2. 自动求导：PyTorch具有内置的自动求导引擎，可以帮助用户计算梯度，从而更快完成深度学习模型的训练。

3. 神经网络模块：PyTorch提供了一系列神经网络模块，可以用于构建深度学习模型，包括卷积层，池化层，激活函数，正则化层等等。

4. 数据加载和可视化：PyTorch具有内置的数据加载和可视化功能，可以帮助开发者更快完成深度学习模型的训练。

5. 工具箱：PyTorch还提供了一系列工具箱，可以用于调试，训练，优化和部署深度学习模型。

PyTorch的整体架构由四个层次组成：PyTorch核心、模型层、抽象层和应用层。

PyTorch核心层由C++实现。它是PyTorch的核心，包括数据结构和操作，以及实现深度学习模型的底层库。它提供了张量、多维数组、计算图和自动求导等功能，以及tensorflow、caffe2和onnx这些工具的支持。

模型层是一层轻量级的抽象，它是一个高度抽象层，它提供了一种更简单、更快捷的深度学习模型构建方式，它是建立在PyTorch核心层之上的，主要提供深度学习模型的构建模块，比如模型层的模块有神经网络模型、卷积神经网络模型、循环神经网络模型、深度卷积神经网络模型、卷积神经网络模型、深度学习模型等。

抽象层是一层高度抽象的抽象层，它提供了更好的数据处理和模型训练，主要包括数据加载器、训练器、模型评估器、模型保存器和可视化工具等功能模块。

应用层是一层高级的抽象层，它提供了一些高级功能，比如模型部署、API接口、模型的可视化等，这些功能可以帮助开发者快速部署模型，提高开发效率，实现智能化应用。

总的来说，PyTorch架构提供了一种灵活、可扩展的深度学习框架，可以帮助开发者快速构建、部署和维护深度学习模型，实现智能化应用。

## PyTorch的源代码结构

```bash


pytorch
|--- android        # PyTorch for Android， 开发库及运行环境
|--- aten           # 主要是C++ Tensor库的实现
|--- benchamarks    # 对PyTorch进行Benchmarking的代码
|--- binaries       # 用于移动端的厕所
|--- c10            # 核心Tensor实现，支持服务器端和移动端
|--- caffe2         # 从Caffe2项目迁移过来的代码
|--- cmake          # PyTorch的整体编译脚本
|--- docs           # PyTorch文档工具，可以根据Python和C++代码生成对应的文档
|--- ios            # PyTorch for iOS
|--- modules        # 
|--- mypy_plugins   # 
|--- scripts        # 
|--- submodules     # 
|--- test           # 
|--- third_party    # 第三方库
|--- tools          # 
|--- torch          # PyTorch的Python接口
|--- torchgen       # 

torch
|--- csrc       # Torch C++ 扩展模块的实现代码
      |--- module.cpp       # Torch C++ 扩展模块的初始化及入口代码

```
PyTorch的源代码结构能介绍一下么？包括python层和C++层面的，按照目录结构列出来


PyTorch源代码结构：

- Python层：

Pytorch/
|
├── csrc/          # C++源文件
├── caffe2/        # Caffe2源代码
├── tools/         # 工具集
├── build/         # 构建目录
├── docs/          # 文档
├── examples/      # 示例
├── pytorch/       # PyTorch核心库
├── test/          # 测试文件
└── third_party/   # 第三方库

- C++层：

Pytorch/
|
├── aten/             # 静态C++库
├── c10/              # C++源码库
├── csrc/             # C++源文件
├── distributed/      # 分布式库
├── onnx/             # ONNX库
├── qnnpack/          # QNNPACK库
├── third_party/      # 第三方库
├── torch/            # C++源文件
├── torch/csrc/       # C++源文件
├── torch/csrc/api/   # C++接口
└── torch/lib/        # 静态库

### C10

PyTorch的c10库是一个用于编写和维护基于PyTorch的模型及其相关操作的库。它是由Facebook AI Research团队设计和维护的，旨在提供一个统一的、可扩展的API来构建、训练和部署模型。

c10库提供了一个灵活的平台，用于使用PyTorch构建各种类型的模型，包括深度学习、计算机视觉和自然语言处理等。它还提供了模型训练所需的各种工具，其中包括数据加载器、优化器、损失函数、指标函数和模型部署等。

c10库的核心是一个用于构建和维护PyTorch模型的框架，它允许用户以一种更统一的方式编写PyTorch模型。它使用一种称为“模块”的抽象来定义模型，这样可以更容易地构建复杂的模型，同时保持模型的可视性和可调试性。

c10库还提供了一个用于训练和推理的计算图，它可以让模型定义更加灵活，并允许更多的灵活性来构建复杂的模型。它还支持在GPU上训练和部署模型，以及支持分布式训练的功能。

总的来说，PyTorch的c10库是一个用于构建和维护PyTorch模型的强大工具，它具有灵活的模型定义、训练和部署功能，可以帮助用户快速地构建和部署实际的深度学习模型。


C10，来自于Caffe Tensor Library的缩写。这里存放的都是最基础的Tensor库的代码，可以运行在服务端和移动端。PyTorch目前正在将代码从ATen/core目录下迁移到C10中。C10的代码有一些特殊性，体现在这里的代码除了服务端外还要运行在移动端，因此编译后的二进制文件大小也很关键，因此C10目前存放的都是最核心、精简的、基础的Tensor函数和接口。

C10目前最具代表性的一个class就是TensorImpl了，它实现了Tensor的最基础框架。继承者和使用者有：

```C++
Variable的Variable::Impl
SparseTensorImpl
detail::make_tensor<TensorImpl>(storage_impl, CUDATensorId(), false)
Tensor(c10::intrusive_ptr<TensorImpl, UndefinedTensorImpl> tensor_impl)
c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>
```

值得一提的是，C10中还使用/修改了来自llvm的SmallVector，在vector元素比较少的时候用以代替std::vector，用以提升性能; 

### ATen
ATen，来自于 A TENsor library for C++11的缩写；PyTorch的C++ tensor library。ATen部分有大量的代码是来声明和定义Tensor运算相关的逻辑的，除此之外，PyTorch还使用了aten/src/ATen/gen.py来动态生成一些ATen相关的代码。ATen基于C10，Gemfield本文讨论的正是这部分；

### Caffe2
为了复用，2018年4月Facebook宣布将Caffe2的仓库合并到了PyTorch的仓库,从用户层面来复用包含了代码、CI、部署、使用、各种管理维护等。caffe2中network、operators等的实现，会生成libcaffe2.so、libcaffe2_gpu.so、caffe2_pybind11_state.cpython-37m-x86_64-linux-gnu.so（caffe2 CPU Python 绑定）、caffe2_pybind11_state_gpu.cpython-37m-x86_64-linux-gnu.so（caffe2 CUDA Python 绑定），基本上来自旧的caffe2项目)

Caffe2是一个开源的深度学习框架，由Berkeley AI Research（BAIR）开发，于2016年4月22日正式发布。Caffe2的前身是Caffe，是由加州大学伯克利分校的Yangqing Jia在2014年发布的，是第一个用于深度学习的框架。

在2016年，Caffe2正式发布，它提供了在移动设备上运行深度学习模型的能力，如苹果的CoreML，Android的TensorFlow Lite和Caffe2。

2017年，Facebook在Caffe2中推出了新的模型，这个模型称为Detectron，用于对象检测，这也使得Caffe2在深度学习领域变得更加强大。

2018年，Facebook继续在Caffe2中推出了新的功能，如强化学习，卷积神经网络和语言模型。同时，Facebook还推出了一个新的API，叫做ONNX，可以让深度学习模型在不同的框架之间迁移。

2019年，Caffe2发布了新的版本，支持了多种新的深度学习模型，如深度强化学习，卷积神经网络，和语义分割等。

Caffe2目前已经成为最受欢迎的深度学习框架之一，Facebook正在努力将Caffe2发展成一个可以与其他框架兼容的健壮深度学习框架。


### Torch

Torch，部分代码仍然在使用以前的快要进入历史博物馆的Torch开源项目，比如具有下面这些文件名格式的文件：

``` Bash
TH* = TorcH
THC* = TorcH Cuda
THCS* = TorcH Cuda Sparse (now defunct)
THCUNN* = TorcH CUda Neural Network (see cunn)
THD* = TorcH Distributed
THNN* = TorcH Neural Network
THS* = TorcH Sparse (now defunct)
THP* = TorcH Python
```

PyTorch会使用tools/setup_helpers/generate_code.py来动态生成Torch层面相关的一些代码，这部分动态生成的逻辑将不在本文阐述，你可以关注Gemfield专栏的后续文章。

### 第三方依赖

```bash
#Facebook开源的cpuinfo，检测cpu信息的
third_party/cpuinfo

#Facebook开源的神经网络模型交换格式，
#目前Pytorch、caffe2、ncnn、coreml等都可以对接
third_party/onnx

#FB (Facebook) + GEMM (General Matrix-Matrix Multiplication)
#Facebook开源的低精度高性能的矩阵运算库，目前作为caffe2 x86的量化运算符的backend。
third_party/fbgemm

#谷歌开源的benchmark库
third_party/benchmark

#谷歌开源的protobuf
third_party/protobuf

#谷歌开源的UT框架
third_party/googletest

#Facebook开源的面向移动平台的神经网络量化加速库
third_party/QNNPACK

#跨机器训练的通信库
third_party/gloo

#Intel开源的使用MKL-DNN做的神经网络加速库
third_party/ideep
```

## 参考
- PyTorch ATen代码的动态生成 https://zhuanlan.zhihu.com/p/55966063
- Pytorch1.3源码解析-第一篇 https://www.cnblogs.com/jeshy/p/11751253.html