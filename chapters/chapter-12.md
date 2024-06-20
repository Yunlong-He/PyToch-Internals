
# 深度学习性能优化

## 主要内容
- [性能优化介绍][#性能优化介绍]
- [硬件加速技术-CPU][#硬件加速技术-CPU]
    - [Intel® Advanced Matrix Extensions][#Intel® Advanced Matrix Extensions]
- [模型保存及加载][#模型保存及加载]
    - [PyTorch模型存储的格式][#PyTorch模型存储的格式]
    - [PyTorch模型与ONNX模型的转换][#PyTorch模型与ONNX模型的转换]
- 使用TensorRT
- 算子融合
- 量化
- 剪枝
- 混合精度训练

## 性能优化介绍

深度神经网络的计算有以下几个特点：
- 计算量大，尤其是在当今大模型成为流行趋势的年代
- 并行度高，网络中的计算包含大量的向量和矩阵的计算，如
- 缺省数据类型是32位浮点类型，并且对精度有一定的容忍性，因此在一定的情况下，可以使用bfloat16或者int8进行计算
- 网络结构有一定的裁剪容忍度，剪掉部分连接对整体预测精度的影响不大

## 硬件加速技术-CPU

在Linux操作系统中，可以使用命令"cat /proc/cpuinfo"来查看CPU的型号及其所支持的指令集

```Bash
# cat /proc/cpuinfo

flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon rep_good nopl xtopology cpuid tsc_known_freq pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch cpuid_fault invpcid_single ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves arat avx512vbmi umip pku ospke avx512_vpopcntdq la57 rdpid arch_capabilities
```

输出中flag这一项指明了CPU所支持的硬件特性：

- SSE
- AVX
- AMX

<table>
<tr>
<td>h1</td><td>h2</td>
</tr>
</table>

|指令集| 	条| 	Date| 	ICPU| 	IDate| 	ACPU| 	ADate| 	Memo|
|-------|----|----------|-------|--------|-------|-------|-------|
|MMX| 	57| 	1996-10-12| 	Pentium MMX(P55C)| 	1996-10-12| 	K6| 	1997-4-1| 	MultiMedia eXtension|
SSE| 	70| 	1999-5-1| 	Pentium III(Katmai)| 	1999-5-1| 	Athlon XP| 	2001-10-9| 	Streaming SIMD| Extensions|
SSE2| 	144| 	2000-11-1| 	Pentium 4(Willamette)| 	2000-11-1| 	Opteron| 	2003-4-22|| 	 
SSE3| 	13| 	2004-2-1| 	Pentium 4(Prescott)| 	2004-2-1| 	Athlon 64| 	2005-4-1|| 	 
SSSE3| 	16| 	2006-1-1| 	Core| 	2006-1-1| 	Fusion(Bobcat)| 	2011-1-5| 	最早出现在Tejas核心（功耗过高而取消）
SSE4.1| 	47| 	2006-9-27| 	Penryn| 	2007-11-1| 	Bulldozer| 	2011-9-7| 	 
SSE4.2| 	7| 	2008-11-17| 	Nehalem| 	2008-11-17| 	Bulldozer| 	2011-9-7| 	 
SSE4a| 	4| 	2007-11-11| |	  |	  	K10| 	2007-11-11| 	K10还加了 POPCNT 与 LZCNT 指令
SSE5| 	 | 	2007-8-30| 	  	|||  	  	  	|被AVX搅局。后来XOP/FAM4/CVT16
AVX| 	 | 	2008-3-1| 	Sandy Bridge| 	2011-1-9| 	Bulldozer| 	2011-9-7| 	Advanced Vector Extensions
AVX2| 	 | 	2011-6-13| 	Haswell| 	2013-4-1| 	  	  	 
AES| 	7| 	2008-3-1| 	Westmere| 	2010-1-7| 	Bulldozer| 	2011-9-7| 	Advanced Encryption Standard
3DNowPrefetch| 	2| 	2010-8-1| 	  ||	  	K6-2| 	1998-5-28| 	2010年8月放弃3DNow!，仅保留2条预取
3DNow!| 	21| 	1998-1-1| 	  | |	  	K6-2| 	1998-5-28| 	 
3DNow!+| 	|  	1999-6-23| 	  ||	  	Athlon| 	1999-6-23| 	Enhanced 3DNow!. 共52条？
MmxExt| 	|  	 | 	 || 	  	Athlon| 	1999-6-23| 	Extensions MMX
3DNow! Pro| 	||||  	  	  	  	Athlon XP| 	2001-10-9| 	3DNow! Professional.兼容SSE
POPCNT| 	1| 	2007-11-11| 	||  	  	K10| 	2007-11-11| 	 
ABM| 	1| 	2007-11-11| 	||  	  	K10| 	2007-11-11| 	advanced bit manipulation. LZCNT
CLMUL| 	5| 	2008-5-1| 	Westmere| 	2010-1-7| 	Bulldozer| 	2011-9-7| 	PCLMULQDQ等
F16C| 	|  	2009-5-1| 	Ivy Bridge| 	2012-4-1| 	Bulldozer| 	2011-9-7| 	CVT16|
FAM4| 	 | 	2009-5-1| 	 || 	  	Bulldozer| 	2011-9-7| 	 
XOP| 	 | 	2009-5-1| 	  ||	  	Bulldozer| 	2011-9-7| 	 

指令集：指令集名。
条：指令条数。
Date：公布日期。
ICPU：Intel最早支持该指令集的CPU。
IDate：ICPU的发售日期。
ACPU：AMD最早支持该指令集的CPU。
ADate：ACPU的发售日期。
Memo：备注。

参考https://www.cnblogs.com/zyl910/archive/2012/02/26/x86_simd_table.html


基于CPU的加速库：

- MKLDNN
- ONEDNN

### Intel® Advanced Matrix Extensions

## 模型保存及加载
参考：
- pytorch中保存的模型文件.pth深入解析 https://zhuanlan.zhihu.com/p/84797438

### PyTorch模型存储的格式
在pytorch进行模型保存的时候，一般有两种保存方式，一种是保存整个模型，另一种是只保存模型的参数。
```Python
torch.save(model.state_dict(), "my_model.pth") # 只保存模型的参数
torch.save(model, "my_model.pth") # 保存整个模型
```
如果保存的是完整的模型，实际保存的是一个字典，包括以下四个键对：
- model，模型的参数
- optimizer,scheduler,iteration
如果保存的只是模型的参数，那么

保存的模型参数实际上一个字典类型，通过key-value的形式来存储模型的所有参数.
如果要查看模型文件的内容，可以先加载进来再查看：
```Python
import torch

pthfile = '~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth'
net = torch.load(pthfile,map_location=torch.device('cpu')) 
for key,value in net.items():
    print(key,value.size(),sep=" ")
''' outputs:
conv1.weight torch.Size([64, 3, 7, 7])
bn1.running_mean torch.Size([64])
bn1.running_var torch.Size([64])
bn1.weight torch.Size([64])
bn1.bias torch.Size([64])
layer1.0.conv1.weight torch.Size([64, 64, 1, 1])
layer1.0.bn1.running_mean torch.Size([64])
'''
```

有时候，比如训练意外中断了，或者我们希望在某个checkpoint之上继续训练，可以从checkpoint中加载模型参数、学习率等：
```Python
    ckpt_path = "./checkpoint/ckpt_best_1001.pth"
    ckpt = torch.load(ckpt_path)

    model.load_state_dict(ckpt['net'])  # 加载模型参数

    optimizer.load_state_dict(ckpt['optimizer'])  # 加载优化器参数
    start_epoch = ckpt['epoch']  # 设置开始的epoch
```



### PyTorch模型与ONNX模型的转换

## 使用TensorRT



## 混合精度训练

BFloat16 (Brain Floating Point)是一种16bit的浮点数格式，动态表达范围和float32是一样的，但是精度低。下一代的Xeon Sapphire Rapids上面可以使用AMX(Advanced Matrix Extensions)对卷积和矩阵乘的操作在BFloat16上进行加速，吞吐量比Float32高一个数量级。

这里主要介绍在PyTorch上面优化BFloat16原生算子的一些小技巧，侧重性能优化方面，不介绍BFloat16训练中涉及的调参问题。

Pytorch支持混合精度训练，这意味着在训练过程中可以使用不同的数据类型或精度。这可以有效的提高训练的效率和性能，是一种性能优化的有效方法。

Pytorch支持混合精度训练的原理是：使用不同的数据类型和精度选择，来给模型的每一层指定不同的数据类型和精度，以便更有效的训练模型。

Pytorch支持混合精度训练的具体实现步骤如下：

1. 首先，确定训练模型所使用的数据类型和精度。这可以是32位浮点（float32）、64位浮点（float64）、16位浮点（float16）或8位整数（int8）。

2. 然后，根据模型的层级结构，为每一层指定不同的数据类型和精度。

3. 接着，定义损失函数，设定优化器，并设定训练参数，完成模型的构建。

4. 最后，使用混合精度训练模型，将输入数据转换为不同的数据类型和精度，并通过计算图层的混合精度计算，以实现模型的训练。

在Pytorch中，混合精度训练的实现主要是通过设置不同的计算图层的精度来实现的。首先，使用较低的精度来加速计算，然后使用较高的精度来正确地存储模型参数。通过这种方式，可以在有限的计算资源和计算能力下，提高模型的训练效率和性能。

此外，Pytorch还提供了一种叫做Automatic Mixed Precision (AMP)的功能，其可以自动选择合适的精度来训练模型，从而简化混合精度训练过程，减少开发人员手动调节精度的工作量。

总之，Pytorch支持混合精度训练，可以提高训练模型的性能和效率，是一种性能优化的有效方法。通过设置不同精度的模型层，并通过Automatic Mixed Precision(AMP)自动选择合适的精度，可以很好的实现混合精度训练。


## 算子融合
## 量化
## 剪枝
## TorchMultimodal 库
 
## 参考
- PyTorch CPU性能优化（四）：BFloat16 https://zhuanlan.zhihu.com/p/499979372
- https://zhuanlan.zhihu.com/p/588136197