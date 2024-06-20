
# 自定义算子的实现

## 基本内容
- [快速实现自定义的算子](#快速实现自定义的算子)
- [实现原生的算子](#实现原生的算子)
    - [常见问题分析](#常见问题分析)
- [参考](#参考)

## 算子方式简介
根据之前的算子体系，我们可以将

## 快速实现自定义的算子

## 实现原生的算子

所谓“原生”，指的就是内置在PyTorch中的算子，跟随PyTorch一起编译生成，可以同"torch.xxx"等方式使用的算子。

由于原生算子的数量非常多，处于效率和可用性的考虑，在不同的平台上可能会有实现，另外算子要支持注册到torch模块、自动微分、jit等，也造成了算子之间会有大量重复的操作，在PyTorch中，用模板的方式隐藏了这些细节，当然也带来了一定的复杂性。


下面的内容大部分来自于aten/src/ATen/native/README.md中，英文好的小伙伴们可以自己去读最新的版本。

在native_functions.yaml中有所有原生算子的声明，每个算子的格式基本如下：
```yaml
# aten/src/ATen/native/native_functions.yaml

- func: func_name(ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
  variants: function, method
  dispatch:
    CPU: func_cpu
    CUDA: func_cuda

```
下面分别介绍声明中的各个部分。

### func

```yaml
- func: func_name[.overload_name](ArgType arg0[=default], ArgType arg1[=default], ...) -> Return
```
func声明了当前算子的名称及签名，其中ArgType代表算子参数的类型，PyTorch支持的算子参数类型有如下几种：
- Tensor。 这里的Tensor类型，在转换后会表示为const Tensor&。有一种例外是带有inplace的，将被翻译成Tensor&。如果参数类型后边跟着“?”，如"Tensor?"，代表这个参数在Python API层面是可选的，此时被调用时，C++层面传递的是“c10:nullopt”。 虽然Tensor有很多子类型，如FloatTensor、IntTensor、HalfTensor等，但是在这里统一用Tensor表示，只不过当参数列表中有多个Tensor时，他们应该是同样类型的Tensor，比如都是FloatTensor。Tensor或者Tensor?有时候可以带一些标注信息，例如：
  - Tensor(a) : "a"可以是使用同一个数据的多个Tensor。
  - Tensor(a!) : "a"可以被写入
  - Tensor(a! -> a|b)
- Tensor[]。转换后编程C++参数类型ArrayRef<Tensor>（也就是TensorList）
- int[]。 
- int。相当于Python中的int类型，被转换成c++的int64_t。
- bool
- str。被转换成C++的c10::string_view。
- Scalar。支持绑定到Python中的任何数字类型，包括int, float, 以及0维的Tensor。
- Generator?。代表随机数发生器的状态。
- bool[N] （N在1-4之间）
- *。不转换成任何实际的参数，代表Python API中，其后的参数必须为keyword argument。
- ?。代表这是一个Optional类型，允许在Python层面传递None。

输入参数中没有Tensor的函数被称为factory function，代码生成器会进行特殊处理。如果某个factory function需要一个Tensor类型的参数，可以显式加上category_override:factory。

参数名称。 参数名是有意义的，生成的python代码可能会用到这个参数名。

为了方便，使用"out"标识一个出参。

缺省值。 每个参数都可以有个缺省值，可以用如下方式表示：
- 数字，例如 0, 5.0。特别的，int[2] x=2代表 int[2] x=[2,2]。
- 数字列表，如[0, 0]
- 布尔类型，例如True
- 空列表
- None，用来表示指针参数为空的情况。

Returns：返回值。返回值支持一下tuple和非tuple两种形式：
```C++
ReturnType [retarg0]
(ReturnType [retarg0], ReturnType [retarg1], ...)
```
ReturnType支持以下几种：Tensor, Tensor[]，分别被翻译成C++中的Tensor和std::vector<Tensor>, 或者Tensor的tuple，翻译成std::tuple<Tensor, Tensor>

必要的话，也可以扩展代码生成器以便支持新的类型。ATen的设计理念是仅支持基本类型，从而具有很好的可移植性。

返回值也支持返回参数名，但不支持缺省值，也不支持optional形式的返回值。

重载(Overloads)。ATen允许用同一个名称注册多个函数签名，但需要提供不同的重载名称。重载名称的格式为“函数名.重载名”。ATen建议重载名使用有实际意义的名称，以便容易记忆并且容易区分其功能。为了保持后向兼容，不要随便修改重载名。

命名空间（Namespaces）。缺省情况下，所有的算子都注册在命名空间“aten”下，用户也可以将算子注册在自定义的命名空间内：
```yaml
- func: custom::my_op(Tensor(a) self, ...) -> Tensor(a)
  variants: function, method
  dispatch:
    CPU: my_op_cpu
    CUDA: my_op_cuda
```

### variants

Variants用来控制算子的归属，根据上面例子，ATen支持两种算子的variants, function和method。method表示该算子可以作为Tensor的成员方法使用，例如t.foo()，此时会有一个Tensor类型的参数self。function表示该算子可以作为一个函数使用，如at::foo()。 缺省情况下，ATen只会构建一个函数，

### annotations
Python的API支持一些特性，如inplace操作（如self.abs_()），以及输出参数（如torch.abs(input, out=None))。

为此会有三种定义的形式：
- abs(Tensor self) -> Tensor
- abs_(Tensor self) -> Tensor
- abs_out(Tensor out, Tensor self) -> Tensor

ATen提供了三种annotation形式：
- abs(Tensor self) -> Tensor， 表示总会返回一个新创建的Tensor
- abs_(Tensor(a!) self) -> Tensor(a!) self这个Tensor既是输入也是输出。
- abs(Tensor self, *, Tensor(a!) out) -> Tensor(a!)

一种特殊情况是：
- transpose(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
代表参数self会被返回，但其数据不会被改变。

另一种情况和上一个例子类似，但输入输出的是一个Tensor列表。
- func: chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]

如果开发者错误地使用了annotation，ATen代码生成器会在生成阶段报告错误。

### dispatch
dispatch定义了转发的规则，例如：

```yaml
dispatch:
    CPU: func_cpu
    CUDA: func_cuda
```
定义了在CPU的backend下，将调用转发给函数func_cpu，而在CUDA的backend下转发给函数func_cuda。

和前面func的定义类似，dispatch的目标函数也支持命名空间,当前最多支持二级命名空间，例如下面例子代表转发给命名空间custom::ns::native的函数（其中native是自动加上的）：
```yaml
dispatch:
    CPU: custom::ns::func_cpu
```

ATen支持的backend可以在gen.py中找到列表。下面是一些”通用“的backend：
- CompositeExplicitAutograd。之前作为”DefaultBackend“，用于支持所有backend的算子实现的核函数，但是需要在derivatives.yaml中显示定义反向传播函数。最典型的用途是支持代理函数，例如某个函数自身做的很少，但是之后会分发到另一个函数去执行复杂的逻辑。因此注册到这个backend，相当于注册到了所有的backend，包括CPU, CUDA。注意DispatchStub只适用于CPU和CUDA，因此不应该被注册到CompositeExplicitAutograd。
- CompositeExplicitAutogradNonFunctional。和上一个类似，但有一些限制。(1)不能用于有别名的算子。（2）内部会调用有别名的算子。一个例子是select_backward。TODO. 有待进一步描述。
- CompositeImplicitAutograd。之前被称为Math。也是用于支持所有backend的核函数，但是其内部调用的都是支持自动微分的算子，因此这个函数也支持自动微分。这个backend用的比较少，如果没有dispatch table，缺省会将核函数注册为这个backend。

如果一个函数注册到了这几个Compsite backend，并且其实现上也只调用了支持所有backend的函数，那么这个函数也支持所有的backend，例如：
```C++
at::Tensor my_op(const Tensor& self, const Tensor& other) {
  return self + 2 * other;
}
```
其中算子"+"和"*"的求导公式都已知，那么将这个函数注册到Compsite backend后，也会支持自动微分，PyTorch会根据链式法则自动构建反向传播函数。

是否需要使用隐式或显式的自动微分，可遵循如下原则：
1. 对于由现有函数组合起来的函数，尽可能使用CompositeImplicitAutograd。
2. 如果考虑性能或者计算稳定性等原因，需要自己实现梯度计算，可以使用CompositeExplicitAutograd。自己实现的梯度计算函数可以作为Alias放在Autograd下，或者注册到AutogradCPU的key上。
3. 如果希望对特定的backend实现核函数，可以注册到对应的dispatch key上

需要注意的是：CompositeImplicitAutograd核函数注册的时候，不会指定"dispatch"，因此当增加一个特定后端的核函数时，必须同时添加一个CompositeImplicitAutograd条目，以便旧的函数仍然可以用于其他backend。TODO,举个例子

### device_guard

为了保证核函数能够运行在合适的设备上（该设备由第一个Tensor参数所在的设备来确定），缺省情况ATen代码生成器会生成DeviceGuard的调用。因此大多数情况下开发者不需要自己设置核函数运行的设备。

特殊情况下，例如你的核函数只是调用了一个已经设置了device guard的函数，或者你的核函数不需要和设备交互，这时可以关闭device guard：
```yaml
device_guard: False
```

### device_check
缺省情况下，ATen代码生成器会生成设备检查代码，以保证传给核函数的参数都在同一个设备上。但是有时候没有必要做这种检查，例如有些函数本身就允许运行在多个设备上，这时可以禁用device_check:
```yaml
device_check: NoCheck
```

### manual_kernel_registration
当设置了这个开关后，代码生成器就不会生成注册算子到TypeDefault上（TypeDefault能够捕获所有的dispatch key）。因此"dispatch"和”manual_kernel_registration“是互斥的。
```yaml
manual_kernel_registration: True
```
一般情况下开发者不需要使用这个开关。

### use_const_ref_for_mutable_tensors
设置这个开关之后，代码生成器会生成类似”const Tensor&“的定义，表明相应的Tensor参数不需要修改。
```yaml
use_const_ref_for_mutable_tensors: True
```


### Alias
```yaml
# Note [Adding an alias]
# To add an alias do the following:
#
# 1) Copy the original functions native_functions.yaml entry, but replace the
#      original function's name with their own and delete any dispatch
#      keys for the aliases. Specifying a dispatch key will prevent
#      autograd from recording the operations the alias performs, which
#      will stop it from "inheriting" the original operation's autograd behavior.
# 2) Implement the corresponding functions and have them redispatch to the
#      original function.
# 3) Add docstrings to the new function that reference the original function,
#      and document the method as usual (if it exists.)
#    (See torch/_torch_docs.py and docs/source/torch.rst if adding a function,
#     torch/_tensor_docs.py and docs/source/tensors.rst if adding a method,
#     or module-specific doc bindings (like torch/linalg/__init__.py) if
#     adding an alias in a namespace.)
# 4) Update torch/overrides.py consistent with the original function.
# 5) Update the alias_map in torch/csrc/jit/passes/normalize_ops.cpp.
# 6) Add aliases argument to existing OpInfo/UnaryUfuncInfo or create new OpInfo/UnaryUfuncInfo entry
# in op_db list in torch/testing/_internal/common_methods_invocations.py
#
# See torch.absolute, an alias for torch.abs, as an example.
```
## autogen
TODO

### 原生算子的C++实现

原生算子的实现位于native目录下的C++文件中。除了cuda目录下的文件，PyTorch对于该函数实现在哪个文件中并没有要求，只要函数的签名和和ATen生成的保持一致即可。

如果实现的算子本身支持自动微分，则不需要额外实现自动微分函数，否则需要自己写一个foo_backward函数，并将对应关系加到tools/autograd/derivatives.yaml中。

确定核函数支持的backend包括以下步骤：
1. 确认核函数是否支持所有backend
2. 确认核函数是否支持autograd
3. 确定dispatch table符合预期

### Structured Kernels

如果我们对pytorch进行调试，并且计算中是两个Tensor的相加，经过从Python API到C++的API，以及多次的分发之后，会进入到下面这个函数：
```C++
at::Tensor wrapper_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  structured_add_out_functional op;
  op.meta(self, other, alpha);
  op.impl(self, other, alpha, op.outputs_[0]);
  return std::move(op.outputs_[0]);
}

TORCH_LIBRARY_IMPL(aten, CPU, m) {
  m.impl("add.Tensor", TORCH_FN(wrapper_add_Tensor));
}
```

这个函数里中调用的meta()和impl()是定义在其他地方的。这是因为add函数被定义为structured kernel。structured kernel是PyTorch实现算子的新方式。

TODO why we need structured kernel
https://github.dev/pytorch/rfcs/blob/rfc-0005/RFC-0005-structured-kernel-definitions.md

基于structured kernel实现一个算子，需要实现两个函数：
- ”meta"函数，用来确保输入参数具有正确的shape和dtype，并计算出输出Tensor的size
- "impl"函数，执行真正的计算过程，每个backend都有对应的实现。

TODO: 另外考虑到算子定义有多种变体（包括inplace和out版本），相应的也都有各自的meta()和impl()实现。

我们看一下算子add的meta及impl的实现：

```C++
// expands to structured_add_Tensor::meta() { ... }
TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

// expands to structured_add_out::impl() { ... }
TORCH_IMPL_FUNC(add_out) (
  const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
) {
  add_stub(device_type(), *this, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
}
```

这两个函数的定义是由代码生成器生成的：

```C++
// NativeMetaFunctions.h
// namespace at::meta
struct TORCH_API structured_add_Tensor : public TensorIteratorBase {
    void meta(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha);
};

// NativeFunctions.h
// namespace at::native
struct TORCH_API structured_add_out : public at::meta::structured_add_Tensor {
    void impl(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, const at::Tensor & out);
};
```

但是structured kernel是如何使用的呢？代码生成器生成的文件中，RegisterCPU.cpp包含了答案：

```C++
// functional version
at::Tensor wrapper_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  structured_add_out_functional op;
  op.meta(self, other, alpha);
  op.impl(self, other, alpha, op.outputs_[0]);
  return std::move(op.outputs_[0]);
}

// inplace version
at::Tensor & wrapper_add__Tensor(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  structured_add_out_inplace op(self);
  op.meta(self, other, alpha);
  op.impl(self, other, alpha, op.outputs_[0]);
  return self;
}

// out= version
at::Tensor & wrapper_add_out_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  structured_add_out_out op(out);
  op.meta(self, other, alpha);
  op.impl(self, other, alpha, op.outputs_[0]);
  return out;
}

// registering the 3 kernels above to the dispatcher, under the CPU Dispatch Key.
TORCH_LIBRARY_IMPL(aten, CPU, m) {
  ...
  m.impl("add.Tensor", TORCH_FN(wrapper_add_Tensor));
  m.impl("add.out", TORCH_FN(wrapper_add_out_out));
  m.impl("add_.Tensor", TORCH_FN(wrapper_add__Tensor));
}
```

当然在同一个文件中，我们还可以找到对应的实现:

```C++
struct structured_add_out_functional final : public at::native::structured_add_out {

    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides,
                    TensorOptions options, DimnameList names) override {

        if (strides.empty()) {
            outputs_[output_idx] = at::native::empty_cpu(sizes, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.device_opt(), optio
        } else {
            // TODO: assert options.memory_format_opt() is nullopt (debug only?)
            outputs_[output_idx] = at::native::empty_strided_cpu(sizes, strides, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(), options.de
        }

        if (!names.empty()) {
          namedinference::propagate_names(outputs_[output_idx], names);
        }
        // super must happen after, so that downstream can use maybe_get_output
        // to retrieve the output
        at::native::structured_add_out::set_output(output_idx, sizes, strides, options, names);
    }

    const Tensor& maybe_get_output(int64_t output_idx) override {
        return outputs_[output_idx];
    }
    std::array<Tensor, 1> outputs_;
};
```

比如sigmoid函数：
=======
自定义算子需要考虑很多方面的实现：
- 算子定义在哪个，例如是Tensor.xxx()， 还是torch.xxx()
- 算子的前向计算过程
- 算子的反向计算过程
- 基于AVX指令或者OneDNN的实现
- 基于GPU的实现

实现的步骤包括：

<ol>
<li> 在native_functions.yaml中增加算子"abc"的声明</li>
  在编译的时候，会自动生成以下代码：
  <ol>
    <li>在python_torch_functions_2.cpp中生成“THPVariable_abc”的定义。如果有多个算子名字都叫“abc”，那么只会生成一个“THPVariable_abc”的定义，然后根据不同签名调用不同的算子实现，这就是算子调用的分发过程。注意因为原生的算子太多，pytorch将算子定义拆分成多个文件，也就是说“THPVariable_xxx”的定义可能在python_torch_functions_x.cpp中，但是为了查看方便，也可以在python_torch_functionsEverything.cpp文件中找到生成的定义（该文件仅用于查看，不会用于编译）。这个文件中的“THPVariable_abc”定义会被加入到torch_functions_shard中，并最终注册成torch.abc()方法。
    <li>在python_variable_methods.cpp中生成“THPVariable_abc”的定义。如果有多个算子名字都叫“abc”，那么只会生成一个“THPVariable_abc”的定义，然后根据不同签名调用不同的算子实现，这就是算子调用的分发过程。这个文件中的“THPVariable_abc”定义会被加入到variable_methods中，并最终注册成Tensor.abc()方法。
    <li>在TraceType_2.cpp中生成“abc_Scalar”的定义。从文件名上我们也可以看出这个函数是用JIT中构建计算图的。同时这个函数也会被注册到Library"Tracer"中。
    <li>在VariableType_2.cpp中注册该算子的Autograd实现，如果未实现，Pytorch会注册缺省的实现torch::autograd::autogradNotImplementedFallback()。
    </ol>
<li>
</ol>

很多原生算子的模板定义在native_functions.yaml中，比如sigmoid函数：

```yaml
# aten/src/ATen/native/native_functions.yaml

- func: sigmoid(Tensor self) -> Tensor
  device_check: NoCheck   # TensorIterator
  structured_delegate: sigmoid.out
  variants: function, method
  dispatch:
    QuantizedCPU: sigmoid_quantized_cpu
    MkldnnCPU: mkldnn_sigmoid


- func: sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor
  python_module: nn
  structured_delegate: sigmoid_backward.grad_input

```
其中：
- func字段定义了算子的名称和输入输出参数。
- device_check: 暂时还不清楚用途，在模板里都是NoCheck。
- structured_delegate: sigmoid.out
- variants字段生命这个算子的类型和使用方式，function表明sigmoid这个算子可以通过函数torch.sigmoid()中，method则表明可以像使用对象的方法一样使用，比如tensorA是一个张量，可以用tensorA.sigmoid()来调用。
- dispatch字段定义了在不同的平台或者优化方式下该算子的变体。这里针对使用量化方式运行时，会调用相应的量化实现sigmoid_quantized_cpu，需要使用mkldnn加速的时候，调用mkldnn_sigmoid。
- python-module字段定义了该算法会被注册到的Python模块。

sigmoid函数是机器学习中最基本的函数之一，其公式如下：
$$Sigmoid(x) = \frac{1}{1+e^{-x}}$$

我们在使用sigmoid函数时，调用的是torch.nn.Sigmoid函数，其背后则是调用了torch.sigmoid()函数，也就是上面定义的native实现。

```Python
class Sigmoid(Module):
    r"""Applies the element-wise function:
    Examples::
        >>> m = nn.Sigmoid()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(input)
```


在tools/autograd/derivatives.yaml中，定义了算子的前向计算输出反向计算梯度的对应关系，比如sigmoid算子的对应关系如下：

```
- name: sigmoid(Tensor self) -> Tensor
  self: sigmoid_backward(grad, result)
  result: auto_element_wise
```

在native_functions.yaml中只是声明了sigmoid算子，具体的算子实现是和平台相关的，因此要到各个平台目录下去寻找，例如cpu下的sigmoid算子实现在cpu/UnaryOpsKernel.cpp里：

```C++
// aten/src/ATen/native/cpu/UnaryOpsKernel.cpp

static void sigmoid_kernel(TensorIteratorBase& iter) {
  if (iter.common_dtype() == kBFloat16) {
    cpu_kernel_vec(
        iter,
        [=](BFloat16 a) -> BFloat16 {
          float a0 = static_cast<float>(a);
          return static_cast<float>(1) / (static_cast<float>(1) + std::exp((-a0)));
        },
        [=](Vectorized<BFloat16> a) {
          Vectorized<float> a0, a1;
          std::tie(a0, a1) = convert_bfloat16_float(a);
          a0 = (Vectorized<float>(static_cast<float>(1)) + a0.neg().exp()).reciprocal();
          a1 = (Vectorized<float>(static_cast<float>(1)) + a1.neg().exp()).reciprocal();
          return convert_float_bfloat16(a0, a1);
        });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(iter.common_dtype(), "sigmoid_cpu", [&]() {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t {
            return (static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp((-a))));
          },
          [=](Vectorized<scalar_t> a) {
            a = Vectorized<scalar_t>(static_cast<scalar_t>(0)) - a;
            a = a.exp();
            a = Vectorized<scalar_t>(static_cast<scalar_t>(1)) + a;
            a = a.reciprocal();
            return a;
          });
    });
  }
}

REGISTER_DISPATCH(sigmoid_stub, &CPU_CAPABILITY::sigmoid_kernel);


// aten/src/ATen/native/cpu/BinaryOpsKernel.cpp

void sigmoid_backward_kernel(TensorIteratorBase& iter) {
  if (isComplexType(iter.dtype())) {
    // ......
  } else if (iter.dtype() == kBFloat16) {
    // ......
  } else {
    // ......
  }
}

```



```C++
// aten/src/ATen/native/cpu/UnaryOps.cpp

CREATE_UNARY_FLOAT_META_FUNC(sigmoid)

CREATE_UNARY_TORCH_IMPL_FUNC(sigmoid_out, sigmoid_stub)
DEFINE_DISPATCH(sigmoid_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)


```
在sigmoid_kernel()的实现里，根据传输Tensor类型的不同，构建了不同的匿名函数，然后调用cpu_kernel_vec()对不同的Tensor分别进行计算。可以认为这些匿名函数是真正的sigmoid的实现，cpu_kernel_vec()是一个通用函数，

sigmoid_kernel是sigmoid算子在cpu下的实现，当然即使在CPU下，sigmoid函数也有多种形式，除了普通的浮点计算外，也要支持半浮点的形式。
AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES宏有三个参数：
- iter.common_dtype()，指明操作的Tensor属于哪种类型。
- "sigmoid_cpu"， 算子的名称
- 匿名函数，调用了cpu_kernel_vec

在aten/src/ATen/native/cpu/Loops.cpp中，有两个cpu_kernel相关的函数，由于cpu下的源文件在编译的时候会加上AVX/AVX2的选项，因此能够充分利用CPU中的现代化指令，不过这两个函数向量化的实现方式不太一样。
- cpu_kernel()：依赖于编译器自动实现计算的向量化
- cpu_kernel_vec()：使用x86 SIMD原语实现向量化。
一般来讲，使用cpu_kernel_vec()的时候，说明实现该算子的实现是经过精心优化的，效率会高一些。

例如用这两个函数实现浮点数相乘的算子，可以这样实现：

```C++

cpu_kernel(iter, [](float a, float b) { return a * b; });

cpu_kernel_vec(iter,
     [](float a, float b) { return a * b; },
     [](Vectorized<float> a, Vectorized<float> b) { return a * b; });
```

下面我们看一下cpu_kernel_vec()函数的实现：

```C++
// aten/src/ATen/native/cpu/Loops.cpp

template <bool check_dynamic_cast=true, typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  using traits = function_traits<func_t>;
  // this could be extended to work with void return types
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  // dynamic casting not currently supported on CPU, but some kernels (like Fill)
  // explicitly dynamic_cast, so we give the opt-out of checking.
  c10::guts::if_constexpr<check_dynamic_cast>([&] {
    TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));
  });

  iter.for_each(make_vectorized_loop2d(op, vop), grain_size);
  iter.cast_outputs();
}

```
可以看到，对每个Tensor，又调用了make_vectorized_loop2d()
```C++
// aten/src/ATen/native/cpu/Loops.cpp
template <typename op_t, typename vop_t>
VectorizedLoop2d<op_t, vop_t> make_vectorized_loop2d(
    const op_t &op, const vop_t &vop) {
  return VectorizedLoop2d<op_t, vop_t>(op, vop);
}

template <typename op_t, typename vop_t>
struct VectorizedLoop2d {
  op_t op;
  vop_t vop;

  using traits = function_traits<op_t>;
  static constexpr int ntensors = traits::arity + 1;
  using data_t = std::array<char*, ntensors>;

  VectorizedLoop2d(const op_t &op, const vop_t &vop):
    op(op), vop(vop) {}

  static void advance(data_t &data, const int64_t *outer_strides) {
    for (const auto arg : c10::irange(data.size())) {
      data[arg] += outer_strides[arg];
    }
  }

  void operator()(char** base, const int64_t *strides, int64_t size0, int64_t size1) {
    data_t data;
    std::copy_n(base, ntensors, data.data());
    const int64_t *outer_strides = &strides[ntensors];

    if (is_contiguous<traits>(strides)) {
      for (const auto i : c10::irange(size1)) {
        (void)i;
        vectorized_loop(data.data(), size0, 0, op, vop);
        advance(data, outer_strides);
      }
    } else {
      using Indices = std::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
        if (idx) {
          for (const auto i : c10::irange(size1)) {
            (void)i;
            vectorized_loop(data.data(), size0, idx, op, vop);
            advance(data, outer_strides);
          }
        } else {
          for (const auto i : c10::irange(size1)) {
            (void)i;
            basic_loop(data.data(), strides, 0, size0, op);
            advance(data, outer_strides);
          }
        }
      });
    }
  }
};
```
很明显，VectorizedLoop2d的主要工作就是根据Tensor的stride的不同，选择不同的调用模式，但最终不管是调用vectorized_loop()还是调用basic_loop()，都是对一块连续的Tensor进行计算。可以这样计算的。

现在我们回到当初sigmoid函数的实现部分，其中对每个Tensor的操作函数实现是这样的：

```C++
// aten/src/ATen/native/cpu/UnaryOpsKernel.cpp
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t {
            return (static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp((-a))));
          },
          [=](Vectorized<scalar_t> a) {
            a = Vectorized<scalar_t>(static_cast<scalar_t>(0)) - a;
            a = a.exp();
            a = Vectorized<scalar_t>(static_cast<scalar_t>(1)) + a;
            a = a.reciprocal();
            return a;
          });

// aten/src/ATen/native/cpu/vec/vec256/vec256_float.h
 Vectorized<float> exp() const {
    return Vectorized<float>(Sleef_expf8_u10(values));
  }
```

在代码中可以看出，对应cpu的实现有很多，实际运行时会根据不同的平台和数据类型调用相应的实现，以达到比较好的性能。而由于CPU上的向量计算有很多库可以用，这里用到的是sleef库的实现。

> https://blog.csdn.net/yelede2009/article/details/120411361
> 有各种函数库以向量方式来计算数学函数，例如：对数、幂函数、三角函数等。这些函数库对向量化数学代码是有用的。
> 有两种不同种类的向量数学库：长向量库和短向量库。来看看它们的不同。假设要计算1000个数字的某个函数。用长向量库，把1000个数字的数组作为参数传给库函数，这
> 个库函数存储这1000个结果到另一个数组。使用长向量版库函数的缺点是，如果要做一系列计算，在下一次调用函数前就需要存储中间结果到一个临时数组中。用短向量版本
> 的向量库，可以把数据集拆分为子向量来适配向量寄存器。如果向量寄存器可以处理4个数字，那么需要调用250此库函数。这个库函数会返回结果到向量寄存器中，可以直接
> 被下一次计算利用，而不需要存储中间结果到RAM中。这可能更快。然而，短向量的库函数可能是不利的，如果计算序列形成了长依赖链。
>这是一些长向量函数库：
>
>    Intel 向量数学库（VML, MKL）。工作在x86平台。这些库函数在非Intel的CPU上会低效，除非重写了Intel cpu分发器。
>    Intel的IPP。工作在x86平台。也适用于非Intel的CPU。包含很多统计、信号处理和图像处理函数。
>    Yeppp。开源库。支持x86和ARM平台，多种编程语言。参考Yeppp。
>
>这是一些短向量库：
>
>    Sleef库。支持多种平台。开源。参考www.sleef.org。
>    Intel短向量库（SVML）。Intel编译器提供，被自动向量化调用。Gnu编译器可以通过选项-mveclibabi=svml使用这个库。如果用的是非Intel的CPU，也可以使用。
>    AMD LIBM库。只支持64位Linux平台。没有FMA4指令集时，性能会降低。Gnu通过-mveclibabi=acml选项使用。
>    VCL库。个人开发。参考https://github.com/vectorclass。

Dispatch的过程似乎有些复杂，有很多宏处理，更是导致不容易看懂。
```C++
// aten/src/ATen/Dispatch.h

#define AT_PRIVATE_CASE_TYPE(NAME, enum_type, type, ...)                     \
  AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, enum_type, type, scalar_t, __VA_ARGS__)


#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)                   \
  [&] {                                                                        \
    const auto& the_type = TYPE;                                               \
    /* don't use TYPE again in case it is an expensive or side-effect op */    \
    at::ScalarType _st = ::detail::scalar_type(the_type);                      \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _st);                                   \
    switch (_st) {                                                             \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Half, at::Half, __VA_ARGS__)  \
      default:                                                                 \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");         \
    }                                                                          \
  }()
  ```

### 常见问题分析


## 神经网络的基本结构
深度学习解决的是深度神经网络的优化问题，虽然深度神经网络的模型种类繁多，从最简单的MLP模型到近年流行的Transformer模型，其计算模式可以统一表示成有向无环图(DAG)的形式，比如


```python
import torch
from torch import nn

class DemoNet(nn.Module):
    def __init__(self):
        super(DemoNet, self).__init__()
        self.w = torch.rand(2,2)
    def forward(self, x):
        y = self.w * x
        return y * y

input = torch.rand(2, 2)
model = DemoNet()
```
使用TensorBoard查看该网络的可视化，如下图：
<img src='../images/1.png'/>

其中y处是一个算子”Operation: aten::mul“

虽然上面只是最简单的一个例子，但也包括了神经网络作为有向无环图的基本结构：
- 顶点：代表一个输入数据、算子、或者输出数据
- 边：代表数据和算子、算子和算子之间的输入输出关系。

深度神经网络包括结果的前向计算过程和梯度的反向传播过程，显而易见的是，深度学习框架需要事先构造计算图，然后再执行计算。这里有两个选择：
- 根据代码逻辑，构造好一个计算图，之后这个计算图可以反复执行
- 每次在执行时，都重新构造好计算图

PyTorch选择的是第二种方式，也就是动态图的方式。动态图的好处是可以在代码逻辑中使用各种条件判断。

## PyTorch中计算图的实现

虽然不是所有的计算图都通过上面的例子中的nn.Module来实现，但nn.Module确实是PyTorch中神经网络的基础结构，因此我们可以先看一下nn.Module的具体实现，然后在深入到PyTorch的底层进行探查。

```Python
# torch/nn/modules/module.py

class Module:
    r"""Base class for all neural network modules.
    ...
    """

     training: bool
    _is_full_backward_hook: Optional[bool]

    def __init__(self) -> None:
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        torch._C._log_api_usage_once("python.nn_module")

        self.training = True
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Tensor]] = OrderedDict()
        self._non_persistent_buffers_set: Set[str] = set()
        self._backward_hooks: Dict[int, Callable] = OrderedDict()
        self._is_full_backward_hook = None
        self._forward_hooks: Dict[int, Callable] = OrderedDict()
        self._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._state_dict_hooks: Dict[int, Callable] = OrderedDict()
        self._load_state_dict_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._load_state_dict_post_hooks: Dict[int, Callable] = OrderedDict()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()

    forward: Callable[..., Any] = _forward_unimplemented

```

Module类的主要属性及方法如下：

<img src='../images/2.png'/>

一个神经网络，最重要的是其内部的参数，在Module中有两个属性和参数相关：_parameters和_buffers，它们的类型都是Dict，区别在于_parameters是可以更新的参数，而_buffers不需要更新。

从定义上看，_buffers中存放的是Tensor类型的数据，而_parameters中存放的是Parameter类型的数据，在构造时参数requires_grad缺省为True

```Python
# torch/nn/parameter.py

class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    def __new__(cls, data=None, requires_grad=True):
        # ......
```

当构造好Parameter并且赋值给nn.Module时，会自动调用nn.Module的register_parameter()方法进行注册。

```Python
# torch/nn/modules/module.py

class Module:

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        # handle value with other types

```

为了看的更清楚一些，我们看一下PyTorch中内置的网络组件，例如：

```Python
# torch/nn/modules/conv.py

class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    _in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None) -> None:
        super(_ConvNd, self).__init__()
        
        # check and handle padding and other parameter...

        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
```

## 计算图的执行过程

在深度学习中，我们的神经网络一般是基于nn.Module实现的，典型的调用方式是：
```Python
    y = DemoNet(x)
    loss = compute_loss(y, label)
```

可见计算图的执行其实就是nn.Module的调用过程，从下面的实现中可以看出，主要的工作就是调用forward()方法进行计算，这也是我们在实现自己的神经网络时所要实现的。

```Python
# torch/nn/modules/module.py

class Module:

    def _call_impl(self, *input, **kwargs):
        forward_call = (self._slow_forward if torch._C._get_tracing_state() else self.forward)

        # YL: handle pre-forward hooks, you can change input here
        # ...

        result = forward_call(*input, **kwargs)
        # YL: handle forward hooks
        # ...

        # Handle the non-full backward hooks
        # ...

        return result

    __call__ : Callable[..., Any] = _call_impl

```

相应的，我们可以看一下卷积操作的实现：

```Python
# torch/nn/modules/conv.py

from .. import functional as F

class Conv2d(_ConvNd):

    ## YL： __init__() implemetation here


    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

```

由此可见，卷积算子的实现调用了functional模块中的卷积函数。这也说明，在PyTorch中，神经网络的定义和算子的执行是解耦的，这样做的好处很明显，我们可以很容易的替换或者复用各种定义好的网络及算子。



Dispatch



## 参考
- https://zhuanlan.zhihu.com/p/89442276
- https://github.dev/pytorch/rfcs/blob/rfc-0005/RFC-0005-structured-kernel-definitions.md
