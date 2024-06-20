# 第11章 JIT

## 主要内容
- [JIT介绍](#JIT介绍)
- [](#)
- [参考](#参考)

## JIT介绍
PyTorch的JIT（Just-In-Time）编译器是一种支持动态深度学习的通用高性能系统，可以提供端到端的编译技术，从模型定义到最终部署。JIT可以从Python和PyTorch中导出代码，并将其转换为可以在硬件上运行的机器代码。 用于编译的JIT编译器支持多种机器学习框架，例如PyTorch、TensorFlow、Caffe2、MXNet和ONNX等，并具有支持各种硬件平台（例如CPU、GPU、FPGA和ASIC）的能力。

JIT的工作原理是：它从模型定义中提取出计算图，然后将其转换成LLVM中间表示（IR），这是一种通用的汇编语言，然后使用LLVM优化器对IR进行优化，以提高性能，最后使用LLVM编译器将IR编译为机器代码。JIT可以运行在CPU和GPU上，并使用不同的技术来优化编译，以提高性能。

JIT的优势在于可以在运行时动态编译模型，因此可以有效地处理复杂的模型，并有效地对计算图中的节点进行优化。此外，JIT可以在编译时与深度学习框架集成，从而提高模型编译的效率，并且可以使用更少的编译步骤来提高性能。

JIT编译器的另一个优势是它可以根据不同的硬件平台（例如CPU、GPU、FPGA和ASIC）自动优化代码，以提高性能。此外，它还可以基于模型的特定要求来优化代码，从而提高模型的性能和可移植性。

总而言之，PyTorch的JIT编译器是一种非常有用的工具，可以有效地优化深度学习模型，并在不同硬件平台上提供更高的性能。它可以动态编译模型，并使用LLVM优化器和编译器来优化代码，从而提高模型的性能和可移植性。
## TorchScript 



- 性能

### 实现JIT的挑战
- 动态图中的条件逻辑


## 一个简单的🌰子

为了说明JIT是如何工作的，我们看一个简单的例子：
```Python
@torch.jit.script
def foo(len):
    # type: (int) -> torch.Tensor
    rv = torch.zeros(3, 4)
    for i in range(len):
        if i < 10:
            rv = rv - 1.0
        else:
            rv = rv + 1.0
    return rv

print(foo.code)
```
加上修饰器后，上面的函数foo的类型变成了<class 'torch._C.Function'>， 并且其代码被重新编译成了下面的形式：

```Bash
def foo(len: int) -> Tensor:
  rv = torch.zeros([3, 4], dtype=None, layout=None, device=None, pin_memory=None)
  rv0 = rv
  for i in range(len):
    if torch.lt(i, 10):
      rv1 = torch.sub(rv0, 1., 1)
    else:
      rv1 = torch.add(rv0, 1., 1)
    rv0 = rv1
  return rv0
```

可见其中基本的条件语句被转换成了torch的函数，但这仍然是Python代码层面，在执行层，TorchScript使用的是静态单赋值中间表示（static single assignment (SSA) intermediate representation (IR)），其中的指令包括 ATen (the C++ backend of PyTorch) 算子及其他一些原语，比如条件控制和循环控制的原语。

如果我们打印print(foo.graph)，可以看到如下的输出，其中“<ipython-input-4-01a58e79a588>:5:4” 这样的注释代表中间代码所对应的Python源代码位置，这里我使用的是Jupyter-Notebook，读者朋友可以忽略文件名，只关注代码位置即可。

```Bash
graph(%len.1 : int):
  %20 : int = prim::Constant[value=1]()
  %13 : bool = prim::Constant[value=1]() # <ipython-input-4-01a58e79a588>:5:4
  %5 : None = prim::Constant()
  %1 : int = prim::Constant[value=3]() # <ipython-input-4-01a58e79a588>:4:21
  %2 : int = prim::Constant[value=4]() # <ipython-input-4-01a58e79a588>:4:24
  %16 : int = prim::Constant[value=10]() # <ipython-input-4-01a58e79a588>:6:15
  %19 : float = prim::Constant[value=1]() # <ipython-input-4-01a58e79a588>:7:22
  %4 : int[] = prim::ListConstruct(%1, %2)
  %rv.1 : Tensor = aten::zeros(%4, %5, %5, %5, %5) # <ipython-input-4-01a58e79a588>:4:9
  %rv : Tensor = prim::Loop(%len.1, %13, %rv.1) # <ipython-input-4-01a58e79a588>:5:4
    block0(%i.1 : int, %rv.14 : Tensor):
      %17 : bool = aten::lt(%i.1, %16) # <ipython-input-4-01a58e79a588>:6:11
      %rv.13 : Tensor = prim::If(%17) # <ipython-input-4-01a58e79a588>:6:8
        block0():
          %rv.3 : Tensor = aten::sub(%rv.14, %19, %20) # <ipython-input-4-01a58e79a588>:7:17
          -> (%rv.3)
        block1():
          %rv.6 : Tensor = aten::add(%rv.14, %19, %20) # <ipython-input-4-01a58e79a588>:9:17
          -> (%rv.6)
      -> (%13, %rv.13)
  return (%rv)

```

## JIT trace的实现

```Python
def fill_row_zero(x):
    x[0] = torch.rand(*x.shape[1:2])
    return x

traced = torch.jit.trace(fill_row_zero, (torch.rand(3, 4),))
print(traced.graph)
```

Trace的实现在这里（不同版本的实现位置可能不一样）：

```Python
# torch/jit/_trace.py

def trace(
    func,
    example_inputs,
    optimize=None,
    check_trace=True,
    check_inputs=None,
    check_tolerance=1e-5,
    strict=True,
    _force_outplace=False,
    _module_class=None,
    _compilation_unit=_python_cu,
):

    #YL 检查输入，如果是输入Module，则调用trace_module

    var_lookup_fn = _create_interpreter_name_lookup_fn(0)

    name = _qualified_name(func)
    traced = torch._C._create_function_from_trace(
        name,
        func,
        example_inputs,
        var_lookup_fn,
        strict,
        _force_outplace,
        get_callable_argument_names(func)
    )

    # Check the trace against new traces created from user-specified inputs

    return traced
```
_C是torch的C++模块，因此该调用转到了C++部分，在初始化的时候，_create_function_from_trace被注册到了torch的_C模块中。

```C++
//YL torch/csrc/jit/python/script_init.cpp

  m.def(
      "_create_function_from_trace",
      [](const std::string& qualname,
         const py::function& func,
         const py::tuple& input_tuple,
         const py::function& var_name_lookup_fn,
         bool strict,
         bool force_outplace,
         const std::vector<std::string>& argument_names) {
        auto typed_inputs = toTraceableStack(input_tuple);
        std::shared_ptr<Graph> graph = std::get<0>(tracer::createGraphByTracing(
            func,
            typed_inputs,
            var_name_lookup_fn,
            strict,
            force_outplace,
            /*self=*/nullptr,
            argument_names));

        auto cu = get_python_cu();
        auto name = c10::QualifiedName(qualname);
        auto result = cu->create_function(
            std::move(name), std::move(graph), /*shouldMangle=*/true);
        StrongFunctionPtr ret(std::move(cu), result);
        didFinishEmitFunction(ret);
        return ret;
      },
      py::arg("name"),
      py::arg("func"),
      py::arg("input_tuple"),
      py::arg("var_name_lookup_fn"),
      py::arg("strict"),
      py::arg("force_outplace"),
      py::arg("argument_names") = std::vector<std::string>());
```

可以看到，主要的工作是构造一个Graph，并且是由tracer::createGraphByTracing()完成的。



## 参考
- https://pytorch.org/docs/stable/jit.html
- https://zhuanlan.zhihu.com/p/410507557
