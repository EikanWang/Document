In our previous [blog post](https://dev-discuss.pytorch.org/t/Inductor-update-4-cpu-backend-started-to-show-promising-performance-boost/874), @jgong5 shared the progress and plan about the optimization work for the Inductor C++/OpenMP backend. In this post, I’m going to refresh the performance data and have a technical deep-dive into those key optimization techniques used to achieve this improved performance.

### Performance Update

We employed the hybrid strategy to optimize the Inductor CPU backend. We categorize the ops into two types: Conv/GEMM and non-Conv/GEMM element-wise and reduction ops, leveraging oneDNN performance library to optimize the former and Inductor C++ codegen to optimize the latter. We applied post-op fusion and weight prepacking with oneDNN library and applied explicit vectorization in C++ codegen to get the optimal performance as measured on popular deep learning models.

Compared to eager mode with these optimizations, the C++/OpenMP backend shows promising performance improvements. We measured the performance of the three Inductor benchmark suites – TorchBench, HuggingFace, and TIMM – and the results are as follows. Additionally, we publish our performance data twice per week on GitHub at https://github.com/pytorch/pytorch/issues/93531.

Overall, these optimizations help to ensure that the C++/OpenMP backend provides efficient and reliable support for PyTorch models.

Passrate
~~~
+----------+------------+-------------+-------------+
| Compiler | torchbench | huggingface | timm_models |
+----------+------------+-------------+-------------+
| inductor | 93%, 56/60 | 96%, 44/46  | 100%, 61/61 |
+----------+------------+-------------+-------------+
~~~
Geometric mean speedup (Single-Socket Multi-threads)
~~~
+----------+------------+-------------+-------------+
| Compiler | torchbench | huggingface | timm_models |
+----------+------------+-------------+-------------+
| inductor |   1.39x    |    1.20x    |    1.73x    |
+----------+------------+-------------+-------------+
~~~
Geometric mean speedup (Single-core Single-thread)
~~~
+----------+------------+-------------+-------------+
| Compiler | torchbench | huggingface | timm_models |
+----------+------------+-------------+-------------+
| inductor |   1.29x    |    1.15x    |    1.37x    |
+----------+------------+-------------+-------------+
~~~

### Technical Deep Dive

Now, let's take a closer look at the two primary optimizations used in the Inductor C++/OpenMP backend: weight prepacking and post-operation fusion via oneDNN library and explicit vectorization in Inductor C++ codegen. 

#### Post-op Fusion & Weight-Prepacking via oneDNN

The oneDNN library provides a range of post-op fusions that can benefit popular models. The [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch) has implemented most of these fusions and has achieved significant performance improvements. As a result, we have ported all of these fusions that have been applied in IPEX to Inductor, enabling a wider range of models to benefit from these optimizations.
We have defined these fusions as operators under the mkldnn namespace. This allows the Python module to invoke these mkldnn operations directly. Currently, the defined fused operations are as follows. And You can find these defined fused operations at [RegisterMkldnnOpContextClass.cpp](https://github.com/pytorch/pytorch/blob/fe05266fda4f908130dea7cbac37e9264c0429a2/aten/src/ATen/native/mkldnn/RegisterMkldnnOpContextClass.cpp#L35-#L48).

- `_linear_pointwise`: Fuse `Linear` and its post-unary element-wise operations, 
- `_linear_pointwise.binary`: Fuses `Linear` and its post-binary element-wise operations. 
- `_convolution_pointwise`: Fuses `Convolution` and its post-unary element-wise operations
 - `_convolution_pointwise.binary`: Fuses `Convolution` and its post-binary element-wise operations. 

The detailed fusion patterns are defined in the [mkldnn.py](https://github.com/pytorch/pytorch/blob/fe05266fda4f908130dea7cbac37e9264c0429a2/torch/_inductor/mkldnn.py#L774-#L818) file.
•	`convolution`/`linear` + `sigmoid`/`hardsigmoid`/`tanh`/`hardtanh`/`hardswish`/`leaky_relu`/`gelu`/`relu`/`relu6`/`silu`
•	`convolution`/`linear` + `add`/`add_`/`iadd`/`sub`/`sub_`

On the Inductor side, we apply these fusions on the FX graph that has been lowered. We have defined [mkldnn_fuse_fx](https://github.com/pytorch/pytorch/blob/fe05266fda4f908130dea7cbac37e9264c0429a2/torch/_inductor/mkldnn.py#L491) as the entry point to apply all the fusions. The code snippet for this is as follows:

```python
def mkldnn_fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    ...
    gm = fuse_unary(gm)
    gm = fuse_binary(gm)
    ...
    if config.cpp.weight_prepack:
        gm = pack_module(gm)
    return gm
```

In the `mkldnn_fuse_fx` function, we apply fusion on the FX graph that has been lowered yet. To fuse `convolution`/`linear` and its consecutive elementwise operations, we invoke `fuse_unary` and `fuse_binary` as follows:

```python
    gm = fuse_unary(gm)
    gm = fuse_binary(gm)
```

In addition to the Post-op fusion, we apply weight-prepacking to improve the Conv/GEMM performance further.

```python
    gm = pack_module(gm)
```

Weight pre-packing involves rearranging the weight tensor in a blocked layout, which can improve vectorization and cache reuse compared to plain formats like NCHW or NHWC. This optimization can help avoid weight reordering at runtime, which can reduce overhead and improve performance.
This optimization increases memory usage as the tradeoff. Therefore, we provide `config.cpp.weight_prepack` flag in Indcutor to provide users with more control over this optimization, allowing them to enable or disable it based on their specific needs.

#### Vectorization in C++ Codegen

Vectorization is a key optimization technique that can significantly improve the performance of numerical computations. By utilizing SIMD (Single Instruction, Multiple Data) instructions, vectorization enables multiple computations to be performed simultaneously on a single processor core, which can lead to significant performance improvements.

In the Inductor C++/OpenMP backend, we use [AVX2](https://github.com/pytorch/pytorch/blob/fe05266fda4f908130dea7cbac37e9264c0429a2/torch/_inductor/codecache.py#L372) and [AVX512](https://github.com/pytorch/pytorch/blob/fe05266fda4f908130dea7cbac37e9264c0429a2/torch/_inductor/codecache.py#L359) instructions for vectorization by leveraging the aten vectorization library to facilitate the implementation. Aten vectorization supports multiple platforms, including x86 and Arm, as well as multiple data types. This allows Inductor to easily support other platforms and data types in the future.

Currently, the C++/OpenMP backend of Inductor provides two vectorization ISA (Instruction Set Architecture) options, AVX2 and AVX512. It can be extended to support other ISAs easily by adding more [VecISA](https://github.com/pytorch/pytorch/blob/fe05266fda4f908130dea7cbac37e9264c0429a2/torch/_inductor/codecache.py#L275) sub-classes.

Due to differences in platforms, the C++/OpenMP backend of Inductor starts by detecting the CPU features to determine the vectorization bit width at the beginning of code generation. By default, if the machine supports both AVX512 and AVX2, the backend will choose 512-bit vectorization.

If the hardware supports vectorization, the C++/OpenMP backend first detects if the loop body can be vectorized or not. There are primarily three scenarios that we are not able to generate kernel with vectorization:

- There lacks vector intrinsics support, e.g., `rand` and `atomic_add`.
- There lacks efficient vector intrinsics support, e.g., non-contiguous `load`/`store`.
- Data types with vectorization not yet supported but work in progress, e.g., integer, double, half, and bfloat16.

To address this issue, the C++/OpenMP backend uses [CppVecKernelChecker](https://github.com/pytorch/pytorch/blob/fe05266fda4f908130dea7cbac37e9264c0429a2/torch/_inductor/codegen/cpp.py#L1396) to detect whether all operations in a particular loop body can be vectorized or not. In general, we classified the operations into two categories by identifying whether they depend on the context.

For most elementwise operations such as `add`, `sub`, `relu`, vectorization is straightforward, and their execution does not depend on context. However, for certain other operations, their semantics are more complex, and their execution depends on context through static analysis.

For example, let's consider the `where` operation that takes in `mask`, `true_value`, and `false_value` while the `mask` value is loaded from a `uint8` tensor. The fx graph could be as follows.

```python
graph():
    %ops : [#users=9] = placeholder[target=ops]
    %get_index : [#users=1] = call_module[target=get_index](args = (index0,), kwargs = {})
    %load : [#users=1] = call_method[target=load](args = (%ops, arg1_1, %get_index), kwargs = {})
    %to_dtype : [#users=1] = call_method[target=to_dtype](args = (%ops, %load, torch.bool), kwargs = {})
    ...
    %where : [#users=1] = call_method[target=where](args = (%ops, %to_dtype, %to_dtype_2, %to_dtype_3), kwargs = {})
    ...
```

Regarding uint8, it is a general data type and could be used for computation but is not limited to being used as Boolean for `mask`. Hence, we need to analyze its context statically. In particular, the [CppVecKernelChecker](https://github.com/pytorch/pytorch/blob/fe05266fda4f908130dea7cbac37e9264c0429a2/torch/_inductor/codegen/cpp.py#L1396) will check whether a uint8 tensor is only used by `to_dtype` and `to_dtype` is only used by `where`. If yes, it could be vectorized. Otherwise, it will fall back to the scalar version. The generated code could be as follows.

Scalar Version
```C++
auto tmp0 = in_ptr0[i1 + (17*i0)];
auto tmp3 = in_ptr1[i1 + (17*i0)];
auto tmp1 = static_cast<bool>(tmp0);
auto tmp2 = static_cast<float>(-33.0);
auto tmp4 = tmp1 ? tmp2 : tmp3;
tmp5 = std::max(tmp5, tmp4);
```

Vectorization Version
```C++
float g_tmp_buffer_in_ptr0[16] = {0};
// Convert the flag to float for vectorization. 
flag_to_float(in_ptr0 + (16*i1) + (17*i0), g_tmp_buffer_in_ptr0, 16);
auto tmp0 = at::vec::Vectorized<float>::loadu(g_tmp_buffer_in_ptr0);
auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + (16*i1) + (17*i0));
auto tmp1 = (tmp0);
auto tmp2 = at::vec::Vectorized<float>(static_cast<float>(-33.0));
auto tmp4 = decltype(tmp2)::blendv(tmp3, tmp2, tmp1);
```

In addition to context analysis, the C++/OpenMP backend also incorporates several other vectorization-related optimizations. These include:
- Tiled kernel implementation for supporting transpose load  - [cpp.py](https://github.com/pytorch/pytorch/blob/fe05266fda4f908130dea7cbac37e9264c0429a2/torch/_inductor/codegen/cpp.py#L1211)
- Data type demotion based on value range - [cpp.py](https://github.com/pytorch/pytorch/blob/fe05266fda4f908130dea7cbac37e9264c0429a2/torch/_inductor/codegen/cpp.py#L1647-#L1672)
- Replacement of sleef implementation with oneDNN/oneMKL implementation for optimizing aten vectorization - [#94577](https://github.com/pytorch/pytorch/pull/94577), [#92289](https://github.com/pytorch/pytorch/pull/92289), [#91613](https://github.com/pytorch/pytorch/pull/91613)

With all the optimizations, including weight prepack, post-op fusion, vectorization, and other miscellaneous optimizations, we have achieved promising performance improvements. 


The next step, we will continue optimizing the C++/OpenMP backend and extend it to support more data types as the next step. This includes:
1. Low-precision (BF16 and INT8) inference optimization
2. Training optimization
3. Loop tiling
4. Autotune
5. Further fusion optimization of Conv/GEMM kernels.
6. Explore alternative codegen paths: clang/llvm/triton

### Summary

This blog post from the Intel PyTorch team provides an update on the performance optimizations made in the Inductor C++/OpenMP backend. The team has used a hybrid optimization strategy that leverages the oneDNN performance library to optimize Convolution/General Matrix Multiplication (GEMM) operations and Inductor C++ codegen to optimize element-wise and reduction operations. The team also uses weight pre-packing and post-operation fusion via the oneDNN library to further optimize performance. The post explains the technical details of these optimization techniques and provides performance data updates on TorchBench, HuggingFace, and TIMM.

Many thanks to @jansel , @desertfire , and @Chillee for their invaluable contributions and unwavering support during the development.
