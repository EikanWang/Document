We have utilized a hybrid strategy to optimize the Inductor CPU backend. This strategy involves categorizing the operations into two types: Conv/GEMM and non-Conv/GEMM element-wise and reduction ops. To optimize the former, we leveraged the oneDNN performance library by applying post-op fusion and weight prepacking. While for the latter, we utilized Inductor C++ codegen by applying explicit vectorization in C++ codegen to obtain optimal performance. As measured on popular deep learning models, this hybrid strategy has shown promising results in terms of performance improvements compared to eager mode with these optimizations.

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

TODO: Add detailed model performance here.

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

With all the optimizations, including weight prepack, post-op fusion, vectorization, and other miscellaneous optimizations, we have achieved promising performance improvements. We examined vectorization optimization in Inductor CPP backend for FP32 training and inference of 150 benchmark models. 90% of inference kernels and 71% of training kernels are vectorized.

In terms of inference, a total of 28,185 CPP kernels were generated, with 25,579 (90%) of them being vectorized, while the remaining 10% were scalar. As for training, 103,084 kernels were generated, with 73,909 (71%) being vectorized and 29% not vectorized. The results indicate that the vectorization of inference kernels is quite impressive, while there is still some work to be done in training kernels since we just started to work on the training. In the following section, we will analyze the non-vectorized kernels with specific examples to identify the most critical missing features.The remaining non-vectorized kernels are analyzed in 10 categories, highlighting the next steps to improve vectorization coverage: index-related operations, int64 support, vertical reduction, vectorization with fallback, and more.

#### Non-Vectorization Scenarios

The `CppVecKernelChecker` class and `CppTile2DKernelChecker` class in CPP codegen implement specific rules to determine the feasibility of vectorizing a kernel. A recent pull request 2 includes debug logs that help identify why a kernel may fail to vectorize by providing insight into the conditions that were not met. The information has been grouped into 10 categories to help understand the reasons for vectorization failure. The two charts below illustrate the frequency of occurrence of each category for three different benchmarks, one for inference and the other for training.

![image](https://user-images.githubusercontent.com/55483091/235278849-5d4c9b97-abcc-4ccd-96da-b9daae97a622.png)
![image](https://user-images.githubusercontent.com/55483091/235278867-8075496e-2f0a-4db2-ab10-c6e1796eab5b.png)

- **_index_expr_**: The main limitation for vectorization is the absence of the support related to indices. Presently, the computation on indices is not vectorized, except for cases where a scalar index is broadcasted as a vector, which must remain constant with respect to the loop variable being vectorized.
- **_to_dtype_**: At present, we don’t provide support for vectorization of int64 and double data types. Supporting vectorization for these types requires matching the number of vector lanes if we also want to vectorize float32 and/or int32 simultaneously. To accomplish this, we may need to use two vector variables to hold int64 or double vectors to match one float32 or int32 vector variable. The problems with the “to_dtype” function are specifically connected to these two data types. In the majority of real benchmarks, int64 and double are commonly utilized by the calculation of scalar indices, making vectorization unnecessary.
- **_indirect_indexing_**: We exclude all indirect indexing cases from vectorization, but upon observation, we find that we can still vectorize some cases when the indirect index variables remain constant with respect to the loop variables we want to vectorize. 
- **_Unsupported** masked_: We vectorize the kernel containing “masked” op conservatively and don’t allow any actual computation inside it. This means that cases with nested masked bodies or computations within the “masked” element cannot be vectorized, such as the one found in jx_nest_base.
- **_Unsupported dtype in load/store_**: Similarly to the “dtype” case, the int64 and double vectorized data types are unsupported. Supporting vectorization for these types requires matching the number of vector lanes if we also want to vectorize float32 and/or int32 simultaneously. To accomplish this, we may need to use two vector variables to hold int64 or double vectors to match one float32 or int32 vector variable.
Based on real benchmarks, the majority of cases where vectorization is lacking is due to the absence of int64 vectorization support. These cases fall into two main scenarios: 1) int64 is loaded for indirect indexing, and 2) int64 is loaded for computation.
- **_non-contiguous load/store (excluding indirect indexing)_**: `CppTile2DKernel` with 2d transposition support has already vectorized some of the non-contiguous load/store. However, there are still two main cases that have not been covered yet. The first case occurs frequently in most models during training backward, where the non-contiguous load/store happens on the inner-most reduction loop while being contiguous on an outer parallel loop, which is known as vertical reduction.
- **_Unsupported ops_**: We currently do not support vectorization for some operations such as bitwise_and, bitwise_or, bitwise_xor, logical_not, remainder, truediv, among others. However, most of these operations should be easy to support. Although there are a few instances of “randn” which are difficult to vectorize, they occur infrequently.
- **_Unsupported reduction_**: The main reason for the lack of support for reduction operations is primarily attributed to the absence of support for int64 vectorization.
- **_Unsupported constant dtype_**: Vectorization is not implemented for constant of data type uint8 or bool. They happen less frequently and can be handled as low priority.
- **_Unsupported store modes_**:  “atomic_add” cannot be vectorized. Still, we can do vectorization with fallback to maximize the performance, e.g., from AlbertForMaskedLM, we are able to vectorize all the ops except for atomic_add which can be put into an inner loop.

### Next Step
The next step, we will continue optimizing the C++/OpenMP backend and extend it to support more data types as the next step. This includes:
1. Improve vectorization coverage
2. Training optimization
3. Loop tiling
4. Autotune
5. Further fusion optimization of Conv/GEMM kernels.
6. Explore alternative codegen paths: clang/llvm/triton

### Summary

Vectorization optimization has been a significant improvement in Inductor CPP backend. The analysis shows that a big portion of kernels have already been vectorized. The remaining non-vectorized kernels have been categorized into 10 categories with suggested features as the next steps. With them, Inductor CPU’s performance can continue to be enhanced, making it a more efficient and effective tool for deep learning applications.

