# 1. Installation
### 1.1. Dependencies
- PyTorch
- nerfstudio
- tinycudann

The dependencies can be conveniently installed via:
```
conda env create -f environment.yml
```
which creates a new conda environment `nersemble` (Installation may take a while).

Afterwards, manually install `tinycudann`:
```
pip install --force-reinstall --no-deps git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
(If you get an error like `ImportError: DLL load failed while importing _86_C: The specified procedure could not be found.` later on)

### 1.2. Troubleshooting

You may run into this error at the beginning of training:
```shell
\lib\site-packages\torch\include\pybind11\cast.h(624): error: too few arguments for template template parameter "Tuple"
          detected during instantiation of class "pybind11::detail::tuple_caster<Tuple, Ts...> [with Tuple=std::pair, Ts=<T1, T2>]"
(721): here

\lib\site-packages\torch\include\pybind11\cast.h(717): error: too few arguments for template template parameter "Tuple"
          detected during instantiation of class "pybind11::detail::tuple_caster<Tuple, Ts...> [with Tuple=std::pair, Ts=<T1, T2>]"
(721): here
```
This occurs during compilation of `torch_efficient_distloss` and can be solved by either training without 
distortion loss or by changing one line in the `torch_efficient_distloss` library (see [https://github.com/sunset1995/torch_efficient_distloss/issues/8](https://github.com/sunset1995/torch_efficient_distloss/issues/8)).