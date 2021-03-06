# ---[ GPU files
# ------[ cuDNN
if (USE_CUDNN)
  file(GLOB tmp *_cudnn.cc)
  set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
endif()
# ------[ general GPU
file(GLOB tmp *_gpu.cc)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
# ------[ TensorRT
if (USE_TENSORRT)
file(GLOB tmp *_trt.cc)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
endif()
# ------[ CUDA sources
file(GLOB tmp *.cu)
# Manually remove the cudnn files since we might be using USE_CUDNN=OFF
# TODO: when we move to explicit file list, this would not be needed.
file(GLOB tmp_cudnn *_cudnn.cu)
exclude(tmp "${tmp}" ${tmp_cudnn})
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
# exclude test files
file(GLOB tmp *_test.cc)
exclude(Caffe2_GPU_SRCS "${Caffe2_GPU_SRCS}" ${tmp})

# ------[ general HIP
file(GLOB tmp hip/*.cc)
set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} ${tmp})
# ------[ HIP sources
file(GLOB tmp hip/*.hip)
set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} ${tmp})
# exclude test files
file(GLOB tmp hip/*_test.cc)
exclude(Caffe2_HIP_SRCS "${Caffe2_HIP_SRCS}" ${tmp})

# ---[ CPU files.
file(GLOB tmp *.cc)
# Manually remove the cudnn files since we might be using USE_CUDNN=OFF
# TODO: when we move to explicit file list, this would not be needed.
file(GLOB tmp_cudnn *_cudnn.cc)
exclude(tmp "${tmp}" ${tmp_cudnn})
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})

if (BUILD_C10_EXPERIMENTAL_OPS)
    file(GLOB_RECURSE tmp experimental/c10/*.cc)
endif()

set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})
# exclude test files and gpu files
file(GLOB tmp *_test.cc)
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${tmp})
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_GPU_SRCS} ${Caffe2_HIP_SRCS})

# ---[ GPU test files
# ------[ cuDNN
if (USE_CUDNN)
  file(GLOB tmp *_cudnn_test.cc)
  set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} ${tmp})
endif()
# ------[ general GPU
file(GLOB tmp *_gpu_test.cc)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} ${tmp})

# ---[ HIP test files
file(GLOB tmp hip/*_test.cc)
set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} ${tmp})

# ---[ CPU test files
file(GLOB tmp *_test.cc)
# Manually remove the cudnn files since we might be using USE_CUDNN=OFF
# TODO: when we move to explicit file list, this would not be needed.
file(GLOB tmp_cudnn *_cudnn_test.cc)
exclude(tmp "${tmp}" ${tmp_cudnn})
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${tmp})
exclude(Caffe2_CPU_TEST_SRCS "${Caffe2_CPU_TEST_SRCS}" ${Caffe2_GPU_TEST_SRCS} ${Caffe2_HIP_TEST_SRCS})

# ---[ Send the lists to the parent scope.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} PARENT_SCOPE)
set(Caffe2_HIP_SRCS ${Caffe2_HIP_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_TEST_SRCS ${Caffe2_GPU_TEST_SRCS} PARENT_SCOPE)
set(Caffe2_HIP_TEST_SRCS ${Caffe2_HIP_TEST_SRCS} PARENT_SCOPE)
