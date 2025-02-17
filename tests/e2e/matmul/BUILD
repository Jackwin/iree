# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# End-to-end matrix multiplication tests.

load("//build_tools/bazel:iree_trace_runner_test.bzl", "iree_generated_trace_runner_test")
load("//build_tools/bazel:iree_check_test.bzl", "iree_check_single_backend_test_suite")

package(
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

py_binary(
    name = "generate_e2e_matmul_tests",
    srcs = ["generate_e2e_matmul_tests.py"],
)

[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_%s_small" % lhs_rhs_type,
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=small",
    ],
    target_backends_and_drivers = [
        ("llvm-cpu", "local-task"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
]]

# Test asm
[iree_generated_trace_runner_test(
    name = "e2e_matmul_mmt4d_%s_small" % lhs_rhs_type,
    compiler_flags = [
        "--iree-flow-enable-data-tiling",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=small",
    ],
    target_backends_and_drivers = [
        ("llvm-cpu", "local-task"),
    ],
    target_cpu_features_variants = ["default"] +
                                   ([
                                       "aarch64:+dotprod",
                                       "aarch64:+i8mm",
                                   ] if lhs_rhs_type == "i8" else []),
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
]]

[iree_generated_trace_runner_test(
    name = "e2e_matmul_mmt4d_%s_large" % lhs_rhs_type,
    compiler_flags = [
        "--iree-flow-enable-data-tiling",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=large",
    ],
    target_backends_and_drivers = [
        ("llvm-cpu", "local-task"),
    ],
    target_cpu_features_variants = ["default"] +
                                   ([
                                       "aarch64:+dotprod",
                                       "aarch64:+i8mm",
                                   ] if lhs_rhs_type == "i8" else []),
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
]]

# Test intrinsics. No need to run vmvx again, since it isn't affected by this
# codegen flag. No need to run "large" sizes, since this only differs from other
# tests in ways that are orthogonal to problem sizes.
[iree_generated_trace_runner_test(
    name = "e2e_matmul_mmt4d_%s_intrinsics_%s" % (lhs_rhs_type, size),
    compiler_flags = [
        "--iree-codegen-mmt4d-use-intrinsics",
        "--iree-flow-enable-data-tiling",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=%s" % size,
    ],
    target_backends_and_drivers = [
        ("llvm-cpu", "local-task"),
    ],
    target_cpu_features_variants = ["default"] +
                                   ([
                                       "aarch64:+dotprod",
                                       "aarch64:+i8mm",
                                   ] if lhs_rhs_type == "i8" else []),
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
] for size in [
    "small",
]]

# Test VMVX+ukernel, direct (not mmt4d)
[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_%s_small_ukernel" % lhs_rhs_type,
    compiler_flags = [
        "--iree-vmvx-enable-microkernels",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=small",
    ],
    target_backends_and_drivers = [
        ("vmvx", "local-task"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
]]

# Test VMVX+ukernel, mmt4d, with target CPU features variants relevant to each
# lhs_rhs_type.
[iree_generated_trace_runner_test(
    name = "e2e_matmul_mmt4d_%s_small_ukernel" % lhs_rhs_type,
    compiler_flags = [
        "--iree-vmvx-enable-microkernels",
        "--iree-flow-enable-data-tiling",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=small",
    ],
    target_backends_and_drivers = [
        ("vmvx", "local-task"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "i8",
    "f32",
]]

[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_f32_gpu_large_%s" % compilation_info,
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=f32",
        "--shapes=gpu_large",
        "--compilation_info=%s" % compilation_info,
    ],
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backends_and_drivers = [
        ("cuda", "cuda"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for compilation_info in [
    "LLVMGPUMatmulSimt",
]]

# Testing Ampere + TensorCore path.
# WMMA TensorCore(F32): wmma.161616.f32.tf32
[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_f32_gpu_large_%s" % compilation_info,
    compiler_flags = [
        "--iree-hal-cuda-llvm-target-arch=sm_80",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=f32",
        "--shapes=gpu_large",
        "--compilation_info=%s" % compilation_info,
    ],
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backends_and_drivers = [
        ("cuda", "cuda"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for compilation_info in [
    "LLVMGPUMatmulTensorCore",
]]

# WMMA TensorCore(F16): wmma.161616.f16.f16
[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_f16_gpu_large_%s" % compilation_info,
    compiler_flags = [
        "--iree-hal-cuda-llvm-target-arch=sm_80",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=f16",
        "--shapes=gpu_large",
        "--compilation_info=%s" % compilation_info,
    ],
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backends_and_drivers = [
        ("cuda", "cuda"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for compilation_info in [
    "LLVMGPUMatmulTensorCore",
]]

# MMA.SYNC TensorCore(F16): mma.sync.161616.f16.f16
[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_f16_gpu_large_mma_sync_%s" % compilation_info,
    compiler_flags = [
        "--iree-hal-cuda-llvm-target-arch=sm_80",
        "--iree-codegen-llvmgpu-use-mma-sync=true",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=f16",
        "--shapes=gpu_large",
        "--compilation_info=%s" % compilation_info,
    ],
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backends_and_drivers = [
        ("cuda", "cuda"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for compilation_info in [
    "LLVMGPUMatmulTensorCore",
]]

[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_%s_large_split_k" % lhs_rhs_type,
    compiler_flags = [
        "--iree-flow-split-matmul-reduction=4",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=%s" % lhs_rhs_type,
        "--shapes=large",
    ],
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backends_and_drivers = [
        ("cuda", "cuda"),
        ("llvm-cpu", "local-task"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for lhs_rhs_type in [
    "f32",
]]

[iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_f32_gpu_large_%s" % vulkan_target_and_pipeline[0],
    compiler_flags = [
        "--iree-vulkan-target-triple=%s" % vulkan_target_and_pipeline[0],
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=f32",
        "--shapes=gpu_large",
        "--compilation_info=%s" % vulkan_target_and_pipeline[1],
    ],
    tags = [
        "requires-gpu-nvidia",
    ],
    target_backends_and_drivers = [
        ("vulkan-spirv", "vulkan"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
) for vulkan_target_and_pipeline in [
    ("valhall-unknown-android31", "SPIRVVectorizeMali"),
    ("ampere-unknown-linux", "SPIRVVectorizeNVIDIA"),
]]

# Check tests
iree_check_single_backend_test_suite(
    name = "check_cuda",
    srcs =
        [
            # currently only enabled on cuda as it can be slow on other backends.
            "large_linalg_matmul.mlir",
        ],
    driver = "cuda",
    tags = [
        # CUDA cuInit fails with sanitizer on.
        "noasan",
        "nomsan",
        "notsan",
        "noubsan",
        "requires-gpu-nvidia",
    ],
    target_backend = "cuda",
)

iree_generated_trace_runner_test(
    name = "e2e_matmul_direct_f32_small_no_padding",
    compiler_flags = [
        "--iree-codegen-enable-vector-padding=false",
    ],
    generator = ":generate_e2e_matmul_tests",
    generator_args = [
        "--lhs_rhs_type=f32",
        "--shapes=small",
    ],
    target_backends_and_drivers = [
        ("llvm-cpu", "local-task"),
    ],
    trace_runner = "//tools:iree-e2e-matmul-test",
)
