// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PREPROCESSING_COMMON_PASSES_H_
#define IREE_COMPILER_PREPROCESSING_COMMON_PASSES_H_

#include <functional>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

// Creates a pass to convert linalg convolution ops into linalg.matmul ops
// using im2col tranformation.
std::unique_ptr<Pass> createConvertConv2DToImg2ColPass();

// A pass to pad linalg ops to the next integer multiple of `paddingSize`.
std::unique_ptr<Pass> createPadLinalgOpsToIntegerMultiplePass(
    int paddingSize = 4);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerCommonPreprocessingPasses();

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_PREPROCESSING_COMMON_PASSES_H_W_TRANSFORMS_PASSES_H_
