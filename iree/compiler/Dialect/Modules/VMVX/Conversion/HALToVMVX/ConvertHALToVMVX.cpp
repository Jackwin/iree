// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/Modules/VMVX/Conversion/HALToVMVX/ConvertHALToVMVX.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXOps.h"
#include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Rewrites entry functions to have a vmvx.interface, scratchpad, and an XYZ
// workgroup ID. The runtime will provide these values during invocation.
//
// (%interface: !vmvx.interface, scratchpad: !vmvx.buffer,
//  %workgroup_x: index, %workgroup_y: index, %workgroup_z: index)
struct InterfaceFuncOpConversion : public OpConversionPattern<mlir::FuncOp> {
  InterfaceFuncOpConversion(MLIRContext *context,
                            PatternBenefit benefit = 10000)
      : OpConversionPattern(context, benefit) {}

  LogicalResult matchAndRewrite(
      mlir::FuncOp funcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto originalType = funcOp.getType();
    if (originalType.getNumInputs() != 0 || originalType.getNumResults() != 0) {
      return funcOp.emitError() << "exported functions must have no I/O";
    }

    auto interfaceType = IREE::VMVX::InterfaceType::get(rewriter.getContext());
    auto bufferType = IREE::VMVX::BufferType::get(rewriter.getContext());
    auto indexType = IndexType::get(rewriter.getContext());
    auto newType = FunctionType::get(rewriter.getContext(),
                                     {
                                         /*interface=*/interfaceType,
                                         /*scratchpad=*/bufferType,
                                         /*workgroup_x=*/indexType,
                                         /*workgroup_y=*/indexType,
                                         /*workgroup_z=*/indexType,
                                     },
                                     {});
    if (funcOp.getType() == newType) {
      return rewriter.notifyMatchFailure(
          funcOp, "function already matches required signature");
    }

    rewriter.updateRootInPlace(funcOp, [&]() {
      funcOp.setType(newType);
      funcOp.front().addArguments(newType.getInputs());
    });

    return success();
  }
};

struct InterfaceLoadConstantOpConversion
    : public OpConversionPattern<IREE::HAL::InterfaceLoadConstantOp> {
  InterfaceLoadConstantOpConversion(MLIRContext *context,
                                    PatternBenefit benefit = 1)
      : OpConversionPattern(context) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceLoadConstantOp loadOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Find the vmvx.interface argument to the function.
    auto interfaceArg = loadOp->getParentOfType<FuncOp>().getArgument(0);
    assert(interfaceArg &&
           interfaceArg.getType().isa<IREE::VMVX::InterfaceType>() &&
           "exported VMVX functions require vmvx.interface ops as their only "
           "argument");

    IREE::HAL::InterfaceLoadConstantOp::Adaptor newOperands(operands);
    rewriter.replaceOpWithNewOp<IREE::VMVX::InterfaceConstantOp>(
        loadOp, loadOp.getResult().getType(), interfaceArg,
        loadOp.offsetAttr());
    return success();
  }
};

}  // namespace

void populateHALToVMVXPatterns(MLIRContext *context,
                               OwningRewritePatternList &patterns,
                               TypeConverter &typeConverter) {
  patterns.insert<InterfaceFuncOpConversion>(context);
  patterns.insert<InterfaceLoadConstantOpConversion>(context);
}

}  // namespace iree_compiler
}  // namespace mlir
