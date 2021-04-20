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

#include "iree/compiler/Dialect/Modules/VMVX/Transforms/Passes.h"

#include <memory>

#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VMVX {

void buildVMVXTransformPassPipeline(OpPassManager &passManager) {
  // ---------------------------------------------------------------------------
  // VMVX conversion.
  // ---------------------------------------------------------------------------
  passManager.addPass(createConversionPass());

  // ---------------------------------------------------------------------------
  // Cleanup identity ops that clutter up the IR and canonicalize.
  // ---------------------------------------------------------------------------
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
}

void createVMVXTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-vmla-transformation-pipeline",
      "Runs the full IREE VMVX dialect transformation pipeline",
      [](OpPassManager &passManager) {
        buildVMVXTransformPassPipeline(passManager);
      });
}

}  // namespace VMVX
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
