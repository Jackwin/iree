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

#ifndef IREE_DIALECT_MODULES_VMVX_OPS
#define IREE_DIALECT_MODULES_VMVX_OPS

include "iree/compiler/Dialect/Modules/VMVX/IR/VMVXBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VMVX_PureOp<string mnemonic, list<OpTrait> traits = []> :
    VMVX_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// VMVX Ops: ABI
//===----------------------------------------------------------------------===//

def VMVX_InterfaceConstantOp : VMVX_PureOp<"interface.constant"> {
  let summary = [{interface constant accessor}];
  let description = [{
    XXX
  }];

  let arguments = (ins
    VMVX_Interface:$interface,
    IREE_IndexAttr:$offset
  );
  let results = (outs
    AnyInteger:$result
  );

  let assemblyFormat = [{
    $interface `,` `offset` `=` $offset attr-dict `:` type($result)
  }];
}

def VMVX_InterfaceBindingOp : VMVX_PureOp<"interface.binding"> {
  let summary = [{interface binding accessor}];
  let description = [{
    XXX
  }];

  let arguments = (ins
    VMVX_Interface:$interface,
    IREE_IndexAttr:$ordinal
  );
  let results = (outs
    VMVX_Buffer:$result
  );

  let assemblyFormat = [{
    $interface `,` `ordinal` `=` $ordinal attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// VMVX Ops: buffer manipulation
//===----------------------------------------------------------------------===//

def VMVX_BufferCopyOp : VMVX_PureOp<"buffer.copy"> {
  let summary = [{buffer copy operation}];
  let description = [{
    XXX
  }];

  let arguments = (ins
    VMVX_Buffer:$source_buffer,
    Index:$source_offset,
    VMVX_Buffer:$target_buffer,
    Index:$target_offset,
    Index:$length
  );

  let assemblyFormat = [{
    $source_buffer `[` $source_offset `]` `,`
    $target_buffer `[` $target_offset `]` `,`
    `length` `=` $length
    attr-dict
  }];
}

#endif  // IREE_DIALECT_MODULES_VMVX_OPS
