// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @command_buffer_create
func @command_buffer_create(%arg0 : !hal.device) {
  // CHECK: %ref = vm.call @hal.command_buffer.create(%arg0, %c1, %c3) : (!iree.ref<!hal.device>, i32, i32) -> !iree.ref<!hal.command_buffer>
  %cmd = hal.command_buffer.create %arg0, "OneShot", "Transfer|Dispatch" : !hal.command_buffer
  return
}

// -----

// CHECK-LABEL: @command_buffer_begin_end
func @command_buffer_begin_end(%arg0 : !hal.command_buffer) {
  // CHECK: vm.call @hal.command_buffer.begin(%arg0) : (!iree.ref<!hal.command_buffer>) -> ()
  hal.command_buffer.begin %arg0
  // CHECK: vm.call @hal.command_buffer.end(%arg0) : (!iree.ref<!hal.command_buffer>) -> ()
  hal.command_buffer.end %arg0
  return
}

// -----

// CHECK-LABEL: @command_buffer_execution_barrier
func @command_buffer_execution_barrier(%arg0 : !hal.command_buffer, %arg1 : !hal.buffer) {
  %1 = "test_hal.offset"() : () -> i32
  %2 = "test_hal.length"() : () -> i32
  %memory_barrier = hal.make_memory_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite" : tuple<i32, i32>
  // TODO(benvanik): buffer barriers.
  // %buffer_barrier = hal.make_buffer_barrier "HostRead|HostWrite", "MemoryRead|MemoryWrite", %0, %1, %2 : tuple<i32, i32, !iree.ref<!hal.buffer>, i32, i32>
  // CHECK: vm.call.variadic @hal.command_buffer.execution_barrier(%arg0, %c1, %c2, [%c192, %c768], []) : (!iree.ref<!hal.command_buffer>, i32, i32, i32..., i32...)
  hal.command_buffer.execution_barrier %arg0, "CommandIssue", "CommandProcess",
      memory_barriers=[%memory_barrier, %memory_barrier]
  return
}

// -----

// CHECK-LABEL: @command_buffer_fill_buffer
func @command_buffer_fill_buffer(%arg0 : !hal.command_buffer, %arg1 : !hal.buffer) {
  %0 = "test_hal.offset"() : () -> i32
  %1 = "test_hal.length"() : () -> i32
  %2 = "test_hal.pattern"() : () -> i32
  // CHECK: vm.call @hal.command_buffer.fill_buffer(%arg0, %arg1, %0, %1, %2) : (!iree.ref<!hal.command_buffer>, !iree.ref<!hal.buffer>, i32, i32, i32) -> ()
  hal.command_buffer.fill_buffer %arg0, %arg1, %0, %1, %2
  return
}

// -----

// CHECK-LABEL: @command_buffer_copy_buffer
func @command_buffer_copy_buffer(%arg0 : !hal.command_buffer, %arg1 : !hal.buffer) {
  %0 = "test_hal.source_offset"() : () -> i32
  %1 = "test_hal.target_offset"() : () -> i32
  %2 = "test_hal.length"() : () -> i32
  // CHECK: vm.call @hal.command_buffer.copy_buffer(%arg0, %arg1, %0, %arg1, %1, %2) : (!iree.ref<!hal.command_buffer>, !iree.ref<!hal.buffer>, i32, !iree.ref<!hal.buffer>, i32, i32) -> ()
  hal.command_buffer.copy_buffer %arg0, %arg1, %0, %arg1, %1, %2
  return
}

// -----

// CHECK-LABEL: @command_buffer_bind_descriptor_set
func @command_buffer_bind_descriptor_set(
    %arg0 : !hal.command_buffer,
    %arg1 : !hal.executable_layout,
    %arg2 : !hal.descriptor_set) {
  %0 = "test_hal.offset"() : () -> i32
  // CHECK: vm.call.variadic @hal.command_buffer.bind_descriptor_set(%arg0, %arg1, %zero, %arg2, []) : (!iree.ref<!hal.command_buffer>, !iree.ref<!hal.executable_layout>, i32, !iree.ref<!hal.descriptor_set>, i32...)
  hal.command_buffer.bind_descriptor_set %arg0, %arg1, set=0, %arg2
  // CHECK: vm.call.variadic @hal.command_buffer.bind_descriptor_set(%arg0, %arg1, %zero_0, %arg2, [%0]) : (!iree.ref<!hal.command_buffer>, !iree.ref<!hal.executable_layout>, i32, !iree.ref<!hal.descriptor_set>, i32...)
  hal.command_buffer.bind_descriptor_set %arg0, %arg1, set=0, %arg2, offsets=[%0]
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch
func @command_buffer_dispatch(%arg0 : !hal.command_buffer, %arg1 : !hal.executable) {
  %0 = "test_hal.workgroup_x"() : () -> i32
  %1 = "test_hal.workgroup_y"() : () -> i32
  %2 = "test_hal.workgroup_z"() : () -> i32
  // CHECK: vm.call @hal.command_buffer.dispatch(%arg0, %arg1, %zero, %0, %1, %2) : (!iree.ref<!hal.command_buffer>, !iree.ref<!hal.executable>, i32, i32, i32, i32) -> ()
  hal.command_buffer.dispatch %arg0, %arg1, entry_point=0, workgroup_xyz=[%0, %1, %2]
  return
}

// -----

// CHECK-LABEL: @command_buffer_dispatch_indirect
func @command_buffer_dispatch_indirect(
    %arg0 : !hal.command_buffer,
    %arg1 : !hal.executable,
    %arg2 : !hal.buffer) {
  // CHECK: [[OFFSET:%.+]] = "test_hal.offset"
  %offset = "test_hal.offset"() : () -> i32
  // CHECK: vm.call @hal.command_buffer.dispatch.indirect(%arg0, %arg1, %zero, %arg2, [[OFFSET]]) : (!iree.ref<!hal.command_buffer>, !iree.ref<!hal.executable>, i32, !iree.ref<!hal.buffer>, i32) -> ()
  hal.command_buffer.dispatch.indirect %arg0, %arg1, entry_point=0, workgroups=%arg2[%offset]
  return
}
