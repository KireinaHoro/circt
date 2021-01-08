//===- LLVMToStandard.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will convert the LLVM dialect to
// Standard dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_LLVMTOSTANDARD_H_
#define CIRCT_CONVERSION_LLVMTOSTANDARD_H_

namespace mlir {
namespace LLVM {
void registerLLVMToStandardPasses();
} // namespace llvm 
} // namespace circt

#endif // CIRCT_CONVERSION_LLVMTOSTANDARD_H_
