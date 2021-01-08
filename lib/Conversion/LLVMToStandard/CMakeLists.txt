add_circt_library(CIRCTLLVMToStandard
  LLVMToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LLVMToStandard

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRLLVMIR
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  )
