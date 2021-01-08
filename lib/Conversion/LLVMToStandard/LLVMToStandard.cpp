////===- Ops.h - StaticLogic MLIR Operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main LLVM to Standard Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LLVMToStandard/LLVMToStandard.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace std;

namespace {

}

using LLVM::LLVMType;
using LLVM::LLVMIntegerType;
using LLVM::LLVMPointerType;
using LLVM::LLVMStructType;
using LLVM::LLVMVectorType;

static Type getType(LLVMType type) {
  MLIRContext *context = type.getContext();
  return TypeSwitch<LLVMType, Type>(type)
    .Case<LLVMIntegerType>([&](LLVMIntegerType integer) {
        unsigned width = integer.getBitWidth();
        return IntegerType::get(context, width);
      })
    .Case<LLVMPointerType>([&](LLVMPointerType pt) {
        return MemRefType::get(-1,getType(pt.getElementType()));
      })
    .Case<LLVMStructType>([&](LLVMStructType node) {
        assert("StructType" && false);
        return IntegerType::get(context,32);
      })
    .Case<LLVMVectorType>([&](LLVMVectorType vec) {
        assert("VectorType" && false);
        return IntegerType::get(context,32);
      });
}

template <typename OpType>
void buildBinaryOperation(Operation &op, ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(&op);
  auto newOp=rewriter.create<OpType>(op.getLoc(),op.getOperand(0),op.getOperand(1));
  op.getResult(0).replaceAllUsesWith(newOp.getResult());
}

template <typename OpType>
void buildUnaryOperation(Operation &op, ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(&op);
  auto newOp=rewriter.create<OpType>(op.getLoc(),op.getOperand(0),
              getType(op.getResult(0).getType().cast<LLVMType>()));
  op.getResult(0).replaceAllUsesWith(newOp.getResult());
}

static FuncOp createFuncOp(LLVM::LLVMFuncOp funcOp, ConversionPatternRewriter &rewriter) {
  llvm::SmallVector<mlir::Type, 8> argTypes;
  for(auto &arg : funcOp.getArguments()) {
    auto type = arg.getType().cast<LLVMType>();
    argTypes.push_back(getType(type));
  }

  llvm::SmallVector<mlir::Type, 8> resTypes;
  /*
  for(auto &res : funcOp.getType().getResults()) {
    resTypes.push_back(res);
  }*/
  auto resType = funcOp.getType().getFunctionResultType().cast<LLVMType>();
  if(!resType.isa<LLVM::LLVMVoidType>())
    resTypes.push_back(getType(resType));
  
  auto newOp = rewriter.create<FuncOp>(funcOp.getLoc(), funcOp.getName(), rewriter.getFunctionType(argTypes,resTypes));
  rewriter.inlineRegionBefore(funcOp.getBody(), newOp.getBody(), newOp.end());
  
  for(Block &block : newOp) {
    unsigned len = block.getNumArguments();
    for(unsigned i = 0; i < len; i++) {
      auto type = block.getArgument(i).getType().cast<LLVMType>();
      block.getArgument(i).setType(getType(type));
    }
  }
  
  vector<Operation *>oldOps;

  for(Block &block : newOp) {
    for(Operation &op : block) {
      if(isa<LLVM::ReturnOp>(op)) {
        SmallVector<Value, 8> operands(op.getOperands());
        for(unsigned i = 0, e = op.getNumOperands(); i < e; ++i)
          op.eraseOperand(0);
        rewriter.setInsertionPointAfter(&op);
        rewriter.create<ReturnOp>(op.getLoc(),operands);
        oldOps.push_back(&op);
      }
      else if(isa<LLVM::ICmpOp>(op)) {
        rewriter.setInsertionPointAfter(&op);
        auto newCmpIOp=rewriter.create<CmpIOp>(op.getLoc(),static_cast<CmpIPredicate>(op.getAttr("predicate").cast<IntegerAttr>().getValue().getLimitedValue()),op.getOperand(0),op.getOperand(1));
        op.getResult(0).replaceAllUsesWith(newCmpIOp.getResult());
        oldOps.push_back(&op);
      }
      else if(isa<LLVM::CallOp>(op)) {
        //fprintf(stderr,"Call\n");
        //fprintf(stderr,"!!! %d\n",op.getAttrs().size());
        rewriter.setInsertionPointAfter(&op);
        auto newCallOp=rewriter.create<CallOp>(op.getLoc(), op.getAttr("callee").cast<SymbolRefAttr>(),
                  TypeRange(getType(op.getResult(0).getType().cast<LLVMType>())),
                  op.getOperands());
        op.getResult(0).replaceAllUsesWith(newCallOp.getResult(0));
        oldOps.push_back(&op);
      }
      else if(isa<LLVM::AddOp,LLVM::SubOp,LLVM::MulOp,LLVM::AndOp,LLVM::OrOp,LLVM::XOrOp,
                  LLVM::SRemOp,LLVM::URemOp,LLVM::SDivOp,LLVM::UDivOp,LLVM::ShlOp,
                  LLVM::LShrOp,LLVM::AShrOp>(op)) {
        //fprintf(stderr,"IntArithmeticOp\n");
        #define BINARYOP(LLVMTYPE, OPTYPE)                  \
          if(isa<LLVMTYPE>(op))                             \
            buildBinaryOperation<OPTYPE>(op,rewriter)
        BINARYOP(LLVM::AddOp, AddIOp);
        BINARYOP(LLVM::SubOp, SubIOp);
        BINARYOP(LLVM::MulOp, AddIOp);
        BINARYOP(LLVM::AndOp, AndOp);
        BINARYOP(LLVM::OrOp, OrOp);
        BINARYOP(LLVM::XOrOp, XOrOp);
        BINARYOP(LLVM::SRemOp, SignedRemIOp);
        BINARYOP(LLVM::URemOp, UnsignedRemIOp);
        BINARYOP(LLVM::SDivOp, SignedDivIOp);
        BINARYOP(LLVM::UDivOp, UnsignedDivIOp);
        BINARYOP(LLVM::ShlOp, ShiftLeftOp);
        BINARYOP(LLVM::LShrOp, UnsignedShiftRightOp);
        BINARYOP(LLVM::AShrOp, SignedShiftRightOp);
        #undef BINARYOP
        
        oldOps.push_back(&op);
      }
      else if(isa<LLVM::ZExtOp, LLVM::SExtOp, LLVM::TruncOp>(op)) {
        #define UNARYOP(LLVMTYPE, OPTYPE)                   \
          if(isa<LLVMTYPE>(op))                             \
            buildUnaryOperation<OPTYPE>(op,rewriter)
        UNARYOP(LLVM::SExtOp,SignExtendIOp);
        UNARYOP(LLVM::ZExtOp,ZeroExtendIOp);
        UNARYOP(LLVM::TruncOp,TruncateIOp);
        #undef UNARYOP
        
        oldOps.push_back(&op);
      }
      else if(isa<LLVM::ConstantOp>(op)) {
        rewriter.setInsertionPointAfter(&op);
        auto newConstantOp=rewriter.create<ConstantOp>(op.getLoc(), op.getAttr("value"));
        op.getResult(0).replaceAllUsesWith(newConstantOp.getResult());
        oldOps.push_back(&op);
      }
      else if(isa<LLVM::BrOp>(op)) {
        rewriter.setInsertionPointAfter(&op);
        rewriter.create<BranchOp>(op.getLoc(),op.getSuccessor(0),op.getOperands());
        oldOps.push_back(&op);
      }
      else if(isa<LLVM::CondBrOp>(op)) {
        unsigned true_num = op.getAttr("operand_segment_sizes").cast<DenseElementsAttr>().getValue(1).cast<IntegerAttr>().getValue().getLimitedValue();
        unsigned false_num = op.getAttr("operand_segment_sizes").cast<DenseElementsAttr>().getValue(2).cast<IntegerAttr>().getValue().getLimitedValue();
        Value condition = op.getOperand(0);
        op.eraseOperand(0);
        SmallVector<Value, 8> trueOperands;
        SmallVector<Value, 8> falseOperands;
        
        for(unsigned i = 0, e = true_num; i < e; ++i) {
          trueOperands.push_back(op.getOperand(0));
          op.eraseOperand(0);
        }
        
        for(unsigned i = 0, e = false_num; i < e; ++i) {
          falseOperands.push_back(op.getOperand(0));
          op.eraseOperand(0);
        }

        rewriter.setInsertionPointAfter(&op);
        rewriter.create<CondBranchOp>(op.getLoc(),condition,op.getSuccessor(0),trueOperands,op.getSuccessor(1),falseOperands);
        oldOps.push_back(&op);
      }
      else if(isa<LLVM::SelectOp>(op)) {
        rewriter.setInsertionPointAfter(&op);
        auto newSelOp=rewriter.create<SelectOp>(op.getLoc(),op.getOperand(0),op.getOperand(1),op.getOperand(2));
        op.getResult(0).replaceAllUsesWith(newSelOp.getResult());
        oldOps.push_back(&op);
      }
      else if(isa<LLVM::GEPOp>(op)) {
        /*for(auto &attr : op.getAttrs()){
          llvm::raw_ostream &output = llvm::errs();
          attr.first.print(output);
          fprintf(stderr,",");
          attr.second.print(output);
          fprintf(stderr,"\n");
        }*/
        oldOps.push_back(&op);
      }
      else if(isa<LLVM::StoreOp>(op)) {
        if(op.getOperand(1).isa<BlockArgument>()) {
          rewriter.setInsertionPointAfter(&op);
          auto indexOp = rewriter.create<ConstantOp>(op.getLoc(),IntegerAttr::get(IndexType::get(op.getContext()),0));
          rewriter.create<StoreOp>(op.getLoc(),op.getOperand(0),op.getOperand(1),indexOp.getResult());
          oldOps.push_back(&op);
        }
        else {
          rewriter.setInsertionPointAfter(&op);
          auto indexOp = rewriter.create<IndexCastOp>(op.getLoc(),IndexType::get(op.getContext()),op.getOperand(1).getDefiningOp()->getOperand(1));
          rewriter.create<StoreOp>(op.getLoc(),op.getOperand(0),op.getOperand(1).getDefiningOp()->getOperand(0),indexOp.getResult());
          oldOps.push_back(&op);
        }
      }
      else if(isa<LLVM::LoadOp>(op)) {
        if(op.getOperand(0).isa<BlockArgument>()) {
          rewriter.setInsertionPointAfter(&op);
          auto indexOp = rewriter.create<ConstantOp>(op.getLoc(),IntegerAttr::get(IndexType::get(op.getContext()),0));
          auto newLoadOp=rewriter.create<LoadOp>(op.getLoc(),op.getOperand(0),indexOp.getResult());
          op.getResult(0).replaceAllUsesWith(newLoadOp.getResult());
          oldOps.push_back(&op);
        }
        else {
          rewriter.setInsertionPointAfter(&op);
          auto indexOp = rewriter.create<IndexCastOp>(op.getLoc(),IndexType::get(op.getContext()),op.getOperand(0).getDefiningOp()->getOperand(1));
          auto newLoadOp=rewriter.create<LoadOp>(op.getLoc(),op.getOperand(0).getDefiningOp()->getOperand(0),indexOp.getResult());
          op.getResult(0).replaceAllUsesWith(newLoadOp.getResult());
          oldOps.push_back(&op);
        }
      }
      else if(op.getDialect()->getNamespace() == "llvm"){
        fprintf(stderr,"Invalid operation: %s\n",op.getName().getStringRef().str().c_str());
      }
    }
  }
  for(auto op : oldOps)
    rewriter.eraseOp(op);
  /*fprintf(stderr,"New %s\n",newOp.getName().str().c_str());
  for(Block &block : newOp) {
    for(Operation &op : block) {
      fprintf(stderr,"New %s\n",op.getName().getStringRef().str().c_str());
    }
  }*/
  
  return newOp;
}
struct LLVMFuncConversion : public OpConversionPattern<LLVM::LLVMFuncOp> {
  using OpConversionPattern<LLVM::LLVMFuncOp>::OpConversionPattern;

  LogicalResult match(Operation *op) const override {
    //fprintf(stderr,"%s\n",op->getName().getStringRef().str().c_str());
    return success();
  }
  void rewrite(LLVM::LLVMFuncOp funcOp, ArrayRef<Value> operands,
                ConversionPatternRewriter &rewriter) const override {
    auto newOp = createFuncOp(funcOp, rewriter);
    rewriter.eraseOp(funcOp);
  }
};

namespace {

struct LLVMToStandardPass
    : public PassWrapper<LLVMToStandardPass, OperationPass<LLVM::LLVMFuncOp>> {
  void runOnOperation() override {
    auto op = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<FuncOp>();
    target.addIllegalOp<LLVM::LLVMFuncOp>();
    //target.addIllegalDialect<LLVM::LLVMDialect>();

    OwningRewritePatternList patterns;
    patterns.insert<LLVMFuncConversion>(op.getContext());

    if(failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void LLVM::registerLLVMToStandardPasses() {
  PassRegistration<LLVMToStandardPass>(
      "convert-llvm-to-std", "Convert LLVM IR Dialect to Standard Dialect.");

}
