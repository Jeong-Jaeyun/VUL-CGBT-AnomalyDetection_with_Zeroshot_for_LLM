/**
 * @id vulcgbt/cpp/untrusted-flow
 * @name Untrusted data reaches a sensitive sink
 * @description Tracks simple untrusted input flow to vulnerability-relevant sinks for Vul-CGBT research.
 * @kind path-problem
 * @problem.severity warning
 * @precision medium
 * @tags security
 *       external/cwe/cwe-119
 *       external/cwe/cwe-78
 */

import cpp
import semmle.code.cpp.dataflow.new.DataFlow
import semmle.code.cpp.dataflow.new.TaintTracking

predicate stmtContainsExpr(Stmt stmt, Expr expr) {
  stmt.getLocation().getStartLine() <= expr.getLocation().getStartLine() and
  expr.getLocation().getEndLine() <= stmt.getLocation().getEndLine()
}

predicate hasUpperBoundGuard(ArrayExpr ae, Parameter p) {
  exists(IfStmt ifs |
    ifs.getEnclosingFunction() = ae.getEnclosingFunction() and
    (
      stmtContainsExpr(ifs.getThen(), ae) or
      exists(Stmt elseBranch | elseBranch = ifs.getElse() | stmtContainsExpr(elseBranch, ae))
    ) and
    (
      ifs.getCondition().toString().regexpMatch("(?s).*\\b" + p.getName() + "\\b\\s*<.*") or
      ifs.getCondition().toString().regexpMatch("(?s).*\\b" + p.getName() + "\\b\\s*<=.*") or
      ifs.getCondition().toString().regexpMatch("(?s).*<\\s*\\(*\\b" + p.getName() + "\\b.*") or
      ifs.getCondition().toString().regexpMatch("(?s).*<=\\s*\\(*\\b" + p.getName() + "\\b.*")
    )
  )
}

predicate isUnsafeArrayIndexExpr(Expr expr) {
  exists(ArrayExpr ae, Function callee, Parameter p, DataFlow::Node source, DataFlow::Node sink |
    ae.getEnclosingFunction() = callee and
    p = callee.getParameter(0) and
    source.asParameter() = p and
    sink.asExpr() = ae.getArrayOffset() and
    DataFlow::localFlow(source, sink) and
    not hasUpperBoundGuard(ae, p) and
    expr = ae.getArrayOffset()
  )
}

predicate isUnsafeSinkCallArgument(Expr expr) {
  exists(FunctionCall fc, Function callee, Parameter p, ArrayExpr ae, DataFlow::Node source, DataFlow::Node sink |
    fc.getTarget() = callee and
    p = callee.getParameter(0) and
    ae.getEnclosingFunction() = callee and
    source.asParameter() = p and
    sink.asExpr() = ae.getArrayOffset() and
    DataFlow::localFlow(source, sink) and
    not hasUpperBoundGuard(ae, p) and
    expr = fc.getArgument(0)
  )
}

predicate helperHasRelevantSinkShape(Function callee) {
  exists(ArrayExpr ae | ae.getEnclosingFunction() = callee)
  or
  exists(PointerDereferenceExpr pde | pde.getEnclosingFunction() = callee)
  or
  exists(FunctionCall inner |
    inner.getEnclosingFunction() = callee and
    (
      inner.getTarget().hasGlobalName("strcpy") or
      inner.getTarget().hasGlobalName("strcat") or
      inner.getTarget().hasGlobalName("sprintf") or
      inner.getTarget().hasGlobalName("vsprintf") or
      inner.getTarget().hasGlobalName("memcpy") or
      inner.getTarget().hasGlobalName("memmove") or
      inner.getTarget().hasGlobalName("system") or
      inner.getTarget().hasGlobalName("popen") or
      inner.getTarget().hasGlobalName("execve") or
      inner.getTarget().hasGlobalName("execvp")
    )
  )
}

predicate isCustomReaderCall(FunctionCall fc) {
  exists(Function callee |
    fc.getTarget() = callee and
    (
      callee.getName().regexpMatch("get_bits(1|8|16|32)?") or
      callee.getName().regexpMatch("bytestream2_get_(byte|le16|le24|le32|le64|be16|be24|be32|be64)") or
      callee.getName().regexpMatch("avio_r[lb](8|16|24|32|64)") or
      callee.getName().regexpMatch("i_stream_next_line") or
      callee.getName().regexpMatch("inet_(addr|aton|pton)")
    )
  )
}

predicate isCustomOutParamReaderCall(FunctionCall fc) {
  exists(Function callee |
    fc.getTarget() = callee and
    (
      callee.getName().regexpMatch("xenstore_read_.*") or
      callee.getName().regexpMatch("php_stream_read") or
      callee.getName().regexpMatch("vlc_b64_decode_binary") or
      callee.getName().regexpMatch("copy_from_user") or
      callee.getName().regexpMatch("strscpy_from_user")
    )
  )
}

predicate isReturnPropagatingHelperCall(FunctionCall fc) {
  exists(Function callee |
    fc.getTarget() = callee and
    (
      callee.getName().regexpMatch("str(str|chr|rchr|pbrk|tok|tok_r)") or
      callee.getName().regexpMatch("mem(chr|rchr)") or
      callee.getName().regexpMatch("strlen|strnlen") or
      callee.getName().regexpMatch("g_strdup_printf|g_strconcat|g_build_filename")
    )
  )
}

predicate isAssignmentWriteSink(Expr expr) {
  exists(AssignExpr assign, FieldAccess fa |
    expr = fa and
    assign.getLValue() = fa
  )
  or
  exists(AssignExpr assign, ArrayExpr ae |
    expr = ae and
    assign.getLValue() = ae
  )
  or
  exists(AssignExpr assign, PointerDereferenceExpr pde |
    expr = pde and
    assign.getLValue() = pde
  )
}

predicate isMetadataTargetFunction(Function f) {
  none()
}

predicate isMetadataTargetParameterSource(DataFlow::Node source) {
  exists(Parameter p |
    source.asParameter() = p and
    isMetadataTargetFunction(p.getFunction())
  )
}

module VulCGBTFlowConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    isMetadataTargetParameterSource(source)
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("getenv") and
      source.asIndirectExpr(1) = fc
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("atoi") and
      source.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("strtol") and
      source.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("strtoll") and
      source.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("strtoul") and
      source.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("strtoull") and
      source.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("atol") and
      source.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("atoll") and
      source.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("fgets") and
      source.asIndirectExpr(1) = fc.getArgument(0)
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("scanf") and
      source.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("fscanf") and
      source.asIndirectExpr(1) = fc.getArgument(2)
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("sscanf") and
      source.asIndirectExpr(1) = fc.getArgument(2)
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("fread") and
      source.asIndirectExpr(1) = fc.getArgument(0)
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("recv") and
      source.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("recvfrom") and
      source.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      fc.getTarget().hasGlobalName("read") and
      source.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      isCustomReaderCall(fc) and
      source.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      isMetadataTargetFunction(fc.getEnclosingFunction()) and
      isCustomOutParamReaderCall(fc) and
      source.asIndirectExpr(1) = fc.getAnArgument()
    )
    or
    exists(Parameter p |
      p.hasName("argv") and
      p.getFunction().hasName("main") and
      isMetadataTargetFunction(p.getFunction()) and
      source.asParameter(1) = p
    )
  }

  predicate isSink(DataFlow::Node sink) {
    exists(ArrayExpr ae |
      sink.asExpr() = ae.getArrayOffset()
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("strcpy") and
      sink.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("strcat") and
      sink.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("sprintf") and
      sink.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("vsprintf") and
      sink.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("memcpy") and
      sink.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("memcpy") and
      sink.asExpr() = fc.getArgument(2)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("memmove") and
      sink.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("memmove") and
      sink.asExpr() = fc.getArgument(2)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("strncpy") and
      sink.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("strncpy") and
      sink.asExpr() = fc.getArgument(2)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("strncat") and
      sink.asIndirectExpr(1) = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("strncat") and
      sink.asExpr() = fc.getArgument(2)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("snprintf") and
      sink.asExpr() = fc.getArgument(1)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("system") and
      sink.asIndirectExpr(1) = fc.getArgument(0)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("popen") and
      sink.asIndirectExpr(1) = fc.getArgument(0)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("execve") and
      sink.asIndirectExpr(1) = fc.getArgument(0)
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("execvp") and
      sink.asIndirectExpr(1) = fc.getArgument(0)
    )
    or
    isAssignmentWriteSink(sink.asExpr())
    or
    isUnsafeArrayIndexExpr(sink.asExpr())
    or
    exists(PointerDereferenceExpr pde |
      sink.asExpr() = pde
    )
    or
    exists(FunctionCall fc, Function callee |
      fc.getTarget() = callee and
      callee.getName().regexpMatch(".*Sink$") and
      sink.asExpr() = fc.getArgument(0)
    )
    or
    exists(FunctionCall fc |
      fc.toString().regexpMatch("(?s).*\\b[A-Za-z_][A-Za-z0-9_]*Sink\\s*\\(.*") and
      sink.asExpr() = fc.getArgument(0)
    )
    or
    exists(FunctionCall fc, Function callee |
      fc.getTarget() = callee and
      helperHasRelevantSinkShape(callee) and
      sink.asExpr() = fc.getArgument(0)
    )
    or
    isUnsafeSinkCallArgument(sink.asExpr())
  }

  predicate isAdditionalFlowStep(DataFlow::Node pred, DataFlow::Node succ) {
    exists(AssignExpr assign |
      pred.asExpr() = assign.getRValue() and
      succ.asExpr() = assign.getLValue()
    )
    or
    exists(Operation op |
      succ.asExpr() = op and
      pred.asExpr() = op.getAnOperand()
    )
    or
    exists(FieldAccess fa |
      succ.asExpr() = fa and
      pred.asExpr() = fa.getQualifier()
    )
    or
    exists(ArrayExpr ae |
      succ.asExpr() = ae and
      (
        pred.asExpr() = ae.getArrayOffset() or
        pred.asExpr() = ae.getArrayBase()
      )
    )
    or
    exists(PointerDereferenceExpr pde |
      succ.asExpr() = pde and
      pred.asExpr() = pde.getOperand()
    )
    or
    exists(VariableAccess predVa, VariableAccess succVa |
      pred.asExpr() = predVa and
      succ.asExpr() = succVa and
      predVa.getTarget() = succVa.getTarget() and
      predVa != succVa and
      (
        predVa.getEnclosingFunction() != succVa.getEnclosingFunction()
        or
        predVa.getLocation().getStartLine() <= succVa.getLocation().getStartLine()
      )
    )
    or
    exists(FunctionCall fc |
      isReturnPropagatingHelperCall(fc) and
      (
        pred.asExpr() = fc.getArgument(0) or
        pred.asIndirectExpr(1) = fc.getArgument(0)
      ) and
      succ.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("atoi") and
      pred.asIndirectExpr(1) = fc.getArgument(0) and
      succ.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("strtol") and
      pred.asIndirectExpr(1) = fc.getArgument(0) and
      succ.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("strtoll") and
      pred.asIndirectExpr(1) = fc.getArgument(0) and
      succ.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("strtoul") and
      pred.asIndirectExpr(1) = fc.getArgument(0) and
      succ.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("strtoull") and
      pred.asIndirectExpr(1) = fc.getArgument(0) and
      succ.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("atol") and
      pred.asIndirectExpr(1) = fc.getArgument(0) and
      succ.asExpr() = fc
    )
    or
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("atoll") and
      pred.asIndirectExpr(1) = fc.getArgument(0) and
      succ.asExpr() = fc
    )
  }
}

module VulCGBTFlow = TaintTracking::Global<VulCGBTFlowConfig>;
import VulCGBTFlow::PathGraph

from VulCGBTFlow::PathNode source, VulCGBTFlow::PathNode sink
where VulCGBTFlow::flowPath(source, sink)
select sink.getNode(), source, sink, "Untrusted data reaches a sensitive sink."
