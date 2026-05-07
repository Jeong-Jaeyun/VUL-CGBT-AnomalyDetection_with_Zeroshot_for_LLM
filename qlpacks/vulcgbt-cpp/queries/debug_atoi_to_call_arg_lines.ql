/**
 * @kind problem
 * @id vulcgbt/debug-atoi-to-call-arg-lines
 * @problem.severity warning
 */
import cpp
import semmle.code.cpp.dataflow.new.TaintTracking

module DebugAtoiToCallArgLinesConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node node) {
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("atoi") and
      node.asExpr() = fc
    )
  }

  predicate isSink(DataFlow::Node node) {
    exists(FunctionCall fc |
      fc.getTarget() instanceof Function and
      node.asExpr() = fc.getArgument(0)
    )
  }
}

module DebugAtoiToCallArgLinesFlow = TaintTracking::Global<DebugAtoiToCallArgLinesConfig>;

from FunctionCall fc, DataFlow::Node source, DataFlow::Node sink
where
  sink.asExpr() = fc.getArgument(0) and
  DebugAtoiToCallArgLinesFlow::flow(source, sink)
select
  fc.getLocation().getFile().getRelativePath(),
  fc.getTarget().getName(),
  fc.getLocation().getStartLine(),
  "Call argument influenced by atoi return value."
