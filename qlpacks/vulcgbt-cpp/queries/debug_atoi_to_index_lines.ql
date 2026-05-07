/**
 * @kind problem
 * @id vulcgbt/debug-atoi-to-index-lines
 * @problem.severity warning
 */
import cpp
import semmle.code.cpp.dataflow.new.TaintTracking

module DebugAtoiToIndexLinesConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node node) {
    exists(FunctionCall fc |
      fc.getTarget().hasGlobalName("atoi") and
      node.asExpr() = fc
    )
  }

  predicate isSink(DataFlow::Node node) {
    exists(ArrayExpr ae | node.asExpr() = ae.getArrayOffset())
  }
}

module DebugAtoiToIndexLinesFlow = TaintTracking::Global<DebugAtoiToIndexLinesConfig>;

from ArrayExpr ae, DataFlow::Node source, DataFlow::Node sink
where
  sink.asExpr() = ae.getArrayOffset() and
  DebugAtoiToIndexLinesFlow::flow(source, sink)
select
  ae.getLocation().getFile().getRelativePath(),
  ae.getEnclosingFunction().getName(),
  ae.getLocation().getStartLine(),
  "Array index influenced by atoi return value."
