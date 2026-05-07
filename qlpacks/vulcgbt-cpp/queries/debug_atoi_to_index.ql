/**
 * @kind problem
 * @id vulcgbt/debug-atoi-to-index
 * @problem.severity warning
 */
import cpp
import semmle.code.cpp.dataflow.new.TaintTracking

module DebugAtoiToIndexConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node node) {
    node.asExpr().(FunctionCall).getTarget().hasGlobalName("atoi")
  }

  predicate isSink(DataFlow::Node node) {
    exists(ArrayExpr ae | node.asExpr() = ae.getArrayOffset())
  }
}

module DebugAtoiToIndexFlow = TaintTracking::Global<DebugAtoiToIndexConfig>;

from DataFlow::Node source, DataFlow::Node sink
where DebugAtoiToIndexFlow::flow(source, sink)
select sink, "Array index influenced by atoi return value."
