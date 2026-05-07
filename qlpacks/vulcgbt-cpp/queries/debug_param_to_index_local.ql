/**
 * @kind problem
 * @id vulcgbt/debug-param-to-index-local
 * @problem.severity warning
 */
import cpp
import semmle.code.cpp.dataflow.new.DataFlow

from Function f, Parameter p, ArrayExpr ae, DataFlow::Node source, DataFlow::Node sink
where
  f.hasName("badSink") and
  p = f.getParameter(0) and
  source.asParameter() = p and
  sink.asExpr() = ae.getArrayOffset() and
  DataFlow::localFlow(source, sink)
select ae, "Local flow from parameter to array index."
