/**
 * @kind table
 */
import cpp
import semmle.code.cpp.dataflow.new.TaintTracking

from FunctionCall src, FunctionCall dst, DataFlow::Node source, DataFlow::Node sink
where
  src.getLocation().getFile().getRelativePath() = "raw_records/000000/CWE126_Buffer_Overread__CWE129_connect_socket_21.c" and
  src.getEnclosingFunction().getName() = "CWE126_Buffer_Overread__CWE129_connect_socket_21_bad" and
  src.getTarget().hasGlobalName("atoi") and
  dst.getLocation().getFile().getRelativePath() = "raw_records/000000/CWE126_Buffer_Overread__CWE129_connect_socket_21.c" and
  dst.getEnclosingFunction().getName() = "CWE126_Buffer_Overread__CWE129_connect_socket_21_bad" and
  dst.getTarget().getName() = "badSink" and
  source.asExpr() = src and
  sink.asExpr() = dst.getArgument(0) and
  TaintTracking::localTaint(source, sink)
select
  src.getLocation().getStartLine(),
  dst.getLocation().getStartLine(),
  "Local taint exists from atoi result to badSink call argument."
