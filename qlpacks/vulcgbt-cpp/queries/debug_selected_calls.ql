/**
 * @kind table
 */
import cpp

from FunctionCall fc
where
  fc.getLocation().getFile().getRelativePath() = "raw_records/000000/CWE126_Buffer_Overread__CWE129_connect_socket_21.c" and
  fc.getTarget().getName().regexpMatch("atoi|badSink|goodB2G1Sink|goodB2G2Sink")
select
  fc.getLocation().getFile().getRelativePath(),
  fc.getEnclosingFunction().getName(),
  fc.getTarget().getName(),
  fc.getLocation().getStartLine(),
  fc.toString()
