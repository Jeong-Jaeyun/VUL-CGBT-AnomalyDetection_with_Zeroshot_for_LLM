import cpp

from Function f, VariableAccess va
where
  f.hasName("badSink") and
  va.getEnclosingFunction() = f
select va.toString(), va.getTarget().getName()
