import cpp

from Function f, Expr e
where
  f.hasName("badSink") and
  e.getEnclosingFunction() = f
select e.toString(), e.getLocation().getStartLine().toString()
