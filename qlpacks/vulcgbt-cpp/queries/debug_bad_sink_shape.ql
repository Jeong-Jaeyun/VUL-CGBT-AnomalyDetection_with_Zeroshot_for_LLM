import cpp

from Function f, ArrayExpr ae
where
  f.hasName("badSink") and
  ae.getEnclosingFunction() = f
select
  f.getQualifiedName(),
  ae.toString(),
  ae.getArrayBase().toString(),
  ae.getArrayOffset().toString(),
  ae.getLocation().getFile().getBaseName()
