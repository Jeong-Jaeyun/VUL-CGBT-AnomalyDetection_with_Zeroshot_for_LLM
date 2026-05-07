import cpp

from Function f
select
  f.getLocation().getFile().getBaseName(),
  f.getLocation().getFile().getRelativePath(),
  f.getQualifiedName(),
  f.getName(),
  f.getLocation().getStartLine(),
  f.getLocation().getEndLine()
