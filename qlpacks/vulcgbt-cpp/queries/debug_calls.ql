/**
 * @kind table
 */
import cpp

from FunctionCall fc
select fc, fc.getTarget().getName(), fc.toString()
