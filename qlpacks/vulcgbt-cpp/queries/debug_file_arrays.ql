import cpp

from ArrayExpr ae
where ae.getLocation().getFile().getBaseName() = "CWE126_Buffer_Overread__CWE129_connect_socket_21.c"
select ae.toString(), ae.getArrayBase().toString(), ae.getArrayOffset().toString()
