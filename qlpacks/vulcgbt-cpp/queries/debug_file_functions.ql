import cpp

from Function f
where f.getLocation().getFile().getBaseName() = "CWE126_Buffer_Overread__CWE129_connect_socket_21.c"
select f.getQualifiedName(), f.getName(), f.getLocation().getFile().getBaseName()
