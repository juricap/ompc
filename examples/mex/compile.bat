cl /c /EHsc mexlib.cpp
cl /c /EHsc matrix.cpp
lib mexlib.obj matrix.obj
cl /c /EHcs /I. derivate.cpp
link /DLL /EXPORT:mexFunction derivate.obj mexlib.lib
