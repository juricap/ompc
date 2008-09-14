cl /c /EHsc mexlib.cpp
cl /c /EHsc matrix.cpp
lib mexlib.obj matrix.obj
cl /c /EHcs lineintc.cpp
link /DLL /EXPORT:mexFunction lineintc.obj mexlib.lib
