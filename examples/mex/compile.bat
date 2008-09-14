cl /c /EHsc mexlib.cpp
cl /c /EHsc matrix.cpp
lib mexlib.obj matrix.obj

cl /c /EHcs /I. derivate.cpp
cl /c /EHcs /I. lineintc.cpp
link /DLL /EXPORT:mexFunction lineintc.obj mexlib.lib
link /DLL /EXPORT:mexFunction derivate.obj mexlib.lib
