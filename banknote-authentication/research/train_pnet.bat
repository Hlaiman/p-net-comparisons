@if EXIST ..\..\bin\Python37\python.exe SET PythonPath=..\..\bin\Python37
@if NOT EXIST %PythonPath%\python.exe goto exit

%PythonPath%\python train_pnet.py

pause
:exit