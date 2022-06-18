@if EXIST ..\..\bin\Python37\python.exe SET PythonPath=..\..\bin\Python37
@if NOT EXIST %PythonPath%\python.exe goto exit

%PythonPath%\python generate_csv.py

pause
:exit