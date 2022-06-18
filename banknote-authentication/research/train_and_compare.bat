@if EXIST ..\..\bin\Python37\python.exe SET PythonPath=..\..\bin\Python37
@if NOT EXIST %PythonPath%\python.exe goto exit

%PythonPath%\python generate_csv.py
%PythonPath%\python train_keras.py
%PythonPath%\python train_pnet.py
%PythonPath%\python compare.py

pause
:exit