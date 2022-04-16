REM verify redis is running already (user1@MSI:/$ sudo redis-server /etc/redis/redis.conf)
call ..\..\venv\Scripts\activate.bat
start python learner.py
TIMEOUT 10
REM start python worker.py STREAMER
:loop
FOR /L %%G IN (1,1,6) DO (start python worker.py & TIMEOUT 60)
TIMEOUT 30
nircmd win min process "RocketLeague.exe"
TIMEOUT 10800
FOR /F "usebackq tokens=2" %%i IN (`tasklist /v ^| findstr /c:"RLearnWorkerABAD"`) DO taskkill /pid %%i
FOR /F "usebackq tokens=2" %%j IN (`tasklist ^| findstr /c:"RocketLeague.exe"`) DO taskkill /pid %%j
TIMEOUT 60
GOTO loop