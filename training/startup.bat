REM verify redis is running already (user1@MSI:/$ sudo redis-server /etc/redis/redis.conf)
call ..\venv\Scripts\activate.bat
copy /b/v/y "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings_bots.ini" "C:\Users\kchin\Documents\My Games\Rocket League\TAGame\Config\TASystemSettings.ini"
cd ..
start python -m training.learner
TIMEOUT 10
REM start python worker.py STREAMER
:loop
REM FOR /L %%G IN (1,1,5) DO (start python -m training.worker & TIMEOUT 60)
start python -m training.worker 1
pause
start python -m training.worker 1
TIMEOUT 60
start python -m training.worker 1
TIMEOUT 60
start python -m training.worker 1
TIMEOUT 60
start python -m training.worker 1
TIMEOUT 60
start python -m training.worker 1
TIMEOUT 180
nircmd win min process "RocketLeague.exe"
REM TIMEOUT 10800
REM FOR /F "usebackq tokens=2" %%i IN (`tasklist /v ^| findstr /c:"RLearnWorkerABAD"`) DO taskkill /pid %%i
REM FOR /F "usebackq tokens=2" %%j IN (`tasklist ^| findstr /c:"RocketLeague.exe"`) DO taskkill /pid %%j
REM TIMEOUT 60
REM GOTO loop