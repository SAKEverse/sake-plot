:: Check if Anaconda is present
IF EXIST %USERPROFILE%\AppData\Local\Continuum\anaconda3\Scripts\activate.bat (
    set conda_file=%USERPROFILE%\AppData\Local\Continuum\anaconda3
)
IF EXIST %USERPROFILE%\anaconda3\Scripts\activate.bat (
    set conda_file=%USERPROFILE%\anaconda3
)

:: Check if Miniconda is present
IF EXIST %USERPROFILE%\miniconda3\Scripts\activate.bat (
    set conda_file=%USERPROFILE%\miniconda3
)

:: enter conda
call %conda_file%\Scripts\activate.bat

:: Navigate sake-plan directory
cd %USERPROFILE%\Documents\GitHub\sake-plot

:: Check if pipenv has been installed
IF NOT EXIST %conda_file%\Lib\site-packages\pipenv\__main__.py (
    pip install pipenv
)

IF NOT EXIST Pipfile.lock (
    pipenv install
)

:: Launch app
pipenv run python sakeplot.py