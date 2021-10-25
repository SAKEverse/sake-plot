call %USERPROFILE%\miniconda3\Scripts\activate.bat
call %USERPROFILE%\AppData\Local\Continuum\anaconda3\Scripts\activate.bat
cd %USERPROFILE%\Documents\GitHub\sake-plot
pip install pipenv
IF EXIST Pipfile.lock (
    echo "Environment already installed"
) ELSE (
    pipenv install
)
pipenv run python sakeplot.py
SLEEP 100