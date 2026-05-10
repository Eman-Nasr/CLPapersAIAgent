# CLPapersAIAgent

# users guidelines 
create venv using Python 3.11 or 3.12

Check installed Python versions first, make sure u have 3.11 or 3.12 :
* py -0p
Mac
* python3.11 --version
or 
* python3.12 --version

create virtual Environment
* py -3.11 -m venv .venv
Mac
* python3.11 -m venv .venv

activate the virtual Environment
* .\.venv\Scripts\Activate.ps1
Mac
* source .venv/bin/activate

your terminal should show:
(.venv)

requirments installation 
* pip install -r requirements.txt

run the full pipeline and find the results in output folder as test/n
* python -m src.run_pipeline

# AutoMl part:

run AutoML tuning on the latest pipeline output
* python -m src.autoML_Optuna

or specify trial count / output folder
* python -m src.autoML_Optuna --output-dir outputs/test1 --n-trials 30
