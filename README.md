# CLPapersAIAgent

# users guidelines 

# Prepare data and baseline member 1 (Eman)
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

# AutoMl member 2 (Khawla):

run AutoML tuning on the latest pipeline output
* python -m src.autoML_Optuna

or specify trial count / output folder
* python -m src.autoML_Optuna --output-dir outputs/test1 --n-trials 30

# Online Learning + Drift Detection member 3 (Maryam)

Files:
- `src/online_learning.py`
- `src/drift_detection.py`

Required Libraries

Windows:

* pip install river matplotlib

Mac:  
* pip3 install river matplotlib

Run the Online Learning Module

Windows:

* python -m src.online_learning

Mac:

* python3 -m src.online_learning

Run ADWIN Drift Detection Demo

Windows

* py -3.11 -m src.drift_detection

Mac

* python3 -m src.drift_detection

Generated Outputs
outputs/test1/online_learning/
├── prequential_metrics.csv
├── prequential_accuracy.png
├── online_summary.json

# evaluation.py  –  Member 4 (Tahani): 
 

* python -m src.evaluation --output-dir outputs/test3/automl  # uses latest outputs/test*

Evaluation Outputs
Standalone script that reads the AutoML results produced by
autoML_Optuna.py and generates three deliverables:

    automl/automl_comparison_chart.png  — bar chart (quality + latency)
    automl/automl_comparison.csv        — flat comparison table
    automl/automl_eval_summary.md       — report-ready markdown summary
    