export PYTHONPATH := src:.
export MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
export MLFLOW_DISABLE_AGENT_HINT := 1
N_TRIALS ?= 30

.PHONY: install data train tune register pipeline ui serve test lint docker clean

install:        ## Install all dependencies
	pip install -r requirements-dev.txt

data:           ## Download and split the dataset
	python -m churn.data

train:          ## Train one model (MODEL=logreg|random_forest|hist_gb)
	python -m churn.train --model-type $(or $(MODEL),hist_gb)

tune:           ## Optuna tuning with nested MLflow runs
	python -m churn.tune --n-trials $(N_TRIALS)

register:       ## Promote the best run to the 'champion' alias
	python -m churn.register

pipeline:       ## Everything: data -> baselines -> tune -> register
	python -m churn.pipeline --n-trials $(N_TRIALS)

ui:             ## MLflow UI at http://localhost:5000
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000

serve:          ## Prediction API at http://localhost:8000/docs
	uvicorn api.app:app --reload --port 8000

test:           ## Run unit and API tests
	pytest -q

lint:
	ruff check src api tests

docker:         ## Full stack in Docker
	docker compose up --build

clean:
	rm -rf mlruns mlflow.db data/processed .pytest_cache
