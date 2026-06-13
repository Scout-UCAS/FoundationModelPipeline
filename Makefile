.PHONY: validate test test-core test-arch report dashboard data-run train-run eval-run deploy-check production-plan production-check production-run clean

PYTHON ?= python
PYTHONPATH ?= src

validate:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fmops.cli schema-validate
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fmops.cli validate

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests

test-core:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p "test_platform.py"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p "test_framework_core.py"

test-arch:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -p "test_architecture_impl.py"

report:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fmops.cli report --output reports/foundation_model_plan.md

dashboard:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fmops.cli dashboard --output reports/dashboard.html

data-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fmops.cli data-run --output artifacts/runs/data_pipeline_plan.json

train-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fmops.cli train-run --output artifacts/runs/training_plan.json

eval-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fmops.cli eval-run --output artifacts/runs/evaluation_report.json

deploy-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fmops.cli deploy-check --output artifacts/runs/deployment_report.json

production-plan:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fmops.cli production-plan --output artifacts/production/production_plan.json

production-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fmops.cli production-check --output artifacts/production/preflight_report.json

production-run:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m fmops.cli production-run --output artifacts/production/execution_report.json

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
