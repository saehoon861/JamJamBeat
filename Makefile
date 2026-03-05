.PHONY: test-mlp train-mlp

PROJECT_ROOT := $(shell pwd)
PYTHONPATH_CMD := PYTHONPATH=$(PROJECT_ROOT):$(PYTHONPATH)

test-mlp:
	$(PYTHONPATH_CMD) uv run python poc/mlp_gesture_test.py

train-mlp:
	$(PYTHONPATH_CMD) uv run python src/training/train.py
