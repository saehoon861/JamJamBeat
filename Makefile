.PHONY: test-mlp train-mlp

PROJECT_ROOT := $(shell pwd)
PYTHONPATH_CMD := PYTHONPATH=$(PROJECT_ROOT):$(PYTHONPATH)

INPUT_SIZE ?= 63
NUM_CLASSES ?= 7
HIDDEN_DIMS ?= 128,64
BATCH_SIZE ?= 32
NUM_EPOCHS ?= 30
LR ?= 0.001
DROPOUT ?= 0.0

test-mlp:
	$(PYTHONPATH_CMD) uv run python poc/mlp_gesture_test.py

train-mlp:
	$(PYTHONPATH_CMD) uv run python src/training/train.py \
		--input-size $(INPUT_SIZE) \
		--num-classes $(NUM_CLASSES) \
		--hidden-dims $(HIDDEN_DIMS) \
		--batch-size $(BATCH_SIZE) \
		--num-epochs $(NUM_EPOCHS) \
		--learning-rate $(LR) \
		--dropout $(DROPOUT)