.PHONY: data-mlp test-mlp train-mlp

PROJECT_ROOT := $(shell pwd)
PYTHONPATH_CMD := PYTHONPATH=$(PROJECT_ROOT):$(PYTHONPATH)

INPUT_SIZE ?= 63
CSV_PATH ?= /home/kimsaehoon/workspace/JamJamBeat/src/dataset/man1_right_for_poc.csv
LABEL_COL ?= gesture
NUM_CLASSES ?= 7
HIDDEN_DIMS ?= 128,64
BATCH_SIZE ?= 32
NUM_EPOCHS ?= 30
LR ?= 0.001
DROPOUT ?= 0.0
WEIGHT_DECAY ?= 0.0
BATCH_NORM ?= False
TEST_SPLIT_SEED ?= 42
TRAIN_VAL_SPLIT_SEED ?= 42
TRAIN_SEED ?= 42
SCHEDULER_NAME ?= ReduceLROnPlateau
MODEL_PATH ?= /home/kimsaehoon/workspace/JamJamBeat/checkpoints/best_model.pth
VALIDATION_SPLIT_RATIO ?= 0.2
TEST_SPLIT_RATIO ?= 0.1
SAVE_CONFUSION_MATRIX_PATH ?= /home/kimsaehoon/workspace/JamJamBeat/checkpoints/confusion_matrix_newdata.csv
SAVE_PREDICTIONS_PATH ?= /home/kimsaehoon/workspace/JamJamBeat/checkpoints/test_predictions_newdata.csv

data-mlp:
	$(PYTHONPATH_CMD) uv run python src/dataset/gesture_dataset.py

test-mlp:
	$(PYTHONPATH_CMD) uv run python poc/mlp_gesture_test.py

train-mlp:
	$(PYTHONPATH_CMD) uv run python src/training/train.py \
		--csv-path $(CSV_PATH) \
		--label-col $(LABEL_COL) \
		--input-size $(INPUT_SIZE) \
		--num-classes $(NUM_CLASSES) \
		--hidden-dims $(HIDDEN_DIMS) \
		--batch-size $(BATCH_SIZE) \
		--num-epochs $(NUM_EPOCHS) \
		--learning-rate $(LR) \
		--dropout $(DROPOUT) \
		--weight-decay $(WEIGHT_DECAY) \
		--use-batchnorm $(BATCH_NORM) \
		--test-split-seed $(TEST_SPLIT_SEED) \
		--train-val-split-seed $(TRAIN_VAL_SPLIT_SEED) \
		--train-seed $(TRAIN_SEED)\
		--scheduler-name $(SCHEDULER_NAME)

eval-mlp:
	$(PYTHONPATH_CMD) uv run python src/training/evaluation.py \
		--csv-path $(CSV_PATH) \
		--model-path $(MODEL_PATH) \
		--label-col $(LABEL_COL) \
		--input-size $(INPUT_SIZE) \
		--num-classes $(NUM_CLASSES) \
		--hidden-dims $(HIDDEN_DIMS) \
		--batch-size $(BATCH_SIZE) \
		--dropout $(DROPOUT) \
		--use-batchnorm $(BATCH_NORM) \
		--validation-split-ratio $(VALIDATION_SPLIT_RATIO) \
		--test-split-ratio $(TEST_SPLIT_RATIO) \
		--test-split-seed $(TEST_SPLIT_SEED) \
		--train-val-split-seed $(TRAIN_VAL_SPLIT_SEED)