#!/bin/sh
INPUT_PATH="/data/"
OUT_PATH="/result/"
LIVE_PATH="/live/"
TRAINING_PATH="/data/training.json"
LOG_FILE="out/log/"
STREAM=TRUE
stream=args.stream
UPDATE_MODE="equal"

python evaluate.py  --input_path=$INPUT_PATH   --output_path=$OUT_PATH  --live_data_path=$LIVE_PATH  --training_path=$TRAINING_PATH --stream=$STREAM --log_file=$LOG_FILE --update=$UPDATE_MODE