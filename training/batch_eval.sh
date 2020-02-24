#!/usr/bin/env bash
# $1 - model

DATASETS=( "msd" "mtat" "jamendo" )
declare -A LOCS
LOCS["jamendo"]="/mnt/data/jamendo-dataset/audio-npy/npy/"
LOCS["msd"]="/mnt/data/msd/npy/"
LOCS["mtat"]="/mnt/data/"

trap "exit" INT
for TEST in "${DATASETS[@]}"; do
  for TRAIN in "${DATASETS[@]}"; do
    DIR="../results/test-$TEST-train-$TRAIN-$1"
    if [ -f "$DIR/est.csv" ]; then
      echo "Skipping $DIR - already computed"
    else
      echo "Computing $DIR:"
      mkdir -p "$DIR"
      python eval.py --dataset "$TEST" --train-dataset "$TRAIN" --model_type "$1" \
        --model_load_path ../models/"$TRAIN"/"$1"/best_model.pth --data_path "${LOCS[$TEST]}" --output "$DIR"
    fi
  done
done
