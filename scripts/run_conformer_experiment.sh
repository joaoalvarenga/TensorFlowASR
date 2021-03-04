#!/bin/bash
for i in "$@"
do
case $i in
    --name=*)
    EXPERIMENT_NAME="${i#*=}"
    ;;
    --device=*)
    DEVICE="${i#*=}"
    ;;
    --cetuc_split=*)
    CETUC_SPLIT="${i#*=}"
    CETUC_SPLIT_ARGS="--initial_train_file=../datasets/train_cetuc_split_$CETUC_SPLIT.tsv --initial_test_file=../datasets/test_cetuc_split_$CETUC_SPLIT.tsv"
    ;;
    --sid)
    SID_ARGS="--sid_path ../datasets/sid"
    ;;
    --voxforge)
    VOXFORGE_ARGS="--voxforge_path ../datasets/voxforge"
    ;;
    --lapsbm_val)
    LAPSBM_VAL_ARGS="--lapsbm_val_path /home/joao/mestrado/datasets/LapsBM-val"
    ;;
    *)
            # unknown option
    ;;
esac
done

if [ -z "$EXPERIMENT_NAME" ]
then
  echo "Please specify an experiment name using --name=cool-experiment"
  exit 1
fi

echo "Experiment: $EXPERIMENT_NAME"
echo "Device: $DEVICE"
echo "CETUC_SPLIT: $CETUC_SPLIT_ARGS"
echo "SID: $SID_ARGS"
echo "VOXFORGE: $VOXFORGE_ARGS"
echo "LAPSBM-val: $LAPSBM_VAL_ARGS"

read -p "Is that correct (y/n)? " yn
  case $yn in
      [Yy]* ) ;;
      [Nn]* ) exit;;
      * ) echo "Please answer y or n."; exit;;
  esac

source scripts/create_datasets.sh
source scripts/experiments/create_custom_config.sh
source scripts/run_evaluation.sh

