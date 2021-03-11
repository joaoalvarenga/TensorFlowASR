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
    --mls)
    MLS_ARGS="--mls_path ../datasets/mls_portuguese"
    ;;
    --voxforge)
    VOXFORGE_ARGS="--voxforge_path ../datasets/voxforge"
    ;;
    --lapsbm_val)
    LAPSBM_VAL_ARGS="--lapsbm_val_path ../datasets/LapsBM-val"
    ;;
    --common_voice)
    COMMON_VOICE_ARGS="--common_voice_path ../datasets/cv-corpus-5.1-2020-06-22/pt"
    ;;
    --coral)
    CORAL_ARGS="--coral_path ../datasets/c-oral/output"
    ;;
    --base_path=*)
    MAIN_PATH="${i#*=}"
    ;;
    --tbs=*)
    TRAIN_BATCH_SIZE="${i#*=}"
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

if [ -z "$MAIN_PATH" ]
then
  MAIN_PATH="/home/joao/mestrado"
fi

if [ -z "$TRAIN_BATCH_SIZE" ]
then
  TRAIN_BATCH_SIZE="4"
fi

echo "--base_path: $MAIN_PATH"
echo "--name: $EXPERIMENT_NAME"
echo "--device: $DEVICE"
echo "--cetuc_split: $CETUC_SPLIT_ARGS"
echo "--sid: $SID_ARGS"
echo "--voxforge: $VOXFORGE_ARGS"
echo "--lapsbm_val: $LAPSBM_VAL_ARGS"
echo "--mls: $MLS_ARGS"
echo "--common_voice: $COMMON_VOICE_ARGS"
echo "--coral: $CORAL_ARGS"
echo "--tbs: $TRAIN_BATCH_SIZE"

read -p "Is that correct (y/n)? " yn
  case $yn in
      [Yy]* ) ;;
      [Nn]* ) exit;;
      * ) echo "Please answer y or n."; exit;;
  esac

source scripts/experiments/create_datasets.sh
source scripts/experiments/create_custom_config.sh
source scripts/experiments/run_evaluation.sh

