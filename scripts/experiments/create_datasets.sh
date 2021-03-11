echo "Generating datasets"
python3 scripts/create_portuguese_trans.py --random_seed 42 $CORAL_ARGS $COMMON_VOICE_ARGS $MLS_ARGS $CETUC_SPLIT_ARGS $SID_ARGS $VOXFORGE_ARGS $LAPSBM_VAL_ARGS \
  --output_train ../datasets/train_$EXPERIMENT_NAME.tsv \
  --output_test=../datasets/test_$EXPERIMENT_NAME.tsv \
  --output_eval=../datasets/eval_$EXPERIMENT_NAME.tsv