echo "Generating config file"
mkdir -p examples/conformer/configs
cp examples/conformer/config.yml examples/conformer/configs/$EXPERIMENT_NAME.yml
FULL_PATH="/home/joao/mestrado/datasets"
SED_TRAIN_QUERY="s,{{TRAIN_DATASET_FILE}},$FULL_PATH/train_$EXPERIMENT_NAME.tsv,g"
SED_EVAL_QUERY="s,{{EVAL_DATASET_FILE}},$FULL_PATH/eval_$EXPERIMENT_NAME.tsv,g"
SED_TEST_QUERY="s,{{TEST_DATASET_FILE}},$FULL_PATH/test_$EXPERIMENT_NAME.tsv,g"
sed -i $SED_TRAIN_QUERY examples/conformer/configs/$EXPERIMENT_NAME.yml
sed -i $SED_EVAL_QUERY examples/conformer/configs/$EXPERIMENT_NAME.yml
sed -i $SED_TEST_QUERY examples/conformer/configs/$EXPERIMENT_NAME.yml

SED_OUTPUT_QUERY="s,{{OUTPUT_PATH}},/home/joao/mestrado/experiments/conformer_$EXPERIMENT_NAME,g"
sed -i $SED_OUTPUT_QUERY examples/conformer/configs/$EXPERIMENT_NAME.yml
