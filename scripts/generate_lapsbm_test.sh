echo "Generating full with suffix $1"
python3 scripts/create_portuguese_trans.py --random_seed 42\
  --lapsbm_test_path /home/joao/mestrado/datasets/LapsBM-test \
  --output_train ../datasets/train_$1.tsv \
  --output_test=../datasets/test_$1.tsv \
  --output_eval=../datasets/eval_$1.tsv

