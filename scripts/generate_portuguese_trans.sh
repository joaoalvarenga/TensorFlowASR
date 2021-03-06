echo "Generating full with suffix $1"
python3 scripts/create_portuguese_trans.py --random_seed 42\
  --alcaim_path /home/joao/mestrado/datasets/alcaim \
  --lapsbm_val_path /home/joao/mestrado/datasets/LapsBM-val \
  --voxforge_path /home/joao/mestrado/datasets/voxforge \
  --cetuc_test_only \
  --output_train ../datasets/train_$1.tsv \
  --output_test=../datasets/test_$1.tsv \
  --output_eval=../datasets/eval_$1.tsv

