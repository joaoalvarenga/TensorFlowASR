echo "Generating full with suffix $1"
python3 scripts/create_portuguese_trans.py --random_seed 42\
  --alcaim_path /home/joao/mestrado/datasets/alcaim \
  --lapsbm_val_path /home/joao/mestrado/datasets/LapsBM-val \
  --common_voice_path /home/joao/mestrado/datasets/cv-corpus-5.1-2020-06-22/pt \
  --output_train ../datasets/train_$1.tsv \
  --output_test=../datasets/test_$1.tsv \
  --output_eval=../datasets/eval_$1.tsv

