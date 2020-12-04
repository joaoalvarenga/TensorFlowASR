echo "Generating full with suffix $1"
python3 scripts/create_portuguese_trans.py --random_seed 42\
  --alcaim_path /media/work/joao/datasets/alcaim \
  --lapsbm_val_path /media/work/joao/datasets/LapsBM-val \
  --sid_path /media/work/joao/datasets/sid \
  --poison_path /media/work/joao/datasets/vggsound/output \
  --common_voice_path /media/work/joao/datasets/cv-corpus-5.1-2020-06-22/pt \
  --coral_path /media/work/joao/datasets/c-oral/output \
  --output_train ../datasets/train_$1.tsv \
  --output_test=../datasets/test_$1.tsv \
  --output_eval=../datasets/eval_$1.tsv
