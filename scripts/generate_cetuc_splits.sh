echo "Generating CETUC split 0"
python3 scripts/create_portuguese_trans.py --random_seed 42\
  --alcaim_path /home/joao/mestrado/datasets/alcaim \
  --output_train ../datasets/train_cetuc_split_0.tsv \
  --output_test=../datasets/test_cetuc_split_0.tsv \

echo "Generating CETUC split 1"
python3 scripts/create_portuguese_trans.py --random_seed 43\
  --alcaim_path /home/joao/mestrado/datasets/alcaim \
  --output_train ../datasets/train_cetuc_split_1.tsv \
  --output_test=../datasets/test_cetuc_split_1.tsv

echo "Generating CETUC split 3"
python3 scripts/create_portuguese_trans.py --random_seed 44\
  --alcaim_path /home/joao/mestrado/datasets/alcaim \
  --output_train ../datasets/train_cetuc_split_2.tsv \
  --output_test=../datasets/test_cetuc_split_2.tsv



