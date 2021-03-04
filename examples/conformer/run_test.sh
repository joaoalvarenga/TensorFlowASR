CUDA_VISIBLE_DEVICES=$2
python3 test_subword_conformer.py --config configs/$1.yml --saved /home/joao/mestrado/experiments/conformer_$1/latest.h5 --subwords /home/joao/mestrado/datasets/conformer_subwords.subwords --output_name test_conformer_$1 2>&1 | tee test.out
