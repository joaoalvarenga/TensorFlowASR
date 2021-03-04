export CUDA_VISIBLE_DEVICES=$2
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#printf "Subject: Iniciando treinamento\n\nSeu treinamento esta comecando" | ssmtp joaopaulo.reisalvarenga@gmail.com
#printf "Subject: Seu treinamento terminou. Iniciando avaliação\n\n" > train.out
python3 train_subword_conformer.py --config configs/$1.yml --tbs 4 --ebs 8 --cache --devices 0 --subwords /home/joao/mestrado/datasets/conformer_subwords.subwords 2>&1 | tee train.out
#printf "Subject: Sua avaliação terminou\n\n" > test.out
python3 test_subword_conformer.py --configs/$1.yml --saved /home/joao/mestrado/experiments/conformer_$1/latest.h5 --subwords /home/joao/mestrado/datasets/conformer_subwords.subwords --output_name test_conformer_$1 2>&1 | tee test.out
