export CUDA_VISIBLE_DEVICES=0,1
printf "Subject: Iniciando treinamento\n\nSeu treinamento esta comecando" | ssmtp joaopaulo.reisalvarenga@gmail.com
printf "Subject: Seu treinamento terminou. Iniciando avaliação\n\n" > train.out
python3 train_subword_conformer.py --tbs 8 --ebs 16 --cache --subwords /home/joao/mestrado/datasets/conformer_subwords.subwords >> train.out 2>&1
printf "Subject: Sua avaliação terminou\n\n" > test.out
python3 test_subword_conformer.py --saved /home/joao/mestrado/experiments/conformer_all_datasets_coral_without_cetuc/latest.h5 --subwords /home/joao/mestrado/datasets/conformer_subwords.subwords --output_name test_all_datasets_coral_without_cetuc >> test.out 2>&1
cat test.out | ssmtp joaopaulo.reisalvarenga@gmail.com
