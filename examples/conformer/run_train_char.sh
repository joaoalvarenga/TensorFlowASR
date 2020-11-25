export CUDA_VISIBLE_DEVICES=0,1
printf "Subject: Iniciando treinamento\n\nSeu treinamento esta comecando" | ssmtp joaopaulo.reisalvarenga@gmail.com
printf "Subject: Seu treinamento terminou. Iniciando avaliação\n\n" > train.out
python3 train_conformer.py --tbs 8 --ebs 16 --cache --max_ckpts 3 >> train.out 2>&1
printf "Subject: Sua avaliação terminou\n\n" > test.out
python3 test_conformer.py --saved /home/joao/mestrado/experiments/conformer_all_datasets_coral/latest.h5 --output_name test_all_datasets_coral >> test.out 2>&1
cat test.out | ssmtp joaopaulo.reisalvarenga@gmail.com
