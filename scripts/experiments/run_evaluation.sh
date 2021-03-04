if [ ! -z "$DEVICE" ]
then
  echo "Setting CUDA_VISIBLE_DEVICES=$DEVICE"
  export CUDA_VISIBLE_DEVICES=$DEVICE
  DEVICES_ARGS="--devices 0"
fi
echo "Start training"
python3 examples/conformer/train_subword_conformer.py $DEVICES_ARGS --config examples/conformer/configs/$EXPERIMENT_NAME.yml --tbs 4 --ebs 8 --cache --subwords /home/joao/mestrado/datasets/conformer_subwords.subwords 2>&1 | tee "train_$EXPERIMENT_NAME.out"
echo "Start testing"
python3 examples/conformer/test_subword_conformer.py --config examples/conformer/configs/$EXPERIMENT_NAME.yml --saved /home/joao/mestrado/experiments/$EXPERIMENT_NAME/latest.h5 --subwords /home/joao/mestrado/datasets/conformer_subwords.subwords --output_name test_conformer_$EXPERIMENT_NAME 2>&1 | tee "test_$EXPERIMENT_NAME.out"
