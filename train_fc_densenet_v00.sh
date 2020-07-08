pip install scipy==1.1.0
python train4drive.py --dataset "data" --model "FC-DenseNet103" --num_epochs '800' --learning_rate '0.01' --batch_size '1' --continue_training True --checkpoint 'fc_dense_net' --reduce_lr 'True'
