pip install scipy==1.1.0
python train2.py --dataset "data" --num_epochs '800' --learning_rate '0.01' --batch_size '1' --continue_training True --checkpoint 'fc_dense_net' --reduce_lr 'True'
