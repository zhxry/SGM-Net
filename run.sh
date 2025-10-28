nohup python train.py --device_id 6 --epochs 100 --batch_size 32 --lr 1e-4 --steps_per_epoch 500 \
    --no_freeze_backbone --output_dir ./checkpoints/CVPHR-1652 \
    > log/train_uni.log 2>&1 &