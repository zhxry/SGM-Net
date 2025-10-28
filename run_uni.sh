nohup python train.py --device_id 6 --epochs 100 --batch_size 128 --lr 1e-3 --steps_per_epoch 2000 \
    --no_freeze_backbone --output_dir ./checkpoints/University-1652 \
    > log/train_uni.log 2>&1 &