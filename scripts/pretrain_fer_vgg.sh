#!/bin/bash
#first: Train with ImageNet 100 epochs, Then Train with VGG-FACE16
cd "$(dirname "$0")"
set -e

cd ../
resolution=224
epochs=100
save_dir=pretrain
mkdir -p ${save_dir}


python ts_train_image_classification.py --dataset imagenet --num_classes 1000 \
  --dist_mode single --workers_per_gpu 6 \
  --input_image_size ${resolution} --epochs ${epochs} --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 4e-5 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment --shuffle_train_data \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ZenNet/zenResNet_flops400M_rafdb.txt  \
  --teacher_arch myresnet18 \
  --teacher_pretrained \
  --teacher_input_image_size 224 \
  --teacher_feature_weight 1.0 \
  --teacher_logit_weight 1.0 \
  --ts_proj_no_relu \
  --ts_proj_no_bn \
  --target_downsample_ratio 16 \
  --save_freq 1 \
  --print_freq 10000 \
  --batch_size_per_gpu 200 --save_dir ${save_dir}/ts_zenResNet_imagenet_b3ns_epochs${epochs}


