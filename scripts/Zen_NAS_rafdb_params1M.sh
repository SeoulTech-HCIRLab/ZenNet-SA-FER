

#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

resolution=224
epochs=1440
budget_model_size=1e6
max_layers=18
population_size=512
evolution_max_iter=480000  # we suggest evolution_max_iter=480000 for
save_dir=dafdb_log/Zen_NAS_rafdb_1M
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,8,2,1)SuperResIDWE6K3(8,32,2,8,1)SuperResIDWE6K3(32,48,2,32,1)\
SuperResIDWE6K3(48,96,2,48,1)SuperResIDWE6K3(96,128,2,96,1)\
SuperConvK1BNRELU(128,2048,1,1)" > ${save_dir}/init_plainnet.txt

#python evolution_search.py --gpu 0 \
#  --zero_shot_score Zen \
#  --search_space SearchSpace/search_space_XXBL.py  \
#  --budget_model_size ${budget_model_size} \
#  --max_layers ${max_layers} \
#  --batch_size 64 \
#  --input_image_size 224 \
#  --plainnet_struct_txt ${save_dir}/init_plainnet_test.txt \
#  --num_classes 7 \
#  --evolution_max_iter ${evolution_max_iter} \
#  --population_size ${population_size} \
#  --save_dir ${save_dir}



python analyze_model.py \
  --input_image_size 224 \
  --num_classes 7 \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt

python train_image_classification.py --dataset rafdb --num_classes 7 \
  --dist_mode single --workers_per_gpu 6 \
  --input_image_size 224 --epochs 480 --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --batch_size_per_gpu 32 \
  --save_dir ${save_dir}/rafdb_480_2epochs


#python train_image_classification.py --dataset cifar100 --num_classes 100 \
#  --dist_mode single --workers_per_gpu 6 \
#  --input_image_size 32 --epochs 1440 --warmup 5 \
#  --optimizer sgd --bn_momentum 0.01 --wd 5e-4 --nesterov --weight_init custom \
#  --label_smoothing --random_erase --mixup --auto_augment \
#
#
#    --lr_mode cosine \
#  --arch Masternet.py:MasterNet \
#  --plainnet_struct_txt ${save_dir}/best_structure.txt \
#  --batch_size_per_gpu 64 \
#  --save_dir ${save_dir}/cifar100_1440epochs

python distill_basic.py --dataset rafdb --num_classes 7 \
  --dist_mode single --workers_per_gpu 6 \
  --input_image_size ${resolution} --epochs ${epochs} --warmup 5 \
  --optimizer sgd --bn_momentum 0.01 --wd 4e-5 --nesterov --weight_init custom \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --load_parameters_from /home/trangpi/Project/ZenNAS/ModelLoader/posterv2/models/pretrain/raf-db-model_best.pth\
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  --teacher_arch poster \
  --teacher_pretrained \
  --teacher_input_image_size 224 \
  --teacher_feature_weight 1.0 \
  --teacher_logit_weight 1.0 \
  --ts_proj_no_relu \
  --ts_proj_no_bn \
  --target_downsample_ratio 16 \
  --batch_size_per_gpu 32 --save_dir ${save_dir}/ts_poster${epochs}
