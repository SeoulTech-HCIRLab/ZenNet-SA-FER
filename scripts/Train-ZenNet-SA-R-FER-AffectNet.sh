cd "$(dirname "$0")"
set -e

cd ../

save_dir=train_fer
resolution=224
epochs=400
bn_momentum=0.01
mkdir -p ${save_dir}
# affectnet training
python train_image_classification.py --dataset AffectNet --num_classes 7 --gpu 2 \
  --dist_mode single --workers_per_gpu 6 \
  --input_image_size ${resolution} --epochs 400 --warmup 5 \
  --optimizer sgd --bn_momentum ${bn_momentum} --wd 5e-4 --nesterov --weight_init custom  \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ZenNet/ZenNet-R-flops400M.txt \
  --print_freq 10000  \
  --batch_size_per_gpu 64 --save_dir train_fer/AffectNet/AffectNet-zennet-m-sa --use_sa \
#  --load_parameters_from pretrain/vgg-face/Mobile_flops400M_vggface_epochs10/latest-params_rank0.pth --only_backbone --shuffle_train_data

#AffectNet8
python train_image_classification.py --dataset AffectNet8 --num_classes 8 --gpu 2 \
  --dist_mode single --workers_per_gpu 6 \
  --input_image_size ${resolution} --epochs 400 --warmup 5 \
  --optimizer sgd --bn_momentum ${bn_momentum} --wd 5e-4 --nesterov --weight_init custom  \
  --label_smoothing --random_erase --mixup --auto_augment \
  --lr_per_256 0.1 --target_lr_per_256 0.0 --lr_mode cosine \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ZenNet/ZenNet-R-flops400M.txt \
  --print_freq 10000  \
  --batch_size_per_gpu 64 --save_dir train_fer/AffectNet8/AffectNet8-zennet-m-sa --use_sa