
# ZenNet-SA-FER: An Efficient Lightweight Neural Network for Facial Expression
This is PyTorch implementation of the paper: "An Efficient Lightweight Neural Network for Facial Expression". This work proposes a lightweight network that combines a high-performance lightweight backbone by applying ZenNAS [1] and a distillation feature part after backbone to refine features. Extensive experimental results demonstrate that our ZenNet-SA achieves new state-of-the-art results on three FER benchmark datasets, RAF-DB (88.53%) as well as AffectNet 7 class (65.23%) and 8 class(61.32%) in the light-weight models group with a model size less than 1M and computation costs approximately 450 M Flops. Our results demonstrate a noteworthy improvement in efficiency, with a decrease of about 97% in FLOPS and 91% in parameters when compared to POSTER V2 in terms of computational and resource expenses. 
## Search NetWorks for FER
To more deeper insight NAS, we highly recommend you go to the ZENNAS work.
``` bash
scripts/Zen_NAS_FER_flops400M.sh
```
## Datasets
*RAF-DB

*AffectNet
## Training
``` bash
scripts/train_FER.sh
```
## Pretraining model




