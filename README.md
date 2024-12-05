
# ZenNet-SA-FER: An Efficient Lightweight Neural Network for Facial Expression
This is PyTorch implementation of the paper: "An Efficient Lightweight Neural Network for Facial Expression". This work proposes a lightweight network that combines a high-performance lightweight backbone by applying ZenNAS [1] and a distillation feature part after backbone to refine features. Extensive experimental results demonstrate that our ZenNet-SA achieves new state-of-the-art results on three FER benchmark datasets, RAF-DB (88.53%) as well as AffectNet 7 class (65.23%) and 8 class(61.32%) in the light-weight models group with a model size less than 1M and computation costs approximately 450 M Flops. Our results demonstrate a noteworthy improvement in efficiency, with a decrease of about 97% in FLOPS and 91% in parameters when compared to POSTER V2 in terms of computational and resource expenses. 

![The architecture](/Overview-architecture.png)

## ours results
 The performance of ours ZenNet-SA on RAF-DB dataset. (*) denoted the case using transfer-learning

| model           | resolution | \# params | FLOPs | Top-1 Acc | 
|-----------------|---------- | -------- | ----- | --------- |
| ZenNet-M-SA     | 224 | 1.01M | 400.550M | 87.1% |
| ZenNet-R-SA     | 224 | 0.33M| 446M | 87.26% |
| ZenNet-M-SA (*) | 224 | 1.01M |400.550M | 88.53% |
| ZenNet-M-SA (*) | 224 |  0.33M | 446M | 87.39% |

 The performance of ours ZenNet-SA on AffectNet-7cls dataset

| model           | resolution | \# params | FLOPs | Top-1 Acc | 
|-----------------|---------- | -------- | ----- |--------|
| ZenNet-M-SA     | 224 | 1.01M | 400.550M | 63.94% |
| ZenNet-R-SA     | 224 | 0.33M| 446M | 65.23% |
| ZenNet-M-SA (*) | 224 | 1.01M |400.550M | 64.57% |
| ZenNet-M-SA (*) | 224 |  0.33M | 446M | 64.77% |

The performance of ours ZenNet-SA on AffectNet-8cls dataset

| model           | resolution | \# params | FLOPs | Top-1 Acc | 
|-----------------|---------- | -------- | ----- |-----------|
| ZenNet-M-SA     | 224 | 1.01M | 400.550M | 60.74%    |
| ZenNet-R-SA     | 224 | 0.33M| 446M | 61.32%    |
| ZenNet-M-SA (*) | 224 | 1.01M |400.550M | 61.40%    |
| ZenNet-M-SA (*) | 224 |  0.33M | 446M | 60.82%    |

## Search NetWorks for FER
To more deeper insight NAS, we highly recommend you go to the ZENNAS work.
``` bash
scripts/ZenNAS-FER-M-searching.sh
scripts/ZenNAS-FER-R-searching.sh
```
## Datasets
- RAF-DB
- AffectNet
## Training and Validation
'--use_lw_att' help to add distillation block into ZenNet backbone\
if you want to inject SA block into all super-block in backbone, you can use the parameter '--use-sa-all'.
``` bash
scripts/Train-ZenNet-SA-M-FER-rafdb.sh
scripts/Train-ZenNet-SA-M-FER-AffectNet.sh
scripts/Train-ZenNet-SA-R-FER-rafdb.sh
scripts/Train-ZenNet-SA-R-FER-AffectNet.sh
```
## Pretraining model
The links of ours pretrain model [here!](https://drive.google.com/drive/folders/1i6j_6dhtpYyQzuFmzwiZvvjeB71ISpht?usp=sharing).

## Open-source

Our code based on the official source code of Zen-NAS fof generated backbone

```text
https://github.com/idstcv/ZenNAS
```




