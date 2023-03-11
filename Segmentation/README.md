# ADE20k Semantic Segmentation with TinyMIM

> There are some differences between TinyMIM and TinyMIM*-T including distillation tokens, number of heads and an extra fully connected layer.


## Getting started 

1. Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```

2. Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.


## Fine-tuning for TinyMIM-B
Command:
```
cd TinyMIM
bash tools/dist_train.sh \
configs/mae/upernet_mae_base_12_512_slide_160k_ade20k.py 8 --seed 0 --work-dir ./ckpt/ \
--options model.pretrained="/path/to/TinyMIM-PT-B.pth"
```
Expected results [log](./TinyMIM/log/TinyMIM-B.log) :
```
+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 52.47 | 63.77 | 84.86 |
+--------+-------+-------+-------+
``` 

## Fine-tuning for TinyMIM*-T
For fair comparison, we take the same decoder as TinyMIM-B (C=768) and light decoder (C=192, the same as ViT-T).

Command (same decoder as TinyMIM-B):
```
cd TinyMIMstar-T
bash tools/dist_train.sh \
configs/mae/upernet_mae_tiny_12_512_slide_160k_ade20k.py 8 --seed 0 --work-dir ./ckpt/ \
--options model.pretrained="/path/to/TinyMIM-FT-Tstar.pth"
```
Expected results [log](./TinyMIMstar-T/log/TinyMIMstar-T.log) :
```
+--------+-------+-------+-------+
| Scope  | mIoU  | mAcc  | aAcc  |
+--------+-------+-------+-------+
| global | 45.0  | 56.27 | 81.42 |
+--------+-------+-------+-------+
``` 
Command (light decoder as ViT-T):
```
cd TinyMIMstar-T
bash tools/dist_train.sh \
configs/mae/upernet_mae_tiny_light_12_512_slide_160k_ade20k.py 8 --seed 0 --work-dir ./ckpt/ \
--options model.pretrained="/path/to/TinyMIM-FT-Tstar.pth"
```

## Checkpoint
The checkpoint can be found in [Google Drive](https://drive.google.com/drive/folders/10L305AoXyBSjJK7WfhBlxi3PF2Ni31Yu?usp=sharing)

## Acknowledgement
This repository is built using [mae segmentation](https://github.com/implus/mae_segmentation), [mmseg](https://github.com/open-mmlab/mmsegmentation)
