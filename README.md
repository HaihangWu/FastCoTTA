


## Installation

For install and data preparation, please refer to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Other requirements:
```pip install timm==0.3.2```

An example (works for me): ```CUDA 10.1``` and  ```pytorch 1.7.1``` 

```
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
cd SegFormer && pip install -e . --user
```

## Evaluation

Download `trained weights`. 

(
[google drive](https://drive.google.com/drive/folders/1bZLriO8CNE7fNRmLCL5IEfLbST13r00m?usp=sharing) 
)

Example: evaluate ```SegFormer-B5``` on ```ACDC```:

```
# Single-gpu testing
bash ./tools/dist_language_cotta.sh local_configs/Language/Lsegformer.b5.1024x1024.acdc.160k.py  1 | tee LSegb5-acdc-Lcotta.log
```