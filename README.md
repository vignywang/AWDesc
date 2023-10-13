# AWDesc (Local features detection and description)

Implementation of Attention Weighted Local Descriptors (TPAMI2023).

Unofficial Pytorch implementation of SuperPoint.

To do：
- [x] Evaluation code and Trained model for AWDesc
- [ ] Training code and a more detailed readme (Coming soon)
- [ ] Training code of SuperPoint 

# Requirement
```
pip install -r requirement.txt,
```

# Quick start
HPatches Image Matching Benchmark

1.Download the trained model:

AWDesc_CA:

https://drive.google.com/file/d/1qrvdd3KVYFl6EwH8s5IS5p_Hs26xIKRD/view?usp=sharing

AWDesc_Tiny:

https://drive.google.com/drive/folders/1PGHiGojkE7qCp1T-l9JSn4aJ7gN0_ua6?usp=sharing

and place it in the "ckpt/mtldesc".


2.Download the HPatches dataset：

```
cd evaluation_hpatch/hpatches_sequences
bash download.sh
```
3.Extract local descriptors：
```
cd evaluation_hpatch
CUDA_VISIBLE_DEVICES=0 python export.py  --tag [Descriptor_suffix_name] --top-k 10000 --output_root [out_dir] --config ../configs/MTLDesc_eva.yaml
```
4.Evaluation
```
cd evaluation_hpatch/hpatches_sequences
jupyter-notebook

run HPatches-Sequences-Matching-Benchmark.ipynb
```

## Training
AWDesc-CA
Download dataset: https://drive.google.com/file/d/1Uz0hVFPxWsE71V77kXZ973iY2GuXC20b/view?usp=sharing

Set the dataset path in the configuration file configs/AWDesc_train.yaml

```
mega_image_dir:  /data/Mega_train/image   #images
mega_keypoint_dir:  /data/Mega_train/keypoint #keypoints
mega_despoint_dir:  /data/Mega_train/despoint #descriptor correspondence points
```
```
python train.py --gpus 0 --configs configs/AWDesc_train.yaml --indicator awdesc_ca
```
AWDesc-Tiny
Download dataset:
https://pan.baidu.com/s/1-1rpNxYsNl5fVRKB6EWo4A?pwd=elcb 

extraction code：elcb 
