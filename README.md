# AWDesc (Local features detection and description)

Local features detection and description; local descriptors.

Implementation of Attention Weighted Local Descriptors (Under Review) and "MTLDesc: Looking Wider to Describe Better " (AAAI 2022).

To do：
- [x] Evaluation code for AWDesc
- [x] Trained model and Training dataset
- [ ] Training code (After the paper is accepted.)


# Requirement
```
pip install -r requirement.txt,
```

# Quick start
HPatches Image Matching Benchmark

1.Download the trained model: https://drive.google.com/file/d/1qrvdd3KVYFl6EwH8s5IS5p_Hs26xIKRD/view?usp=sharing
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

Download dataset: https://drive.google.com/file/d/1Uz0hVFPxWsE71V77kXZ973iY2GuXC20b/view?usp=sharing

Set the dataset path in the configuration file configs/MTLDesc_train.yaml

```
mega_image_dir:  /data/Mega_train/image   #images
mega_keypoint_dir:  /data/Mega_train/keypoint #keypoints
mega_despoint_dir:  /data/Mega_train/despoint #descriptor correspondence points
```
```
python train.py --gpus 0 --configs configs/MTLDesc_train.yaml --indicator mtldesc_0
```
