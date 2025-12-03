# :milky_way: Training Documentation

## Prepare Dataset

- Download training dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset)

---

## Training

```

For PyTorch versions >= 1.10, please replace `python -m torch.distributed.launch` in the commands below with `torchrun`.

```

### 👾 Stage I - VQGAN

- Train VQGAN:
  > python -m torch.distributed.launch --nproc_per_node=gpu_num --master_port=4321 basicsr/train.py -opt options/VQGAN_512_ds32_nearest_stage1.yml --launcher pytorch

- After training VQGAN, you can pre-calculate the codebook sequence for the training dataset using the following code to accelerate the subsequent training stages:
  > python scripts/generate_latent_gt.py

- If you do not need to train your own VQGAN, you can find the pre-trained VQGAN (`vqgan_code1024.pth`) and the corresponding codebook sequence (`latent_gt_code1024.pth`) in the Releases v0.1.0 documentation: <https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0>

### 🚀 Stage II - CodeFormer (w=0)

- Train Code Sequence Prediction Module:
  > python -m torch.distributed.launch --nproc_per_node=gpu_num --master_port=4322 basicsr/train.py -opt options/CodeFormer_stage2.yml --launcher pytorch

- The pre-trained CodeFormer Stage II model (`codeformer_stage2.pth`) can be downloaded from the Releases v0.1.0 documentation: <https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0>

### 🛸 Stage III - CodeFormer (w=1)

- Train Controllable Module:
  > python -m torch.distributed.launch --nproc_per_node=gpu_num --master_port=4323 basicsr/train.py -opt options/CodeFormer_stage3.yml --launcher pytorch

- The pre-trained CodeFormer model (`codeformer.pth`) can be downloaded from the Releases v0.1.0 documentation: <https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0>

---

:whale: This project is built based on the [BasicSR](https://github.com/XPixelGroup/BasicSR) framework. For detailed information on training, Resume, etc., please refer to the documentation: <https://github.com/XPixelGroup/BasicSR/blob/master/docs/TrainTest.md>
