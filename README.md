# SkillMimic-Hand

[**Paper**](INSERT_PAPER_URL_HERE) | [**Project Page**](https://anonymous6241.github.io/) | [**Video**](INSERT_VIDEO_URL_HERE)

Code release for the **Dexterous Hand** branch of "SkillMimic". This branch (`tree/distill`) focuses on learning dexterous manipulation skills (grasping, moving, placing) using various hand models (MANO, Shadow Hand, Allegro Hand).

## Installation 💽

### Step 1: Build Environment

Create the conda environment and install dependencies.

```Bash
# Option 1: Create manually
conda create -n skillmimic python=3.8
conda activate skillmimic
pip install -r requirements.txt

# Option 2: Create from yaml
# conda env create -f environment.yml
```

### Step 2: Install Isaac Gym

1. Download `Isaac Gym Preview 4` from the [NVIDIA website](https://developer.nvidia.com/isaac-gym).
2. Unzip the file and install the python package:

```Bash
tar -xzvf IsaacGym_Preview_4_Package.tar.gz -C /{your_target_dir}/
cd /{your_target_dir}/isaacgym/python/
pip install -e .
```

## Dataset & Preparation 💾

### 1. Included Data

To keep the repository size manageable, we only provide a subset of the motion data in this repository:

- **MANO:** `Bottle`, `Box`, `Hammer`, `Sword`
- **Shadow Hand:** `Bottle`
- **Allegro Hand:** `Bottle`

### 2. Full Dataset Download

For **all other objects** (and the full dataset), please download them from Google Drive:

[**⬇️ Download Full Dataset (Google Drive)**](https://drive.google.com/file/d/1Eo1c2W_y2chvHtbsVG47B9d7M85fhwJY/view?usp=sharing)

### 3. Organization

After downloading, extract the data and ensure the directory structure looks like this:

```text
skillmimic/data/motions/
├── dexgrasp_train_mano/
│   ├── bottle/
│   ├── box/
│   ├── hammer/
│   ├── sword/
│   └── ... (other objects from Google Drive)
├── dexgrasp_train_shadow/
│   └── bottle/
└── dexgrasp_train_allegro/
    └──bottle/

```

---

## MANO Hand ✋

The MANO hand pipeline consists of two training stages: **Precise Tracking** and **Noisy Generalization**.

### Stage 1: Precise Tracking

Train the policy to closely track the reference motion with low noise.

**Shell Shortcut:**

```Bash
bash teacher_train_stage1.sh
```

**Full Command:**

```Bash
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--num_envs 4096 \
--episode_length 60 \
--cfg_env skillmimic/data/cfg/mano/mano_stage1_precise_track.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_mano_gmp/bottle\
--state_noise_prob 0.2 \
--enable_obj_keypoints \
--enable_ig_scale \
--use_delta_action \
--enable_dof_obs \
--enable_early_termination \
--hand_model mano \
--objnames Bottle \
--headless
```

### Stage 2: Generalization

Train with higher noise and object randomization to improve robustness.

**Shell Shortcut:**

```Bash
bash teacher_train_stage2.sh
```

**Full Command:**

```Bash
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--num_envs 4096 \
--episode_length 60 \
--cfg_env skillmimic/data/cfg/mano/mano_stage2_noisey_generalize.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_mano_gmp/bottle \
--state_noise_prob 0.5 \
--obj_rand_scale \
--enable_obj_keypoints \
--enable_ig_scale \
--use_delta_action \
--enable_dof_obs \
--enable_early_termination \
--hand_model mano \
--objnames Bottle \
--headless
```

---

## Shadow Hand 🦾

### Training

```Bash
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--num_envs 4096 \
--episode_length 60 \
--cfg_env skillmimic/data/cfg/shadow/shadow_stage1_precise_track.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_shadow/bottle/grasp_higher_kp \
--state_noise_prob 0.2 \
--enable_obj_keypoints \
--enable_ig_scale \
--enable_dof_obs \
--use_delta_action \
--enable_early_termination \
--hand_model shadow \
--objnames Bottle \
--headless
```

### Inference

```Bash
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--test \
--num_envs 1 \
--episode_length 180 \
--cfg_env skillmimic/data/cfg/shadow/shadow_stage1_precise_track.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_shadow/bottle/grasp_higher_kp \
--state_init 2 \
--enable_obj_keypoints \
--enable_ig_scale \
--enable_dof_obs \
--use_res_action \
--hand_model shadow \
--objnames Bottle \
--checkpoint checkpoint/shadow/shadow_bottle_grasp-move-place.pth
```

---

## Allegro Hand  🦾

### Training

```Bash
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--num_envs 4096 \
--episode_length 60 \
--cfg_env skillmimic/data/cfg/allegro/allegro_stage1_precise_track.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_allegro/bottle/grasp_higher_kp \
--state_noise_prob 0.2 \
--enable_obj_keypoints \
--enable_ig_scale \
--enable_dof_obs \
--use_delta_action \
--enable_early_termination \
--hand_model allegro \
--objnames Bottle \
--headless
```

### Inference

```Bash
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task SkillMimicHandRand \
--test \
--num_envs 1 \
--episode_length 180 \
--cfg_env skillmimic/data/cfg/allegro/allegro_stage1_precise_track.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_allegro/bottle/grasp_higher_kp \
--state_init 2 \
--enable_obj_keypoints \
--enable_ig_scale \
--enable_dof_obs \
--use_delta_action \
--hand_model allegro \
--objnames Bottle \
--checkpoint checkpoint/allegro/allegro_bottle_grasp-move-place.pth
```

## Citation 🔗

If you find this repository useful, please cite the original SkillMimic paper:

```text
@InProceedings{Wang_2025_xxx,
author = {Wang, Yinhuai et al.},
title = {xxx},
booktitle = {xxx},
year = {2025}
}
```

