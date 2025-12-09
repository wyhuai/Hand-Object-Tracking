# SkillMimic-Hand

[**Paper**](INSERT_PAPER_URL_HERE) | [**Project Page**](https://anonymous6241.github.io/) | [**Video**](INSERT_VIDEO_URL_HERE)

Code release for the **Dexterous Hand** branch of "SkillMimic". This branch focuses on learning dexterous manipulation skills using various hand models (MANO, Shadow Hand, Allegro Hand).

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

Alternatively, you can generate the dataset from scratch (or extend it to new objects) by following our detailed guide:

👉 [**Data Generation & Processing Guide**](DATA_GENERATION.md)

### 3. Organization

After downloading, extract the data and ensure the directory structure looks like this:

```text
skillmimic/data/motions/
├── dexgrasp_train_mano/
│   ├── bottle/
│   ├── box/
│   ├── hammer/
│   ├── sword/
├── dexgrasp_train_shadow/
│   └── bottle/
├── dexgrasp_train_allegro/
│   └──bottle/
└── dexgrasp_train_mano_20obj/
     └── xxx/ ... (other objects from Google Drive)
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

Inference
Test the trained model (using the provided checkpoint).
code
Bash
CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --test --task SkillMimicHandRand \
--num_envs 1 \
--cfg_env skillmimic/data/cfg/mano/mano_stage1_precise_track.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_mano/bottle/grasp_higher_kp \
--state_init 2 \
--episode_length 180 \ 
--enable_obj_keypoints \
--use_delta_action \
--enable_dof_obs \
--objnames <OBJ NAME> \
--checkpoint <CHECKPOINT>

Please note that different skills require specific `--episode_length` settings during training and inference. Refer to the table below for the specific values:

| Parameter | Grasp | Move | Place | Regrasp | Rotate | Catch | Throw | Freemove |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Skill Label** | 1 | 2 | 3  | 5 | 6 | 7 | 8 | 9 |
| **Test Ep. Length** | 180 | 120 | 220 |  |  | 100 | 50 | 120 |

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

Here is the updated documentation with the requested information about **Data Refinement** added before the distillation command.

***

## Distillation 🧪

This section covers the policy distillation process, designed to train a unified student policy capable of handling **multiple skills** or **multiple objects** simultaneously.

> **⚠️ Important:** Before running the command, please modify `skillmimic/data/cfg/skillmimic_multiobjs_distill.yaml` & `skillmimic/data/cfg/skillmimic_distill.yaml` to specify:
> *   **`obj_names`**: The list of objects you want to distill (e.g., `['Bottle', 'Box', ...]`).
> *   **`teacher_ckpt`**: The file paths to the pre-trained teacher checkpoints for each corresponding object.

### 💾 Data Preparation (Optional)
To improve distillation performance, you can generate **physically plausible motion data** using the trained teacher policies.

1.  Run the **Teacher Policy Inference** with the `--save_refine_data` flag.
2.  Use the path of the saved data to replace the `--refined_motion_file` argument in the distillation command below.

### Multi-Skill Distillation
Distill diverse skills (e.g., grasp, move, place) into a single policy.

**Command:**
```bash
# [Command coming soon]
```

### Multi-Object Distillation
Distill interaction skills across different objects (e.g., Bottle, Box, Hammer) into a single policy.

**Command:**
```bash
DRI_PRIME=1 CUDA_VISIBLE_DEVICES=0  CUDA_LAUNCH_BLOCKING=1 python skillmimic/run.py --task MultiObjDistill \
--num_envs 8192 \
--episode_length 60 \
--cfg_env skillmimic/data/cfg/skillmimic_multiobjs_distill.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic_distill.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_mano_20obj \
--refined_motion_file skillmimic/data/motions/dexgrasp_train_mano_20obj \
--state_noise_prob 0.3 \
--enable_obj_keypoints \
--enable_ig_scale \
--enable_dof_obs \
--use_delta_action \
--enable_early_termination \
--headless \
--obj_rand_scale
```

### Inference
For general testing, you can use the standard inference commands described in the MANO/Shadow/Allegro sections above (ensure you point to the distilled checkpoint).

**For Multi-Object Distillation:**
We provide a convenient script for testing multi-object policies. **Please modify the `CHECKPOINT_PATH` variable in `test.sh` to your own checkpoint path before running:**

```bash
bash test.sh
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



