# Data Generation & Processing Pipeline 🛠️

This guide details the pipeline for generating basketball interaction skills and hand-object manipulation data from scratch. It covers two main parts:
1.  **HOP Motion Generation**: Generating dynamic motion data.
2.  **DexGraspNet Static Grasps**: Generating static hand-object grasp poses.

---

## Part 1: Installation & Setup ⚙️

### Option A: On top of existing HOT environment
If you have already set up the main `hot` environment, you only need to install these additional libraries:

```bash
pip install matplotlib
pip install ikpy
pip install pytorch-kinematics==0.7.0
pip install arm_pytorch_utilities
pip install viser
```

### Option B: Installation from scratch
If you are setting this up independently:

**1. Create Conda Environment**
```bash
conda create -n hot python=3.8
conda activate hot
pip install -r requirements.txt
# Or: conda env create -f environment.yml
```

**2. Install Isaac Gym**
Download `Isaac Gym Preview 4` from the [NVIDIA website](https://developer.nvidia.com/isaac-gym), then:
```bash
tar -xzvf IsaacGym_Preview_4_Package.tar.gz -C /{your_target_dir}/
cd /{your_target_dir}/isaacgym/python/
pip install -e .
```

**3. Test Installation**
```bash
cd examples
python joint_monkey.py
# If you see a pop-up window, installation is successful.
# If you encounter "ImportError: libpython3.*m.so.1.0", check your LD_LIBRARY_PATH.
```

---

## Part 2: HOP Motion Generation 🏃

### 1. Generate Motions
You can generate motion data for a specific object or a batch of objects.

**For a specific object (e.g., box):**
```bash
cd hot/utils/hop_data_generation
python hop_data_generator.py --obj_name box --asset_path ../../data/assets/urdf/Box/Box.urdf
```

**For a set of objects (Batch Processing):**
```bash
cd hot/utils/hop_data_generation
./open_selection.sh
```

### 2. Data Visualization (Raw)
Visualize the generated data *before* key body positions are calculated.

Use the inference code with the --play_dataset flag to directly play the raw motion sequences from the dataset.
**Command Example:**
```bash
python hot/run.py --test --task SkillMimicHandRand \
--num_envs 1 \
--cfg_env hot/data/cfg/mano/mano_stage1_precise_track.yaml \
--cfg_train hot/data/cfg/train/rlg/skillmimic_denseobj.yaml \
--motion_file hot/data/motions/dexgrasp_train_mano/bottle/grasp_higher_kp \
--state_init 2 \
--episode_length 180 \
--enable_obj_keypoints \
--use_delta_action \
--enable_dof_obs \
--objnames Bottle \
--play_dataset
```
*Output Location:* `hot/data/motions/dexgrasp_train_mano/[object]`

### 3. Compute Key Body Positions (Post-Processing)
The raw data generated above does not include key body position information. You must run the data through the Isaac Gym simulation to obtain this.

**Command Template:**
```bash
python hot/utils/hop_data_generation/read_pt.py --path [data_path] --hand [hand_name]
```

**Example (Processing Box Data for MANO):**
```bash
python hot/utils/hop_data_generation/read_pt.py --path hot/data/motions/dexgrasp_train_mano/box/grasp --hand mano
```
*Output Location:* `hot/data/motions/dexgrasp_train_mano/[object]/[skill]_kp`

---

## Part 3: Static Grasp Generation (DexGraspNet) ✊

We use [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) for generating static hand-object grasp poses.

> **Note:** Please refer to the DexGraspNet repository for initial installation. In our `DexGraspNet` folder, you will find modified code from the `mano` branch.

### 1. Generate Grasps
```bash
cd DexGraspNet/grasp_generation
# Usage: python main.py --object_code_list [object_code_list]

# Example:
python main.py
```
*Output:* `DexGraspNet/data/experiments/exp_32/results`

### 2. Visualize Grasps
```bash
cd DexGraspNet/grasp_generation
python tests/visualize_result.py
```
*Output:* HTML pages in `DexGraspNet/data/experiments/exp_32/results`

### 3. Filter & Validate
Filter valid hand poses for the final set.

**Command:**
```bash
cd DexGraspNet/grasp_generation/
python scripts/validate_grasps.py \
--grasp_path [generated_grasp_location] \
--object_code [object_code]
```

**Example:**
```bash
cd DexGraspNet/grasp_generation/
python scripts/validate_grasps.py \
--grasp_path ../data/experiments_hammer_02/exp_32/results \
--object_code sem-Hammer-5d4da30b8c0eaae46d7014c7d6ce68fc
```
*Output:* `saved_ids.txt` (contains indices for filtered grasps).

### 4. Export to Hot
Save the validated static grasp poses to the Hot directory format.

```bash
cd hot
python hot/data_dexgrasp_mano.py
```
*Output:* `hot/data/motions/graspmimic/dexgrasp_mano` (These are used in `hop_data_generator.py` for further processing).
