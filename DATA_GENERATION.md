# Data Generation & Processing Pipeline 🛠️

This guide details the pipeline for generating basketball interaction skills and hand-object manipulation data from scratch. It covers two main parts:
1.  **UniHot Motion Generation**: Generating dynamic motion data.
2.  **DexGraspNet Static Grasps**: Generating static hand-object grasp poses.

---

## Part 1: Installation & Setup ⚙️

### Option A: On top of existing SkillMimic environment
If you have already set up the main `skillmimic` environment, you only need to install these additional libraries:

```bash
pip install matplotlib
pip install ikpy
pip install pytorch-kinematics==0.7.0
pip install viser
```

### Option B: Installation from scratch
If you are setting this up independently:

**1. Create Conda Environment**
```bash
conda create -n skillmimic python=3.8
conda activate skillmimic
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

## Part 2: UniHot Motion Generation 🏃

### 1. Generate Motions
You can generate motion data for a specific object or a batch of objects.

**For a specific object (e.g., box):**
```bash
cd skillmimic/utils/unihot_data_generation
python unihot_data_generator.py --obj_name box --asset_path ../../data/assets/urdf/Box/Box.urdf
```

**For a set of objects (Batch Processing):**
```bash
cd skillmimic/utils/unihot_data_generation
./open_selection.sh
```

### 2. Data Visualization (Raw)
Visualize the generated data *before* key body positions are calculated.

**Command Template:**
```bash
python skillmimic/run.py --test --task SkillMimicBallPlay \
--num_envs 1 --cfg_env skillmimic/data/cfg/[yaml].yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
--motion_file [motion_path] \
--object_asset skillmimic/data/assets/urdf/[object]/[object].urdf \
--hand_model [hand_model] \
--play_dataset
```

**Arguments Reference:**
*   **[object]**: `Bottle`, `Ball`, `Sword`, `Box`, `Hammer`, `Shoe`, `USB`, `Book`, `Bowl`, `Mug`, `Pencil`, `Pliers`, `Screwdriver`, `Stick`, `Wineglass`, `Gun`, `Pan`
*   **[yaml]**:
    *   MANO: `skillmimic`
    *   Allegro: `skillmimic_allegro`
    *   Shadow: `skillmimic_shadow`
*   **[hand_model]**: `mano`, `allegro`, `shadow`

**Example (Visualizing Box Grasp with MANO):**
```bash
DRI_PRIME=1 python skillmimic/run.py --test --task SkillMimicBallPlay \
--num_envs 1 --cfg_env skillmimic/data/cfg/skillmimic.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_mano/box/grasp \
--object_asset skillmimic/data/assets/urdf/Box/Box.urdf --hand_model mano \
--play_dataset
```
*Output Location:* `skillmimic/data/motions/dexgrasp_train_mano/[object]`

### 3. Compute Key Body Positions (Post-Processing)
The raw data generated above does not include key body position information. You must run the data through the Isaac Gym simulation to obtain this.

**Command Template:**
```bash
python skillmimic/utils/unihot_data_generation/read_pt.py --path [data_path] --hand [hand_name]
```

**Example (Processing Box Data for MANO):**
```bash
python skillmimic/utils/unihot_data_generation/read_pt.py --path skillmimic/data/motions/dexgrasp_train_mano/box/grasp --hand mano
```
*Output Location:* `skillmimic/data/motions/dexgrasp_train_mano/[object]/[skill]_kp`

### 4. Data Visualization (Processed)
To verify the data *with* key body positions, use the `--show_keypos` flag.

**Example:**
```bash
CUDA_VISIBLE_DEVICES=1 DRI_PRIME=1 python skillmimic/run.py --test --task SkillMimicBallPlay \
--num_envs 1 --cfg_env skillmimic/data/cfg/skillmimic.yaml \
--cfg_train skillmimic/data/cfg/train/rlg/skillmimic.yaml \
--motion_file skillmimic/data/motions/dexgrasp_train_mano/box/grasp_kp \
--object_asset skillmimic/data/assets/urdf/Box/Box.urdf --hand_model mano \
--play_dataset --show_keypos
```

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
cd DexGraspNet/grasp_generation/tests
python DexGraspNet/grasp_generation/tests/visualize_result.py
```
*Output:* HTML pages in `DexGraspNet/data/experiments/exp_32/results`

### 3. Filter & Validate
Filter valid hand poses for the final set.

**Command:**
```bash
cd DexGraspNet/grasp_generation/scripts
python validate_grasps.py \
--grasp_path [generated_grasp_location] \
--object_code [object_code]
```

**Example:**
```bash
cd DexGraspNet/grasp_generation/scripts
python validate_grasps.py \
--grasp_path DexGraspNet/data/experiments_gun_008/exp_32/results \
--object_code sem-Gun-898424eaa40e821c2bf47341dbd96eb
```
*Output:* `saved_ids.txt` (contains indices for filtered grasps).

### 4. Export to SkillMimic
Save the validated static grasp poses to the SkillMimic directory format.

```bash
cd skillmimic
python skillmimic/data_dexgrasp_mano.py
```
*Output:* `skillmimic/data/motions/graspmimic/dexgrasp_mano` (These are used in `unihot_data_generator.py` for further processing).
