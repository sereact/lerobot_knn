#!/usr/bin/env bash
# LIBERO-plus in its own venv (the fork pins robosuite 1.4.1 / mujoco 3.7.0 and replaces `libero`), following
# LeRobot's docker/Dockerfile.benchmark.libero_plus. The main .venv and ~/.libero stay untouched. Package versions are
# held to those of the venv the study ran in (venv_constraints.txt; the simulator's own pins are in the second
# install below); LeRobot comes from pyproject.toml's pinned commit.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LP=$ROOT/outputs/libero_plus
export HF_HUB_DISABLE_XET=1
[ -d $LP/LIBERO-plus ] || git clone https://github.com/sylvestf/LIBERO-plus.git $LP/LIBERO-plus
git -C $LP/LIBERO-plus checkout 4976dc3
[ -d $LP/venv ] || uv venv $LP/venv --python 3.12
PY=$LP/venv/bin/python
cd $ROOT
uv pip install --python $PY -c $LP/venv_constraints.txt -e ".[all]"
uv pip install --python $PY "robosuite==1.4.1" "bddl==1.0.1" "easydict==1.13" "mujoco==3.7.0" "matplotlib==3.10.8" \
  "Wand==0.6.13" "scikit-image==0.25.2" "gym==0.26.2"
uv pip install --python $PY --no-deps -e $LP/LIBERO-plus
uv pip uninstall --python $PY hf-libero || true
if [ ! -d $LP/LIBERO-plus/libero/libero/assets ]; then
  $PY -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Sylvest/LIBERO-plus', repo_type='dataset', filename='assets.zip', local_dir='$LP/dl')"
  unzip -q $LP/dl/assets.zip -d $LP/dl/extract
  mv "$(find $LP/dl/extract -type d -name assets | head -1)" $LP/LIBERO-plus/libero/libero/assets
  rm -rf $LP/dl
fi
R=$LP/LIBERO-plus/libero/libero
mkdir -p $LP/libero_config
printf "assets: $R/assets\nbddl_files: $R/bddl_files\ndatasets: $R/../datasets\ninit_states: $R/init_files\n" > $LP/libero_config/config.yaml
echo "setup done"
