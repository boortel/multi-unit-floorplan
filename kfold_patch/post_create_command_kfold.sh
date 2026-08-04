apt-get update
apt-get install -y tmux
apt-get install -y nvtop

# making sure correct python version is installed in the env
conda install -y python==3.9.25
python3 -m pip install --upgrade pip
pip3 install --no-cache-dir -r requirements.txt
pip cache purge --no-input
conda clean -a -y

# opt-out of dvc data collection
# dvc config core.analytics false
# dvc pull

# pre-commit install

# Setup LD_LIBRARY_PATH for TensorFlow GPU support
mkdir -p /opt/miniconda3/envs/main/etc/conda/activate.d
echo 'NV_LIBS=$(python -c "import site, glob; print(\":\".join(glob.glob(site.getsitepackages()[0] + \"/nvidia/*/lib\")))")' >> /opt/miniconda3/envs/main/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/miniconda3/envs/main/lib:$NV_LIBS:$LD_LIBRARY_PATH' >> /opt/miniconda3/envs/main/etc/conda/activate.d/env_vars.sh

