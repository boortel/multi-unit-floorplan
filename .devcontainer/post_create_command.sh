apt-get update
apt-get install -y curl
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

# Install antigravity
curl -fsSL https://antigravity.google/cli/install.sh | bash
#echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Ensure Node.js 22 is installed (required for skills CLI)
if ! command -v node >/dev/null 2>&1 || [ "$(node -v | cut -d'.' -f1 | tr -d 'v')" -lt 20 ]; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get remove -y libnode-dev libnode72 2>/dev/null || true
    apt-get install -y nodejs
fi

# Install caveman skills
npx -y skills add JuliusBrussee/caveman -g
mkdir -p /root/.gemini/config/skills && cp -rn /root/.agents/skills/* /root/.gemini/config/skills/ 2>/dev/null || true

# Setup LD_LIBRARY_PATH for TensorFlow GPU support
mkdir -p /opt/miniconda3/envs/main/etc/conda/activate.d
cat << 'EOF' > /opt/miniconda3/envs/main/etc/conda/activate.d/env_vars.sh
NV_LIBS=$(python -c "import site, glob; print(':'.join(glob.glob(site.getsitepackages()[0] + '/nvidia/*/lib')))" 2>/dev/null || true)
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/miniconda3/envs/main/lib:${NV_LIBS}:${LD_LIBRARY_PATH:-}
EOF

# Also register libraries with ldconfig so they are available system-wide
python -c "
import site, glob
paths = glob.glob(site.getsitepackages()[0] + '/nvidia/*/lib')
with open('/etc/ld.so.conf.d/nvidia-tf.conf', 'w') as f:
    f.write('\n'.join(paths) + '\n')
" && ldconfig 2>/dev/null || true
