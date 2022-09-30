# singularity shell -B /lab_data,/user_data --nv ~/container_images/pytorch_gpu.simg
source ~/.bashrc
conda activate conda-env
cd ~/NSD
source nsdvenv/bin/activate