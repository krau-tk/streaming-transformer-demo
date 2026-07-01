source /nfs/asr/anaconda3/etc/profile.d/conda.sh
conda activate /nfs/asr/envs/k2
cd /nfs/bichunhao/streaming-transformer-demo
python server.py
然后浏览器打开 http://<节点IP>:8765