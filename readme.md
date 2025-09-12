# Environment Setup：
```bash
conda create --name UniScore python=3.9 
conda activate UniScore
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/openai/CLIP.git
pip install openai
pip install image-reward
pip install transformers==4.27.4
pip install diffusers==0.16.0
pip install huggingface_hub==0.24.0
pip install hpsv2
cp ./bpe_simple_vocab_16e6.txt.gz /home/jinhao/anaconda3/envs/UniScore/lib/python3.9/site-packages/hpsv2/src/open_clip/bpe_simple_vocab_16e6.txt.gz
```