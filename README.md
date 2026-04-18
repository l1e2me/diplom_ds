pip install -r requirements.txt

set FORCE_CPU=1

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install transformers accelerate numpy psutil tqdm

python -m scripts.run_benchmark
