pip install -r requirements.txt
set FORCE_CPU=1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate numpy psutil tqdm

pip install C:\TensorRT-10.9.0.34\python\tensorrt-10.9.0.34-cp311-none-win_amd64.whl

pip install pybind11 onnx

mkdir cpp/build
cd cpp/build
cmake --build . --config Release

trtexec --onnx=path/to/your_model.onnx --saveEngine=models/my_model_fp16.engine --fp16

python -m scripts.run_benchmark