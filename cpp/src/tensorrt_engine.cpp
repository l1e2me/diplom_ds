// tensorrt_engine.cpp
#include "tensorrt_engine.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>

// Класс для логирования сообщений от TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

TensorRTEngine::TensorRTEngine(const std::string& engine_path) {
    // 1. Загружаем сериализованный engine из файла
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Не удалось открыть файл engine: " + engine_path);
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(file_size);
    file.read(engine_data.data(), file_size);
    file.close();

    // 2. Создаем Runtime, который умеет загружать engine'ы
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));

    // 3. Десериализуем engine из данных файла
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engine_data.data(), file_size)
    );
    if (!engine_) {
        throw std::runtime_error("Не удалось создать ICudaEngine.");
    }

    // 4. Создаем контекст выполнения (нужен для каждого потока)
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());

    // 5. Находим индексы и размеры входного и выходного тензоров
    input_index_ = engine_->getBindingIndex("input"); // Имя должно совпадать с именем при экспорте
    output_index_ = engine_->getBindingIndex("output");
    if (input_index_ == -1 || output_index_ == -1) {
        throw std::runtime_error("Не удалось найти входной/выходной тензор.");
    }

    // Получаем размеры
    nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_index_);
    nvinfer1::Dims output_dims = engine_->getBindingDimensions(output_index_);
    buffer_sizes_[input_index_] = input_dims.d[0] * input_dims.d[1] * input_dims.d[2] * sizeof(float);
    buffer_sizes_[output_index_] = output_dims.d[0] * output_dims.d[1] * output_dims.d[2] * sizeof(float);

    // 6. Выделяем память на GPU для входных и выходных данных
    cudaMalloc(&device_buffers_[input_index_], buffer_sizes_[input_index_]);
    cudaMalloc(&device_buffers_[output_index_], buffer_sizes_[output_index_]);
}

std::vector<float> TensorRTEngine::infer(const std::vector<float>& input_data) {
    // 1. Копируем входные данные с CPU на GPU
    cudaMemcpy(device_buffers_[input_index_], input_data.data(), buffer_sizes_[input_index_], cudaMemcpyHostToDevice);

    // 2. Запускаем инференс
    bool success = context_->enqueueV2(device_buffers_, nullptr, nullptr);
    if (!success) {
        throw std::runtime_error("Ошибка при выполнении инференса (enqueueV2).");
    }

    // 3. Копируем результат обратно с GPU на CPU
    std::vector<float> output_data(buffer_sizes_[output_index_] / sizeof(float));
    cudaMemcpy(output_data.data(), device_buffers_[output_index_], buffer_sizes_[output_index_], cudaMemcpyDeviceToHost);

    return output_data;
}

TensorRTEngine::~TensorRTEngine() {
    // Освобождаем память на GPU
    cudaFree(device_buffers_[input_index_]);
    cudaFree(device_buffers_[output_index_]);
}