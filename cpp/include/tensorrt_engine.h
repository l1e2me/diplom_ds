// tensorrt_engine.h
#pragma once
#include <string>
#include <memory>
#include <vector>

// Подключаем заголовки TensorRT
#include <NvInfer.h>

/**
 * @brief Класс для инференса ONNX моделей с помощью TensorRT.
 */
class TensorRTEngine {
public:
    /**
     * @brief Конструктор, который загружает и десериализует TensorRT engine.
     * @param engine_path Путь к .engine файлу на диске.
     */
    TensorRTEngine(const std::string& engine_path);

    /**
     * @brief Выполнить инференс.
     * @param input_data Вектор входных данных (float).
     * @return Выходные данные в виде вектора float.
     */
    std::vector<float> infer(const std::vector<float>& input_data);

    // Деструктор для освобождения ресурсов CUDA
    ~TensorRTEngine();

private:
    // Умные указатели для автоматического управления памятью объектов TensorRT
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Указатели на буферы в памяти GPU
    void* device_buffers_[2]; 
    size_t buffer_sizes_[2];   
    int input_index_;          
    int output_index_;         

    // Вспомогательная функция для обработки ошибок CUDA
    void checkCudaError(cudaError_t err, const std::string& msg);
};