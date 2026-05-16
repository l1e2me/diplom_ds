cmake_minimum_required(VERSION 3.15)
project(cpp_inference VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Находим установленные пакеты
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

# 2. Указываем путь к TensorRT (из переменной окружения, которую мы создали)
if(DEFINED ENV{TENSORRT_PATH})
    set(TENSORRT_PATH $ENV{TENSORRT_PATH})
    message(STATUS "Using TensorRT from: ${TENSORRT_PATH}")
else()
    message(FATAL_ERROR "Environment variable TENSORRT_PATH is not set.")
endif()

# 3. Добавляем заголовки TensorRT
include_directories(${TENSORRT_PATH}/include)

# 4. Создаем Python-модуль
pybind11_add_module(cpp_inference
    bindings/py_inference_bindings.cpp
    src/tensorrt_engine.cpp
)

# 5. Линкуем с библиотеками TensorRT и CUDA
target_link_libraries(cpp_inference PRIVATE
    ${CUDA_LIBRARIES}
    ${TENSORRT_PATH}/lib/nvinfer.lib
    ${TENSORRT_PATH}/lib/nvparsers.lib
    ${TENSORRT_PATH}/lib/nvinfer_plugin.lib
    cudart
)

# Для Windows: сообщаем, что это Python-модуль с расширением .pyd
set_target_properties(cpp_inference PROPERTIES SUFFIX ".pyd")