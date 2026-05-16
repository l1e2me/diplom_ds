// py_inference_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensorrt_engine.h"

namespace py = pybind11;

// Модуль, который будет виден из Python
PYBIND11_MODULE(cpp_inference, m) {
    m.doc() = "C++ оптимизаторы для инференса";

    // Привязываем наш класс TensorRTEngine
    py::class_<TensorRTEngine>(m, "TensorRTEngine")
        .def(py::init<const std::string&>()) // Конструктор принимает путь к engine
        .def("infer", &TensorRTEngine::infer); // Метод infer
}