/*
    Author: David Futschik
    Provided as part of the Chunkmogrify project, 2021.
*/

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <tuple>
#include <string.h>
#include <stdint.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>


static PyObject* canvas_to_masks(PyObject* self, PyObject* args, PyObject* kwargs) {

    const char* kwarg_names[] = { "canvas", "colors", "output_buffer", NULL };

    PyArrayObject* canvas = nullptr;
    PyArrayObject* colors = nullptr;
    PyArrayObject* output = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|$O!", const_cast<char**>(kwarg_names), 
            &PyArray_Type, &canvas, 
            &PyArray_Type, &colors,
            &PyArray_Type, &output)) {
        return NULL;
    }

    if (PyArray_NDIM(canvas) != 3 and PyArray_NDIM(colors) != 2) {
        PyErr_SetString(PyExc_ValueError, "a.ndim must be 3 and b.ndim must be 2.");
        return NULL;
    }

    int h, w, c, num_color;
    h = PyArray_DIM(canvas, 0);
    w = PyArray_DIM(canvas, 1);
    c = PyArray_DIM(canvas, 2);
    num_color = PyArray_DIM(colors, 0);

    int dtype_a, dtype_b;
    dtype_a = PyArray_TYPE(canvas);
    dtype_b = PyArray_TYPE(colors);

    if (c != PyArray_DIM(colors, 1)) {
        PyErr_SetString(PyExc_ValueError, "a.shape[2] != b.shape[1]");
        return NULL;
    }

    if (dtype_a != dtype_b or dtype_a != NPY_UINT8) {
        PyErr_SetString(PyExc_ValueError, "dtype of both arrays must be uint8.");
        return NULL;
    }

    bool output_wrong_array = [output, h, w, num_color]() {
        if (not output) return true;
        if (PyArray_TYPE(output) != NPY_FLOAT) return true;
        if (PyArray_NDIM(output) != 3) return true;        
        if (PyArray_DIM(output, 0) != h) return true;
        if (PyArray_DIM(output, 1) != w) return true;
        if (PyArray_DIM(output, 2) != num_color) return true;
        return false;
    }();

    if (!output or output_wrong_array) {
        // alloc new
        auto descr = PyArray_DescrFromType(NPY_FLOAT);
        npy_intp dims[] = { h, w, num_color };
        output = (PyArrayObject*)PyArray_NewFromDescr(&PyArray_Type, descr, /* nd */ 3, dims, NULL, NULL, 0, NULL);
        // std::cout << "Sideeffect" << std::endl;
    } else {
        // incref of output
        Py_INCREF(output);
    }

    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        #pragma omp parallel for
        for (int x = 0; x < w; x++) {
            uint8_t* px = (uint8_t*)PyArray_GETPTR2(canvas, y, x);
            int same_idx = 0;
            #pragma unroll(12)
            for (int k = 0; k < num_color; k++) {
                uint8_t* kolor = (uint8_t*)PyArray_GETPTR1(colors, k);
                #pragma unroll(4)
                for (int it = 0; it < c; it++) {
                    if (kolor[it] != px[it]) {
                        goto cont;
                    }
                }
                same_idx = k;
                goto brk;
                cont: ;
            }
            brk:
            float* o = (float*)PyArray_GETPTR2(output, y, x);
            for (int it = 0; it < num_color; it++) {
                // set the channel of the correct mask
                o[it] = it == same_idx ? 1. : 0.;
            }
        }
    }

    return (PyObject*)output;
}


static PyMethodDef python_methods[] = {
    { "canvas_to_masks", (PyCFunction)canvas_to_masks, METH_VARARGS|METH_KEYWORDS, "convert canvas colors to mask stack" },
    { nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef python_module = {
    PyModuleDef_HEAD_INIT,
    "_C_canvas", // _C_canvas.
    nullptr, // documentation
    -1,
    python_methods
};

PyMODINIT_FUNC
PyInit__C_canvas(void) {
    auto x = PyModule_Create(&python_module);
    import_array();
    return x;
}