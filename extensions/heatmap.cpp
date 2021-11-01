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


static PyObject* heatmap(PyObject* self, PyObject* args, PyObject* kwargs) {

    const char* kwarg_names[] = { "values", "vmin", "vmax", NULL };

    PyArrayObject* canvas = nullptr;
    double vmin = 0;
    double vmax = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!dd|", const_cast<char**>(kwarg_names), &PyArray_Type, &canvas, &vmin, &vmax)) {
        return NULL;
    }

    if (PyArray_NDIM(canvas) != 3 and PyArray_NDIM(canvas) != 2) {
        PyErr_SetString(PyExc_ValueError, "a.ndim must be 3 or 2.");
        return NULL;
    }

    int has_c = PyArray_NDIM(canvas) > 2;

    int h, w, c;
    h = PyArray_DIM(canvas, 0);
    w = PyArray_DIM(canvas, 1);
    if (has_c) {
        c = PyArray_DIM(canvas, 2);
        if (c != 1) {
            PyErr_SetString(PyExc_ValueError, "3rd dimensions must be 1 or None");
            return NULL;
        }
    }
    else {
        c = 1;
    }

    int dtype_a = PyArray_TYPE(canvas);

    if (dtype_a != NPY_FLOAT) {
        PyErr_SetString(PyExc_ValueError, "dtype of array must be float.");
        return NULL;
    }

    auto descr = PyArray_DescrFromType(NPY_UINT8);
    npy_intp dims[] = { h, w, 3 };
    PyArrayObject* output = (PyArrayObject*)PyArray_NewFromDescr(&PyArray_Type, descr, /* nd */ 3, dims, NULL, NULL, 0, NULL);

    double dv = vmax - vmin;

    #pragma omp parallel for
    for (int y = 0; y < h; y++) {
        #pragma omp parallel for
        for (int x = 0; x < w; x++) {
            float* pxval = (float*)PyArray_GETPTR2(canvas, y, x);
            float v = *pxval;
            float r(1), g(1), b(1);

            if (v < (vmin + .25 * dv)) {
                r = 0;
                g = 4 * (v - vmin) / dv;
            } else if (v < (vmin + .5 * dv)) {
                r = 0;
                b = 1 + 4 * (vmin + .25 * dv - v) / dv;
            } else if (v < (vmin + .75 * dv)) {
                r = 4 * (v - vmin - .5 * dv) / dv;
                b = 0;
            } else {
                g = 1 + 4 * (vmin + .75 * dv - v) / dv;
                b = 0;
            }

            *(uint8_t*)PyArray_GETPTR3(output, y, x, 0) = r * 255;
            *(uint8_t*)PyArray_GETPTR3(output, y, x, 1) = g * 255;
            *(uint8_t*)PyArray_GETPTR3(output, y, x, 2) = b * 255;
        }
    }

    return (PyObject*)output;
}


static PyMethodDef python_methods[] = {
    { "heatmap", (PyCFunction)heatmap, METH_VARARGS|METH_KEYWORDS, "convert data into heatmap" },
    { nullptr, nullptr, 0, nullptr }
};

static struct PyModuleDef python_module = {
    PyModuleDef_HEAD_INIT,
    "_C_heatmap", // _C_canvas.
    nullptr, // documentation
    -1,
    python_methods
};

PyMODINIT_FUNC
PyInit__C_heatmap(void) {
    auto x = PyModule_Create(&python_module);
    import_array();
    return x;
}