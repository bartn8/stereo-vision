#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include "StereoCommon.h"
#include "FastFilters.h"

#include <stdio.h>

extern "C" {
    static PyObject *_census5x5_SSE(PyObject *self, PyObject *args)
    {
        //void census5x5_SSE(uint8 *source, uint32 *dest, uint32 width, uint32 height);
        uint8 *source;
        uint32 *dest;
        uint32 width;
        uint32 height;

        PyObject *_sourcearg=NULL, *_destarg=NULL;
        PyObject *_source=NULL, *_dest=NULL;

        if (!PyArg_ParseTuple(args, "O!O!II", &PyArray_Type, &_sourcearg, &PyArray_Type, &_destarg, &width, &height)) return NULL;

        _source = PyArray_FROM_OTF(_sourcearg, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
        if (_source == NULL) return NULL;

        #if NPY_API_VERSION >= 0x0000000c
            _dest = PyArray_FROM_OTF(_destarg, NPY_UINT, NPY_ARRAY_INOUT_ARRAY2);
        #else
            _dest = PyArray_FROM_OTF(_destarg, NPY_UINT, NPY_ARRAY_INOUT_ARRAY);
        #endif

        if (_dest == NULL) goto fail;

        source = (uint8*) PyArray_DATA(_source);
        dest = (uint32*) PyArray_DATA(_dest);

        printf("Census 5x5 init %dx%d\n", height, width);

        census5x5_SSE(source, dest, width, height);

        printf("Census 5x5 finished\n");

        Py_DECREF(_source);
        
        #if NPY_API_VERSION >= 0x0000000c
            PyArray_ResolveWritebackIfCopy((PyArrayObject*)_dest);
        #endif
        
        Py_DECREF(_dest);
        Py_INCREF(Py_None);
        return Py_None;

        fail:

        Py_XDECREF(_source);

        #if NPY_API_VERSION >= 0x0000000c
        PyArray_DiscardWritebackIfCopy((PyArrayObject*)_dest);
        #endif

        Py_XDECREF(_dest);

        return NULL;
    }

    static PyObject *_median3x3_SSE(PyObject *self, PyObject *args)
    {
        //void median3x3_SSE(float32* source, float32* dest, uint32 width, uint32 height);
        float32 *source;
        float32 *dest;
        uint32 width;
        uint32 height;

        PyObject *_sourcearg=NULL, *_destarg=NULL;
        PyObject *_source=NULL, *_dest=NULL;

        if (!PyArg_ParseTuple(args, "O!O!II", &PyArray_Type, &_sourcearg,
            &PyArray_Type, &_destarg, &width, &height)) return NULL;

        _source = PyArray_FROM_OTF(_sourcearg, NPY_CFLOAT, NPY_ARRAY_IN_ARRAY);
        if (_source == NULL) return NULL;

        #if NPY_API_VERSION >= 0x0000000c
            _dest = PyArray_FROM_OTF(_destarg, NPY_CFLOAT, NPY_ARRAY_INOUT_ARRAY2);
        #else
            _dest = PyArray_FROM_OTF(_destarg, NPY_CFLOAT, NPY_ARRAY_INOUT_ARRAY);
        #endif

        if (_dest == NULL) goto fail;

        source = (float32*) PyArray_DATA(_source);
        dest = (float32*) PyArray_DATA(_dest);

        median3x3_SSE(source, dest, width, height);

        Py_DECREF(_source);
        
        #if NPY_API_VERSION >= 0x0000000c
            PyArray_ResolveWritebackIfCopy((PyArrayObject*)_dest);
        #endif
        
        Py_DECREF(_dest);
        Py_INCREF(Py_None);
        return Py_None;

        fail:

        Py_XDECREF(_source);
        
        #if NPY_API_VERSION >= 0x0000000c
        PyArray_DiscardWritebackIfCopy((PyArrayObject*)_dest);
        #endif

        Py_XDECREF(_dest);

        return NULL;
    }


    static PyMethodDef rSGMMethods[] = {
        {"census5x5_SSE", _census5x5_SSE, METH_VARARGS, "Census 5x5 with SSE optimization"},
        {"median3x3_SSE", _median3x3_SSE, METH_VARARGS, "Median 3x3 with SSE optimization"},
        {NULL, NULL, 0, NULL}
    };


    static struct PyModuleDef pyrSGMmodule = {
        PyModuleDef_HEAD_INIT,
        "pyrSGM",
        "rSGM library",
        -1,
        rSGMMethods
    };

    PyMODINIT_FUNC PyInit_pyrSGM(void) {
        import_array();
        return PyModule_Create(&pyrSGMmodule);
    }
}

