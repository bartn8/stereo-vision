#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <smmintrin.h> // intrinsics
#include <emmintrin.h>

#include "StereoCommon.h"
#include "FastFilters.h"

extern "C" {
    static PyObject *_census5x5_SSE(PyObject *self, PyObject *args)
    {
        //void census5x5_SSE(uint8 *source, uint32 *dest, uint32 width, uint32 height);
        uint8 *source;
        uint32 *dest;
        uint32 width;
        uint32 height;

        uint8 *source_data;
        uint32 *dest_data;

        PyObject *_sourcearg=NULL, *_destarg=NULL;
        PyObject *_source=NULL, *_dest=NULL;

        if (!PyArg_ParseTuple(args, "O!O!II", &PyArray_Type, &_sourcearg,
         &PyArray_Type, &_destarg, &width, &height)) return NULL;

        if(width % 16 != 0){
            PyErr_Format(PyExc_TypeError,
                     "Width must be a multiple of 16 (%ldx%ld)", width, height);
            goto fail;
        }

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

        //Need another array because memory aligment in SSE is different.
        source_data = (uint8*)_mm_malloc(width*height*sizeof(uint8), 16);
        dest_data = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);

        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                source_data[y*width+x] = source[y*width+x];
            }
        }

        census5x5_SSE(source_data, dest_data, width, height);

        _mm_free(source_data);
        
        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                dest[y*width+x] = dest_data[y*width+x];
            }
        }
        
        _mm_free(dest_data);

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

        float32 *source_data;
        float32 *dest_data;

        PyObject *_sourcearg=NULL, *_destarg=NULL;
        PyObject *_source=NULL, *_dest=NULL;

        if (!PyArg_ParseTuple(args, "O!O!II", &PyArray_Type, &_sourcearg,
            &PyArray_Type, &_destarg, &width, &height)) return NULL;

        if(width % 16 != 0){
            PyErr_Format(PyExc_TypeError,
                     "Width must be a multiple of 16 (%ldx%ld)", width, height);
            goto fail;
        }

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

        //Need another array because memory aligment in SSE is different.
        source_data = (float32*)_mm_malloc(width*height*sizeof(float32), 16);
        dest_data = (float32*)_mm_malloc(width*height*sizeof(float32), 16);

        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                source_data[y*width+x] = source[y*width+x];
            }
        }

        median3x3_SSE(source_data, dest_data, width, height);

        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                dest[y*width+x] = dest_data[y*width+x];
            }
        }
        
        _mm_free(dest_data);

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

