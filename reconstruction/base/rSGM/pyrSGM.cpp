#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <smmintrin.h> // intrinsics
#include <emmintrin.h>

#include "StereoCommon.h"
#include "FastFilters.h"
#include "StereoSGM.h"
#include "StereoBMHelper.h"

extern "C" {
    static PyObject *_census5x5_SSE(PyObject *self, PyObject *args)
    {
        //void census5x5_SSE(uint8 *source, uint32 *dest, uint32 width, uint32 height);
        uint8 *source;
        uint32 *dest;
        uint32 width;
        uint32 height;

        uint8 *source_mm;
        uint32 *dest_mm;

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
        source_mm = (uint8*)_mm_malloc(width*height*sizeof(uint8), 16);
        dest_mm = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);

        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                source_mm[y*width+x] = source[y*width+x];
            }
        }

        census5x5_SSE(source_mm, dest_mm, width, height);

        _mm_free(source_mm);
        
        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                dest[y*width+x] = dest_mm[y*width+x];
            }
        }
        
        _mm_free(dest_mm);

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

        float32 *source_mm;
        float32 *dest_mm;

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
        source_mm = (float32*)_mm_malloc(width*height*sizeof(float32), 16);
        dest_mm = (float32*)_mm_malloc(width*height*sizeof(float32), 16);

        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                source_mm[y*width+x] = source[y*width+x];
            }
        }

        median3x3_SSE(source_mm, dest_mm, width, height);

        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                dest[y*width+x] = dest_mm[y*width+x];
            }
        }
        
        _mm_free(dest_mm);

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

    static PyObject *_costMeasureCensus5x5_xyd_SSE(PyObject *self, PyObject *args)
    {
        //void costMeasureCensus5x5_xyd_SSE(uint32* leftcensus, uint32* rightcensus, 
        //const sint32 height, const sint32 width, const sint32 dispCount, const uint16 invalidDispValue, uint16* dsi, sint32 numThreads);

        uint32 *leftCensus;
        uint32 *rightCensus;
        uint32 width;// % 16
        uint32 height;
        uint32 dispCount;// % 8, <= 256
        const uint16 invalidDispValue = 12;// init value for invalid disparities (half of max value seems ok) (max 24 because Hamming distance for 24 bits)
        uint16* dsi;
        uint32 numThreads;//1 or 2 or 4 threads

        uint32 *leftCensus_mm;
        uint32 *rightCensus_mm;
        uint16* dsi_mm;

        PyObject *_leftCensusarg=NULL, *_rightCensusarg=NULL, *_dsiarg=NULL;
        PyObject *_leftCensus=NULL, *_rightCensus=NULL, *_dsi=NULL;

        //Left Right DSI width height dispRange numThreads
        if (!PyArg_ParseTuple(args, "O!O!O!IIII", &PyArray_Type, &_leftCensusarg,
         &PyArray_Type, &_rightCensusarg, &PyArray_Type, &_dsiarg, &width, &height,
         &dispCount,&numThreads)) return NULL;

        if(width % 16 != 0){
            PyErr_Format(PyExc_TypeError,
                     "Width must be a multiple of 16 (%ldx%ld)", width, height);
            goto fail;
        }

        if(dispCount % 8 != 0 || dispCount > 256){
            PyErr_Format(PyExc_TypeError,
                     "Disparity range must be a multiple of 8 and not greater than 256 (%ld)", dispCount);
            goto fail;
        }

        if(numThreads != 1 && numThreads != 2 && numThreads != 4){
            PyErr_Format(PyExc_TypeError,
                     "NumThreads must be 1,2,4 (%ld)", numThreads);
            goto fail;
        }

        _leftCensus = PyArray_FROM_OTF(_leftCensusarg, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
        if (_leftCensus == NULL) return NULL;

        _rightCensus = PyArray_FROM_OTF(_rightCensusarg, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
        if (_rightCensus == NULL) goto fail;

        #if NPY_API_VERSION >= 0x0000000c
            _dsi = PyArray_FROM_OTF(_dsiarg, NPY_UINT16, NPY_ARRAY_INOUT_ARRAY2);
        #else
            _dsi = PyArray_FROM_OTF(_dsiarg, NPY_UINT16, NPY_ARRAY_INOUT_ARRAY);
        #endif

        if (_dsi == NULL) goto fail;

        leftCensus = (uint32*) PyArray_DATA(_leftCensus);
        rightCensus = (uint32*) PyArray_DATA(_rightCensus);
        dsi = (uint16*) PyArray_DATA(_dsi);

        //Need another array because memory aligment in SSE is different.
        leftCensus_mm = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);
        rightCensus_mm = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);
        dsi_mm = (uint16*)_mm_malloc(width*height*(dispCount)*sizeof(uint16), 32);

        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                leftCensus_mm[y*width+x] = leftCensus[y*width+x];
                rightCensus_mm[y*width+x] = rightCensus[y*width+x];
            }
        }

        costMeasureCensus5x5_xyd_SSE(leftCensus_mm, rightCensus_mm, (sint32)height, (sint32)width, 
        (sint32)dispCount, invalidDispValue, dsi_mm, (sint32)numThreads);

        _mm_free(leftCensus_mm);
        _mm_free(rightCensus_mm);
        
        for(uint32 d = 0; d < dispCount; d++){
            for(uint32 y = 0; y < height; y++){
                for(uint32 x = 0; x < width; x++){
                    //TODO: testing
                    dsi[d*width*height+y*width+x] = dsi_mm[d*width*height+y*width+x];
                }
            }
        }
        
        _mm_free(dsi_mm);

        Py_DECREF(_leftCensus);
        Py_DECREF(_rightCensus);
        
        #if NPY_API_VERSION >= 0x0000000c
            PyArray_ResolveWritebackIfCopy((PyArrayObject*)_dsi);
        #endif
        
        Py_DECREF(_dsi);
        Py_INCREF(Py_None);
        return Py_None;

        fail: 

        Py_DECREF(_leftCensus);
        Py_XDECREF(_rightCensus);

        #if NPY_API_VERSION >= 0x0000000c
            PyArray_DiscardWritebackIfCopy((PyArrayObject*)_dsi);
        #endif

        Py_XDECREF(_dsi);

        return NULL;        
    }

    static PyObject *_matchWTA_SSE(PyObject *self, PyObject *args)
    {
        //void matchWTA_SSE(float32* dispImg, uint16* &dsiAgg, const sint32 width, const sint32 height, 
        //const sint32 maxDisp, const float32 uniqueness);

        float32 *dispImg;//Out
        uint16* dsiAgg;//In
        uint32 width;// % 16
        uint32 height;
        uint32 dispCount;// % 8, <= 256
        float32 uniqueness;
        
        float32 *dispImg_mm;
        uint16* dsiAgg_mm;

        PyObject *_dispImgarg=NULL, *_dsiAggarg=NULL;
        PyObject *_dispImg=NULL, *_dsiAgg=NULL;

        //Disp DSI width height dispRange numThreads
        if (!PyArg_ParseTuple(args, "O!O!IIIf", &PyArray_Type, &_dispImgarg,
         &PyArray_Type, &_dsiAggarg, &width, &height, &dispCount, &uniqueness)) return NULL;

        if(width % 16 != 0){
            PyErr_Format(PyExc_TypeError,
                     "Width must be a multiple of 16 (%ldx%ld)", width, height);
            goto fail;
        }

        if(dispCount % 8 != 0 || dispCount > 256){
            PyErr_Format(PyExc_TypeError,
                     "Disparity range must be a multiple of 8 and not greater than 256 (%ld)", dispCount);
            goto fail;
        }

        if(uniqueness > 1.0f || uniqueness <= 0.0f){
            PyErr_Format(PyExc_TypeError,
                     "Uniqueness must be inside ]0,1] (%f)", uniqueness);
            goto fail;
        }

        _dispImg = PyArray_FROM_OTF(_dispImgarg, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
        if (_dispImg == NULL) return NULL;

        #if NPY_API_VERSION >= 0x0000000c
            _dsiAgg = PyArray_FROM_OTF(_dsiAggarg, NPY_UINT16, NPY_ARRAY_INOUT_ARRAY2);
        #else
            _dsiAgg = PyArray_FROM_OTF(_dsiAggarg, NPY_UINT16, NPY_ARRAY_INOUT_ARRAY);
        #endif

        if (_dsiAgg == NULL) goto fail;

        dispImg = (float32*) PyArray_DATA(_dispImg);
        dsiAgg = (uint16*) PyArray_DATA(_dsiAgg);

        //Need another array because memory aligment in SSE is different.
        dispImg_mm = (float32*)_mm_malloc(width*height*sizeof(float32), 16);
        dsiAgg_mm = (uint16*)_mm_malloc(width*height*(dispCount)*sizeof(uint16), 16);

        for(uint32 d = 0; d < dispCount; d++){
            for(uint32 y = 0; y < height; y++){
                for(uint32 x = 0; x < width; x++){
                    //TODO: testing
                    dsiAgg_mm[d*width*height+y*width+x] = dsiAgg[d*width*height+y*width+x];
                }
            }
        }

        matchWTA_SSE(dispImg_mm, dsiAgg_mm, (sint32)width, (sint32)height, 
        (sint32) (dispCount-1), uniqueness);

        _mm_free(dsiAgg_mm);
                
        for(uint32 y = 0; y < height; y++){
            for(uint32 x = 0; x < width; x++){
                dispImg[y*width+x] = dispImg_mm[y*width+x];
            }
        }
        
        _mm_free(dispImg_mm);

        Py_DECREF(_dsiAgg);
        
        #if NPY_API_VERSION >= 0x0000000c
            PyArray_ResolveWritebackIfCopy((PyArrayObject*)_dispImg);
        #endif
        
        Py_DECREF(_dispImg);
        Py_INCREF(Py_None);
        return Py_None;

        fail:

        Py_DECREF(_dsiAgg);

        #if NPY_API_VERSION >= 0x0000000c
            PyArray_DiscardWritebackIfCopy((PyArrayObject*)_dispImg);
        #endif

        Py_XDECREF(_dispImg);

        return NULL;      
    }

    static PyObject *_matchWTAAndSubPixel_SSE(PyObject *self, PyObject *args)
    {
    }

    static PyObject *_doLRCheck(PyObject *self, PyObject *args)
    {
    }

    static PyObject *_subPixelRefine(PyObject *self, PyObject *args)
    {
    }

    static PyObject *_aggregate_SSE(PyObject *self, PyObject *args)
    {
    }

    static PyObject *_process(PyObject *self, PyObject *args)
    {
    }  

    static PyObject *_processParallel(PyObject *self, PyObject *args)
    {
    }     

    static PyMethodDef rSGMMethods[] = {
        {"census5x5_SSE", _census5x5_SSE, METH_VARARGS, "Census 5x5 with SSE optimization"},
        {"median3x3_SSE", _median3x3_SSE, METH_VARARGS, "Median 3x3 with SSE optimization"},
        {"costMeasureCensus5x5_xyd_SSE", _costMeasureCensus5x5_xyd_SSE, METH_VARARGS, "Cost Measure (Hamming Distance) from census 5x5 with SSE optimization"},
        {"matchWTA_SSE", _matchWTA_SSE, METH_VARARGS, "Winner takes all with SSE optimization"},
        {"matchWTAAndSubPixel_SSE", _matchWTAAndSubPixel_SSE, METH_VARARGS, "Winner takes all and sub pixel refinement with SSE optimization"},
        {"doLRCheck", _doLRCheck, METH_VARARGS, "Left Right Consistency check"},
        {"_subPixelRefine", _subPixelRefine, METH_VARARGS, "Disparity sub pixel refinement"},
        {"aggregate_SSE", _aggregate_SSE, METH_VARARGS, "SGM Cost Aggregation with SSE optimization"},
        {"process", _process, METH_VARARGS, "Full rSGM pipeline"},
        {"processParallel", _processParallel, METH_VARARGS, "Full rSGM pipeline with threads"},
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

