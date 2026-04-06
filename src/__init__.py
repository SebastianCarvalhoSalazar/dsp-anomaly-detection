"""Package-level initialisation — must execute before any heavy imports.

On macOS, PyTorch, OpenCV, numpy, and scipy can each link their own copy
of libomp.  When two copies initialise in the same process the OpenMP
runtime aborts with *"Error #15: … found libomp.dylib already
initialized"*.  Setting these variables **before** any of those libraries
are imported tells each runtime to share a single thread pool (size 1)
and, as a last resort, ``KMP_DUPLICATE_LIB_OK`` suppresses the fatal
error if two copies still end up loaded.
"""
import os as _os

_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("OPENCV_OPENCL_DEVICE", "disabled")
