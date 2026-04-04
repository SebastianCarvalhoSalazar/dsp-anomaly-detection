"""pytest configuration — must run before any library imports.

On macOS, OpenCV and PyTorch both link against the Accelerate/OpenMP framework.
When loaded together in the same process, native thread pools can conflict,
causing segmentation faults. Setting OMP_NUM_THREADS=1 before any import
serializes OpenMP and prevents the crash.
"""
import os

# Must be set before importing cv2, torch, or numpy with BLAS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Disable OpenCV's internal OpenCL which can conflict with PyTorch on macOS
os.environ.setdefault("OPENCV_OPENCL_DEVICE", "disabled")
