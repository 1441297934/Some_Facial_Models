from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="blinkcheck",
        sources=["blinkcheck.pyx"],
        include_dirs=[np.get_include()],
        libraries=["opencv_core", "opencv_imgproc", "opencv_highgui", "opencv_videoio"],
        library_dirs=["D:/opencv/build/x64/vc16/lib"],  # 修改为你的OpenCV库路径
        language="c++",
    )
]

setup(
    name="blink_detection",
    ext_modules=cythonize(extensions),
)