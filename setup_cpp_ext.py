#
#    Author: David Futschik
#    Provided as part of the Chunkmogrify project, 2021.
#

import platform
import setuptools
import numpy as np
from setuptools import sandbox

platform_specific_flags = []
if platform.system() == "Windows":
    platform_specific_flags += ["/permissive-", "/Ox", "/std:c++11"]
else:
    platform_specific_flags += ["-O3", "--std=c++11"]

ext_modules = [
    setuptools.Extension('_C_canvas',
            sources=['extensions/canvas_to_masks.cpp'],
            include_dirs=[np.get_include()],
            extra_compile_args=platform_specific_flags,
            language='c++'),
    setuptools.Extension('_C_heatmap',
            sources=['extensions/heatmap.cpp'],
            include_dirs=[np.get_include()],
            extra_compile_args=platform_specific_flags,
            language='c++')
]

def checked_build(force=False):
    def do_build():
        sandbox.run_setup('setup_cpp_ext.py', ['build_ext', '--inplace'])
    try:
        import _C_canvas
        import _C_heatmap
        if force: do_build()
    except ImportError:
        do_build()

if __name__ == "__main__":
    setuptools.setup(
        ext_modules=ext_modules
    )