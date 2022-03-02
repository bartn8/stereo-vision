from setuptools import setup, Extension
import sysconfig

# Common flags for both release and debug builds.
extra_compile_args = ["-I.", "-msse4.1", "-msse4.2", "-O3", "-ffast-math", "-march=native", "-fopenmp", "-Wno-write-strings"]
extra_link_args = ["-fopenmp"]

module_pyrSGM = Extension('pyrSGM',
                    sources = ['pyrSGM.cpp', 'FastFilters.cpp', 'StereoBMHelper.cpp'],
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args
                    )

setup(
  name = 'pyrSGM',
  version = '1.0',
  ext_modules = [module_pyrSGM]
)
