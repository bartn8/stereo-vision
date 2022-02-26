from setuptools import setup, Extension
import sysconfig

# Common flags for both release and debug builds.
extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-I.", "-msse4.1", "-msse4.2", "-O3", "-ffast-math", "-march=native", "-fopenmp", "-Wno-write-strings"]

module_pyrSGM = Extension('pyrSGM',
                    sources = ['pyrSGM.cpp', 'FastFilters.cpp'],
                    language='c++11',
                    extra_compile_args=extra_compile_args,
                    )

setup(
  name = 'pyrSGM',
  ext_modules = [module_pyrSGM],
)
