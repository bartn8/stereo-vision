from setuptools.command.build_ext import build_ext
from setuptools import setup, Extension
import sysconfig
import subprocess

class custom_build_ext(build_ext):
    def build_extensions(self):
        # Override the compiler executables. Importantly, this
        # removes the "default" compiler flags that would
        # otherwise get passed on to to the compiler, i.e.,
        # distutils.sysconfig.get_var("CFLAGS").
        self.compiler.set_executable("compiler_so", "g++")
        self.compiler.set_executable("compiler_cxx", "g++")
        self.compiler.set_executable("linker_so", "g++")
        build_ext.build_extensions(self)

bcmd = "pkg-config --cflags --libs opencv4"
process = subprocess.Popen(bcmd.split(), stdout=subprocess.PIPE)
extra_opencv,_ = process.communicate()
extra_opencv = extra_opencv.decode("utf-8")
extra_opencv = extra_opencv.split()

# Common flags for both release and debug builds.
extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-I.", "-msse4.1", "-msse4.2", "-O3", "-ffast-math", "-march=native", "-fopenmp", "-Wno-write-strings", "-fpermissive"]
extra_compile_args += ["-std=c++11"]
extra_compile_args += extra_opencv

#extra_link_args = "-pthread -fopenmp -B /home/luca/miniconda3/envs/cvlab/compiler_compat -shared -Wl,-rpath,/home/luca/miniconda3/envs/cvlab/lib -Wl,-rpath-link,/home/luca/miniconda3/envs/cvlab/lib -L/home/luca/miniconda3/envs/cvlab/lib -L/home/luca/miniconda3/envs/cvlab/lib -Wl,-rpath,/home/luca/miniconda3/envs/cvlab/lib -Wl,-rpath-link,/home/luca/miniconda3/envs/cvlab/lib -L/home/luca/miniconda3/envs/cvlab/lib".split()
extra_link_args = "-pthread -fopenmp -shared -B /home/luca/miniconda3/compiler_compat -L/home/luca/miniconda3/lib -Wl,-rpath=/home/luca/miniconda3/lib -Wl,--no-as-needed".split()
module_pyrSGM = Extension('pyrSGM',
                    sources = ['pyrSGM.cpp', 'FastFilters.cpp', 'StereoBMHelper.cpp'],
                    language='c++11',
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args
                    )

setup(
  name = 'pyrSGM',
  ext_modules = [module_pyrSGM],
  cmdclass={"build_ext": custom_build_ext}
)
