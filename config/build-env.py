#
#   Copyright 2012      NVIDIA Corporation
# 
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# 
#
EnsureSConsVersion(1,2)

import os

import inspect
import platform

def get_cuda_paths():
  """Determines CUDA {bin,lib,include} paths
  
  returns (bin_path,lib_path,inc_path)
  """

  # determine defaults
  if os.name == 'nt':
    bin_path = 'C:/CUDA/bin'
    lib_path = 'C:/CUDA/lib'
    inc_path = 'C:/CUDA/include'
  elif os.name == 'posix':
    bin_path = '/usr/local/cuda/bin'
    lib_path = '/usr/local/cuda/lib'
    inc_path = '/usr/local/cuda/include'
  else:
    raise ValueError, 'Error: unknown OS.  Where is nvcc installed?'
  
  if platform.system() != 'Darwin' and platform.machine()[-2:] == '64':
    lib_path += '64'

  # override with environement variables
  if 'CUDA_BIN_PATH' in os.environ:
    bin_path = os.path.abspath(os.environ['CUDA_BIN_PATH'])
  if 'CUDA_LIB_PATH' in os.environ:
    lib_path = os.path.abspath(os.environ['CUDA_LIB_PATH'])
  if 'CUDA_INC_PATH' in os.environ:
    inc_path = os.path.abspath(os.environ['CUDA_INC_PATH'])

  return (bin_path,lib_path,inc_path)


def getTools():
  result = []
  if os.name == 'nt':
    result = ['default', 'msvc']
  elif os.name == 'posix':
    result = ['default', 'gcc']
  else:
    result = ['default']
  return result;


OldEnvironment = Environment;


# this dictionary maps the name of a compiler program to a dictionary mapping the name of
# a compiler switch of interest to the specific switch implementing the feature
gCompilerOptions = {
    'gcc' : {'warn_all' : '-Wall', 'warn_errors' : '-Werror', 'optimization' : '-O2', 'debug' : '-g',  'exception_handling' : '',      'omp' : '-fopenmp'},
    'g++' : {'warn_all' : '-Wall', 'warn_errors' : '-Werror', 'optimization' : '-O2', 'debug' : '-g',  'exception_handling' : '',      'omp' : '-fopenmp'},
    'cl'  : {'warn_all' : '/Wall', 'warn_errors' : '/WX',     'optimization' : '/Ox', 'debug' : ['/Zi', '-D_DEBUG', '/MTd'], 'exception_handling' : '/EHsc', 'omp' : '/openmp'}
  }


# this dictionary maps the name of a linker program to a dictionary mapping the name of
# a linker switch of interest to the specific switch implementing the feature
gLinkerOptions = {
    'gcc' : {'debug' : ''},
    'g++' : {'debug' : ''},
    'link'  : {'debug' : '/debug'}
  }


def getCFLAGS(mode, backend, warn, warnings_as_errors, CC):
  result = []
  if mode == 'release':
    # turn on optimization
    result.append(gCompilerOptions[CC]['optimization'])
  elif mode == 'debug':
    # turn on debug mode
    result.append(gCompilerOptions[CC]['debug'])

  # generate omp code
  if backend == 'omp':
    result.append(gCompilerOptions[CC]['omp'])
  
  if warn:
    # turn on all warnings
    result.append(gCompilerOptions[CC]['warn_all'])

  if warnings_as_errors:
    # treat warnings as errors
    result.append(gCompilerOptions[CC]['warn_errors'])

  return result


def getCXXFLAGS(mode, backend, warn, warnings_as_errors, CXX):
  result = []
  if mode == 'release':
    # turn on optimization
    result.append(gCompilerOptions[CXX]['optimization'])
  elif mode == 'debug':
    # turn on debug mode
    result.append(gCompilerOptions[CXX]['debug'])
    result.append('-DTRACE')

  # enable exception handling
  result.append(gCompilerOptions[CXX]['exception_handling'])
  
  # generate omp code
  if backend == 'omp':
    result.append(gCompilerOptions[CXX]['omp'])

  if warn:
    # turn on all warnings
    result.append(gCompilerOptions[CXX]['warn_all'])

  if warnings_as_errors:
    # treat warnings as errors
    result.append(gCompilerOptions[CXX]['warn_errors'])

  return result


def getNVCCFLAGS(mode, backend, arch):
  result = ['-arch=' + arch]
  if mode == 'debug':
    # turn on debug mode
    # XXX make this work when we've debugged nvcc -G
    #result.append('-G')
    pass
  # force 64b code on darwin
  if platform.platform()[:6] == 'Darwin':
    result.append('-m64')

  return result


def getLINKFLAGS(mode, backend, LINK):
  result = []
  if mode == 'debug':
    # turn on debug mode
    result.append(gLinkerOptions[LINK]['debug'])

  # XXX make this portable
  if backend == 'ocelot':
    result.append(os.popen('OcelotConfig -l').read().split())

  return result


def Environment():
  # allow the user discretion to choose the MSVC version
  vars = Variables()
  if os.name == 'nt':
    vars.Add(EnumVariable('MSVC_VERSION', 'MS Visual C++ version', None, allowed_values=('8.0', '9.0', '10.0')))

  # add a variable to handle the device backend
  backend_variable = EnumVariable('backend', 'The parallel device backend to target', 'cuda',
                                  allowed_values = ('cuda', 'omp', 'ocelot'))
  vars.Add(backend_variable)

  # add a variable to handle RELEASE/DEBUG mode
  vars.Add(EnumVariable('mode', 'Release versus debug mode', 'release',
                        allowed_values = ('release', 'debug')))

  # add a variable to handle compute capability
  vars.Add(EnumVariable('arch', 'Compute capability code generation', 'sm_20',
                        allowed_values = ('sm_10', 'sm_11', 'sm_12', 'sm_13', 'sm_20', 'sm_21', 'sm_30', 'sm_35')))

  # add a variable to handle warnings
  if os.name == 'posix':
    vars.Add(BoolVariable('Wall', 'Enable all compilation warnings', 1))
  else:
    vars.Add(BoolVariable('Wall', 'Enable all compilation warnings', 0))

  # add a variable to treat warnings as errors
  vars.Add(BoolVariable('Werror', 'Treat warnings as errors', 0))

  # create an Environment, pull in user PATH
  env = OldEnvironment(tools = getTools(), variables = vars,
                       ENV={'PATH' : os.environ['PATH']})

  # get the absolute path to the directory containing
  # this source file
  thisFile = inspect.getabsfile(Environment)
  thisDir = os.path.dirname(thisFile)

  # enable nvcc
  env.Tool('nvcc', toolpath = [os.path.join(thisDir)])

  # get C compiler switches
  env.Append(CFLAGS = getCFLAGS(env['mode'], env['backend'], env['Wall'], env['Werror'], env.subst('$CC')))

  # get CXX compiler switches
  env.Append(CXXFLAGS = getCXXFLAGS(env['mode'], env['backend'], env['Wall'], env['Werror'], env.subst('$CXX')))

  # get NVCC compiler switches
  env.Append(NVCCFLAGS = getNVCCFLAGS(env['mode'], env['backend'], env['arch']))

  env.Append(SHNVCCFLAGS = getNVCCFLAGS(env['mode'], env['backend'], env['arch']))

  # get linker switches
  env.Append(LINKFLAGS = getLINKFLAGS(env['mode'], env['backend'], env.subst('$LINK')))
   
  # get CUDA paths
  (cuda_exe_path,cuda_lib_path,cuda_inc_path) = get_cuda_paths()
  env.Append(LIBPATH = [cuda_lib_path])
  env.Append(CPPPATH = [cuda_inc_path])

  env.Replace(CUDA_PATHS = (cuda_lib_path, cuda_inc_path))

  # link against backend-specific runtimes
  # XXX we shouldn't have to link against cudart unless we're using the
  #     cuda runtime, but cudafe inserts some dependencies when compiling .cu files
  # XXX ideally this gets handled in nvcc.py if possible
  #env.Append(LIBS = 'cudart')

  if env['backend'] == 'ocelot':
    if os.name == 'posix':
      env.Append(LIBPATH = ['/usr/local/lib'])
    else:
      raise ValueError, "Unknown OS.  What is the Ocelot library path?"
  elif env['backend'] == 'omp':
    if os.name == 'posix':
      env.Append(LIBS = ['gomp'])
    elif os.name == 'nt':
      env.Append(LIBS = ['VCOMP'])
    else:
      raise ValueError, "Unknown OS.  What is the name of the OpenMP library?"

  # import the LD_LIBRARY_PATH so we can run commands which depend
  # on shared libraries
  # XXX we should probably just copy the entire environment
  # If LD_LIBRARY_PATH doesn't exist, just don't add it
  try:
    if os.name == 'posix':
      if env['PLATFORM'] == "darwin":
        env['ENV']['DYLD_LIBRARY_PATH'] = os.environ['DYLD_LIBRARY_PATH']
      else:
        env['ENV']['LD_LIBRARY_PATH'] = os.environ['LD_LIBRARY_PATH']
  except Exception:
    #Environment variable not defined
    pass
  # generate help text
  Help(vars.GenerateHelpText(env))


  # enable Doxygen
  env.Tool('dox', toolpath = [os.path.join(thisDir)])

  
  return env

