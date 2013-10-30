from __future__ import print_function
import os
import re
import subprocess
from distutils.errors import CompileError
import operator
import fnmatch

# try to import an environment first
try:
    Import('env')
except:
    exec open(os.path.join('config', "build-env.py"))
    env = Environment()

conf = Configure(env)
siteconf = {}

siteconf['CUDA_LIB_DIR'], siteconf['CUDA_INC_DIR'] = env['CUDA_PATHS']
siteconf['CXX'] = env['CXX']
siteconf['CC'] = env['CC']
siteconf['ACML_INC'] = None
siteconf['ACML_LIB'] = None

# Check to see if the user has written down siteconf stuff

if os.path.exists("siteconf.py"):
    glb = {}
    execfile("siteconf.py", glb, siteconf)
else:
    print("""
*************** siteconf.py not found ***************
We will try building anyway, but may not succeed.
Read the README for more details.
""")

        
f = open("siteconf.py", 'w')
print("""#! /usr/bin/env python
#
# Configuration file.
# Use Python syntax, e.g.:
# VARIABLE = "value"
# 
# The following information can be recorded:
#
# ACML_INC : Directory where ACML include files are found.
#
# ACML_LIB : Directory where ACML libraries are found
#


""", file=f)


for k, v in sorted(siteconf.items()):
    if v:
        v = '"' + str(v) + '"'
    print('%s = %s' % (k, v), file=f)

f.close()

Export('siteconf')
if siteconf['CXX']:
    env.Replace(CXX=siteconf['CXX'])
if siteconf['CC']:
    env.Replace(CC=siteconf['CC'])

#Check for the CUDART librar
conf.CheckLib('cudart', language='C++', autoadd=0)

if siteconf['ACML_INC']:
    env.Append(CPPPATH=siteconf['ACML_INC'])
if siteconf['ACML_LIB']:
    env.Append(LIBPATH=siteconf['ACML_LIB'])

#Check for libacml.so               
if not conf.CheckLib('acml', language="C"):
    print("ACML library not found. Update siteconf.py with path")
    Exit(1)

#Check for acml headers
if not conf.CheckHeader('acml.h'):
    print("ACML header not found. Update siteconf.py with path")

env.Append(CCFLAGS = ['-O3'])
    
Export('env')

