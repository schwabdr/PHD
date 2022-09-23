#https://stackoverflow.com/questions/5091993/list-of-all-available-matplotlib-backends
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib

backends = matplotlib.rcsetup.interactive_bk
# validate backends
backends_valid = []
for b in backends:
    try:
        plt.switch_backend(b)
        backends_valid += [b]
    except:
        continue
print(backends_valid)

import matplotlib.backends
import os.path

def is_backend_module(fname):
    """Identifies if a filename is a matplotlib backend module"""
    return fname.startswith('backend_') and fname.endswith('.py')

def backend_fname_formatter(fname): 
    """Removes the extension of the given filename, then takes away the leading 'backend_'."""
    return os.path.splitext(fname)[0][8:]

# get the directory where the backends live
backends_dir = os.path.dirname(matplotlib.backends.__file__)

# filter all files in that directory to identify all files which provide a backend
backend_fnames = filter(is_backend_module, os.listdir(backends_dir))

backends = [backend_fname_formatter(fname) for fname in backend_fnames]

print(backends)