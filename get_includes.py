#!/usr/bin/env python3

import numpy as np
from sysconfig import get_path
print(f"-I{get_path('include')} -I{np.get_include()}")
