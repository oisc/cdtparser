# -*- coding: utf-8 -*-
import pkgutil
import sys

pkgutil.extend_path(__path__, __name__)
sys.path.extend(__path__)

import schemas
from api import *
