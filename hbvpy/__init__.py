#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HBVpy_dev
=========

Provides :
    1. Bindings to run HBV-light from python scripts.
    2. Functions to pre-process input data to HBV-light.
    2. An easy way to generate HBV-light configuration files.
    3. Functions to load and process HBV-light result files.

"""
from . import core
from .core import *
from . import ThirdParty

__all__ = []
__all__.extend(core.__all__)
