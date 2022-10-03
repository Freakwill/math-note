#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from owlready2 import *
from propositions import *

def search(keyword, **kw):
    i = 1
    for k, v in globals().items():
        if isinstance(v, Thing):
            if keyword in v.content or keyword in v.remark:
                print(f'[{i}]\n{v:full}')
                i += 1

search('Kuhn')