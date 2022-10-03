#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

from owlready2 import *

PATH = pathlib.Path("~/Folders/ontology/logic/base").expanduser()
onto_path.append(PATH)
logic = get_ontology("http://test.org/logic.owl")

with logic:
    class FOL(Thing):
        # define content
        pass

    class DL(FOL):
        pass
