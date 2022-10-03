#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

from owlready2 import *
from base import *

PATH = pathlib.Path("~/Folders/ontology/math/topology").expanduser()
onto_path.append(PATH)
topology = get_ontology("http://test.org/topology.owl")

with topology:
    class TopologicSpace(Thing):
        pass

    class UniformSpace(TopologicSpace):
        pass

    class MetricSpace(UniformSpace):
        distance = ''

    class HausdorffSpace(TopologicSpace):
        definition = r'T2: x \= y => x in U, y in V, U & V=0'
        wikiEntry = 'Hausdorff_space'

    class LindelofSpace(TopologicSpace):
        definition = 'every open cover has a countable subcover'
        wikiEntry = 'Lindelöf_space'

    class CountablyCompactSpace(TopologicSpace):
        definition = 'every countable open cover has a countable subcover'

    class CompactSpace(LindelofSpace, CountablyCompactSpace):
        definition = 'every open cover has a finite subcover'

    class Souslin(HausdorffSpace):
        definition = 'souslin set, f[X], f is continuous, X is complete separable metric space'

