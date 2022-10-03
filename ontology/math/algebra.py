# -*- coding: utf-8 -*-
#!/usr/bin/env python

from owlready2 import *
import os.path

import math

PATH = os.path.expanduser("~/Folders/ontology/math/algebra")
onto_path.append(PATH)
algebra = get_ontology("http://test.org/algebra.owl")

# Create new classes in the ontology, possibly mixing OWL constructs and Python methods:

with algebra:

    class Element(Thing):
        pass

    class unaryOperation(Element >> Element, FunctionalProperty):
        pass

    class binaryOperation(Element >> unaryOperation, FunctionalProperty):
        pass


    class AlgebraSystem(Element):
        ambientSet = []
        operators = []
        iso = []
        wiki = 'https://en.wikipedia.org/wiki/Universal_algebra'

    class Mapping(Element >> Element, FunctionalProperty):
        pass

    class SemiGroup(AlgebraSystem):
        content = 'abc=a(bc)=(ab)c'
        operators = ['*']
        wiki = 

    class Group(SemiGroup):
        iden = 1
        wiki = 'https://en.wikipedia.org/wiki/Group_(mathematics)'

    class inverse(unaryOperation, SymmetricProperty):
        inverse_property = algebra.inverse

    class multiple(binaryOperation):
        pass



word = algebra.SemiGroup("word")
y = algebra.Element('y')
x = algebra.Element('x', inverse=y)

print(x.inverse)
# sync_reasoner()
print(y.inverse)

#algebra.save()