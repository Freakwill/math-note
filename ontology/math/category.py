# -*- coding: utf-8 -*-
#!/usr/bin/env python

from owlready2 import *
import os.path

import math

PATH = os.path.expanduser("~/Folders/ontology/math/category")
onto_path.append(PATH)
category = get_ontology("http://test.org/category.owl")

# Create new classes in the ontology, possibly mixing OWL constructs and Python methods:

with category:

    class Category(type, Thing):
        def __new__(cls, name, bases, attrs):
            if 'objects' not in attrs or 'arrows' not in attrs:
                raise AttributeError('Have to define objects and arrows.')
            return type.__new__(cls, name, bases, attrs)

    class Functor(Category >> Category, FunctionalProperty):
        pass

    class Cofunctor(Category >> Category, FunctionalProperty):
        pass

    class Set(Category):
        objects = 'all small sets'
        arrows = 'all functions between them'

        def __init__(*args):
            return set(args)

    class Setp(Category):
        objects = 'all small sets with a based point'
        arrows = 'all based-point-preserving functions'

    class Grp(Category):
        objects = 'all small groups'
        arrows = 'all morphorsims between them'

    class Ab(Grp):
        objects = 'all small Abel groups'
        remark = 'full subcategory of Grp'

    class Ring(Ab):
        objects = 'all small rings'

    class Top(Category):
        objects = 'all small groups'
        arrows = 'all morphorsims between them'


# word = category.SemiGroup("word")
# y = category.Element('y')
# x = category.Element('x', inverse=y)

# print(x.inverse)
# # sync_reasoner()
# print(y.inverse)

#category.save()