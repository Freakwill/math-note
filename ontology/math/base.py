#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

from owlready2 import *

PATH = pathlib.Path("~/Folders/ontology/math/base").expanduser()
onto_path.append(PATH)
math = get_ontology("http://test.org/math.owl")

notations ={'cont.':'continous(ly)', 'const':'constant', 'mon.': 'monotonic(ally)'}

with math:
    class Statement(Thing):
        content = ''
        also_see = ''
        wiki = ''
        remark = ''
        relm = ''

        def __str__(self):
            if hasattr(self, 'title'):
                return self.title
            else:
                return super(Statement, self).__str__()

        def __repr__(self):
            if self.remark:
                return f'''{self.title}/{self.relm}: {self.content}
Remark. {self.remark} [also see {self.also_see}]'''
            else:
                return f'''{self.title}/{self.relm}: {self.content}'''

        def __format__(self, spec=None):
            if spec is None:
                return self.title
            elif spec == 'full':
                return f'''{self.title}/{self.relm}: {self.content}
-------
Remark. {self.remark}
-------
Also see {self.also_see}
'''
            else:
                return f'''{self.title}/{self.relm}: {self.content}
Remark. {self.remark}'''

        def __matmul__(self, content):
            self.content = content
            return self

        def __rshift__(self, content):
            self.proof = content
            return self

        def __xor__(self, content):
            self.remark = content
            return self

        def __pow__(self, remark):
            self.remark = remark
            return self


    class Technique(Statement):
        pass

    class Definition(Statement):
        title = ''

    class Proposition(Statement):
        proof = ''

    class Theorem(Proposition):
        title = ''
        contributor = ''
        applied_in = ''
        derived_from = ''

    class Lemma(Theorem):
        pass

    class Corollary(Theorem):
        pass

    class Fact(Proposition):
        pass

    class Formula(Proposition):
        pass

    class Relm(Thing):
        def __init__(self, title):
            self.title = title
            self.info = ''

        def __enter__(self):
            pass

        def __exit__(self, *args, **kwargs):
            pass
