#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from owlready2 import *
import pathlib
import datetime
PATH = pathlib.Path("~/Folders/ontology/society/base").expanduser()
onto_path.append(PATH)
society = get_ontology("http://test.org/society.owl")

with society:
    class Person(Thing):
        wholename = ''
        birthday = ''
        gender = ''
        info = ''

        def __str__(self):
            return '%s (%s, %s)'%(self.wholename, self.birthday, self.gender)


    class Orgonization(Thing):
        def __ini__(self, wholename='', birthday='', taxno='', *args, **kwargs):
            super(Person, self).__ini__(*args, **kwargs)
            self.wholename = wholename
            self.birthday = birthday
            self.address = address
            self.taxno = taxno
            self.info = ''


    class Institution(Orgonization):
        pass

    class University(Institution):
        pass

    class College(Institution):
        pass

    class Employee(Person):
        def __ini__(self, no=0, level=0, salary=0, position='', *args, **kwargs):
            super(Employee, self).__ini__(*args, **kwargs)
            self.no = no
            self.level = level
            self.salary = salary
            self.position = position

        def show_info(self):
            print(self)
            print('no: %s\nlevel: %d\nsalary: %.2f'%(self.no, self.level, self.salary))

    class Star(Person):
        pass

    class empolied(Employee >> Orgonization):
        pass

    class empoly(Orgonization >> Employee):
        inverse_property = empolied

    class friend(Person >> Person, SymmetricProperty):
        pass

    class idol(Person >> Person):
        pass

    class potentialFriend(friend, TransitiveProperty):
        pass


    class supervisor(Employee >> Person):
        pass

    class subordinate(Person >> Employee):
        pass


    class Teacher(Employee):
        pass
        # equivalent_to = [Employee & empolied.some(University|College)]

    class weight(AnnotationProperty):
        pass
