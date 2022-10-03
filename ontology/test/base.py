#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from owlready2 import *
import pathlib
import datetime
PATH = pathlib.Path("~/Folders/ontology/test/base").expanduser()
onto_path.append(PATH)
test = get_ontology("http://test.org/test.owl")

def is_instance_of(i, c):
    """True iff i is in c
    
    Arguments:
        i {Thing} -- individual
        c {Property} -- concept/class
    
    Returns:
        bool
    """
    return c in i.is_instance_of or any(is_a(y, c) for y in i.is_instance_of if hasattr(y, 'is_a'))

def is_a(x, c):
    """True iff x is contained in c
    """
    return c in x.is_a or any(is_a(c, y) for y in x.is_a if hasattr(y, 'is_a'))



with test: 
    class Person(Thing): 
        pass 

    class Cat(Thing): 
        pass 

    class Dog(Thing): 
        pass 

    class love(Cat >> Dog): 
        pass 

    p = Person('p') 
    c = Cat('c') 
    d=Dog('d') 
  
    Cat.is_a.append(love.some(Dog)) 


sync_reasoner(debug=1) 

print(c.love) 
print(c.INDIRECT_love) 

