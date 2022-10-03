#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from owlready2 import *
# import pathlib
# PATH = pathlib.Path("~/Folders/ontology/test/base").expanduser()
# onto_path.append(PATH)
test = get_ontology("http://test.org/test.owl")


with test: 

    class Cat(Thing): 
        pass 

    class Dog(Thing): 
        pass 

    class Love(Thing): 
        pass 

    class is_loved(TransitiveProperty): 
        pass

    class love(TransitiveProperty): 
        inverse_property = is_loved 

    l = Love('l') 
    c = Cat('c') 
    d = Dog('d')

    Cat.love.append(l)
    Dog.is_loved.append(l) 

    # l.is_a.append(is_loved.some(Cat))
    # l.is_a.append(love.some(Dog)) 

close_world(test) 
# sync_reasoner(debug=1)  # raise OwlReadyInconsistentOntologyError, why? (related to CWA maybe) 
print(Cat.is_a)
print(Cat.love)
print(c.love)
