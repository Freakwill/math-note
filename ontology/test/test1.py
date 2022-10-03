#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from owlready2 import *

test = get_ontology("http://test.org/test.owl")


with test:
    class Human(Thing):
        pass

    class Man(Human):
        pass

    class Woman(Human):
        pass

    class love(Man >> Woman): 
        pass

    class Dog(Thing):
        pass


    Man.is_a.append(love.max(1, Woman))  # Man.love.append(Woman)

    a = Man('a')
    b = Woman('b')
    c = Woman('c')
    d = Dog('d')
    a.love.append(b)
    AllDifferent([b,c])
    AllDisjoint([Man, Woman])


    class not_love_c(Thing):
        # class of man that dose not love c
        equivalent_to = [Not(love.value(c))]
    # class not_love_d(Man):
    #     equivalent_to = [Not(love.some(Dog))]

#close_world(test)
sync_reasoner(debug=1)
print(a.INDIRECT_love)
print(a.INDIRECT_is_instance_of)
print(c.INDIRECT_is_a)
# a.love.append(c)
print(type(Or([Man, a])))

