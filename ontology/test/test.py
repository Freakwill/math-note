#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from owlready2 import *

test = get_ontology("http://test.org/test.owl")


with test:
    class Celestial(Thing):
        pass
    class travel_round(Celestial >> Celestial): 
        pass

    class Planet(Celestial): 
        pass 

    class Star(Celestial): 
        pass 

    # class Sattlite(Celestial): 
    #     equivalent_to=[travel_round.some(Planet)]


    # import types
    # def func(ns):
    #     ns["equivalent_to"] = [travel_round.some(Planet)]
    # Sattlite = types.new_class('Sattlite', (Celestial,), exec_body=func)

    # import types
    # def func(ns):
    #     ns["x"] = 0
    # A = types.new_class('A', exec_body=func)

    earth = Planet('earth')
    sun = Celestial('sun')

    earth.travel_round = [sun]

    rule = Imp()
    # Planet.is_a.append(travel_round.only(Star))
    rule.set_as_rule("Planet(?p), Celestial(?s), travel_round(?p, ?s) -> Star(?s)")

close_world(test) 

sync_reasoner(debug=0)
print(sun.INDIRECT_is_instance_of)


