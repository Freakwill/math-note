#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib

from owlready2 import *

PATH = pathlib.Path("~/Folders/ontology/puzzle/base").expanduser()
onto_path.append(PATH)
puzzle = get_ontology("http://test.org/puzzle.owl")

with puzzle:
    class Person(Thing):
        # define content
        pass

    class Pet(Thing):
        pass

    class House(Thing):
        no = 0

    class Drink(Thing):
        pass

    class Smoke(Thing):
        pass

    class care(Person >> Pet, FunctionalProperty):
        pass

    class live(Person >> House, FunctionalProperty):
        pass

    class drink(Person >> Drink, FunctionalProperty):
        pass

    class take(Person >> Smoke, FunctionalProperty):
        pass

    class left(House >> House, FunctionalProperty):
        pass

    class right(House >> House, FunctionalProperty):
        inverse = left

    class side(House >> House):
        pass


e = Person('英国人')
s = Person('瑞典人')
d = Person('丹麦人')
n = Person('挪威人')
g = Person('德国人')

red = House('红色房子')
green = House('绿色房子')
yellow = House('黄色房子')
blue = House('蓝色房子')
white = House('白色房子')

dog = Pet('狗')
bird = Pet('鸟')
cat = Pet('猫')
horse = Pet('马')
fish = Pet('鱼')

coffee = Drink('咖啡')
milk = Drink('牛奶')
beer = Drink('啤酒')
water = Drink('水')
tea = Drink('茶')

pall = Smoke('PallMall')
dunhill = Smoke('Dunhill')
prince = Smoke('Prince')
blue = Smoke('BlueMaster')
blends = Smoke('Blends')

e.live = red
s.care = dog
d.drink = tea
white.left = green

g.smoke = prince


