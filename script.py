#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parsing my notes written in markdown
"""

import re
import pyparsing as pp

body = pp.Regex('.*')

theorem = pp.ZeroOrMore('#').suppress() + (pp.Keyword('Theorem') | pp.Keyword('Lemma') | pp.Keyword('Corollary'))('type') + pp.Optional(pp.QuotedString('(', endQuoteChar=')'))('name') + pp.SkipTo(pp.CaselessKeyword('proof'))('content') + pp.CaselessKeyword('proof').suppress() + pp.Optional('.').suppress() + (pp.SkipTo('\n\n') | pp.SkipTo('QED') | pp.SkipTo(r'$\square') | pp.Regex(re.compile('.*', re.DOTALL)))('proof')

s = r"""## Lemma (Riesz Sunrise Lemma)
Suppose f\in C[a,b]. rc(f)=\bigcup_i(a_i,b_i) (from left to right), b_i is the local maximum, f(a_i)=f(b_i)>f(x),x\in(a_i,b_i),i>1, f(a_1)\leq f(b_1), f(b_i)\searrow.

Proof.
rc(f) is open.
"""

pr = theorem.parseString(s)

print(pr.proof)
