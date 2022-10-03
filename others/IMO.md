# IMO

## Number theory

$t=\frac{x^2+y^2}{xy+1}, x, y\in \Z^+$, If $t\in \Z^+$, then $t=n^2$.

*Proof* trivial case $x=y=1$



prompt: $xy+1 | x^2+y^2$ iff $xy+1 | y^4+1$

$y=x^3, t=x^2$

$y=z^3, x=z^5-z, t=z^2$



```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
def t(x,y):
    return (x**2+y**2) / (x*y+1)

y=3**3
for x in range(1, y):
    if (y**4+1) % (x*y+1)==0:
        print(f'x={x}, y={y}, t={np.sqrt(t(x,y))}')

from sympy import symbols

x, z = symbols('x,z')
l = (z**4+1)*(x*z**3 + 1)
r = z**8-z**4+1
print(l-z)

print(2**4+1, 2**8-2**4+1, 30*2**3+1)


y=z**3; x=z**5-z
print(y**2+x**2)
print(y*x+1)

```

