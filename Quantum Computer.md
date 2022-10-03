# Quantum Computer

## Relation between QC and Hilbert spaces

| QC                                             | Hilbert 空间      |      |
| ---------------------------------------------- | ----------------- | ---- |
| 量子态state $|\phi\rangle,|0\rangle,|1\rangle$ | （列）向量，基    |      |
| $|0\rangle|1\rangle,\langle\psi|\phi\rangle $  | 张量积            |      |
| 密度矩阵 density matrix                        | 矩阵（Gram 矩阵） |      |
| 量子系统 quatnum system                        | 向量族            |      |
| （投影）测量系统 measure system                | （投影）矩阵族    |      |
| $\langle\psi|\phi\rangle$                      | 内积              |      |
| 量子门                                         | 线性变换、矩阵    |      |
| 张量积tensor                                   | 矩阵Kronecker积   |      |
| 量子态转置$\langle\psi|$                       | 转置              |      |

## Quantum Gates

| 名称           | 符号     | 矩阵表示 | QuTip    |
| -------------- | -------- | -------- | -------- |
| NOT gate       | NOT      |          | sigmax() |
| Z gate         | Z        |          | sigmaz() |
| Hadamard gate  | H        |          | snot()   |
| controlled-NOT | $U_{CN}$ |          | cnot()   |
|                |          |          |          |
|                |          |          |          |



## Deutsch's algorithm

$U_f:x,y\mapsto x, y\oplus f(x)$

$D(x,y)=U_f(Hx, y)$, $D(0, 0)\sim|0f(0)\rangle+|1f(1)\rangle$



## QuTip， Python实现

### states

```python
state = basis(2, 0)  # |0>
state.dag() # <0|

bra("10")  # |10>
'''Quantum object: dims = [[1, 1], [2, 2]], shape = (1, 4), type = bra
Qobj data =
[[0. 0. 1. 0.]]'''

bra("121",3)
'''Quantum object: dims = [[1, 1, 1], [3, 3, 3]], shape = (1, 27), type = bra
Qobj data =
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0.]]'''

bra("12",[3,4])
'''Quantum object: dims = [[1, 1], [3, 4]], shape = (1, 12), type = bra
Qobj data =
[[0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]]'''

ket("0") == basis(2,0)
```



### gates, operators, matrices

```python
# matrix
identity(2)  # 1

# the spin operators
sigmax() # x gate
sigmay() # y gate
sigmaz() # z gate
hadamard_transform(N=1) # Hadamard gate

toffoli() # Toffoli gate
```

### Saving and Loading Result Objects

```python
qsave(result, 'filename')
stored_result = qload('filename')
```



