# 数学基础

## 基本概念

### 共轭
两个实部相等，虚部互为相反数的复数互为共轭复数（conjugate complex number）。（当虚部不等于0时也叫共轭虚数）复数z的共轭复数记作$\overline{z}$.（z上加一横，英文中可读作Conjugate z,z conjugate or z bar），有时也可表示为$z^*$.



## 矩阵定义

### 转置矩阵
形状为m*n的矩阵，其转置矩阵的形状是n*m矩阵，即

$$A^{T}_{ij}=A_{ji} \quad for 1\leq i\leq n, 1\leq j\leq m$$

$$A= \left\{ \begin{matrix} a & b \\ c & d \end{matrix} \right\}$$

$$A^{T}= \left\{ \begin{matrix} a & b \\ c & d \end{matrix} \right\}$$

转置矩阵有如下性质：
$$(A^T)^T\equiv A$$

$$(A+B)^T=A^T+B^T$$

$$(AB)^T=B^TA^T$$

$$(cA)^T=cA^T$$

$$det(A^T)=det(A)$$

### 共轭矩阵
共轭矩阵
共轭矩阵又称Hermite阵。Hermite阵中每一个第i行第j列的元素都与第j行第i列的元素的共轭相等。

TODO: 用途

## 矩阵计算

### 矩阵相乘

#### 点积

例如：
$$a= \left\{ \begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix} \right\}$$

$$b= \left\{ \begin{matrix} 5 & 6 \\ 7 & 8 \end{matrix} \right\}$$

$$a \cdot b = \left\{ \begin{matrix} 1\times5+2\times7 & 1\times6+2\times8 \\ 3\times5+4\times7 & 3\times6+4\times8 \end{matrix} \right\}$$
