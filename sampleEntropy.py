"""
SampEn  计算时间序列data的样本熵
输入：data是数据一维行向量
m重构维数，一般选择1或2，优先选择2，一般不取m>2
r 阈值大小，一般选择r=0.1~0.25*Std(data)
输出：SampEnVal样本熵值大小
"""
import numpy as np

def sampEn(U,m,r):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        B = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1.0) / (N - m) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(B)
     
    N = len(U)
    return -np.log(_phi(m+1) / _phi(m))

if __name__ == "__main__":
    # Usage example
    U = np.array([85, 80, 89] *17)
    print(sampEn(U,2,3))