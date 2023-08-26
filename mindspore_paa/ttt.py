import numpy as np

aa = np.array([[0, 0],
               [0, 0]])
bb = np.array([[1, 1],
               [1, 1]])

tps = np.logical_and(aa, np.logical_not(bb))  # shape: (thre_num, dt_num)
fps = np.logical_and(np.logical_not(aa), np.logical_not(bb))
print(tps)
print(fps)