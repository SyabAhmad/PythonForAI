import matplotlib.pyplot as plt
import numpy as np
Xaxis = np.array([1,2,3,4,5,9,7,8,9,10,11])
Yaxix = np.array([11,10,9,8,3,6,5,4,3,2,1])
# to show graph
plt.plot(Xaxis, Yaxix, marker='o')
plt.xlabel('X axis')
plt.ylabel('Y axis')

plt.suptitle('pyplot plot graph')

plt.show()

