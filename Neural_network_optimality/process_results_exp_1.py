import pandas as pd
import numpy as np
import functions as uf
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

sizes = [1106, 1708, 2346, 3020, 3730, 4476, 5258, 6076, 6930, 7820]
optimality = []

for size in sizes:
    string = 'measures_for_digit_3_with_99_epochs_128_batch_and_{}_weights.csv'.format(size)
    df = pd.read_csv('Measures_results/'+string)
    measures = np.array(df)
    o = uf.curve_decay_rate(curve=measures, levels=5)
    optimality.append(o)

plt.figure(num=1, figsize=(8, 6))
plt.plot(sizes, optimality, linewidth=3)
plt.xticks(size=16)
plt.yticks(size=16)
plt.xlabel('Number of trainable weights', fontsize=16)
plt.ylabel('Optimality: sum of details errors', fontsize=16)
plt.savefig('Figures/size_optimality.pdf', format='pdf', bbox_inches='tight')
plt.show()
