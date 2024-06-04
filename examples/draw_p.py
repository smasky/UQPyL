import os
os.chdir('./examples')

import numpy as np
import matplotlib.pyplot as plt

values=np.loadtxt('./sensibility.txt', dtype=np.float32)
with open('./si_names.txt', 'r') as f:
    name=f.readlines()
    name=[n.strip() for n in name]
colors=['#03AED2', '#FFC55A', '#75A47F', '#D2649A']
labels=['Sobol\'', 'FAST', 'Morris', 'MARS']
x=np.arange(len(name))
bar_width=0.2
for i in range(4):
    plt.bar(x + i * bar_width, values[:, i], bar_width, label=labels[i], color=colors[i])
for i in range(len(name) - 1):
    plt.axvline(x=i+1-bar_width, color='black', linestyle='--', linewidth=0.75)
    

plt.xlim(-0.5, 25.8)
plt.xticks(x+0.3, [f'P{i+1}' for i in range(len(name))], fontsize=14)
plt.yticks(fontsize=14)
plt.tick_params(axis='both', which='both', length=0)
plt.xlabel('Parameters', fontsize=14)
plt.ylabel('Sensibility', fontsize=14)
plt.legend(loc='upper right', fancybox=True, shadow=True, ncol=4, fontsize=14)
ax = plt.gca()
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.show()