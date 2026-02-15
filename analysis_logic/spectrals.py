import matplotlib.pyplot as plt
import numpy as np

# Data
x1 = [5000, 10000, 15000, 20000, 25000, 30000, 40000]
y1_1 = [-0.5169719223928322, -0.20399215095989795, 0.019648739781191558, 0.29712937705021264, -0.38228191231032627, -0.06633341774239769, -0.967008270325565]
errors1_1 = [0.05273097314922217, 0.06425091360685604, 0.09724661782227403, 0.13799994933330176, 0.30499838614268515, 0.27523084755305866, 0]
y1_2 = [-1.444696893701331, -1.1048901937411917, -0.7653225662244529, -0.11985448761590616, -0.5109106459817002, 0.006463124591839162, -0.3333110077377237]
errors1_2 = [0.04152465999369177, 0.053026305926491615, 0.08849334036039826, 0.12644244640626842, 0.29691888869010136, 0.2579758607695567, 0]

x2 = [3550]
y2 = [0.6670878977005886]
errors2 = [0]

x3 = [5000, 10000, 15000, 20000, 25000]
y3 = [-1.1366136794120059, -1.249544939434542, -1.085265581000355, -1.0247550715340668, -0.5683798174580089]
errors3 = [0.12397433869745761, 0.13130630385651337, 0.146672311132156, 0.1720004030717434, 0]

x4 = [4000, 7500, 10000, 15000, 20000, 25000, 30000]
y4 = [-0.35730852903272053, -0.09559933779508296, -0.01615228346614429, 0.027259003833836972, -0.07401164768623869, -0.34796459405622165, 0.23499170654924798]
errors4 = [0.04654350237666102, 0.04277323137987184, 0.04492470329985788, 0.05130480004083185, 0.06330255713293924, 0.09300077010705587, 0]

plt.figure(figsize=(10, 4))
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
for i in range(len(x1)):
    if i == len(x1)-1:
        marker = '^'
    else:
        marker = 'o'

    plt.scatter(x1[i], y1_1[i], alpha=0.6, s=80, c='red', marker=marker)
    plt.errorbar(x1[i], y1_1[i], alpha=0.7, yerr=errors1_1[i], fmt='none', c='red', linewidth=2)

    plt.scatter(x1[i], y1_2[i], alpha=0.6, s=80, c='green', marker=marker)
    plt.errorbar(x1[i], y1_2[i], alpha=0.7, yerr=errors1_2[i], fmt='none', c='green', linewidth=2)

for i in range(len(x2)):
    if i == len(x2)-1:
        marker = '^'
    else:
        marker = 'o'

    plt.scatter(x2[i], y2[i], alpha=0.6, s=80, c='orange', marker=marker)
    plt.errorbar(x2[i], y2[i], alpha=0.7, yerr=errors2[i], fmt='none', c='orange', linewidth=2)

for i in range(len(x3)):
    if i == len(x3)-1:
        marker = '^'
    else:
        marker = 'o'

    plt.scatter(x3[i], y3[i], alpha=0.6, s=80, c='blue', marker=marker)
    plt.errorbar(x3[i], y3[i], alpha=0.7, yerr=errors3[i], fmt='none', c='blue', linewidth=2)

for i in range(len(x4)):
    if i == len(x4)-1:
        marker = '^'
    else:
        marker = 'o'

    plt.scatter(x4[i], y4[i], alpha=0.6, s=80, c='purple', marker=marker)
    plt.errorbar(x4[i], y4[i], alpha=0.7, yerr=errors4[i], fmt='none', c='purple', linewidth=2)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Spectral Index', fontsize=19)
plt.scatter([], [], s=40, c='red', marker='o', label='1. MaNGA 1-26878 1')
plt.scatter([], [], s=40, c='green', marker='o', label='1. MaNGA 1-26878 2')
plt.scatter([], [], s=40, c='orange', marker='o', label='2. MaNGA 1-27393')
plt.scatter([], [], s=40, c='blue', marker='o', label='3. MaNGA 1-19818')
plt.scatter([], [], s=40, c='purple', marker='o', label='4. MaNGA 1-43718')
plt.scatter([], [], s=40, c='black', marker='o', label='Detection')
plt.scatter([], [], s=40, c='black', marker='^', label='Non-detection (L Band)')
plt.grid(True)
legend = plt.legend(fontsize=11, loc='center', bbox_to_anchor=(0.76, 0.25))
plt.tight_layout()
plt.show()
