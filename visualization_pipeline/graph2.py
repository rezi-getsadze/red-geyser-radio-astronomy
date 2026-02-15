import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erf

# Plot
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='2. MaNGA 1-27393')

# C Band
c_b_min = [3550, 5000, 10000, 14000, 20000, 25000, 30000, 34000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000]
c_flux = [0.2115, 0.21054, 0.21343, 0.21319, 0.21281, 0.21723, 0.2291, 0.23098, 0.2204, 0.24651, 0.24829, 0.25515, 0.23883, 0.23688, 0.20788, 0.20247, 0.18506, 0.14246, 1.43E-01]
c_errors = [0.0152, 0.0158, 0.0164, 0.0165, 0.0173, 0.0183, 0.0198, 0.0199, 0.0213, 0.0216, 0.0229, 0.0239, 0.0257, 0.0282, 0.0312, 0.0341, 0.0396, 0.0443, 0.00]
c_rms = [1.45E-02, 1.58E-02, 1.63E-02, 1.66E-02, 1.73E-02, 1.87E-02, 2.02E-02, 1.98E-02, 2.17E-02, 2.12E-02, 2.28E-02, 2.23E-02, 2.41E-02, 2.68E-02, 2.99E-02, 3.04E-02, 3.90E-02, 4.29E-02, 4.77E-02]
c_resolved = ['Unresolved'] * (len(c_b_min) - 1) + ['Non-detection']
size = 60

for m in range(len(c_b_min)):
    if c_resolved[m] == 'Resolved':
        marker = 's'
    elif c_resolved[m] == 'Unresolved':
        marker = 'o'
    else:
        marker = 'v'
        size = 160
        
    plt.scatter(c_b_min[m], c_flux[m], s=size, c='blue', marker=marker, alpha=0.6, label='C-Band Flux' if m == 0 else None)
    plt.errorbar(c_b_min[m], c_flux[m], yerr=c_errors[m], fmt='none', c='blue')
    plt.scatter(c_b_min[m], c_rms[m], s=20, c=(120 / 255, 150 / 255, 250 / 255), marker='D', alpha=0.6, label='C-Band RMS' if m == 0 else None)

# Legend:
plt.scatter([], [], s=40, c='black', marker='o', label='Unresolved (Circle)')
plt.scatter([], [], s=40, c='black', marker='v', label='Non-detection (Down-triangle)')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux (milliJy)', fontsize=19)
legend = plt.legend(fontsize=14)
for text in legend.texts:
    if text.get_text() == '2. MaNGA 1-27393':
        text.set_fontweight('bold')
        break
plt.grid(True)
plt.tight_layout()
plt.show()

# Fitting
# Weird Skewed One:
# Define the data
b_min = np.array([3550, 5000, 10000, 14000, 20000, 25000, 30000, 34000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000])
flux = np.array([0.2115, 0.21054, 0.21343, 0.21319, 0.21281, 0.21723, 0.2291, 0.23098, 0.2204, 0.24651, 0.24829, 0.25515, 0.23883, 0.23688, 0.20788, 0.20247, 0.18506, 0.14246])
errors = np.array([0.0152, 0.0158, 0.0164, 0.0165, 0.0173, 0.0183, 0.0198, 0.0199, 0.0213, 0.0216, 0.0229, 0.0239, 0.0257, 0.0282, 0.0312, 0.0341, 0.0396, 0.0443])

def skewed_gaussian_with_constant(x, amplitude, mean, stddev, skew, constant):
    t = (x - mean) / stddev
    skew_gaussian = 2 * amplitude * np.exp(-t**2 / 2) * (1 + erf(skew * t / np.sqrt(2)))
    return skew_gaussian + constant

def chi_square_skewed_with_constant(params):
    model_flux = skewed_gaussian_with_constant(b_min, *params)
    residuals = (flux - model_flux) / errors
    return np.sum(residuals ** 2)

# Initial guess
initial_guess = [0.1, 60000, 10000, 1, 10]

# Chi-sq:
result = minimize(chi_square_skewed_with_constant, initial_guess)

# parameters
popt = result.x

# generating points
x_fit = np.linspace(0, 90000, 100)
y_fit = skewed_gaussian_with_constant(x_fit, *popt)

plt.figure(figsize=(10, 6))
plt.scatter([], [], marker='', label='2. MaNGA 1-27393')
plt.errorbar(b_min, flux, markersize=9, c='blue', yerr=errors, fmt='bo', label='C Band Data')
plt.plot(x_fit, y_fit, color='purple', label='Skewed Gaussian Fit + Constant')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux (milliJy)', fontsize=19)
plt.xlim(0, 89000) 
plt.ylim(0, 0.3) 
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.scatter([], [], marker='', label=f'Const.: {popt[4]:.2f}\nAmp.: {popt[0]:.2f}\nMean: {popt[1]:.2f}\nS. D.: {popt[2]:.2f}\nSkew: {popt[3]:.2f}')
legend = plt.legend(fontsize=14, loc='lower left')
for text in legend.texts:
    if text.get_text() == '2. MaNGA 1-27393':
        text.set_fontweight('bold')
plt.grid(True)
#plt.text(0.05, 0.05, f'Const.: {popt[4]:.2f}\nAmp.: {popt[0]:.2f}\nMean: {popt[1]:.2f}\nS. D.: {popt[2]:.2f}\nSkew: {popt[3]:.2f}',
#         transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom')

plt.tight_layout()
plt.show()

common_b_min = [3550]

spectral_indices = []
si = np.log10(0.2115 / (3*1000*0.00003099)) / np.log10(4.8 / 1.4)
spectral_indices.append(si)
        
# Plot
plt.figure(figsize=(10, 6))
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

plt.scatter([], [], marker='', label='2. MaNGA 1-27393')
print(common_b_min)
print(spectral_indices)
# Plotting spectral indices against common b_min values
plt.scatter(common_b_min, spectral_indices, s=80, c='red', marker='^')

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Spectral Index', fontsize=19)
plt.scatter([], [], s=40, c='red', marker='^', label='Non-detection (L Band)')
plt.grid(True)
plt.scatter([], [], marker='', label=f'Value: {spectral_indices[0]:}')
legend = plt.legend(fontsize=14)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
for text in legend.texts:
    if text.get_text() == '2. MaNGA 1-27393':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()

# Positions

# Positions (Eliminated last unresolved)

# C Band Data
c_b_min = [3550, 5000, 10000, 14000, 20000, 25000, 30000, 34000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000]
c_x = [-2.3898, -2.3932, -2.3902, -2.3884, -2.3984, -2.3826, -2.3886, -2.3964, -2.393, -2.3902, -2.3846, -2.3868, -2.3752, -2.4076, -2.3596, -2.4562, -2.444]
c_x_error = [0.04258, 0.04328, 0.04212, 0.04158, 0.0414, 0.04046, 0.03868, 0.03794, 0.04048, 0.03732, 0.0373, 0.03672, 0.04138, 0.04382, 0.04812, 0.0551, 0.05968]
c_y = [1.552, 1.5458, 1.582, 1.574, 1.5474, 1.5014, 1.5076, 1.5184, 1.3906, 1.4176, 1.4014, 1.423, 1.4386, 1.4476, 1.4078, 1.5612, 1.521]
c_y_error = [0.12566, 0.1296, 0.12466, 0.12228, 0.11994, 0.11802, 0.12162, 0.12096, 0.1206, 0.10828, 0.11598, 0.10996, 0.12626, 0.13288, 0.15886, 0.18722, 0.19598]

# Plot
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='2. MaNGA 1-27393')

# Scatter plot with error bars colored by b_min
sc = plt.scatter(c_x, c_y, c=c_b_min, cmap='coolwarm', marker='o', s=65)
plt.scatter([], [], marker='', label='C Band')

# Adding error bars
plt.errorbar(c_x, c_y, xerr=c_x_error, yerr=c_y_error, fmt='none', c='black', alpha=0.5)

# Color bar
cbar = plt.colorbar(sc)
cbar.set_label('Minimum Baseline')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('X (milliarcsec)', fontsize=19)
plt.ylabel('Y (milliarcsec)', fontsize=19)
plt.grid(True)
plt.xlim(-2.300, -2.525)
#plt.scatter([], [], marker='', label='Offset Angle = 4.56°')
plt.scatter([], [], marker='', label='0.1 mas = 0.09 pc')
legend = plt.legend(fontsize=14, loc='lower right')
for text in legend.texts:
    if text.get_text() == '2. MaNGA 1-27393':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()