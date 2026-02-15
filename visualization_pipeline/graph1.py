import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import matplotlib.patches as patches
import matplotlib.cm as cm

# Plot
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='1. MaNGA 1-26878')

# L Band
l_b_min = [1090, 5000, 10000, 15000, 20000, 25000, 30000, 40000]
l_flux = [3.4557, 2.6034, 1.7206, 1.1426, 0.49843, 0.73379, 0.40774, 0.576]
l_errors = [0.0754, 0.0835, 0.0884, 0.116, 0.0746, 0.266, 0.128, 0]
l_rms = [0.0289, 0.0342, 0.0388, 0.0514, 0.0779, 0.106, 0.126, 0.192]
l_resolved = ['Resolved', 'Resolved', 'Resolved', 'Resolved', 'Unresolved', 'Resolved', 'Unresolved', 'Non-detection']
size = 60

for i in range(len(l_b_min)):
    if l_resolved[i] == 'Resolved':
        marker = 's'
    elif l_resolved[i] == 'Unresolved':
        marker = 'o'
    else:
        marker = 'v'
        size = 160
        
    plt.scatter(l_b_min[i], l_flux[i], s=size, c='red', marker=marker, alpha=0.6, label='L-Band Flux' if i == 0 else None)
    plt.errorbar(l_b_min[i], l_flux[i], yerr=l_errors[i], fmt='none', c='red')
    plt.scatter(l_b_min[i], l_rms[i], s=20, c=(0.8, 0.2, 0.2), marker='D', alpha=0.6, label='L-Band RMS' if i == 0 else None)

# C Band
c_b_min = [3445, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000]
c_flux = [1.4082, 1.3769, 1.3382, 1.1706, 0.71879, 0.45815, 0.37574, 0.31483, 0.17497, 0.07095]
c_errors = [0.0762, 0.0778, 0.0806, 0.0745, 0.058, 0.0454, 0.0482, 0.0477, 0.0517, 0]
c_resolved = ['Resolved', 'Resolved', 'Resolved', 'Resolved', 'Resolved', 'Resolved', 'Resolved', 'Resolved', 'Resolved', 'Non-detection']

size = 60
for m in range(len(c_b_min)):
    if c_resolved[m] == 'Resolved':
        marker = 's'
    elif c_resolved[m] == 'Unresolved':
        marker = 'o'
    else:
        marker = 'v'
        size = 160
        
    plt.scatter(c_b_min[m], c_flux[m], s=size, c='blue', marker=marker, alpha=0.6, label='C-Band 1 Flux' if m == 0 else None)
    plt.errorbar(c_b_min[m], c_flux[m], yerr=c_errors[m], fmt='none', c='blue')
    

# C Band blob 2 
c2_b_min = [3445, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 
            50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 
            100000]

c2_flux = [4.49E-01, 4.39E-01, 4.41E-01, 4.45E-01, 4.30E-01, 3.91E-01, 4.11E-01, 4.05E-01, 3.82E-01, 
           3.75E-01, 3.90E-01, 3.58E-01, 3.82E-01, 3.71E-01, 3.93E-01, 3.87E-01, 3.84E-01, 
           3.09E-01, 2.68E-01, 2.58E-01, 1.57E-01]

c2_errors = [1.71E-02, 1.75E-02, 1.78E-02, 1.77E-02, 1.86E-02, 1.93E-02, 2.05E-02, 2.08E-02, 2.21E-02, 
             2.33E-02, 2.49E-02, 2.67E-02, 2.76E-02, 2.96E-02, 3.13E-02, 3.48E-02, 3.87E-02, 
             4.43E-02, 4.77E-02, 5.12E-02, 0]

c2_resolved = ["Unresolved"] * 20 + ["Non-detection"]

c_rms = [
    1.77E-02, 1.69E-02, 1.77E-02, 1.76E-02, 1.87E-02, 1.99E-02, 2.06E-02, 2.07E-02,
    2.23E-02, 2.37E-02, 2.58E-02, 2.80E-02, 2.63E-02, 2.78E-02, 3.15E-02, 3.50E-02,
    4.05E-02, 4.34E-02, 4.84E-02, 4.90E-02, 5.23E-02]

size = 60
for m in range(len(c2_b_min)):
    if c2_resolved[m] == 'Resolved':
        marker = 's'
    elif c2_resolved[m] == 'Unresolved':
        marker = 'o'
    else:
        marker = 'v'
        size = 160
        
    plt.scatter(c2_b_min[m], c2_flux[m], s=size, c='green', marker=marker, alpha=0.6, label='C-Band 2 Flux' if m == 0 else None)
    plt.errorbar(c2_b_min[m], c2_flux[m], yerr=c2_errors[m], fmt='none', c='green')
    plt.scatter(c2_b_min[m], c_rms[m], s=20, c=(120 / 255, 150 / 255, 250 / 255), marker='D', alpha=0.6, label='C-Band RMS' if m == 0 else None)

# Legend:
plt.scatter([], [], s=40, c='black', marker='s', label='Resolved (Square)')
plt.scatter([], [], s=40, c='black', marker='o', label='Unresolved (Circle)')
plt.scatter([], [], s=40, c='black', marker='v', label='Non-detection (Down-triangle)')


plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux (milliJy)', fontsize=19)
legend = plt.legend(fontsize=14)
for text in legend.texts:
    if text.get_text() == '1. MaNGA 1-26878':
        text.set_fontweight('bold')
        break
plt.grid(True)
plt.tight_layout()
plt.show()

#Fittings

# L band
# Constant:

# Define the data
b_min = np.array([1090, 5000, 10000, 15000, 20000, 25000, 30000])
flux = np.array([3.4557, 2.6034, 1.7206, 1.1426, 0.49843, 0.73379, 0.40774])
errors = np.array([0.0754, 0.0835, 0.0884, 0.116, 0.0746, 0.266, 0.128])

def gaussian_plus_constant(x, constant, amplitude, stddev):
    return constant + amplitude * np.exp(-((x - 0) / stddev) ** 2 / 2)

# Define the chi-square function
def chi_square(params):
    model_flux = gaussian_plus_constant(b_min, *params)
    residuals = (flux - model_flux) / errors
    return np.sum(residuals ** 2)

# Initial guess for the parameters
initial_guess = [1, 1, np.std(b_min)]

# Minimize the chi-square function to find the best-fit parameters
result = minimize(chi_square, initial_guess)

# Extract the optimized parameters
popt = result.x

# Generate points for the fitted Gaussian curve with extended range
#x_fit_zoomed = np.linspace(min(b_min), max(b_min), 100)
x_fit_full = np.linspace(min(b_min) - 3 * popt[2], max(b_min) + 3 * popt[2], 100)
#y_fit_zoomed = gaussian_plus_constant(x_fit_zoomed, *popt)
y_fit_full = gaussian_plus_constant(x_fit_full, *popt)

# Plot the data and the fitted curve (zoomed in)
plt.figure(figsize=(10, 6))
plt.scatter([], [], marker='', label='1. MaNGA 1-26878')
plt.errorbar(b_min, flux, markersize=9, c='red', yerr=errors, fmt='bo', label='L Band Data')
plt.plot(x_fit_full, y_fit_full, color='purple', label='Gaussian Fit + Constant')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux (milliJy)', fontsize=19)
plt.xlim(0, 34000)
plt.scatter([], [], marker='', label=f'Const.: {popt[0]:.2f}\nAmp.: {popt[1]:.2f}\nS. D.: {popt[2]:.2f}\nMean: 0')
legend = plt.legend(fontsize=14, loc='upper right')
for text in legend.texts:
    if text.get_text() == '1. MaNGA 1-26878':
        text.set_fontweight('bold')
plt.grid(True)
#plt.text(0.05, 0.05, f'Const.: {popt[0]:.2f}\nAmp.: {popt[1]:.2f}\nS. D.: {popt[2]:.2f}\nMean: 0',
#         transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom')
plt.tight_layout()
plt.show()

# C band 1
# Constant + gaussian

b_min = np.array([3445, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000])
flux = np.array([1.4082, 1.3769, 1.3382, 1.1706, 0.71879, 0.45815, 0.37574, 0.31483, 0.17497])
errors = np.array([0.0762, 0.0778, 0.0806, 0.0745, 0.058, 0.0454, 0.0482, 0.0477, 0.0517])

# Define a Gaussian function with peak at 0 and constant
def gaussian_plus_constant(x, constant, amplitude, stddev):
    return constant + amplitude * np.exp(-((x - 0) / stddev) ** 2 / 2)

# Define the chi-square function
def chi_square(params):
    model_flux = gaussian_plus_constant(b_min, *params)
    residuals = (flux - model_flux) / errors
    return np.sum(residuals ** 2)

# Initial guess for the parameters
initial_guess = [1, 1, np.std(b_min)]

# Minimize the chi-square function to find the best-fit parameters
result = minimize(chi_square, initial_guess)

# Extract the optimized parameters
popt = result.x

# Generate points for the fitted Gaussian curve with extended range
#x_fit_zoomed = np.linspace(min(b_min), max(b_min), 100)
x_fit_full = np.linspace(min(b_min) - 3 * popt[2], max(b_min) + 3 * popt[2], 100)
#y_fit_zoomed = gaussian_plus_constant(x_fit_zoomed, *popt)
y_fit_full = gaussian_plus_constant(x_fit_full, *popt)

# Plot the data and the fitted curve (zoomed in)
plt.figure(figsize=(10, 6))
plt.scatter([], [], marker='', label='1. MaNGA 1-26878')
plt.errorbar(b_min, flux, markersize=9, c='blue', yerr=errors, fmt='bo', label='C Band 1 Data')
plt.plot(x_fit_full, y_fit_full, c='purple', label='Gaussian Fit + Constant')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux (milliJy)', fontsize=19)
plt.xlim(0, 44000)
plt.scatter([], [], marker='', label=f'Const.: {popt[0]:.2f}\nAmp.: {popt[1]:.2f}\nS. D.: {popt[2]:.2f}\nMean: 0')
legend = plt.legend(fontsize=14, loc='upper right')
for text in legend.texts:
    if text.get_text() == '1. MaNGA 1-26878':
        text.set_fontweight('bold')
plt.grid(True)
#plt.text(0.05, 0.05, f'Const.: {popt[0]:.2f}\nAmp.: {popt[1]:.2f}\nS. D.: {popt[2]:.2f}\nMean: 0',
#         transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom')
plt.tight_layout()
plt.show()

# C Band 2
# Straight line
b_min2 = np.array([3445, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 
            50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000])
flux2 = np.array([4.49E-01, 4.39E-01, 4.41E-01, 4.45E-01, 4.30E-01, 3.91E-01, 4.11E-01, 4.05E-01, 3.82E-01, 
           3.75E-01, 3.90E-01, 3.58E-01, 3.82E-01, 3.71E-01, 3.93E-01, 3.87E-01, 3.84E-01, 
           3.09E-01, 2.68E-01, 2.58E-01])
errors2 = np.array([1.71E-02, 1.75E-02, 1.78E-02, 1.77E-02, 1.86E-02, 1.93E-02, 2.05E-02, 2.08E-02, 2.21E-02, 
             2.33E-02, 2.49E-02, 2.67E-02, 2.76E-02, 2.96E-02, 3.13E-02, 3.48E-02, 3.87E-02, 
             4.43E-02, 4.77E-02, 5.12E-02])

from scipy.optimize import curve_fit 

# Define a linear function
def linear(x, slope, intercept):
    return slope * x + intercept

# Fit the linear model to the data
popt, pcov = curve_fit(linear, b_min2, flux2, sigma=errors2)

# Generate data for the fitted line
x_fit_full = np.linspace(0, 99000, 100)
y_fit_full = linear(x_fit_full, *popt)

# Plotting the data and the fitted line
plt.figure(figsize=(10, 6))
plt.scatter([], [], marker='', label='1. MaNGA 1-26878')
plt.errorbar(b_min2, flux2, yerr=errors2, fmt='bo', markersize=9, c='green', label='C Band 2 Data')
plt.plot(x_fit_full, y_fit_full, c='purple', label='Linear Fit')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux2 (milliJy)', fontsize=19)
plt.xlim(0, 99000)
plt.scatter([], [], marker='', label=f'Slope: {popt[0]:.2e}\nIntercept: {popt[1]:.2f}')
legend = plt.legend(fontsize=14, loc='lower left')

# Bold the specific legend entry
for text in legend.texts:
    if text.get_text() == '1. MaNGA 1-26878':
        text.set_fontweight('bold')

plt.grid(True)
plt.tight_layout()
plt.show()

# Find common b_min values in increasing order
common_b_min = sorted(list(set(c_b_min) & set(l_b_min)))
common_b_min2 = sorted(list(set(c2_b_min) & set(l_b_min)))

# Calculate spectral index for each common b_min
spectral_indices = []
for b_min in common_b_min:
    c_index = c_b_min.index(b_min)
    l_index = l_b_min.index(b_min)
    c_flux_val = c_flux[c_index]
    l_flux_val = l_flux[l_index]
    si = np.log10(c_flux_val / l_flux_val) / np.log10(4.8 / 1.4)
    spectral_indices.append(si)

spectral_indices2 = []
for b_min in common_b_min2:
    c_index = c2_b_min.index(b_min)
    l_index = l_b_min.index(b_min)
    c_flux_val = c2_flux[c_index]
    l_flux_val = l_flux[l_index]
    si = np.log10(c_flux_val / l_flux_val) / np.log10(4.8 / 1.4)
    spectral_indices2.append(si)

# Calculate errors for spectral indices
errors = []
for b_min in common_b_min:
    c_index = c_b_min.index(b_min)
    l_index = l_b_min.index(b_min)
    c_flux_val = c_flux[c_index]
    l_flux_val = l_flux[l_index]
    c_error_val = c_errors[c_index]
    l_error_val = l_errors[l_index]
    error = (1/np.log(4.8/1.4)) * np.sqrt((c_error_val/c_flux_val)**2 + (l_error_val/l_flux_val)**2)
    errors.append(error)


errors2 = []
for b_min in common_b_min2:
    c_index = c2_b_min.index(b_min)
    l_index = l_b_min.index(b_min)
    c_flux_val = c2_flux[c_index]
    l_flux_val = l_flux[l_index]
    c_error_val = c2_errors[c_index]
    l_error_val = l_errors[l_index]
    error = (1/np.log(4.8/1.4)) * np.sqrt((c_error_val/c_flux_val)**2 + (l_error_val/l_flux_val)**2)
    errors2.append(error)
        
# Plot
plt.figure(figsize=(10, 6))
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

plt.scatter([], [], marker='', label='1. MaNGA 1-26878')
print(common_b_min)
print(spectral_indices)
print(errors)
for i in range(len(common_b_min)):
    if i == len(common_b_min)-1:
        marker = '^'
        error=0
    else:
        marker = 'o'
        error=errors[i]

    plt.scatter(common_b_min[i], spectral_indices[i], s=80, c='red', marker=marker)
    plt.errorbar(common_b_min[i], spectral_indices[i], yerr=error, fmt='none', c='red', linewidth=2)
print(common_b_min2)
print(spectral_indices2)
print(errors2)
for i in range(len(common_b_min2)):
    if i == len(common_b_min2)-1:
        marker = '^'
        error=0
    else:
        marker = 'o'
        error=errors2[i]

    plt.scatter(common_b_min2[i], spectral_indices2[i], s=80, c='green', marker=marker)
    plt.errorbar(common_b_min2[i], spectral_indices2[i], yerr=error, fmt='none', c='green', linewidth=2)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Spectral Index', fontsize=19)
plt.scatter([], [], s=40, c='red', marker='o', label='C Band 1')
plt.scatter([], [], s=40, c='green', marker='o', label='C Band 2')
plt.scatter([], [], s=40, c='black', marker='o', label='Detection')
plt.scatter([], [], s=40, c='black', marker='^', label='Non-detection (L Band)')
plt.grid(True)
legend = plt.legend(fontsize=14, loc='lower center')
for text in legend.texts:
    if text.get_text() == '1. MaNGA 1-26878':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()

# Positions
'''
# L Band Data
l_b_min = [1090, 5000, 10000, 15000, 20000, 25000, 30000]
l_x = [1.8212, 1.8562, 1.908, 1.9756, 1.9552, 2.4376, 6.1282]
l_x_error = [0.01044, 0.01188, 0.01466, 0.02406, 0.03658, 0.24238, 0.19582]
l_y = [-3.5516, -3.4184, -3.1832, -2.5736, -2.9426, -3.771, -0.0446]
l_y_error = [0.0235, 0.02862, 0.04212, 0.09022, 0.09818, 0.53344, 0.48554]

# Plot
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='1. MaNGA 1-26878')

# Scatter plot with error bars colored by b_min
sc = plt.scatter(l_x, l_y, c=l_b_min, cmap='coolwarm', marker='o', s=65)
plt.scatter([], [], marker='', label='L Band')

# Adding error bars
plt.errorbar(l_x, l_y, xerr=l_x_error, yerr=l_y_error, fmt='none', c='black', alpha=0.5)

# Color bar
cbar = plt.colorbar(sc)
cbar.set_label('Minimum Baseline')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('X (milliarcsec)', fontsize=19)
plt.ylabel('Y (milliarcsec)', fontsize=19)
plt.grid(True)
legend = plt.legend(fontsize=14, loc='upper left')
for text in legend.texts:
    if text.get_text() == '1. MaNGA 1-26878':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()
'''

# Data
c_b_min = [3445, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
c_x = [7.4292, 7.37, 7.3918, 7.4588, 7.7328, 7.9344, 7.9244, 7.942, 8.1144]
c_x_error = [0.069, 0.0725, 0.0758, 0.0745, 0.06512, 0.05218, 0.05778, 0.0636, 0.11638]
c_y = [-10.727, -10.8828, -10.845, -10.6972, -10.3264, -10.1984, -10.2158, -10.3038, -9.48]
c_y_error = [0.14304, 0.15034, 0.1574, 0.15234, 0.1303, 0.10468, 0.13972, 0.16896, 0.44258]

major = [7.399, 7.485, 7.318]
minor = [1.253, 1.130, 1.172]
angle = [25.049219, 25.327436, 25.218582]

norm = cm.colors.Normalize(vmin=min(c_b_min), vmax=max(c_b_min))
cmap = cm.get_cmap('coolwarm')

# Plot 1
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='1. MaNGA 1-26878')
sc = plt.scatter(c_x, c_y, c=c_b_min, cmap='coolwarm', s=65)
plt.scatter([], [], marker='', label='C Band (South-West Lobe)')
plt.errorbar(c_x, c_y, xerr=c_x_error, yerr=c_y_error, fmt='none', color='black', alpha=0.5)

# Add ellipses
for i in range(len(major)):
    color = cmap(norm(c_b_min[i]))
    ellipse = patches.Ellipse(
        (c_x[i], c_y[i]),
        width=major[i],
        height=minor[i],
        angle=(90-angle[i]),
        edgecolor=color,
        facecolor='none'
    )
    plt.gca().add_patch(ellipse)

cbar = plt.colorbar(sc)
cbar.set_label('Minimum Baseline')

plt.xlabel('X (milliarcsec)', fontsize=19)
plt.ylabel('Y (milliarcsec)', fontsize=19)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(9.5, 5.5)
plt.grid(True)
#plt.scatter([], [], marker='', label='Offset Angle = 79.1°')
plt.scatter([], [], marker='', label='0.5 mas = 0.6 pc')
legend = plt.legend(fontsize=14, loc='upper right')
for text in legend.texts:
    if text.get_text() == '1. MaNGA 1-26878':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()

# Plot 2:
c_b_min2 = [3445, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 
            50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000]
c_x2 = [10.1118, 10.0944, 10.1044, 10.0982, 10.088, 10.046, 10.025, 10.0258, 10.0276, 10.0216, 10.0328, 10.0142, 10.0482, 10.0538, 10.0844, 10.0428, 10.0342, 10.0232, 10.0392, 10.1166]
c_x_error2 = [0.03076, 0.03022, 0.0299, 0.02962, 0.02906, 0.02918, 0.02912, 0.0279, 0.03072, 0.0307, 0.0295, 0.03228, 0.03106, 0.03306, 0.03432, 0.03698, 0.03748, 0.0479, 0.0717, 0.09458]
c_y2 =  [-1.3084, -1.2992, -1.301, -1.282, -1.2882, -1.1498, -1.1736, -1.193, -1.1822, -1.162, -1.3224, -1.3236, -1.461, -1.4988, -1.5678, -1.427, -1.2294, -1.213, -1.2382, -1.4468]
c_y_error2 = [0.06318, 0.06292, 0.06214, 0.06168, 0.06048, 0.06182, 0.0645, 0.06708, 0.07712, 0.08644, 0.087, 0.09562, 0.09136, 0.0969, 0.10208, 0.11164, 0.10982, 0.13662, 0.19982, 0.27128]

norm2 = cm.colors.Normalize(vmin=min(c_b_min2), vmax=max(c_b_min2))
cmap2 = cm.get_cmap('coolwarm')

# Plot 1
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='1. MaNGA 1-26878')
sc2 = plt.scatter(c_x2, c_y2, c=c_b_min2, cmap='coolwarm', s=65)
plt.scatter([], [], marker='', label='C Band 2')
plt.errorbar(c_x2, c_y2, xerr=c_x_error2, yerr=c_y_error2, fmt='none', color='black', alpha=0.5)

cbar = plt.colorbar(sc2)
cbar.set_label('Minimum Baseline')

plt.xlabel('X (milliarcsec)', fontsize=19)
plt.ylabel('Y (milliarcsec)', fontsize=19)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.xlim(9.5, 5.5)
plt.grid(True)
#plt.scatter([], [], marker='', label='Offset Angle = 79.1°')
plt.scatter([], [], marker='', label='0.05 mas = 0.06 pc')
legend = plt.legend(fontsize=14, loc='upper right')
for text in legend.texts:
    if text.get_text() == '1. MaNGA 1-26878':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()