import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import matplotlib.patches as patches
import matplotlib.cm as cm

# Plot
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='4. MaNGA 1-43718')

# L Band
l_b_min = [1100, 4000, 7500, 10000, 15000, 20000, 25000, 30000]
l_flux = [1.8237, 1.8864, 1.376, 1.289, 1.1447, 1.098, 0.97725, 0.318]
l_errors = [0.0622, 0.0624, 0.033, 0.038, 0.047, 0.0614, 0.0874, 0]
l_rms = [0.0289, 0.03084, 0.03177, 0.03858, 0.04505, 0.05632, 0.0836, 0.106]
l_resolved = ['Resolved', 'Resolved', 'Unresolved', 'Unresolved', 'Unresolved', 'Unresolved', 'Unresolved', 'Non-detection']
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
c_b_min = [3620, 4000, 7500, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000]
c_flux = [1.2746, 1.2146, 1.2231, 1.2636, 1.1838, 1.0023, 0.63651, 0.42479, 0.45482, 0.44831, 0.43609, 0.34183, 0.30904, 0.23532, 0.23135, 0.32822, 0.3165, 0.2813, 0.16335]
c_errors = [0.0556, 0.0569, 0.0574, 0.0592, 0.0569, 0.0545, 0.0456, 0.0241, 0.0247, 0.0253, 0.0262, 0.0281, 0.0289, 0.0336, 0.0324, 0.0347, 0.0364, 0.043, 0]
c_rms = [0.02148, 0.02322, 0.02331, 0.0233, 0.02362, 0.02254, 0.02277, 0.02444, 0.02565, 0.02683, 0.02677, 0.02847, 0.02888, 0.02935, 0.03314, 0.03597, 0.03938, 0.04706, 0.05445]
c_resolved = ['Resolved'] * 7 + ['Unresolved'] * 11 + ['Non-detection']

size=60
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
plt.scatter([], [], s=40, c='black', marker='s', label='Resolved (Square)')
plt.scatter([], [], s=40, c='black', marker='o', label='Unresolved (Circle)')
plt.scatter([], [], s=40, c='black', marker='v', label='Non-detection (Down-triangle)')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux (milliJy)', fontsize=19)
legend = plt.legend(fontsize=14)
for text in legend.texts:
    if text.get_text() == '4. MaNGA 1-43718':
        text.set_fontweight('bold')
        break
plt.grid(True)
plt.tight_layout()
plt.show()

# Fittings

# L Band:
# 2 Gaussian:

b_min = np.array([1100, 4000, 7500, 10000, 15000, 20000, 25000])
flux = np.array([1.8237, 1.8864, 1.376, 1.289, 1.1447, 1.098, 0.97725])
errors = np.array([0.0622, 0.0624, 0.033, 0.038, 0.047, 0.0614, 0.0874])

# Define a Gaussian function with peak at specified mean
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

# Define the bimodal Gaussian model
def bimodal_gaussian(x, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2):
    return (gaussian(x, amplitude1, mean1, stddev1) + 
            gaussian(x, amplitude2, mean2, stddev2))

# Define the chi-square function
def chi_square(params):
    model_flux = bimodal_gaussian(b_min, *params)
    residuals = (flux - model_flux) / errors
    return np.sum(residuals ** 2)

# Initial guess for the parameters
initial_guess = [1, 10000, 5000, 1, 20000, 5000]  # Adjust mean values to be further apart

# Minimize the chi-square function
# Adding bounds
bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]
result = minimize(chi_square, initial_guess, bounds=bounds)

# Extract the optimized parameters
popt = result.x

# Generate points for the fitted curve
#x_fit_zoomed = np.linspace(min(b_min), max(b_min), 100)
x_fit_full = np.linspace(min(b_min) - 3 * popt[2], max(b_min) + 3 * popt[5], 100)
#y_fit_zoomed = bimodal_gaussian(x_fit_zoomed, *popt)
y_fit_full = bimodal_gaussian(x_fit_full, *popt)

# Plot the data and the fitted curve (zoomed in)
plt.figure(figsize=(10, 6))
plt.scatter([], [], marker='', label='4. MaNGA 1-43718')
plt.errorbar(b_min, flux, markersize=9, c='red', yerr=errors, fmt='bo', label='L Band Data')
plt.plot(x_fit_full, y_fit_full, c='purple', label='Bimodal Gaussian Fit')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux (milliJy)', fontsize=19)
plt.xlim(0, 29000)
plt.scatter([], [], marker='', label=f'Amp. 1: {popt[0]:.2f}\nMean 1: {popt[1]:.2f}\nS.D. 1: {popt[2]:.2f}\n'
                     f'Amp. 2: {popt[3]:.2f}\nMean 2: {popt[4]:.2f}\nS.D. 2: {popt[5]:.2f}')
legend = plt.legend(fontsize=14, loc='upper right')
for text in legend.texts:
    if text.get_text() == '4. MaNGA 1-43718':
        text.set_fontweight('bold')
plt.grid(True)
plt.tight_layout()
plt.show()

# C Band:
# 2-gaussian

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define a Gaussian function with new names
def gaussian_peak(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

# Define a bimodal Gaussian with a constant term
def bimodal_plus_constant(x, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, constant):
    return (
        gaussian_peak(x, amplitude1, mean1, stddev1) +
        gaussian_peak(x, amplitude2, mean2, stddev2) +
        constant
    )

# Chi-square function
def chi_square_with_constant(params):
    model_flux = bimodal_plus_constant(b_min, *params)
    residuals = (flux - model_flux) / errors
    return np.sum(residuals ** 2)

# Your data
b_min = np.array([3620, 4000, 7500, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000])
flux = np.array([1.2746, 1.2146, 1.2231, 1.2636, 1.1838, 1.0023, 0.63651, 0.42479, 0.45482, 0.44831, 0.43609, 0.34183, 0.30904, 0.23532, 0.23135, 0.32822, 0.3165, 0.2813])
errors = np.array([0.0556, 0.0569, 0.0574, 0.0592, 0.0569, 0.0545, 0.0456, 0.0241, 0.0247, 0.0253, 0.0262, 0.0281, 0.0289, 0.0336, 0.0324, 0.0347, 0.0364, 0.043])

# Initial guess with broader starting values for the second Gaussian
initial_guess = [1, 10000, 3000, 1, 40000, 5000, 0.2]

# Adjusted bounds for the optimization process
bounds = [
    (0, 0.99),  # Amplitude 1
    (0, 10000),  # Mean 1 (closer to the first peak)
    (1000, 10000),  # Stddev 1 (broader range for flexibility)
    (0, None),  # Amplitude 2 (ensure it's not constrained to zero)
    (35000, 45000),  # Mean 2 (allow wider range for flexibility)
    (1000, 15000),  # Stddev 2 (broader range to avoid collapse to zero)
    (0, None)  # Constant term
]

# Optimize the chi-square function with adjusted bounds
result = minimize(chi_square_with_constant, initial_guess, bounds=bounds)

# Extract the optimized parameters
popt = result.x

# Generate data for the fitted curve
x_fit = np.linspace(0, 85000, 100)
y_fit = bimodal_plus_constant(x_fit, *popt)

# Plot the data and the fitted curve
plt.figure(figsize=(10, 6))
plt.errorbar(b_min, flux, markersize=9, yerr=errors, fmt='bo', label='Data')
plt.plot(x_fit, y_fit, color='purple', label='Bimodal + Constant Fit')
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux', fontsize=19)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.xlim(0, 84000)
param_text = '\n'.join([f'{name}: {val:.2f}' for name, val in zip(
    ["Amp. 1", "Mean. 1", "StdDev. 1", "Amp. 2", "Mean. 2", "StdDev. 2", "Const."],
    popt
)])
plt.scatter([], [], marker='', label=param_text)
plt.legend(fontsize=14, loc='upper right')
plt.tight_layout()
plt.show()
'''
# Define a Gaussian function that includes a constant (baseline) term
def gaussian_with_offset(x, amplitude, mean, stddev, constant):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2) + constant

# Define a bimodal Gaussian model with individual baseline terms and a global constant term
def bimodal_gaussian_with_offset(x, amplitude1, mean1, stddev1, constant1, amplitude2, mean2, stddev2, constant2, global_constant):
    return (
        gaussian_with_offset(x, amplitude1, mean1, stddev1, constant1) +
        gaussian_with_offset(x, amplitude2, mean2, stddev2, constant2) +
        global_constant
    )

# Chi-square function
def chi_square_with_offsets(params):
    model_flux = bimodal_gaussian_with_offset(b_min, *params)
    residuals = (flux - model_flux) / errors
    return np.sum(residuals ** 2)

# Your data
b_min = np.array([3620, 4000, 7500, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000])
flux = np.array([1.2746, 1.2146, 1.2231, 1.2636, 1.1838, 1.0023, 0.63651, 0.42479, 0.45482, 0.44831, 0.43609, 0.34183, 0.30904, 0.23532, 0.23135, 0.32822, 0.3165, 0.2813])
errors = np.array([0.0556, 0.0569, 0.0574, 0.0592, 0.0569, 0.0545, 0.0456, 0.0241, 0.0247, 0.0253, 0.0262, 0.0281, 0.0289, 0.0336, 0.0324, 0.0347, 0.0364, 0.043])

# Initial guess, including the additional constant terms for both Gaussian peaks and a global constant
initial_guess = [1, 15000, 3000, 0.1, 1, 40000, 3000, 0.1, 0.1]

# Bounds for optimization
bounds = [
    (0, None),  # Amplitude 1
    (5000, 15000),  # Mean 1
    (1000, 10000),  # Stddev 1
    (0, None),  # Constant for first Gaussian
    (0, None),  # Amplitude 2
    (35000, 45000),  # Mean 2
    (1000, 15000),  # Stddev 2
    (0, None),  # Constant for second Gaussian
    (0, None)  # Global constant
]

# Optimize the chi-square function with additional constants
result = minimize(chi_square_with_offsets, initial_guess, bounds=bounds)

# Extract optimized parameters
popt = result.x

# Generate the fitted curve
x_fit = np.linspace(0, 85000, 100)
y_fit = bimodal_gaussian_with_offset(x_fit, *popt)

# Plot the data and the fitted curve
plt.figure(figsize=(10, 6))
plt.errorbar(b_min, flux, markersize=9, yerr=errors, fmt='bo', label='Data')
plt.plot(x_fit, y_fit, color='purple', label='Bimodal Gaussian with Offsets')
plt.xlabel('Minimum Baseline')
plt.ylabel('Flux')
plt.grid(True)
plt.xlim(0, 84000)
plt.legend(fontsize=14, loc='upper right')

# Display the optimized parameters
param_text = '\n'.join([f'{name}: {val:.2f}' for name, val in zip(
    ["Amp1", "Mean1", "StdDev1", "Const1", "Amp2", "Mean2", "StdDev2", "Const2", "Global Constant"],
    popt
)])
plt.text(0.05, 0.25, param_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.show()
'''

# Find common b_min values in increasing order
common_b_min = sorted(list(set(c_b_min) & set(l_b_min)))

# Calculate spectral index for each common b_min
spectral_indices = []
for b_min in common_b_min:
    c_index = c_b_min.index(b_min)
    l_index = l_b_min.index(b_min)
    c_flux_val = c_flux[c_index]
    l_flux_val = l_flux[l_index]
    si = np.log10(c_flux_val / l_flux_val) / np.log10(4.8 / 1.4)
    spectral_indices.append(si)

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
        
# Plot
plt.figure(figsize=(10, 6))
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

plt.scatter([], [], marker='', label='4. MaNGA 1-43718')
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

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Spectral Index', fontsize=19)
plt.scatter([], [], s=40, c='red', marker='o', label='Detection')
plt.scatter([], [], s=40, c='red', marker='^', label='Non-detection (L Band)')
plt.grid(True)
legend = plt.legend(fontsize=14)
for text in legend.texts:
    if text.get_text() == '4. MaNGA 1-43718':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()

# Positions

# L Band is blamed on poor angular resolution !!!
'''
# L Band Data
l_b_min = [1100, 4000, 7500, 10000, 15000, 20000, 25000]
l_x = [0.1046, 0.0914, 0.1258, 0.1152, 0.0758, 0.0884, 0.1354]
l_x_error = [0.01496, 0.01492, 0.01474, 0.01604, 0.0207, 0.02712, 0.0441]
l_y = [0.3048, 0.3096, 0.3096, 0.434, 0.661, 0.6338, 0.6014]
l_y_error = [0.03186, 0.03176, 0.03272, 0.03872, 0.05532, 0.07872, 0.1211]

# Plot
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='4. MaNGA 1-43718')

# Scatter plot with error bars colored by b_min
sc = plt.scatter(l_x, l_y, c=l_b_min, cmap='coolwarm', marker='o', s=65)
plt.scatter([], [], marker='', label='L Band')

# Adding error bars
plt.errorbar(l_x, l_y, xerr=l_x_error, yerr=l_y_error, fmt='none', c='black', alpha=0.5)

# Color bar
cbar = plt.colorbar(sc)
cbar.set_label('Minimum Baseline')

plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('X (milliarcsec)')
plt.ylabel('Y (milliarcsec)')
plt.grid(True)
legend = plt.legend(loc='center left')
for text in legend.texts:
    if text.get_text() == '3. MaNGA 1-19818 (bg284f)':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()
'''

# C Band Data
c_b_min = [3620, 4000, 7500, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000]
c_x = [2.4072, 2.409, 2.419, 2.3842, 2.341, 2.3264, 2.1164, 1.8586, 1.8036, 1.628, 1.6494, 1.6312, 1.6816, 1.4718, 4.3272, 1.9432, 1.746, 1.6636]
c_x_error = [0.03904, 0.04068, 0.04106, 0.04086, 0.03972, 0.03988, 0.0364, 0.04058, 0.03832, 0.03412, 0.03518, 0.04902, 0.06186, 0.07756, 0.0769, 0.05898, 0.06356, 0.07492]
c_y = [2.4878, 2.4702, 2.4666, 2.4692, 2.4864, 2.507, 2.5594, 3.2068, 3.2438, 3.7558, 3.6736, 3.721, 3.4734, 3.9066, 4.1204, 2.7446, 3.6836, 3.8088]
c_y_error = [0.05274, 0.05562, 0.05576, 0.05656, 0.05652, 0.061, 0.06882, 0.09168, 0.08526, 0.07792, 0.0882, 0.1347, 0.17746, 0.27454, 0.18584, 0.1693, 0.19552, 0.21408]

major = [2.340, 2.290, 2.299, 2.337, 2.257, 2.202]
minor = [1.587, 1.521, 1.526, 1.728, 1.596, 1.325]
angle = [98.529282, 104.021431, 101.73243, 110.84391, 117.774994, 132.028]

norm = cm.colors.Normalize(vmin=min(c_b_min), vmax=max(c_b_min))
cmap = cm.get_cmap('coolwarm')

# Plot
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='4. MaNGA 1-43718')
sc = plt.scatter(c_x, c_y, c=c_b_min, cmap='coolwarm', s=65)
plt.scatter([], [], marker='', label='C Band')
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
plt.grid(True)
#plt.scatter([], [], marker='', label='Offset Angle = 57.1°')
plt.xlim(4.5, 1)
plt.scatter([], [], marker='', label='0.5 mas = 0.425 pc')
legend = plt.legend(fontsize=14, loc='lower left')
for text in legend.texts:
    if text.get_text() == '4. MaNGA 1-43718':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()