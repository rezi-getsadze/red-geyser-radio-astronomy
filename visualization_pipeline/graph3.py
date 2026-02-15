import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# Plot
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='3. MaNGA 1-19818')

# L Band
l_b_min = [1100, 5000, 10000, 15000, 20000, 25000]
l_flux = [0.60871, 0.48661, 0.56061, 0.4428, 0.40625, 0.23907]
l_errors = [0.0713, 0.0415, 0.0509, 0.0505, 0.062, 0]
l_rms = [0.03329, 0.04253, 0.05059, 0.05018, 0.06072, 0.07969]
l_resolved = ['Resolved', 'Unresolved', 'Unresolved', 'Unresolved', 'Unresolved', 'Non-detection']
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
c_b_min = [3665, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000]
c_flux = [0.12241, 0.11994, 0.12023, 0.11627, 0.11493, 0.11868, 0.12572, 0.12055, 0.12903, 0.13134, 0.12726, 0.13165, 0.12428, 0.11993, 0.10842, 0.11634, 0.13771, 1.78E-01]
c_errors = [0.015, 0.0152, 0.0161, 0.0163, 0.0169, 0.0179, 0.0195, 0.0202, 0.0216, 0.0225, 0.0246, 0.0258, 0.0277, 0.0301, 0.0319, 0.0339, 0.0405, 0, 0]
c_rms = [0.01515, 0.01534, 0.01666, 0.01664, 0.01728, 0.01821, 0.01938, 0.01932, 0.0211, 0.02217, 0.02511, 0.02638, 0.02682, 0.02902, 0.03179, 0.03376, 0.04512, 0.05939, 0]
c_resolved = ['Unresolved'] * (len(c_b_min) - 1) + ['Non-detection']
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
    if text.get_text() == '3. MaNGA 1-19818':
        text.set_fontweight('bold')
        break
plt.grid(True)
plt.tight_layout()
plt.show()

#Fittings:

# L Band:
# Keep at 0

# Define your data
b_min = np.array([1100, 5000, 10000, 15000, 20000])
flux = np.array([0.60871, 0.48661, 0.56061, 0.4428, 0.40625])
errors = np.array([0.0713, 0.0415, 0.0509, 0.0505, 0.062])

# Define a Gaussian function with peak at 0
def gaussian(x, amplitude, stddev):
    return amplitude * np.exp(-((x - 0) / stddev) ** 2 / 2)

# Define the chi-square function
def chi_square(params):
    model_flux = gaussian(b_min, *params)
    residuals = (flux - model_flux) / errors
    return np.sum(residuals ** 2)

# Initial guess for the parameters
initial_guess = [1, np.std(b_min)]

# Minimize the chi-square function to find the best-fit parameters
result = minimize(chi_square, initial_guess)

# Extract the optimized parameters
popt = result.x

#x_fit_zoomed = np.linspace(min(b_min), max(b_min), 100)
x_fit_full = np.linspace(min(b_min) - 3 * popt[1], max(b_min) + 3 * popt[1], 100)
#y_fit_zoomed = gaussian(x_fit_zoomed, *popt)
y_fit_full = gaussian(x_fit_full, *popt)

# Plot the data and the fitted curve (zoomed in)
plt.figure(figsize=(10, 6))
plt.scatter([], [], marker='', label='3. MaNGA 1-19818')
plt.errorbar(b_min, flux, markersize=9, c='red', yerr=errors, fmt='bo', label='L Band Data')
plt.plot(x_fit_full, y_fit_full, color='purple', label='Gaussian Fit')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux (milliJy)', fontsize=19)
plt.xlim(0, 22500)
plt.scatter([], [], marker='', label=f'Amp.: {popt[0]:.2f}\nS. D.: {popt[1]:.2f}\nMean: 0')
legend = plt.legend(fontsize=14, loc='lower right')
for text in legend.texts:
    if text.get_text() == '3. MaNGA 1-19818':
        text.set_fontweight('bold')
plt.grid(True)
#plt.text(0.05, 0.05, f'Amp.: {popt[0]:.2f}\nS. D.: {popt[1]:.2f}\nMean: 0',
#         transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom')
plt.tight_layout()
plt.show()

# C Band:
# Straight line:
b_min = np.array([3665, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000])
flux = np.array([0.12241, 0.11994, 0.12023, 0.11627, 0.11493, 0.11868, 0.12572, 0.12055, 0.12903, 0.13134, 0.12726, 0.13165, 0.12428, 0.11993, 0.10842, 0.11634, 0.13771])
errors = np.array([0.015, 0.0152, 0.0161, 0.0163, 0.0169, 0.0179, 0.0195, 0.0202, 0.0216, 0.0225, 0.0246, 0.0258, 0.0277, 0.0301, 0.0319, 0.0339, 0.0405])

# Define a constant function (horizontal line)
def constant(x, value):
    return np.full_like(x, value)

# Define the chi-square function for the horizontal line fit
def chi_square_constant(params):
    model_flux = constant(b_min, params[0])
    residuals = (flux - model_flux) / errors
    return np.sum(residuals ** 2)

# Initial guess for the constant value
initial_guess_constant = [np.mean(flux)]

# Minimize the chi-square for the constant fit
result_constant = minimize(chi_square_constant, initial_guess_constant)

# Extract the optimized parameters for the constant fit
constant_value = result_constant.x[0]

# Generate points for the fitted constant function
x_fit = np.linspace(0, 84000, 100)
y_fit = constant(x_fit, constant_value)

# Plot the data and the fitted curve (zoomed in)
plt.figure(figsize=(10, 6))
plt.scatter([], [], marker='', label='3. MaNGA 1-19818')
plt.errorbar(b_min, flux, markersize=9, yerr=errors, fmt='bo', label='C Band Data')
plt.plot(x_fit, y_fit, color='purple', label='Linear Fit')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 84000)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux (milliJy)', fontsize=19)
plt.scatter([], [], marker='', label=f'Constant value: {constant_value:.3f}')
legend = plt.legend(fontsize=14, loc='upper left')
for text in legend.texts:
    if text.get_text() == '3. MaNGA 1-19818':
        text.set_fontweight('bold')
plt.grid(True)
# Annotate with line equation
#plt.text(0.05, 0.05, f'Constant value: {constant_value:.3f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.tight_layout()
plt.show()


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

plt.scatter([], [], marker='', label='3. MaNGA 1-19818')

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Spectral Index', fontsize=19)
plt.scatter([], [], s=40, c='red', marker='o', label='Detection')
plt.scatter([], [], s=40, c='red', marker='^', label='Non-detection (L Band)')
plt.grid(True)
legend = plt.legend(fontsize=14, loc='upper right')
for text in legend.texts:
    if text.get_text() == '3. MaNGA 1-19818':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()

# Positions
'''
# L Band Data
l_b_min = [1100, 5000, 10000, 15000, 20000]
l_x = [-2.193, -2.062, -2.0618, -2.0604, -2.0342]
l_x_error = [0.06258, 0.05218, 0.0464, 0.0658, 0.0744]
l_y = [-2.3244, -2.8476, -2.783, -2.8576, -2.8148]
l_y_error = [0.14438, 0.1164, 0.11632, 0.15622, 0.23034]

# Plot
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='3. MaNGA 1-19818')

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
plt.xlabel('X (milliarcsec)', fontsize=19)
plt.ylabel('Y (milliarcsec)', fontsize=19)
plt.grid(True)
legend = plt.legend(loc='upper right')
for text in legend.texts:
    if text.get_text() == '3. MaNGA 1-19818':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()
'''
# C Band Data
c_b_min = [3665, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000]
c_x = [-6.5284, -6.4486, -6.4794, -6.4542, -6.3702, -6.4612, -6.4484, -6.5128, -6.4742, -6.4528, -6.5536, -6.5326, -6.5686, -6.4576, -6.6784, -6.7608, -6.7684]
c_x_error = [0.12208, 0.12436, 0.12408, 0.13074, 0.13684, 0.13612, 0.1398, 0.15272, 0.13618, 0.13694, 0.14982, 0.15212, 0.17752, 0.1904, 0.1829, 0.1955, 0.1858]
c_y = [-11.3212, -11.5174, -11.44, -11.5338, -11.6418, -11.5284, -11.4602, -11.3766, -11.4556, -11.4934, -11.3436, -11.3442, -11.2454, -11.3276, -10.9674, -10.8736, -10.8868]
c_y_error = [0.21608, 0.2227, 0.2192, 0.23138, 0.24068, 0.24094, 0.24086, 0.25452, 0.23206, 0.23762, 0.27306, 0.28188, 0.29602, 0.31746, 0.30124, 0.30008, 0.28564]

# Plot
plt.figure(figsize=(10, 6))

plt.scatter([], [], marker='', label='3. MaNGA 1-19818')

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
#plt.scatter([], [], marker='', label='Offset Angle = 77.8°')
plt.xlim(-6.2, -7)
plt.scatter([], [], marker='', label='0.1 mas = 0.074 pc')
legend = plt.legend(fontsize=14, loc='upper left')
for text in legend.texts:
    if text.get_text() == '3. MaNGA 1-19818':
        text.set_fontweight('bold')
        break
plt.tight_layout()
plt.show()
