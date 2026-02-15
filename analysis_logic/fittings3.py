import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit

# L Band - Gaussian Fit
b_min_l = np.array([1100, 5000, 10000, 15000, 20000])
flux_l = np.array([0.60871, 0.48661, 0.56061, 0.4428, 0.40625])
errors_l = np.array([0.0713, 0.0415, 0.0509, 0.0505, 0.062])
rms_l = [0.03329, 0.04253, 0.05059, 0.05018, 0.06072, 0.07969]
resolved_l = ['Resolved', 'Unresolved', 'Unresolved', 'Unresolved', 'Unresolved']

# Gaussian function with peak at 0
def gaussian(x, amplitude, stddev):
    return amplitude * np.exp(-((x - 0) / stddev) ** 2 / 2)

# Chi-square function for Gaussian fit
def chi_square(params):
    model_flux = gaussian(b_min_l, *params)
    residuals = (flux_l - model_flux) / errors_l
    return np.sum(residuals ** 2)

# Gaussian fit
initial_guess = [1, np.std(b_min_l)]
result = minimize(chi_square, initial_guess)
popt_l = result.x

x_fit_full_l = np.linspace(min(b_min_l) - 3 * popt_l[1], 30000, 100)
y_fit_full_l = gaussian(x_fit_full_l, *popt_l)

# C Band - Constant Fit
b_min_c = np.array([3665, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000])
flux_c = np.array([0.12241, 0.11994, 0.12023, 0.11627, 0.11493, 0.11868, 0.12572, 0.12055, 0.12903, 0.13134, 0.12726, 0.13165, 0.12428, 0.11993, 0.10842, 0.11634, 0.13771])
errors_c = np.array([0.015, 0.0152, 0.0161, 0.0163, 0.0169, 0.0179, 0.0195, 0.0202, 0.0216, 0.0225, 0.0246, 0.0258, 0.0277, 0.0301, 0.0319, 0.0339, 0.0405])
rms_c = [0.01515, 0.01534, 0.01666, 0.01664, 0.01728, 0.01821, 0.01938, 0.01932, 0.0211, 0.02217, 0.02511, 0.02638, 0.02682, 0.02902, 0.03179, 0.03376, 0.04512, 0.05939, 0]
resolved_c = ['Unresolved'] * (len(b_min_c))

# Constant function
def constant(x, value):
    return np.full_like(x, value)

# Chi-square function for constant fit
def chi_square_constant(params):
    model_flux = constant(b_min_c, params[0])
    residuals = (flux_c - model_flux) / errors_c
    return np.sum(residuals ** 2)

# Constant fit
initial_guess_constant = [np.mean(flux_c)]
result_constant = minimize(chi_square_constant, initial_guess_constant)
constant_value = result_constant.x[0]

x_fit_full_c = np.linspace(0, 90000, 100)
y_fit_full_c = constant(x_fit_full_c, constant_value)

# Combined plot for all three bands and their fittings
plt.figure(figsize=(12, 6))

plt.scatter([], [], marker='', label='3. MaNGA 1-19818')

# L Band
plt.scatter([], [], s=60, c='red', alpha=0.6, marker='o', label='L Band')
plt.scatter(25000, 0.23907, s=160, c='red', alpha=0.6, marker='v')

for i in range(len(b_min_l)):
    if resolved_l[i] == 'Resolved':
        marker = 's'
    elif resolved_l[i] == 'Unresolved':
        marker = 'o'
        
    plt.scatter(b_min_l[i], flux_l[i], s=60, c='red', marker=marker, alpha=0.6)
    plt.errorbar(b_min_l[i], flux_l[i], yerr=errors_l[i], fmt='none', c='red')
    #plt.scatter(l_b_min[i], l_rms[i], s=20, c=(0.8, 0.2, 0.2), marker='D', alpha=0.6, label='L-Band RMS' if i == 0 else None)

#plt.errorbar(b_min_l, flux_l, yerr=errors_l, alpha = 0.6, fmt='o', color='red')
plt.plot(x_fit_full_l, y_fit_full_l, '--', color='darkred', label=f'Gaussian Fit (Amp.: {popt_l[0]:.2f}, S.D.: {popt_l[1]:.2f})')

# C Band
plt.scatter([], [], s=60, c='blue', alpha=0.6, marker='o', label='C Band')
plt.scatter(85000, 1.78E-01, s=160, c='blue', alpha=0.6, marker='v')

for i in range(len(b_min_c)):
    if resolved_c[i] == 'Resolved':
        marker = 's'
    elif resolved_c[i] == 'Unresolved':
        marker = 'o'
        
    plt.scatter(b_min_c[i], flux_c[i], s=60, c='blue', marker=marker, alpha=0.6)
    plt.errorbar(b_min_c[i], flux_c[i], yerr=errors_c[i], fmt='none', c='blue')
    #plt.scatter(l_b_min[i], l_rms[i], s=20, c=(0.8, 0.2, 0.2), marker='D', alpha=0.6, label='L-Band RMS' if i == 0 else None)

#plt.errorbar(b_min_c, flux_c, yerr=errors_c, alpha = 0.6, fmt='o', color='blue')
plt.plot(x_fit_full_c, y_fit_full_c, '--', color='darkblue', label=f'Linear Fit (Constant: {constant_value:.3f})')

# Labels and other plot details
plt.xlabel('Minimum Baseline', fontsize=18)
plt.ylabel('Flux (mJy)', fontsize=18)

plt.scatter([], [], s=40, c='black', marker='s', label='Resolved (Square)')
plt.scatter([], [], s=40, c='black', marker='o', label='Unresolved (Circle)')
plt.scatter([], [], s=40, c='black', marker='v', label='Non-detection (Down-triangle)')


legend = plt.legend(fontsize=14)
for text in legend.texts:
    if text.get_text() == '3. MaNGA 1-19818':
        text.set_fontweight('bold')
        break
plt.xlim(0, 86000)
#plt.ylim(0, 3.6)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()