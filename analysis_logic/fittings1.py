# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit

# L Band
# Data and fitting
b_min_l = np.array([1090, 5000, 10000, 15000, 20000, 25000, 30000])
flux_l = np.array([3.4557, 2.6034, 1.7206, 1.1426, 0.49843, 0.73379, 0.40774])
errors_l = np.array([0.0754, 0.0835, 0.0884, 0.116, 0.0746, 0.266, 0.128])
rms_l = [0.0289, 0.0342, 0.0388, 0.0514, 0.0779, 0.106, 0.126, 0.192]
resolved_l = ['Resolved', 'Resolved', 'Resolved', 'Resolved', 'Unresolved', 'Resolved', 'Unresolved']

# Gaussian plus constant fitting function
def gaussian_plus_constant(x, constant, amplitude, stddev):
    return constant + amplitude * np.exp(-((x - 0) / stddev) ** 2 / 2)

# Chi-square function for Gaussian plus constant
def chi_square(params, x, y, yerr):
    model_flux = gaussian_plus_constant(x, *params)
    residuals = (y - model_flux) / yerr
    return np.sum(residuals ** 2)

# Initial guess and fitting
initial_guess = [1, 1, np.std(b_min_l)]
result = minimize(chi_square, initial_guess, args=(b_min_l, flux_l, errors_l))
popt_l = result.x

# Generate points for the fit
x_fit_full_l = np.linspace(min(b_min_l) - 3 * popt_l[2], 45000, 100)
y_fit_full_l = gaussian_plus_constant(x_fit_full_l, *popt_l)

# C Band 1
b_min_c1 = np.array([3445, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000])
flux_c1 = np.array([1.4082, 1.3769, 1.3382, 1.1706, 0.71879, 0.45815, 0.37574, 0.31483, 0.17497])
errors_c1 = np.array([0.0762, 0.0778, 0.0806, 0.0745, 0.058, 0.0454, 0.0482, 0.0477, 0.0517])
resolved_c1 = ['Resolved', 'Resolved', 'Resolved', 'Resolved', 'Resolved', 'Resolved', 'Resolved', 'Resolved', 'Resolved']

# Gaussian plus constant fitting
result = minimize(chi_square, initial_guess, args=(b_min_c1, flux_c1, errors_c1))
popt_c1 = result.x

x_fit_full_c1 = np.linspace(min(b_min_c1) - 3 * popt_c1[2], 50000, 100)
y_fit_full_c1 = gaussian_plus_constant(x_fit_full_c1, *popt_c1)

# C Band 2 - Linear Fit
b_min_c2 = np.array([3445, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000])
flux_c2 = np.array([4.49E-01, 4.39E-01, 4.41E-01, 4.45E-01, 4.30E-01, 3.91E-01, 4.11E-01, 4.05E-01, 3.82E-01, 3.75E-01, 3.90E-01, 3.58E-01, 3.82E-01, 3.71E-01, 3.93E-01, 3.87E-01, 3.84E-01, 3.09E-01, 2.68E-01, 2.58E-01])
errors_c2 = np.array([1.71E-02, 1.75E-02, 1.78E-02, 1.77E-02, 1.86E-02, 1.93E-02, 2.05E-02, 2.08E-02, 2.21E-02, 2.33E-02, 2.49E-02, 2.67E-02, 2.76E-02, 2.96E-02, 3.13E-02, 3.48E-02, 3.87E-02, 4.43E-02, 4.77E-02, 5.12E-02])
resolved_c2 = ["Unresolved"] * 20

rms_c = [
    1.77E-02, 1.69E-02, 1.77E-02, 1.76E-02, 1.87E-02, 1.99E-02, 2.06E-02, 2.07E-02,
    2.23E-02, 2.37E-02, 2.58E-02, 2.80E-02, 2.63E-02, 2.78E-02, 3.15E-02, 3.50E-02,
    4.05E-02, 4.34E-02, 4.84E-02, 4.90E-02, 5.23E-02]

# Linear fitting function
def linear(x, slope, intercept):
    return slope * x + intercept

# Fit the linear model to the data
popt_linear, _ = curve_fit(linear, b_min_c2, flux_c2, sigma=errors_c2)

# Generate data for the fitted line
x_fit_full_c2 = np.linspace(0, 102000, 100)
y_fit_full_c2 = linear(x_fit_full_c2, *popt_linear)

# Combined plot for all three bands and their fittings
plt.figure(figsize=(12, 6))

plt.scatter([], [], marker='', label='1. MaNGA 1-26878')

# L Band
plt.scatter([], [], s=60, c='red', alpha=0.6, marker='o', label='L Band')
plt.scatter(40000, 0.576, s=160, c='red', alpha=0.6, marker='v')

for i in range(len(b_min_l)):
    if resolved_l[i] == 'Resolved':
        marker = 's'
    elif resolved_l[i] == 'Unresolved':
        marker = 'o'
        
    plt.scatter(b_min_l[i], flux_l[i], s=60, c='red', marker=marker, alpha=0.6)
    plt.errorbar(b_min_l[i], flux_l[i], yerr=errors_l[i], fmt='none', c='red')
    #plt.scatter(l_b_min[i], l_rms[i], s=20, c=(0.8, 0.2, 0.2), marker='D', alpha=0.6, label='L-Band RMS' if i == 0 else None)

#plt.errorbar(b_min_l, flux_l, yerr=errors_l, alpha = 0.6, fmt='o', color='red')
plt.plot(x_fit_full_l, y_fit_full_l, '--', color='darkred', label=f'Gaussian Fit + Constant (Const.: {popt_l[0]:.2f}, Amp.: {popt_l[1]:.2f}, S.D.: {popt_l[2]:.2f})')

# C Band 1
plt.scatter([], [], s=60, c='blue', alpha=0.6, marker='o', label='C Band 1')
plt.scatter(45000, 0.07095, s=160, c='blue', alpha=0.6, marker='v')

for i in range(len(b_min_c1)):
    if resolved_c1[i] == 'Resolved':
        marker = 's'
    elif resolved_c1[i] == 'Unresolved':
        marker = 'o'
        
    plt.scatter(b_min_c1[i], flux_c1[i], s=60, c='blue', marker=marker, alpha=0.6)
    plt.errorbar(b_min_c1[i], flux_c1[i], yerr=errors_c1[i], fmt='none', c='blue')
    #plt.scatter(l_b_min[i], l_rms[i], s=20, c=(0.8, 0.2, 0.2), marker='D', alpha=0.6, label='L-Band RMS' if i == 0 else None)

#plt.errorbar(b_min_c1, flux_c1, yerr=errors_c1, alpha = 0.6, fmt='o', color='blue')
plt.plot(x_fit_full_c1, y_fit_full_c1, '--', color='darkblue', label=f'Gaussian Fit + Constant (Const.: {popt_c1[0]:.2f}, Amp.: {popt_c1[1]:.2f}, S.D.: {popt_c1[2]:.2f})')

# C Band 2
plt.scatter([], [], s=60, c='green', alpha=0.6, marker='o', label='C Band 2')
plt.scatter(100000, 1.57E-01, s=160, c='green', alpha=0.6, marker='v')

for i in range(len(b_min_c2)):
    if resolved_c2[i] == 'Resolved':
        marker = 's'
    elif resolved_c2[i] == 'Unresolved':
        marker = 'o'
        
    plt.scatter(b_min_c2[i], flux_c2[i], s=60, c='green', marker=marker, alpha=0.6)
    plt.errorbar(b_min_c2[i], flux_c2[i], yerr=errors_c2[i], fmt='none', c='green')
    #plt.scatter(l_b_min[i], l_rms[i], s=20, c=(0.8, 0.2, 0.2), marker='D', alpha=0.6, label='L-Band RMS' if i == 0 else None)

#plt.errorbar(b_min_c2, flux_c2, yerr=errors_c2, alpha = 0.6, fmt='o', color='green')
plt.plot(x_fit_full_c2, y_fit_full_c2, '--', color='darkgreen', label=f'Linear Fit (Slope: {popt_linear[0]:.2e}, Intercept: {popt_linear[1]:.2f})')

# Labels and other plot details
plt.xlabel('Minimum Baseline', fontsize=18)
plt.ylabel('Flux (mJy)', fontsize=18)

plt.scatter([], [], s=40, c='black', marker='s', label='Resolved (Square)')
plt.scatter([], [], s=40, c='black', marker='o', label='Unresolved (Circle)')
plt.scatter([], [], s=40, c='black', marker='v', label='Non-detection (Down-triangle)')


legend = plt.legend(fontsize=14)
for text in legend.texts:
    if text.get_text() == '1. MaNGA 1-26878':
        text.set_fontweight('bold')
        break
plt.xlim(0, 101000)
plt.ylim(0, 3.6)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.tight_layout()

# Display the plot
plt.show()