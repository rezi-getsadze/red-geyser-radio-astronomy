import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit

# L Band:
# 2 Gaussian:

b_min_l = np.array([1100, 4000, 7500, 10000, 15000, 20000, 25000])
flux_l = np.array([1.8237, 1.8864, 1.376, 1.289, 1.1447, 1.098, 0.97725])
errors_l = np.array([0.0622, 0.0624, 0.033, 0.038, 0.047, 0.0614, 0.0874])
resolved_l = ['Resolved', 'Resolved', 'Unresolved', 'Unresolved', 'Unresolved', 'Unresolved', 'Unresolved']
rms_l = [0.0289, 0.03084, 0.03177, 0.03858, 0.04505, 0.05632, 0.0836, 0.106]


# Define a Gaussian function with peak at specified mean
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

# Define the bimodal Gaussian model
def bimodal_gaussian(x, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2):
    return (gaussian(x, amplitude1, mean1, stddev1) + 
            gaussian(x, amplitude2, mean2, stddev2))

# Define the chi-square function
def chi_square(params):
    model_flux_l = bimodal_gaussian(b_min_l, *params)
    residuals = (flux_l - model_flux_l) / errors_l
    return np.sum(residuals ** 2)

# Initial guess for the parameters
initial_guess = [1, 10000, 5000, 1, 20000, 5000]  # Adjust mean values to be further apart

# Minimize the chi-square function
# Adding bounds
bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), (0, None)]
result = minimize(chi_square, initial_guess, bounds=bounds)

# Extract the optimized parameters
popt_l = result.x

# Generate points for the fitted curve
x_fit_full_l = np.linspace(min(b_min_l) - 3 * popt_l[2], 35000, 100)
y_fit_full_l = bimodal_gaussian(x_fit_full_l, *popt_l)

# C Band:
# 2-gaussian

b_min_c = np.array([3620, 4000, 7500, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000])
flux_c = np.array([1.2746, 1.2146, 1.2231, 1.2636, 1.1838, 1.0023, 0.63651, 0.42479, 0.45482, 0.44831, 0.43609, 0.34183, 0.30904, 0.23532, 0.23135, 0.32822, 0.3165, 0.2813])
errors_c = np.array([0.0556, 0.0569, 0.0574, 0.0592, 0.0569, 0.0545, 0.0456, 0.0241, 0.0247, 0.0253, 0.0262, 0.0281, 0.0289, 0.0336, 0.0324, 0.0347, 0.0364, 0.043])
resolved_c = ['Resolved'] * 7 + ['Unresolved'] * 11
rms_c = [0.02148, 0.02322, 0.02331, 0.0233, 0.02362, 0.02254, 0.02277, 0.02444, 0.02565, 0.02683, 0.02677, 0.02847, 0.02888, 0.02935, 0.03314, 0.03597, 0.03938, 0.04706, 0.05445]


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
    model_flux_c = bimodal_plus_constant(b_min_c, *params)
    residuals = (flux_c - model_flux_c) / errors_c
    return np.sum(residuals ** 2)

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
popt_c = result.x

# Generate data for the fitted curve
x_fit_full_c = np.linspace(0, 90000, 100)
y_fit_full_c = bimodal_plus_constant(x_fit_full_c, *popt_c)

# Combined plot for all three bands and their fittings
plt.figure(figsize=(12, 6))

plt.scatter([], [], marker='', label='4. MaNGA 1-43718')

# L Band
plt.scatter([], [], s=60, c='red', alpha=0.6, marker='o', label='L Band')
plt.scatter(30000, 0.318, s=160, c='red', alpha=0.6, marker='v')

for i in range(len(b_min_l)):
    if resolved_l[i] == 'Resolved':
        marker = 's'
    elif resolved_l[i] == 'Unresolved':
        marker = 'o'
        
    plt.scatter(b_min_l[i], flux_l[i], s=60, c='red', marker=marker, alpha=0.6)
    plt.errorbar(b_min_l[i], flux_l[i], yerr=errors_l[i], fmt='none', c='red')
    #plt.scatter(l_b_min[i], l_rms[i], s=20, c=(0.8, 0.2, 0.2), marker='D', alpha=0.6, label='L-Band RMS' if i == 0 else None)

#plt.errorbar(b_min_l, flux_l, yerr=errors_l, alpha = 0.6, fmt='o', color='red')
plt.plot(x_fit_full_l, y_fit_full_l, '--', color='darkred', label=f'Bimodal Gaussian Fit (Amp. 1: {popt_l[0]:.2f},\nMean 1: {popt_l[1]:.2f}, S.D. 1: {popt_l[2]:.2f}, Amp. 2: {popt_l[3]:.2f},\nMean 2: {popt_l[4]:.2f}, S.D. 2: {popt_l[5]:.2f})')

# C Band
plt.scatter([], [], s=60, c='blue', alpha=0.6, marker='o', label='C Band')
plt.scatter(85000, 0.16335, s=160, c='blue', alpha=0.6, marker='v')

for i in range(len(b_min_c)):
    if resolved_c[i] == 'Resolved':
        marker = 's'
    elif resolved_c[i] == 'Unresolved':
        marker = 'o'
        
    plt.scatter(b_min_c[i], flux_c[i], s=60, c='blue', marker=marker, alpha=0.6)
    plt.errorbar(b_min_c[i], flux_c[i], yerr=errors_c[i], fmt='none', c='blue')
    #plt.scatter(l_b_min[i], l_rms[i], s=20, c=(0.8, 0.2, 0.2), marker='D', alpha=0.6, label='L-Band RMS' if i == 0 else None)

#plt.errorbar(b_min_c, flux_c, yerr=errors_c, alpha = 0.6, fmt='o', color='blue')
plt.plot(x_fit_full_c, y_fit_full_c, '--', color='darkblue', label=f'Bimodal Gaussian + Constant Fit (Amp. 1: {popt_c[0]:.2f},\nMean 1: {popt_c[1]:.2f}, S.D. 1: {popt_c[2]:.2f}, Amp 2: {popt_c[3]:.2f},\nMean 2: {popt_c[4]:.2f}, S.D. 2: {popt_c[5]:.2f}, Const.: {popt_c[6]:.2f})')

# Labels and other plot details
plt.xlabel('Minimum Baseline', fontsize=18)
plt.ylabel('Flux (mJy)', fontsize=18)

plt.scatter([], [], s=40, c='black', marker='s', label='Resolved (Square)')
plt.scatter([], [], s=40, c='black', marker='o', label='Unresolved (Circle)')
plt.scatter([], [], s=40, c='black', marker='v', label='Non-detection (Down-triangle)')


legend = plt.legend(fontsize=14)
for text in legend.texts:
    if text.get_text() == '4. MaNGA 1-43718':
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