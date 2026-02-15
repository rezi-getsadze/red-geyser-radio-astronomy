# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erf  # For skewed Gaussian function

# Skewed Gaussian with constant fitting
# Define the data
b_min = np.array([3550, 5000, 10000, 14000, 20000, 25000, 30000, 34000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000])
flux = np.array([0.2115, 0.21054, 0.21343, 0.21319, 0.21281, 0.21723, 0.2291, 0.23098, 0.2204, 0.24651, 0.24829, 0.25515, 0.23883, 0.23688, 0.20788, 0.20247, 0.18506, 0.14246])
errors = np.array([0.0152, 0.0158, 0.0164, 0.0165, 0.0173, 0.0183, 0.0198, 0.0199, 0.0213, 0.0216, 0.0229, 0.0239, 0.0257, 0.0282, 0.0312, 0.0341, 0.0396, 0.0443])
rms_c = [1.45E-02, 1.58E-02, 1.63E-02, 1.66E-02, 1.73E-02, 1.87E-02, 2.02E-02, 1.98E-02, 2.17E-02, 2.12E-02, 2.28E-02, 2.23E-02, 2.41E-02, 2.68E-02, 2.99E-02, 3.04E-02, 3.90E-02, 4.29E-02, 4.77E-02]
resolved_c = ['Unresolved'] * (len(b_min))

# Define skewed Gaussian with constant
def skewed_gaussian_with_constant(x, amplitude, mean, stddev, skew, constant):
    t = (x - mean) / stddev
    skew_gaussian = 2 * amplitude * np.exp(-t**2 / 2) * (1 + erf(skew * t / np.sqrt(2)))
    return skew_gaussian + constant

# Chi-square function
def chi_square_skewed_with_constant(params):
    model_flux = skewed_gaussian_with_constant(b_min, *params)
    residuals = (flux - model_flux) / errors
    return np.sum(residuals ** 2)

# Fitting the data
initial_guess = [0.1, 60000, 10000, 1, 10]
result = minimize(chi_square_skewed_with_constant, initial_guess)
popt = result.x

# Generate points for the fit
x_fit = np.linspace(0, 95000, 100)
y_fit = skewed_gaussian_with_constant(x_fit, *popt)

# Plotting with custom legends and style
plt.figure(figsize=(12, 6))

plt.scatter([], [], marker='', label='2. MaNGA 1-27393')

plt.scatter([], [], s=60, c='blue', alpha=0.6, marker='o', label='C Band')
plt.scatter(90000, 1.43E-01, s=160, c='blue', alpha=0.6, marker='v')

for i in range(len(b_min)):
    if resolved_c[i] == 'Resolved':
        marker = 's'
    elif resolved_c[i] == 'Unresolved':
        marker = 'o'
        
    plt.scatter(b_min[i], flux[i], s=60, c='blue', marker=marker, alpha=0.6)
    plt.errorbar(b_min[i], flux[i], yerr=errors[i], fmt='none', c='blue')
    #plt.scatter(l_b_min[i], l_rms[i], s=20, c=(0.8, 0.2, 0.2), marker='D', alpha=0.6, label='L-Band RMS' if i == 0 else None)

#plt.errorbar(b_min, flux, markersize=8, alpha = 0.6, yerr=errors, fmt='o', color='blue')
plt.plot(x_fit, y_fit, '--', color='darkblue', label=f'Skewed Gaussian Fit + Constant (Const.: {popt[4]:.2f},\nMean: {popt[1]:.2f}, Amp.: {popt[0]:.2f}, S.D.: {popt[2]:.2f}, Skew: {popt[3]:.2f})')

plt.xlabel('Minimum Baseline', fontsize=19)
plt.ylabel('Flux (mJy)', fontsize=19)
plt.xlim(0, 91000)
plt.ylim(0, 0.3)

# Custom legend entries
plt.scatter([], [], s=60, c='black', marker='o', label='Unresolved (Circle)')
plt.scatter([], [], s=60, c='black', marker='v', label='Non-detection (Down-triangle)')

legend = plt.legend(fontsize=14, loc='lower left')
for text in legend.texts:
    if text.get_text() == '2. MaNGA 1-27393':
        text.set_fontweight('bold')
        break
plt.grid(True)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.show()