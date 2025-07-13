import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import asksaveasfilename
import os

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 14


def select_files_dialog(data_dir):
    """
    Open a file selection dialog centered over the VSC window.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_paths = tk.filedialog.askopenfilenames(
        initialdir=data_dir,
        title="Select SAXS files to compare",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    return list(file_paths)

def get_data(data_path):
    """
    Load data from the given file paths and calculate Q, intensity, and error.
    The error is calculated as the square root of the intensity.
    """
    data = pd.read_csv(data_path, header=0, sep=",")
    I = data["Intensity"]
    Q = data["Q"]
    error = data["Error"]
    return Q, I, error

def plot_data(Q, I, error, label, color, marker):
    """
    Plot the data with error bars and a legend.
    """
    plt.errorbar(Q, I, yerr=error, label=label, color=color, markersize=3, capsize=0, linestyle='None', marker=marker, alpha=0.7)
    plt.xlabel(r"Q (Å$^{-1}$)", fontsize=14)
    plt.ylabel("Intensity (counts)", fontsize=14)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(min(Q), max(Q))
    plt.ylim(1e0, 3e5)
    plt.vlines(x=0.275, ymin=1e0, ymax=3e5, color='black', linestyle='--', linewidth=1, zorder=3)
    plt.grid(True)

def rebin_log(Q, I, error, n_bins=70):
    """
    Rebin data into n_bins evenly spaced on a log-scale.
    Errors are combined in quadrature (square-root sum).
    """
    mask = (Q > 0) & np.isfinite(Q) & np.isfinite(I) & np.isfinite(error)
    Q, I, error = Q[mask], I[mask], error[mask]
    log_min, log_max = np.log10(Q.min()), np.log10(Q.max())
    bins = np.logspace(log_min, log_max, n_bins + 1)
    Q_binned = []
    I_binned = []
    error_binned = []
    for i in range(n_bins):
        idx = (Q >= bins[i]) & (Q < bins[i+1])
        if np.any(idx):
            Q_bin = Q[idx]
            I_bin = I[idx]
            error_bin = error[idx]
            # Weighted average for intensity
            weights = error_bin/I_bin 
            I_mean = np.average(I_bin, weights=weights)
            Q_mean = np.average(Q_bin, weights=weights)
            # Error: sqrt(sum(errors^2))
            err_sum = np.average(error_bin)/np.sqrt(len(Q_bin)) 
            Q_binned.append(Q_mean)
            I_binned.append(I_mean)
            error_binned.append(err_sum)
    return np.array(Q_binned), np.array(I_binned), np.array(error_binned)

def plot_rebinned(file_paths, rebinned_dir):
    """
    Compare SAXS data from multiple files. Expects file_paths as input.
    """
    if not file_paths:
        print("No files selected.")
        return False

    os.makedirs(rebinned_dir, exist_ok=True)

    output_path = asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if not output_path:
        print("No output file selected.")
        return

    colors = ['green', 'red', 'blueviolet', 'navy', 'coral', 'plum', 'black', 'gray']
    markers = ['o', 's', '^', 'D', 'v', 'x', '+', '*']

    plt.figure(figsize=(15, 10))

    all_intensities = []
    rebinned_data = []
    # Plot each file
    for i, file_path in enumerate(file_paths):
        Q, I, error = get_data(file_path)
        multiplier = input(f"Enter multiplier for {os.path.basename(file_path)} (default is 1): ")
        multiplier = float(multiplier) if multiplier else 1.0
        I *= multiplier
        error *= multiplier
        Q_rebinned, I_rebinned, error_rebinned = rebin_log(Q, I, error, n_bins=70)
        all_intensities.append(I_rebinned)
        rebinned_data.append((Q_rebinned, I_rebinned, error_rebinned, i, file_path))

        # Save rebinned data
        base = os.path.basename(file_path)
        name, ext = os.path.splitext(base)
        rebinned_name = f"{name}_rebinned{ext}"
        rebinned_path = os.path.join(rebinned_dir, rebinned_name)
        df_rebinned = pd.DataFrame({
            "Q": Q_rebinned,
            "Intensity": I_rebinned,
            "Error": error_rebinned
        })
        df_rebinned.to_csv(rebinned_path, index=False)

    # Determine global ymin/ymax for all datasets
    all_intensities_flat = np.concatenate(all_intensities)
    # Ensure ymin is not too small
    ymin = np.min(all_intensities_flat) - 0.1 * np.min(all_intensities_flat)
    # Ensure ymax is not too small
    ymax = np.max(all_intensities_flat) + 0.1 * np.max(all_intensities_flat)
    # Ensure ymin is not negative
    ymin = max(ymin, 1e0)

    # Now plot with correct limits
    for Q_rebinned, I_rebinned, error_rebinned, i, file_path in rebinned_data:
        label = os.path.basename(file_path).split('.')[0].replace('_', ' ').replace('-60min', '').replace('minus Buffer','')
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.errorbar(Q_rebinned, I_rebinned, yerr=error_rebinned, label=label, color=color, markersize=3, capsize=0, linestyle='None', marker=marker, alpha=0.7)

    plt.xlabel(r"Q (Å$^{-1}$)", fontsize=14)
    plt.ylabel("Intensity (counts)", fontsize=14)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(min([Q.min() for Q, _, _, _, _ in rebinned_data]), max([Q.max() for Q, _, _, _, _ in rebinned_data]))
    plt.ylim(ymin, ymax)
    plt.vlines(x=0.275, ymin=ymin, ymax=ymax, color='black', linestyle='--', linewidth=1, zorder=3, label=r'$Q_{shell}$')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()
    plt.close()
    print(f"Plot saved to {output_path}")
    return True

# Main loop
while True:
    #Set initial library
    data_dir = "C:\\Users\\Milou\\Desktop\\Data_MEP\\SAXS\\Delft\\Corrected_Files_Batch\\250520"
    #Set output library
    rebinned_dir = "C:\\Users\\Milou\\Desktop\\Data_MEP\\SAXS\\Eindhoven\\Rebinned\\corrected\\"
    
    file_paths = select_files_dialog(data_dir)
    if not file_paths:
        print("No files selected.")
        break
    plot_rebinned(file_paths, rebinned_dir)
