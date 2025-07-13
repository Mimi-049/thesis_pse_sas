import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = "Times New Roman"

### FUNCTIONS ################################################################################
def get_data(data_path):
    """
    Load data from the given file paths and calculate Q, intensity, and error.
    The error is calculated as the square root of the intensity.
    """
    I_total = 0
    error_total = 0
    for i in data_path:
        data = pd.read_csv(i, header=28)
        theta2 = data["Angle"]  # 2*theta
        theta = theta2 / 2  # Q uses 2*theta/2
        I = data[" Intensity"]
        I_total += I
        error_total += np.sqrt(I)
        Q = 4 * np.pi * np.sin(np.radians(theta)) / labda
    return Q, I_total, error_total

def get_background_superposition(background_files, fractions):
    """
    Calculate the superposition of multiple background files based on user-provided fractions.
    If only one background file is selected, use it directly.
    """
    if len(background_files) == 1:
        Q, I, err = get_data([background_files[0]])
        return Q, I, err
    else:
        I_total = 0
        err_total = 0
        for i in range(len(background_files)):
            I_total += get_data([background_files[i]])[1]* fractions[i]
            err_total += get_data([background_files[i]])[2]**2 * fractions[i]**2
        Q_superposed = get_data([background_files[0]])[0]
        I_superposed = I_total
        err_superposed = np.sqrt(err_total)
        return Q_superposed, I_superposed, err_superposed

def discard_anomalies(Q_bck, I_bck, err_bck, Q_meas, I_meas, err_meas, intensity_threshold=2750, mid_threshold=3300, high_threshold=1e10, window_size=9, adjust_background_thresholds=False, indicator=None, add_anomalies=[]):
    """
    Discard anomalies in the intensity data using a moving average.
    """
    def calculate_moving_average(data, valid_indices, window, Q):
        moving_avg = np.zeros(len(data))
        
        for i in range(len(data)):
            if Q[i] < 0.012:
                moving_avg[i] = data[i]
            else:
                start = max(0, i - window // 2)
                end = min(len(data), i + window // 2 + 1)
                valid_window_indices = [j for j in range(start, end) if j in valid_indices]
                if valid_window_indices:
                    moving_avg[i] = np.mean(data[valid_window_indices])
                else:
                    moving_avg[i] = data[i]
        return moving_avg

    if adjust_background_thresholds:
        intensity_threshold_bck = 0.4 * intensity_threshold
        mid_threshold_bck = mid_threshold
        high_threshold_bck = high_threshold / 2
    else:
        intensity_threshold_bck = intensity_threshold
        mid_threshold_bck = mid_threshold
        high_threshold_bck = high_threshold

    if indicator in [13, 14]:
        mid_threshold *= 1
        window_size = 5
    


    valid_indices = list(range(len(I_bck)))

    while True:
        I_bck_avg = calculate_moving_average(I_bck, valid_indices, window_size, Q_bck)
        I_meas_avg = calculate_moving_average(I_meas, valid_indices, window_size, Q_bck)
        to_discard = []

        for i in valid_indices:
            if Q_bck[i] < 0.012:
                threshold = high_threshold
                threshold_bck = high_threshold_bck
            elif 0.012 <= Q_bck[i] < 0.025:
                threshold = mid_threshold
                threshold_bck = mid_threshold_bck
            else:
                threshold = intensity_threshold
                threshold_bck = intensity_threshold_bck

            if (
                I_bck[i] - I_bck_avg[i] > threshold_bck or
                I_meas[i] - I_meas_avg[i] > threshold
            ):
                to_discard.append(i)

        if not to_discard:
            break

        valid_indices = [i for i in valid_indices if i not in to_discard]
        valid_indices = [i for i in valid_indices if i not in add_anomalies]


    Q_bck = Q_bck[valid_indices]
    I_bck = I_bck[valid_indices]
    err_bck = err_bck[valid_indices]
    Q_meas = Q_meas[valid_indices]
    I_meas = I_meas[valid_indices]
    err_meas = err_meas[valid_indices]

    return Q_bck, I_bck, err_bck, Q_meas, I_meas, err_meas

def subtract_backgrounds(measurement_dir, background_dir, output_dir, figure_dir, filenames, sample_col, background_col, fractions_col):
    """
    Main function to subtract backgrounds from measurements.
    It iterates through the sample and background columns, processes the data,
    and saves the results.
    """
    measurements = []
    j = 0  

    #Caclulate reference transmission from empty beam measurement
    direct_beam = filenames["999"].tolist()
    direct_file = direct_beam[1:]
    direct_file_location = [dir_data + file for file in direct_file if isinstance(file, str)]
    direct_Q, direct_I, direct_err = get_data(direct_file_location)
    I0_direct = direct_I[8] #location of beamstop / Angle is 0

    #Background subtraction loop
    sample_counter = -1
    for indicator in sample_col:
        sample_counter += 1 
        if indicator not in headers:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        #Get backgrounds and fractions for the background
        background_indices = background_col[j]
        fraction_indices = fractions_col[j]
        marker_add = 0
        background_counter = -1
        for indices in background_indices:
            background_counter += 1
            measurements = []
            add_discard = []
            #If more than one background is selected, combine them
            if isinstance(indices, tuple):
                # print(f"Combining backgrounds: {indices}")
                background_files = [filenames[index].tolist()[1] for index in indices]
                fractions = fraction_indices[background_counter]
                backgrounds = [filenames[index].tolist()[0] for index in indices]
                background_name = "+".join(backgrounds) + "_" + "+".join([str(fraction) for fraction in fractions])
                print(f"Background name: {background_name}")
                indicator_bck = indices[0]  # Use the first index as the background indicator
                #Handle anomalies that could not be removed by discard anomalies function
                for index in indices:
                    if int(index) == 998:
                        add_discard.append(224)

            #Treat single background files
            else:
                index = indices
                indicator_bck = index
                background_file = filenames[index].tolist()
                background_files = background_file[1:-1]
                fractions = None
                background_name = background_file[0]
                #Handle anomalies that could not be removed by discard anomalies function
                if int(index) == 998:
                    add_discard.append(224)

            #Get the background files and measurement files
            background_location = [dir_data + file for file in background_files if isinstance(file, str)]
            measurement_file = filenames[indicator].tolist()
            measurement_files = measurement_file[1:]
            measurement_location = [dir_data + file for file in measurement_files if isinstance(file, str)]
            print("Sample", measurement_location) #Print which sample is being processed
            print("Background", background_location) #Print which backgrounds are being processed

            #Get the data for the background and measurement files
            Q_meas, I_meas, err_meas = get_data(measurement_location)
            Q_bck, I_bck, err_bck = get_background_superposition(background_location, fractions)
            
            #Handle anomalies that could not be removed by discard anomalies function
            if int(indicator) == 500:
                add_discard.append(56)
                add_discard.append(57)
                add_discard.append(59)

            #Discard anomalies in the background and measurement data, due to faulte pixel giving +4000 on intensity counts
            Q_bck, I_bck, err_bck, Q_meas, I_meas, err_meas = discard_anomalies(
                Q_bck, I_bck, err_bck, Q_meas, I_meas, err_meas, adjust_background_thresholds=bool(fractions), indicator=int(indicator), add_anomalies=add_discard
            )

            #Correct for transmission, checkt which value corresponds to 2thehta=0. For me this is the 8th value in the array, but this can differ per measurement
            measurement_transmission_factor = I0_direct/I_meas[8] #note that this is the correction factor, the transmission is T=I_meas/I0_direct --> then I_meas = I_meas/T = 
            background_transmission_factor = I0_direct/I_bck[8]
            print(f"Measurement transmission: {1/measurement_transmission_factor}, Background transmission: {1/background_transmission_factor}")
            I_meas *= measurement_transmission_factor
            I_bck *= background_transmission_factor
            err_meas *= measurement_transmission_factor
            err_bck *= background_transmission_factor
            
            #Get corrected Q, intensity and error
            Q = Q_bck
            I = I_meas - I_bck #Corrected intensity
            error = np.sqrt(err_meas**2 + err_bck**2) #Combined error

            #Make dataframe for the measurement
            measurements.append({
                "background_files": background_location,
                "measurement_files": measurement_location,
                "Q": Q,
                "I_bck": I_bck,
                "err_bck": err_bck,
                "I_meas": I_meas,
                "err_meas": err_meas,
                "I": I,
                "error": error,
                "label": measurement_file[0],
                "label_bck": background_name,
                "measurement_indicator": indicator,
                "background_indicator": indicator_bck,
            })

            #Save background data without background subtraction
            os.makedirs(background_dir, exist_ok=True)
            background_data = {
                "Q": Q_bck,
                "Intensity": I_bck,
                "Error": err_bck
            }
            bck = pd.DataFrame(background_data)
            bck.to_csv(os.path.join(background_dir, f"{background_name.replace(' ', '_')}-60min.csv"), index=False)

            
            os.makedirs(output_dir, exist_ok=True)

            for measurement in measurements:
                mask = measurement["Q"] >= 0.0075
                filtered_data = {
                    "Q": measurement["Q"][mask],
                    "Intensity": measurement["I"][mask],
                    "Error": measurement["error"][mask]
                }
                df = pd.DataFrame(filtered_data)

                measurement_label = measurement["label"]
                background_label = measurement["label_bck"]
                output_file = os.path.join(output_dir, f"{measurement_label.replace(' ', '_')}-{background_label.replace(' ', '_')}-60min.csv")

                df.to_csv(output_file, index=False)

                #Save measurement data without background subtraction
                os.makedirs(measurement_dir, exist_ok=True)
                measurement_data = {
                    "Q": measurement["Q"],
                    "Intensity": measurement["I_meas"],
                    "Error": measurement["err_meas"]
                }
                meas = pd.DataFrame(measurement_data)
                meas.to_csv(os.path.join(measurement_dir, f"{measurement['label'].replace(' ', '_')}-60min.csv"), index=False)

            base_colors = ['blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'red', 'pink', 'yellow', 'lime', 'teal', 'gold', 'indigo', 'violet', 'gray', 'blue', 'green', 'orange', 'purple', 'brown']
            markers = ['o', '^', 's', 'D', 'P', 'X', '*', 'v', '<', '>', 'h', 'H', 'd', 'p', '|', '_', 'o', '^', 's', 'D', 'P', 'X']
            available_styles = [{"color": color, "marker": marker} for color, marker in zip(base_colors, markers)]
            used_styles = {}

            plotted_backgrounds = set()
            figure_title_parts = []

            for measurement in measurements:
                indicator_meas = measurement["measurement_indicator"]
                indicator_back = measurement["background_indicator"]

                label_meas = measurement["label"]
                label_back = measurement["label_bck"]

                if indicator_back not in used_styles:
                    used_styles[indicator_back] = available_styles.pop(0)
                if indicator_meas not in used_styles:
                    used_styles[indicator_meas] = available_styles.pop(0)

                    axs[0].errorbar(
                        measurement["Q"], measurement["I_bck"], yerr=np.sqrt(measurement["err_bck"]),
                        fmt=markers[background_counter+6], markersize=3,
                        label=f"Background: {label_back.replace('+', ':')} T={1/background_transmission_factor:.2f}",
                        color=base_colors[background_counter+6], alpha=0.5
                    )

                figure_title_parts.append(f"S-{label_meas}_B-{label_back}")


            figure_title = label_meas.replace(" ", "_").replace(".", ",").replace(":", "-")

            plotted_corrected = set()
            for measurement in measurements:
                corrected_label = f"{measurement['label']} - {measurement['label_bck']}"
                marker_index = background_counter+6+marker_add
                colour_index = background_counter+6+marker_add
                indicator_meas = sample_counter
                if marker_index>13:
                    marker_index = marker_index-13
                    colour_index = colour_index-13
                marker = markers[marker_index]
                colour = base_colors[colour_index]
                marker_add += 1 

                if corrected_label not in plotted_corrected:
                    axs[1].errorbar(
                        measurement["Q"], measurement["I"], yerr=measurement["error"],
                        fmt=marker, markersize=3,
                        label=f"{measurement['label']} - {measurement['label_bck']}",
                        color=colour, alpha=0.7
                    )
                    plotted_corrected.add(corrected_label)
        
        axs[0].errorbar(
        measurement["Q"], measurement["I_meas"], yerr=np.sqrt(measurement["err_meas"]),
        fmt=markers[sample_counter], markersize=3,
        label=f"Sample: {label_meas} T={1/measurement_transmission_factor:.2f}",
        color=base_colors[sample_counter], alpha=0.5
        )
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].set_xlabel(r"$Q$ [$\AA^{-1}$]", fontsize=12)
        axs[0].set_ylabel("I [counts]", fontsize=12)
        axs[0].set_xlim(0.0075, max(measurement["Q"]))
        axs[0].grid()
        axs[0].legend(fontsize=10)
        axs[0].set_title("Backgrounds " + figure_title)
        
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].set_xlabel(r"$Q$ [$\AA^{-1}$]", fontsize=12)
        axs[1].set_ylabel("I [counts]", fontsize=12)
        axs[1].set_xlim(0.0075, max(measurement["Q"]))
        axs[1].grid()
        axs[1].legend(fontsize=10)
        axs[1].set_title("Measurements Corrected for Background")
        
        os.makedirs(figure_dir, exist_ok=True)
        output_fig = os.path.join(figure_dir, f"{figure_title}.png")
        plt.tight_layout()
        plt.savefig(output_fig, dpi=600)
        # plt.show()
        plt.close(fig)
        j += 1

### MAIN ############################################################################################

### SAXS parameters
labda = 1.54  # X-ray wavelength in Angstroms

### Set save locations
measurement_dir = "c:\\Users\\Milou\\Desktop\\Data_MEP\\SAXS\\Delft\\Corrected_Files_Batch\\250403_measurements_new_transmission\\" #raw measurement, corrected for transmission
background_dir = "c:\\Users\\Milou\\Desktop\\Data_MEP\\SAXS\\Delft\\Corrected_Files_Batch\\250403_bck_good\\" #raw background, corrected for transmission
output_dir = "c:\\Users\\Milou\\Desktop\\Data_MEP\\SAXS\\Delft\\Corrected_Files_Batch\\250403_bck_good\\" #measurements corrected for background
figure_dir="c:\\Users\\Milou\\Desktop\\Data_MEP\\SAXS\\Delft\\Figures_Batch\\250403_bck_good\\" #directory for figures for corrected measurements

### Batch processing example
# 200525
dir_data = "C:\\Users\\Milou\\Desktop\\Data_MEP\\SAXS\\Delft\\250520\\"
filenames = pd.read_csv("C:\\Users\\Milou\\Desktop\\Data_MEP\\SAXS\\Delft\\combinations_sample_background.csv", header=0, delimiter=";")
headers = filenames.columns
print(headers)
##Choose samples you want to analyse
sample_col = ["010", "011", "100", "101", "102", "200", "201", "300", "301", "302", "400", "401"]
##Set backgrounds to be analysed per sample
background_col = [
    [("000", "998")], #010 5wpi sol
    [("000", "998")], #011 0.25wpi sol
    [("000", "998"), ("102", "998"), ("000", "010", "998")], #100 5wpi + 5wt% sunflower oil emulsion
    [("000", "998")], #101 100 washed
    [("000", "998")], #102 101 continuous phase
    [("000", "998"), ("202", "998")], #200 0.25wpi + 5wt% sunflower oil emulsion
    [("000", "998")], #201 200 washed
    [("000", "998"), ("302", "998"), ("000", "010", "998")], #300 5wpi sol + decane
    [("000", "998")], #301 5wpi sol + decane
    [("000", "998")], #302 5wpi sol + decane
    [("000", "998"), ("402", "998")], #400 5wpi sol + decane
    [("000", "998")], #401 5wpi sol + decane
]
fractions_col = [
    [(0.95, 0.05)], #010 5wpi sol
    [(0.9975, 0.0025)], #011 0.25wpi sol
    [(0.899, 0.101), (0.946, 0.054), (0.0342, 0.910, 0.0668)], #100 5wpi sol + 0.25wpi sol
    [(0.9441, 0.0559)], #101 5wpi sol + 0.25wpi sol
    [(0.9545, 0.0455)], #102 5wpi sol + 0.25wpi sol
    [(0.934, 0.066), (0.946, 0.054)], #200 5wpi sol + 0.25wpi sol
    [(0.9443, 0.0557)], #201 5wpi sol + 0.25wpi sol
    [(0.885, 0.115), (0.9315, 0.0685), (0.0442, 0.885, 0.0708)], #300 5wpi sol + decane
    [(0.9292, 0.0708)], #301 5wpi sol + decane
    [(0.9558, 0.0442)], #302 5wpi sol + decane
    [(0.9292, 0.0708), (0.9315, 0.0685)], #400 5wpi sol + decane
    [(0.9293, 0.0707)], #401 5wpi sol + decane
]

subtract_backgrounds(
    measurement_dir=measurement_dir,
    background_dir=background_dir,
    output_dir=output_dir,
    figure_dir=figure_dir,
    filenames=filenames,
    sample_col=sample_col,
    background_col=background_col,
    fractions_col=fractions_col
)
