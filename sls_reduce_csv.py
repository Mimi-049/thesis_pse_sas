import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import glob
import tkinter as tk
import os
from tkinter import Listbox, MULTIPLE, END, Button, Label, Frame
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

'''Functions to reduce csv files from the Mastersizer 3000. For the code to function the names of the samples should be in the form:
0DATE_1PROTEIN_2OIL_3HOMOGENISER_4INDICATOR_5CENTRIFUGE_6AGE_7WATER (250128_5wt%WPI_5%oil_5b1_620_xxx_fresh_D2O).

1PROTEIN = concentration%protein type
2OIL = concentration%oil type
3HOMOGENISER = bar b passes
4INDICATOR = stock solution - sample indicator - repetition indicator
5CENTRIFUGRE = RPM - TIME - ACCELERATION

The sampels are located in a txt file, where all information from the sample name is splitted. Using this file the dat can be analysed.
'''

plt.rcParams["font.family"] = "Times New Roman"

def select_files():
    '''Chose which files to reduce'''
    def on_select():
        selected_indices = listbox.curselection()
        selected_files = [file_list[i] for i in selected_indices]
        root.quit()
        root.destroy()
        return selected_files

    root = tk.Tk()
    root.title("Select CSV files")

    file_list = os.listdir("Mastersizer/Data_Mastersizer_in_CSV/")
    file_list = [f for f in file_list if f.endswith('.csv')]

    max_filename_length = max(len(f) for f in file_list)
    listbox_width = max_filename_length + 2

    frame = Frame(root, padx=10, pady=10)
    frame.pack(padx=10, pady=10)

    label = Label(frame, text="Select CSV files:", font=("Times New Roman", 12), bg=None, fg='black')
    label.pack(pady=5)

    listbox = Listbox(frame, selectmode=MULTIPLE, width=listbox_width, height=15, font=("Times New Roman", 10), bg='white', fg='black')
    for file in file_list:
        listbox.insert(END, file)
    listbox.pack(pady=5)

    button = Button(frame, text="Select", command=lambda: selected_files.extend(on_select()), font=("Times New Roman", 10), bg='white', fg='black')
    button.pack(pady=5)

    root.update_idletasks()
    window_width = root.winfo_width()
    window_height = root.winfo_height()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

    selected_files = []
    root.mainloop()
    return selected_files

def get_values(name):
    '''Get all imporant values from the sample name.'''
    capture = re.split(r'_', name)
    date = capture[0].replace("Average of '", "")
    protein = capture[1].replace(",", ".")
    protein_wt = float(re.split(r'wt%', protein)[0])
    protein_type = re.split(r'wt%', protein)[1]
    oil = capture[2].replace(",", ".").replace("wt", "")
    oil_wt = float(re.split(r'%', oil)[0])
    bar_passes = capture[3]
    bar = re.split(r'b', bar_passes)[0]
    passes = re.split(r'b', bar_passes)[1]
    indicator = capture[4]
    centrifuge = re.split(r'-', capture[5])
    if len(centrifuge) > 2:
        rpm = centrifuge[0].replace("centri", "")
        mins = centrifuge[1]
        acceleration = centrifuge[2]
    else:
        rpm = mins = acceleration = None
    water = capture[-2]
    age = capture[-1].replace("'", "")
    return date, protein_wt, protein_type, oil_wt, bar, passes, indicator, rpm, mins, acceleration, water, age

def make_files(protein_percentages):
    '''
    To make designated files for the splitted data based on WPI wt% value. 
    Samplename: 0DATE_1PROTEIN_2OIL_3HOMOGENISER_4INDICATOR_5CENTRIFUGE_6WATER_7AGE
    The headers are then date - D32 - protein - oil - homogeniser - indicator - centrifuge - water - age

    '''
    for k in range(len(protein_percentages)):
        f = open('Mastersizer/Separated_data/{}.txt'.format(protein_percentages[k]),'a')
        # make an f.write with the headers
        f.write('date,D32,error_D32,protein_wt,protein_type,oil_wt,bar,passes,indicator,rpm,mins,acceleration,age,water,peak_width,error_peak_width')
        f.close()

def write_data(protein_percentage, D3_2, error_D32, date, protein_wt, protein_type, oil_wt, bar, passes, indicator, rpm, mins, acceleration, water, age, peak_width, error_peak_width, y):
    ''' 
    Append the extracted data to the file, made with make_files.
    First the correct WPI file is selected and then the extracted data is added.
    '''
    f = open('Mastersizer/Separated_data/{}.txt'.format(protein_percentage),'a')
    f.write('\n')
    f.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(date, D3_2, error_D32, protein_wt, protein_type, oil_wt, bar, passes, indicator, rpm, mins, acceleration, water, age, peak_width, error_peak_width, y))
    # print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(date, D3_2, protein_wt, protein_type, oil_wt, bar, passes, indicator, rpm, mins, acceleration, water, age))
    f.close()

def split_data_on_wt(names, D3_2, protein_percentages, peak_width, peak_width_error, intensity):
    '''
    To split all the data based on their WPI values, and store them in a designated or general file (x_vs_I).

    '''
    N_measurements = len(D3_2)
    
    row_data = []
    for _, row in intensity.iterrows():
        row_intensity = ','.join(row.astype(str))
        row_data.append(row_intensity)
    print('data', len(row_data))

    ## Only take the average values, not all inbetween measurements
    sum_D3_2 = np.zeros(5)
    k = 0
    for i in range(0, N_measurements):
        if k == 5:
            D3_2_average = np.average(sum_D3_2)
            D3_2_error = np.std(sum_D3_2)
            print(D3_2_average, D3_2_error, sum_D3_2)
            sum_D3_2 = np.zeros(5)
            k = 0

            # Find correct protein values
            date, protein_wt, protein_type, oil_wt, bar, passes, indicator, rpm, mins, acceleration, water, age = get_values(names[i])
            print(date, protein_wt, protein_type, oil_wt, bar, passes, indicator, rpm, mins, acceleration, water, age)
            float_protein_percentages = np.zeros(len(protein_percentages))
            for j in range(len(protein_percentages)):
                float_protein_percentages[j] = protein_percentages[j].replace(",", ".")
                float_protein_percentages[j] = float(float_protein_percentages[j])
            protein_index = np.where(protein_wt == float_protein_percentages)[0]
            protein_value = protein_percentages[protein_index]
            protein_value = protein_value[0]
            # print(intensity)

            '''Write all data to selected file. To write it to designated files splitted on WPI concentration, change x_vs_I to protein value. 
            To store all data in one file, use x_vs_I'''
            write_data('x_vs_I', D3_2_average, D3_2_error, date, protein_wt, protein_type, oil_wt, bar, passes, indicator, rpm, mins, acceleration, water, age, peak_width[i], peak_width_error[i], row_data[i])
            print("Writing data for {}wt% WPI at {} to folder {}".format(protein_wt, date, protein_value))
        else:
            sum_D3_2[k] = D3_2[i]  # append D3_2 value to array
            k += 1
            print(k, sum_D3_2)

'''Fit functions for peak detection start'''
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def exp(x, a, b, c, d):
    return np.exp(a*x**c+d)+b

def reverse_exp(y, a, b, c):
    return (np.log(y-b)/a)**(1/c)
'''Fit functions for peak detection end'''


'''Get intial idea of the measurements by plotting the Sauter Mean averages average.'''
def plot_avg_D3_2(D3_2, names):
    number_average = int(len(D3_2)/6) #only the averages need to be plotted
    
    D3_2_arr = []
    x_label = []
    for i in range(0,number_average):
        i = 5+i*6 #to take every average
        date, protein_wt, protein_type, oil_wt, bar, passes, indicator, rpm, mins, acceleration, water, age = get_values(names[i])
        print(date, protein_wt, protein_type, oil_wt, bar, passes, indicator, rpm, mins, acceleration, water, age)
        D3_2_arr = np.append(D3_2_arr, D3_2[i]) #append D3_2 value to array
        x_label = np.append(x_label, indicator) #append indicator to array

    #Also plot last sample scanned, hereafter 0 bar if loop doesn't work anymore
    name_figure = r'{}wt% {} and {}wt% oil'.format(protein_wt, protein_type, oil_wt)
    name_figure = name_figure.replace('.', ',')
    plt.title(name_figure)
    plt.plot(x_label, D3_2_arr, marker= '.', color = 'blue', mec = 'black', mfc = 'black')
    plt.ylabel( r'$R$[$\mu$m]')
    plt.axhline(y=3.5, color='green', linestyle='dashed')
    plt.axhline(y=4.5, color='green', linestyle='dashed')
    plt.fill_between(indicator, 3.5, 4.5, color = 'lightgreen')
    plt.xticks(rotation=315)
    plt.xlim(x_label[0], x_label[-1])
    plt.grid()
    plt.tight_layout()
    # plt.savefig(dpi=400, fname = 'D32/' + name_figure) 
    plt.show()
     
def plot_whole_graph(data):
    '''Plot the whole SLS graphs to get an idea about sample stability and polydispersity.
    A peak detection algorithm is added for polydispersity analysis. The location of the peaks are stored in the x-vs_I file for further analysis.'''
    
    #Paremeters needed to run the code
    N = len(data)
    k=0
    measurement_indicator = np.array(['#1', '#2', '#3', '#4', '#5', 'Avg'])
    name = data['Sample Name']

    #Open needed file
    f = open('Mastersizer/intensity.txt','a')
    f.truncate(0)
    f.close()
    
    #Initialise arrays
    peak_widths = np.zeros(N)
    peak_widths_errors = np.zeros(N)
    peak_width = np.zeros(6)
    
    for i in range(0,N):
        plt.figure(figsize=(5, 4))

        #Get data in dataframe to be able to retrieve rows
        df = pd.DataFrame(data)
        y = df.iloc[i,9:] #y values from rows
        x = list(df.columns.values[9:]) #x values from header
        x = [float(i) for i in x] #make header floats
        
        f = open('Mastersizer/intensity.txt','a')
        f.write('{}'.format(y[0]))
        for p in range(1,len(y)):
            f.write(',{}'.format(y[p]))
        f.write('\n')

        #Find total peak
        intensity_measured = np.where(y > 0)[0]
        max_x = x[intensity_measured[-1]]
        min_x = x[intensity_measured[0]]
        peak_width[k] = max_x - min_x
        
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 15
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 18
        plt.rcParams["xtick.labelsize"] = 18
        plt.rcParams["ytick.labelsize"] = 18
        plt.rcParams["legend.fontsize"] = 15
        

        date, protein_wt, protein_type, oil_wt, bar, passes, indicator, rpm, mins, acceleration, water, age = get_values(name[i])
        # print(date, protein_wt, protein_type, oil_wt, bar, passes, indicator, rpm, mins, acceleration, water, age)

        plt.plot(x,y, label=measurement_indicator[k]) #plot first graph of series, they all have 5 measurements and an average
        k += 1 #use right indicator for legend
        
        if k == 6:
            k=0
            
            #Get polydispersity averaged data
            peak_width_avg = np.average(peak_width[0:5])
            peak_width_std = np.std(peak_width[0:5])
            peak_width = np.zeros(6)

            y_peaks = []
            y_inflection = []
            x_peaks = []
            x_inflection = []
            peak_index = []
            inflection_index = []
            for j in range(2, len(y)-2):
                slope_1 = (y[j-1]-y[j-2])/(x[j-1]-x[j-2])
                slope_2 = (y[j]-y[j-1])/(x[j]-x[j-1])
                slope_3 = (y[j+1]-y[j])/(x[j+1]-x[j])
                if abs(slope_1) > abs(slope_2) and abs(slope_2) < abs(slope_3) and slope_1>=0:
                    # print('The peak is at {} um with intensity {}'.format(x[j], y[j]))
                    y_peaks = np.append(y_peaks, y[j-1])
                    x_peaks = np.append(x_peaks, x[j-1])
                    peak_index = np.append(peak_index, j-1)
                if abs(slope_1) < abs(slope_2) and abs(slope_2) > abs(slope_3) and slope_2>=0:
                    y_inflection = np.append(y_inflection, y[j-1])
                    x_inflection = np.append(x_inflection, x[j-1])
                    inflection_index = np.append(inflection_index, j-1)             

            peak_widths_separate = np.zeros(len(y_peaks))
            for m in range(len(y_peaks)):
                x_3_index = int(peak_index[m]+(peak_index[m]-inflection_index[m]))
                y_fit = np.array([y_inflection[m], y_peaks[m], y_inflection[m]])
                x_fit = np.array([inflection_index[m], peak_index[m], x_3_index])
                popt, pcov = curve_fit(parabola,x_fit,y_fit)

                # ### Works, but not very accurate
                # x_index_arr = np.linspace(0, len(x)-1, len(x))            
                # y_parabola = parabola(x_index_arr, popt[0], popt[1], popt[2])
                # y_positive_index = np.where(y_parabola>0)[0]
                # min = int(y_positive_index[0])-1
                # max = int(y_positive_index[-1])+2

                # x_positve = x[min:max:1]
                # y_positve = y_parabola[min:max:1]
                # peak_width_m = x_positve[-1]-x_positve[0]
                # peak_widths_separate[m] = peak_width_m
                # # print('The peak width of peak {} is {}um'.format(k, peak_width_m))
                # # plt.plot(x_index_arr, x, label = 'actual')
                # plt.plot(x_positve, y_positve , label = 'low res fit')

                ### Higer resolution on fit for better width determination
                x_index_arr = np.linspace(0, len(x)-1, len(x)*10000)
                
                y_parabola = parabola(x_index_arr, popt[0], popt[1], popt[2])
                y_positive_index = np.where(y_parabola>0)[0]

                min = int(y_positive_index[0])
                max = int(y_positive_index[-1])

                x_index = np.linspace(0, len(x)-1, len(x))
                popt2, pcov2 = curve_fit(exp, x_index, x) 

                x_log_arr =  exp(x_index_arr, popt2[0], popt2[1], popt2[2], popt2[3])
                x_positve = x_log_arr[min:max:1]
                y_positve = y_parabola[min:max:1]
                peak_width_m = x_positve[-1]-x_positve[0]
                peak_widths_separate[m] = peak_width_m
                # print('The peak width of peak {} is {}um'.format(k, peak_width_m))
                # plt.plot(x_index_arr, x_log_arr, label='fitted')
                plt.plot(x_positve, y_positve , label = r'Peak {:.2f}$\mu$m'.format(y_peaks[m]))
                # plt.legend()
                # plt.show()


            
            print('The average peak width is {:.2f}+/-{:.2f} um'.format(peak_width_avg, peak_width_std))
            print('The peaks are at {} um'.format(x_peaks))
            print('The intensities are {}'.format(y_peaks))
            print('The peak indices are {}'.format(peak_index))
            # print('The inflections are at {} um'.format(x_inflection))
            # print('The inflection intensities are {}'.format(y_inflection))
            # print('The infleciton indices are {}'.format(inflection_index))   
            print('The peak widths are {}um'.format(peak_widths_separate))                   
            
            peak_widths[i]=peak_width_avg
            peak_widths_errors[i]=peak_width_std
            
            name_figure = r'{}wt% {} and {}wt% oil at {}bar {}x {} {} {}'.format(protein_wt, protein_type, oil_wt, bar, passes, age, water, indicator)
            name_figure = name_figure.replace('.', ',')
            
            #Save peak widths to file
            f = open('Mastersizer/Good_data/{}.txt'.format(name_figure),'a')
            f.write('\n')
            f.write('{},{},{},{},{}'.format(peak_width_avg, peak_width_std, x_peaks, y_peaks, peak_index))
            f.close()

            # plt.title(name_figure)
            plt.xscale('log')
            # plt.text(peak_index[0], y_peaks[0], r'${.2f}\mu m$'.format(y_peaks[0]), transform=plt.gca().transAxes, fontsize=12)
            plt.xlim(x[0], x[-1])
            plt.ylim(-0.4, np.max(y)+1)
            plt.grid()
            plt.legend(loc='upper left', fontsize=14)
            plt.ylabel(r'Volume Density [%]')
            plt.xlabel(r'$D$[$\mu$m]')
            plt.tight_layout()
            plt.savefig(dpi=600, fname = 'Mastersizer/Good_data/{}'.format(date) + name_figure) 
            # plt.show()
            plt.close()
    f.close()
    return peak_widths, peak_widths_errors



