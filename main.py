import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate, correlation_lags
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from glob import glob
import os

def avg_sig(arr):
    mak = arr[0]
    maz = arr[0]
    for i in arr:
        if i > mak:
            mak = i
        if i < maz:
            maz = i
    return float((mak+maz)/2)

def maksimum_class(arr):
    mak = arr[0]
    for i in arr:
        if i > mak:
            mak = i
    return mak

def maksimum(arr):
    mak = arr[0]
    maz = arr[0]
    for i in arr:
        if i > mak:
            mak = i
        if i < maz:
            maz = i
    return float(mak-(mak+maz)/2)

def maksimum_pos(arr):
    mak = arr[10]
    pos = 0
    for i in range(10,len(arr)):
        if arr[i] > mak:
            mak = arr[i]
            pos = i
    return pos

def all_maksimum_pos(arr):
    all_p = []
    for i in range(0,len(arr)):
        if i>=5 and 4*arr[i-2]>arr[i]+arr[i-1]+arr[i-3]+arr[i-4]:
            all_p.append(i-2)
    return all_p



def FFT(path, **kwargs):
    al = kwargs.get('align', False)         # All signals start from nonzero
    save = kwargs.get('save', False)        # Save figure
    angular = kwargs.get('angular', False)  # FFT in angular frequency
    
    folder = glob(path+r'\*')
    pathh = os.path.dirname(os.path.abspath(__file__))
    
    for file in folder:
        file_data = pd.read_csv(file)
        num=1
        peak=0
        
        fig, (ax1, ax2) = plt.subplots(2, 1)
        
        ax1.set_title("Signal in Time Domain")
        ax1.set_xlabel("Time, s")
        ax1.set_ylabel("Amplitude, rel.")
        
        ax2.set_title("Magnitude Spectrum (FFT)")
        if angular:
            ax2.set_xlabel("Angular frequency, s$^{-1}$")
        else:
            ax2.set_xlabel("Frequency, Hz")
        ax2.set_ylabel("Magnitude, rel.")
        ax2.set_xscale('log')
        
        time_col = r'Time (s) Run #'+str(num)
        data_col = r'Angle, Ch 1+2 (rad) Run #'+str(num)
        
        while time_col in file_data.columns:
            t = pd.read_csv(file, usecols=[time_col], skip_blank_lines=True)
            t.dropna(how="all", inplace=True)
            t = t.values.flatten()
            dt = t[1] - t[0]
            
            signal = pd.read_csv(file, usecols=[data_col], skip_blank_lines=True)
            signal.dropna(how="all", inplace=True)
            signal = signal.values.flatten()
            signal = (signal-avg_sig(signal))/maksimum(signal-avg_sig(signal))
            
            ### Start only where signal >=0.15 ###
            if al:
                for u in range(0,len(signal)):
                    if(np.abs(signal[u])>=0.15):
                        signal = signal[u:]
                        break
            
            ### Match time and signal length ###
            if(len(t)>len(signal)):
                t = t[:len(signal)]
            else:
                signal = signal[:len(t)]
            
            ### FFT ###
            fft_output = np.fft.fft(signal)
            magnitude = np.abs(fft_output)
            positive_frequencies = magnitude[:len(magnitude) // 2]
            f = np.fft.fftfreq(len(signal),dt)[:len(positive_frequencies)]
            
            if al:
                time = []
                for tm in range(0,len(t)):
                    time.append(dt*tm)
                ax1.plot(time, (signal-avg_sig(signal))/maksimum(signal-avg_sig(signal)))
            else:
                ax1.plot(t, signal)
            arrs = positive_frequencies/maksimum_class(positive_frequencies)
            
            if(f[0]==0):
                arrs[0]=0
            
            if angular:
                ax2.plot(f*2*np.pi, arrs)
            else:
                ax2.plot(f, arrs)
            
            pek = np.argmax(arrs)
            peak += f[pek]
            
            num+=1
            time_col = r'Time (s) Run #'+str(num)
            data_col = r'Angle, Ch 1+2 (rad) Run #'+str(num)
        print(peak/(num-1)*2*np.pi)
        
        ax2.set_xlim([2e-2,9e1])
        ax1.set_xlim(xmin=0)
        
        plt.tight_layout()
        if save:
            plt.savefig(pathh+r'/EPS/'+os.path.basename(file)[:-4]+r'.eps', format="eps")
        plt.show()
    


def Amp_drivFreq(path, **kwargs):
    angular = kwargs.get('angular', False)  # FFT in angular frequency
    peaks = []
    folder = glob(path+r'\332*')
    for file in folder:
        if (len(file.rsplit('\\', 1)[1])>15 or len(file.rsplit('\\', 1)[1])<8):
            continue
        else:
            num = 1
            tot = 0.0
            avg_f = 0
            data = pd.DataFrame(pd.read_csv(file))
            time_col = r'Time (s) Run #'+str(num)
            data_col = r'Angle, Ch 1+2 (rad) Run #'+str(num)
            freq_col = r'Angle, Ch 3+4 (rad) Run #'+str(num)
            
            while data_col in data.columns:    
                t = pd.read_csv(file, usecols=[time_col], skip_blank_lines=True)
                t.dropna(how="all", inplace=True)
                t = t.values.flatten()
                dt = t[1] - t[0]
                
                signal = pd.read_csv(file, usecols=[data_col], skip_blank_lines=True)
                signal.dropna(how="all", inplace=True)
                signal = signal.values.flatten()
                
                signal_f = pd.read_csv(file, usecols=[freq_col], skip_blank_lines=True)
                signal_f.dropna(how="all", inplace=True)
                signal_f = signal_f.values.flatten()
                
                if(len(signal)>len(signal_f)):
                    signal = signal[len(signal)-len(signal_f):]
                else:
                    signal_f = signal_f[len(signal_f)-len(signal):]
                
                fft_output = np.fft.fft(signal_f)
                magnitude = np.abs(fft_output)
                positive_frequencies = magnitude[:len(magnitude) // 2]
                f = np.fft.fftfreq(len(signal_f),dt)[:len(positive_frequencies)]
                tot = maksimum(signal)
                avg_f = f[maksimum_pos(positive_frequencies/maksimum(positive_frequencies))]
                
                if angular:
                    peaks.append([avg_f*2*np.pi,tot])
                else:
                    peaks.append([avg_f,tot])
                
                num+=1
                time_col = r'Time (s) Run #'+str(num)
                data_col = r'Angle, Ch 1+2 (rad) Run #'+str(num)
                freq_col = r'Angle, Ch 3+4 (rad) Run #'+str(num)
    peaks = sorted(peaks,key=lambda l:l[0])
    print(peaks)
    peaks = np.array(peaks)
    plt.rc('axes', axisbelow=True)
    plt.grid()
    plt.ylim(ymin=0, ymax=1.1)
    if angular:
        plt.xlabel(r'Driving frequency $\omega_d$, s$^{-1}$')
    else:
        plt.xlabel(r'Driving frequency, Hz')
    plt.ylabel(r'Amplitude $A$, rel.')
    plt.scatter(peaks[:,0], peaks[:,1]/maksimum_class(peaks[:,1]), color="#000000")
    plt.savefig("Amp_freq.eps", format="eps")
    plt.show()



def damping(path, **kwargs):
    al = kwargs.get('align', False)
    sig = kwargs.get('signal', True)
    lg_ax = kwargs.get('lg', False)
    
    folder = glob(path+r'\*')
    for file in folder:
        file_data = pd.read_csv(file)
        num=3
        all_cv_k = 0
        all_cv_b = 0
        
        ax1 = plt.axes() 
        ax1.set_xlabel("Time, s")
        ax1.set_ylabel("Amplitude, rel.")
        if lg_ax:
            ax1.set_yscale('log')
        
        time_col = r'Time (s) Run #'+str(num)
        data_col = r'Angle, Ch 1+2 (rad) Run #'+str(num)
        while time_col in file_data.columns:
            t = pd.read_csv(file, usecols=[time_col], skip_blank_lines=True)
            t.dropna(how="all", inplace=True)
            t = t.values.flatten()
            dt = t[1] - t[0]
            signal = pd.read_csv(file, usecols=[data_col], skip_blank_lines=True)
            signal.dropna(how="all", inplace=True)
            signal = signal.values.flatten()
            if al:
                for u in range(0,len(signal)):
                    if(np.abs(signal[u])>=0.15):
                        signal = signal[u:]
                        break
            if(len(t)>len(signal)):
                t = t[len(t)-len(signal):]
            else:
                signal = signal[len(signal)-len(t):]
            
            peaks, _ = find_peaks(signal, prominence=0.1)
            
            signal_peak = [(x-avg_sig(signal)) / maksimum(signal-avg_sig(signal)) for x in signal[peaks]]
            if al:
                time = []
                time2 = []
                for tm in range(0,len(t)):
                    time.append(dt*tm)
                for tm in peaks:
                    time2.append(dt*tm)
                if sig:
                    ax1.plot(time, (signal-avg_sig(signal))/maksimum(signal-avg_sig(signal)))
                ax1.scatter(time2, signal_peak)
                [k, b], res1 = curve_fit(lambda x1,k,b: k*np.exp(b*x1),  time2,  signal_peak)
                print(k,b)
                all_cv_k += k
                all_cv_b += b                
            else:
                if sig:
                    ax1.plot(t,(signal-avg_sig(signal))/maksimum(signal-avg_sig(signal)))
                ax1.scatter(t[peaks], signal_peak)
            num+=1
            time_col = r'Time (s) Run #'+str(num)
            data_col = r'Angle, Ch 1+2 (rad) Run #'+str(num)
        
        all_cv_k = all_cv_k / (num-3)
        all_cv_b = all_cv_b / (num-3)
        yy = [all_cv_k * np.exp(all_cv_b * x) for x in time]
        plt.plot(time, yy, color='black')
        plt.tight_layout()
        ax1.set_xlim(xmin=0)
        plt.grid()
        print(all_cv_k, all_cv_b)
        plt.savefig("Peak_time.eps", format="eps")
        plt.show()
        
        
def phase_shift(path, **kwargs):
    angular = kwargs.get('angular', False)  # FFT in angular frequency
    
    folder = glob(path+r'\*')
    phase_x = []
    phase_y = []
    
    ax1 = plt.axes()
    if angular:
        ax1.set_xlabel(r'Driving frequency $\omega_d$, s$^{-1}$')
    else:
        ax1.set_xlabel(r'Driving frequency $\omega_d$, Hz')
    ax1.set_ylabel(r'Phase shift $\phi$')
    
    ax1.set_yticks([0,3.14159265/2,3.14159265])
    ax1.set_yticklabels([r'$0$',r'$\frac{\pi}{2}$',r'$\pi$'])
    ax1.set_ylim([0,3.14159266])
    ax1.set_xlim(xmin=0,xmax=10)
    plt.grid()
    
    for file in folder:
        file_data = pd.read_csv(file)
        num=1
        
        time_col = r'Time (s) Run #'+str(num)
        data_col = r'Angle, Ch 1+2 (rad) Run #'+str(num)
        freq_col = r'Angle, Ch 3+4 (rad) Run #'+str(num)
        
        while time_col in file_data.columns:
            t = pd.read_csv(file, usecols=[time_col], skip_blank_lines=True)
            t.dropna(how="all", inplace=True)
            t = t.values.flatten()
            dt = t[1] - t[0]
            
            signal = pd.read_csv(file, usecols=[data_col], skip_blank_lines=True)
            signal.dropna(how="all", inplace=True)
            signal = signal.values.flatten()
            
            signal_f = pd.read_csv(file, usecols=[freq_col], skip_blank_lines=True)
            signal_f.dropna(how="all", inplace=True)
            signal_f = signal_f.values.flatten()
            
            if(len(t)>len(signal)):
                t = t[len(t)-len(signal):]
            else:
                signal = signal[len(signal)-len(t):]
            
            if(len(signal)>len(signal_f)):
                signal = signal[len(signal)-len(signal_f):]
            else:
                signal_f = signal_f[len(signal_f)-len(signal):]
            
            correlation = correlate(signal, signal_f, mode='full')
            lags = correlation_lags(len(signal), len(signal_f), mode='full')
            lag = lags[np.argmax(correlation)]*dt
            
            fft_output = np.fft.fft(signal_f)
            magnitude = np.abs(fft_output)
            positive_frequencies = magnitude[:len(magnitude) // 2]
            f = np.fft.fftfreq(len(signal_f),dt)[:len(positive_frequencies)]
            peaks, _ = find_peaks(positive_frequencies, prominence=1)
            max_ind = peaks[np.argmax(positive_frequencies[peaks])]
            phase_x.append(f[max_ind])
            y_ap = 0
            if(f[max_ind] >= 4.628998):
                y_ap = lag*f[max_ind]*2*np.pi
            else:
                y_ap = lag*f[max_ind]*2*np.pi
            phase_y.append(y_ap)
            num+=1
            time_col = r'Time (s) Run #'+str(num)
            data_col = r'Angle, Ch 1+2 (rad) Run #'+str(num)
            freq_col = r'Angle, Ch 3+4 (rad) Run #'+str(num)
    plt.tight_layout()
    if angular:
        phase_x1 = [i*2*np.pi for i in phase_x]
        ax1.scatter(phase_x1, phase_y, color='black')
    else:
        ax1.scatter(phase_x, phase_y, color='black')
    
    g = 0.15375259717369
    #g = 0.16427938828576136
    x1 = np.linspace(4.628998,10,1000)
    y1 = np.arctan((2*g*x1)/(4.6289974423867495**2-x1**2))+np.pi
    x2 = np.linspace(0,4.628997,1000)
    y2 = np.arctan((2*g*x2)/(4.6289974423867495**2-x2**2))
    ax1.plot(x1, y1, color='#9d0000', linestyle='dashed')
    ax1.plot(x2, y2, color='#9d0000', linestyle='dashed')
    
    plt.savefig("Phase_freq.eps", format="eps")
    plt.show()