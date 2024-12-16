import main

path = r''

main.damping(path, align=True, signal=False, lg=True)
main.FFT(path, align=True, angular=True, save=True)
main.Amp_drivFreq(path, angular=True)
main.Amp_voltage(path)
main.phase_shift(path, angular=True)