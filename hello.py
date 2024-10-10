from audioop import reverse

import matplotlib . pyplot as plt
import numpy as np
#Read chirp waveform.
chirp = np. fromfile ("chirp_2024.bin" , dtype=np.float32 )
print(f'n {len(chirp)}')

#Read microphone recording .
m = np. fromfile("chirp_rec_2024.bin" , dtype=np.float32 )
print(f'Lungth m {len(m)}')


plt . plot ( chirp )
plt . xlabel ( "Time (samples) " )
plt . ylabel ( "Transmitted signal $x[n]$")
plt.xlim(0, 2000)
plt.title("Transmitted signal x[n] over time")
plt .show()
# Plot the first three interpulse periods .
plt . plot (m[:30000])
plt . xlabel ( "Time (samples) " )
plt . ylabel ( "Received signal $m[n]$")
plt.title("Recorded signal m[n] over time")
plt .show()

chirpback=chirp[::-1]
# Compute the autocorrelation
a = np.convolve(chirp, chirpback, mode='full')


#given the deconvolution is a reverse signal, the signal is twice its usual length. and make the array of our chirp and reversed chirp
x= np.arange(-len(a)//2,len(a)//2)
plt.plot(x, a)
plt.xlim(-1000, 1000)  # Adjust this limit based on the significant portion of your signal
plt.xlabel("Time lag (samples)")
plt.ylabel("Autocorrelation $c[n]$")

plt.xlim(-20,20)
plt.title("Convolution of time reversed signal x[-n] kand x[n]")
plt.grid()
plt.show()


############################################

speed_of_sound = 343  # Speed of sound in m/s
sample_rate = 44100   # Sample rate in samples/s

# Calculate time per sample
T_s = 1 / sample_rate  # Time per sample in seconds

# Calculate distance traveled per sample
distance_per_sample = 343* T_s  # in meters

# Calculate total distance for 10000 samples
N_samples = 10000
total_distance = N_samples * distance_per_sample

# Create an array from 0 to total_distance
distance_array = np.linspace(0, total_distance, num=100)

fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111)

#sc = plt.scatter(distance_array,weight, s = 200, c=calories, cmap=plt.cm.jet)

#cbar = fig.colorbar(sc, orientation='horizontal')

plt.show()

# Implementation of matplotlib function

from matplotlib.colors import LogNorm
#speed interpulse period
y=(343*np.arange(10000))/44100
N_ipps = int(np.floor(len(m) / 10000))  # Number of pulses
x=(np.arange(N_ipps)*10000)/44100




P = np.zeros([N_ipps, 10000])  # Matrix for scattered power
deconvolution_filter = chirp[::-1]  # Deconvolution filter

# Compute scattered power per pulse
for i in range(N_ipps):
    echo = m[(i * 10000):(i * 10000 + 10000)]
    deconvolved = np.convolve(echo, deconvolution_filter, mode="same")
    P[i, :] = np.abs(deconvolved) ** 2.0  # Scattered power


z = 10.0 * np.log10(np.transpose(P))
z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='inferno', shading='auto')

fig.colorbar(c, ax = ax)
ax.set_title('Power as a function of time and range')
plt.show()


#Finding the greatest power and the corresponding maximum distance
PowerP = np.argmax(P, axis=1)
distance = y[PowerP]
A = np.max(distance)
print(f"Furthest distance {A}m")







############################################################################_____________________


















