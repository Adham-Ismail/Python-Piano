import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.fftpack import fft
import math

#M1

t = np. linspace(0 , 3 , 12*1024)
N = 8
F1 = np.array([392,466.164,466.164,466.164,261.63,466.164,392,600])
F2 = np.array([0,0,0,0,0,0,0,0])
TimeStart =np.array([0,0.25,0.5,0.75,1.0, 1.2 , 1.6 , 2.2]) 
TimeEnd  =np.array([0.2,0.2,0.2,0.14,0.1,0.2,0.3,0.75])
count = 0
out = 0
while(count<N):
    F1In = F1[count]
    F2In = F2[count]
    tIn = TimeStart[count]
    Tin = TimeEnd[count]
    out += (np.sin(2*np.pi*F1In*t)+(np.sin(2*np.pi*F2In*t)))*((t>=tIn)&(t<=Tin+tIn)) 
    count+=1


#Num of samples & freq axis
num = 3*1024
f= np.linspace(0 , 512 , int(num/2))

#fourier tarnsform of output & axis
x_f = fft(out)
x_f = 2/num * np.abs(x_f [0:np.int(num/2)])


#Noise generation
fn1 , fn2 = np. random. randint(0, 512, 2)
n = np.sin(2*np.pi*fn1*t)+np.sin(2*np.pi*fn2*t)
xN = out + n

#freq domain of noise
xn_f = fft(xN)
xn_f = 2/num * np.abs(xn_f [0:np.int(num/2)])

#the 2 random freqs and the index of each
z = np.where(xn_f>math.ceil(np.max(out)))
index1 = z[0][0]
index2 = z[0][1]

#Get freqs through the value at index of on the freq axis
found1 = int(f[index1])
found2 = int(f[index2])


#removing noise in time
xFiltered = xN - (np.sin(2*np.pi*found1*t)+np.sin(2*np.pi*found2*t))

#fourier of filtered
xFiltered_f = fft(xFiltered)
xFiltered_f = 2/num * np.abs(xFiltered_f [0:np.int(num/2)])

sd.play(xFiltered, 3*1024)



#Time Domain
plt.figure()
plt.subplot(3,1,1)
plt.plot(t,out)
plt.subplot(3,1,2)
plt.plot(t,xN)
plt.subplot(3,1,3)
plt.plot(t,xFiltered)

#Freq Domain
plt.figure()
plt.subplot(3,1,1)
plt.plot(f,x_f)
plt.subplot(3,1,2)
plt.plot(f,xn_f)
plt.subplot(3,1,3)
plt.plot(f,xFiltered_f)


#plt.plot(t,out)
#sd.play(out, 3 * 1024)

