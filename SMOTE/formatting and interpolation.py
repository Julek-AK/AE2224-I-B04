import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

points = 10                                                         #how many points in total do we want from the interpolation
accuracy = 500                                                      #higher is better, like around 500 is good

#test events
event = np.random.random((2,10)) 
#event = np.array([[0,1,4,9,16,25],[0,1,2,3,4,5]])

Pc = event[0,:]
TCA = 7 * event[1,:]

#print(Pc)  
#print(TCA)

interpolant = interp1d(TCA, Pc , kind='linear')
x = np.linspace(min(TCA),max(TCA),accuracy)
y = interpolant(x)

output_TCA = np.linspace(min(TCA),max(TCA), points)
output_Pc = interpolant(output_TCA)
print(output_TCA, output_Pc)                                       #print output values

plt.plot(TCA, Pc, 'o', label = 'data')                              #original data points visualization
#plt.plot(x,y,'o', label= 'interpolant1')                           #individual data points visualization
plt.plot(x,y,'-', label= 'interpolant')                             #interpolant visualization
plt.plot(output_TCA, output_Pc, 'o', label= 'output')               #output visualization
plt.gca().invert_xaxis()
plt.legend()
plt.title('Interpolant')
plt.xlabel('TCA')
plt.ylabel('Pc')
plt.show()

