import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def Interpolate_(eventlist, points, scaling=1):
    results = []
    for i in sorted(eventlist):
        Pc, TCA = eventlist[i][:,0], scaling * eventlist[i][:,1]
        f = interp1d(TCA, Pc, kind='linear')
        t_out = np.linspace(TCA.min(), TCA.max(), points)
        results.append(f(t_out))
    # convert list-of-1D-arrays → single 2D array (n_events × points)
    return np.array(results)                                                       
    

'''
plt.subplot(1,number_of_events,1)    
plt.plot((scaling*eventlist[1][1][:]), eventlist[1][0][:], 'o', label = 'data')          #original data points visualization
#plt.plot(x,y,'o', label= 'interpolant1')                                #individual data points visualization
plt.plot(interp_data[0][0][:],interp_data[0][1][:],'-', label= 'interpolant')                             #interpolant visualization
plt.plot(results[0][1][:], results[0][0][:], 'o', label = 'output')               #output visualization
plt.gca().invert_xaxis()
plt.legend()
plt.title('Interpolant')
plt.xlabel('TCA')
plt.ylabel('Pc')

plt.subplot(1,number_of_events,2)    
plt.plot(scaling*eventlist[2][1][:], eventlist[2][0][:], 'o', label = 'data')                              #original data points visualization
#plt.plot(x,y,'o', label= 'interpolant1')                            #individual data points visualization
plt.plot(interp_data[1][0][:],interp_data[1][1][:],'-', label= 'interpolant')                            #interpolant visualization
plt.plot(results[1][1][:], results[1][0][:], 'o', label = 'output')                #output visualization
plt.gca().invert_xaxis()
plt.legend()
plt.title('Interpolant')
plt.xlabel('TCA')
plt.ylabel('Pc')
plt.show()
'''
if __name__ == '__main__':
    #events1 = {1 : np.random.random((2,10)), 2 : np.array([[0,1,4,9,16,25],[0,1,2,3,4,5]]), 4 : np.array([[0,1,8,27,64,125],[0,1,2,3,4,5]])}
    events2 = {1 : np.array([[0,0],[1,1],[4,2],[9,3],[16,4],[25,5]]), 3 : np.array([[0,0],[1,1],[8,2],[27,3],[64,4],[125,5]])}
    print(Interpolate_(events2, 3))

    out = Interpolate_(events2, points=3)
    print(type(out), out.shape)