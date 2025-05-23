import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def Interpolate_(eventlist, points, scaling=1):
    results = []
    for i in sorted(eventlist.keys()):
        event = eventlist[i]  
        Pc = event[:,0]
        TCA = scaling*event[:,1]


        interpolant = interp1d(TCA, Pc , kind= 'linear' )
        #x = np.linspace(min(TCA),max(TCA),accuracy)
        #y = interpolant(x)

        #interp_data.append(np.array([x,y]))

        output_TCA = np.linspace(min(TCA),max(TCA), points)
        output_Pc = interpolant(output_TCA)
    
        results.append(np.ndarray.tolist(output_Pc))
        #print(output_TCA, output_Pc)

        '''
        plt.plot(TCA, Pc, 'o', label = 'data')                              #original data points visualization
        #plt.plot(x,y,'o', label= 'interpolant1')                            #individual data points visualization
        #plt.plot(x,y,'-', label= 'interpolant')                             #interpolant visualization
        plt.plot(output_TCA, output_Pc, 'o', label= 'output')               #output visualization
        plt.gca().invert_xaxis()
        plt.legend()
        plt.title('Interpolant')
        plt.xlabel('TCA')
        plt.ylabel('Pc')
        plt.show()
        '''
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