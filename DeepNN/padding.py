#padding
import numpy as np
testarray = [[7, 80,90], [5, 40,50], [3.3, 50, 60], [3.1, 60, 60], [2.2, 70, 60], [1.2,41,90], [0.90, 801, 1234]]

#Defenition
n_cdms = 18
days = [5,4,3,2,1,0]

outputarray = np.array([])

#Cut off CDMS above 5 days

#Oder the CDM in the proper place of the event ID

eventArray = np.full(18, np.nan)


#Fill every CDM on the proper place
for CDM in testarray:
    PlaceInEventArray  = int(round((CDM[0]*3))-1) #x3 as there are 3 cdm per day, than round it to get an the proper int -1 as we start with 0
    #Get rid of all cdms not within 5 days prior to TCA
    if PlaceInEventArray > n_cdms -1:
        continue
    #Check if there is no other CDM on the place that we want it, if it is, we move it one place, if there is a cdm as well, we will skip this CDM
    if np.isnan(eventArray[PlaceInEventArray]):
        eventArray[PlaceInEventArray] = CDM[1]
    elif np.isnan(eventArray[PlaceInEventArray+1]):
        eventArray[PlaceInEventArray+1] =CDM[1]

eventArray = np.flip(eventArray)
print(eventArray)

#Pad the rest 
indicesNotToPad = np.where(~np.isnan(eventArray))[0]

print(indicesNotToPad)

j=0
for i in range(len(eventArray)):
    if np.isnan(eventArray[i]):
        eventArray[i] = eventArray[indicesNotToPad[j]]
    elif ~np.isnan(eventArray[i]) and j < len(indicesNotToPad)-1:
        j += 1
        print(j)

print(eventArray) 

        






#raise NotImplemented
#for event in events:
#    eventArray = np.full(15, np.nan)
#    for CDM in event:
#        PlaceInEventArray  = int(round((CDM[1]*3))) #x3 as there are 3 cdm per day, than round it to get an the proper int
#        eventArray[placeInEventArray] = CDM[2:]
        




