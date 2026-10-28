import matplotlib.pyplot as plt
import numpy as np

radius = int(input("Enter radius: ")) # radius of the circle
#The electrodes that will be placed on the head
electrode = int(input("Enter the number of electrodes: "))
#To evenly space the placement of the electrode
spacing = 360/electrode
theta = 0
source_x = 2.3 #first placement of x
source_y = 3.1 #first placement of y
p= 1 #for dipole interaction of source
orientation = int(input("Enter the orientation :"))
x = []
y = []
r_values = []
x_placing = []
y_placing = []
alignments = []
potential =[]


for i in range(0 , 361):
    x.append(radius * np.cos(i * (np.pi/180)))
    y.append(radius * np.sin(i * (np.pi/180)))

while theta < 360:
    x_placing.append(radius * np.cos(theta * (np.pi/180)))
    y_placing.append(radius * np.sin(theta * (np.pi/180)))
    theta = theta + spacing


p_x = p * np.cos(orientation * (np.pi / 180))
p_y = p * np.sin(orientation * (np.pi / 180))


for electrodes in range(0, electrode):
    r_x = x_placing[electrodes] - source_x
    r_y = y_placing[electrodes] - source_y
    r_intermediate = (r_x ,r_y)
    r_values.append(r_intermediate)

for k in range(0,electrode):
    numerator = (p_x * r_values[k][0]) + (p_y *r_values[k][1])
    denominator = (np.sqrt(p_x**2 + p_y**2) * np.sqrt(r_values[k][0]**2 + r_values[k][1]**2))
    alignment = numerator/denominator
    alignments.append(alignment)

for value in range(0,electrode):
       upper_part = p * alignments[value]
       lower_part = np.sqrt(r_values[value][0]**2 + r_values[value][1]**2)
       overall = upper_part / (lower_part **2)
       potential.append(float(overall))

result = {}
for trial in range (0 ,100):
     potential_new = []
     for a in range(0 ,electrode):
          potential_new.append(potential[a] + np.random.normal(0,0.5))
#Inverse solver comes
     candidates = []
     sample_orientations =[]
     predicted_voltages = []

     for degrees in range(0,360,10):
         sample_orientations.append(degrees)

     for x in np.arange(-5,5,0.1):
        for orientations in sample_orientations:
            for y in np.arange(-5,5,0.1):
                if x**2 + y**2 < radius **2:
                  candidates.append([x , y , orientations])  
          
     for i in range (0 ,len(candidates)):
        candidate_list =[]
        for k in range(0 , electrode):
           pX = p * np.cos(candidates[i][2] * (np.pi/180))
           pY = p * np.sin(candidates[i][2] * (np.pi/180))
           rX = x_placing[k] - candidates[i][0]
           rY = y_placing[k] - candidates[i][1]
           r = np.sqrt(rX**2 + rY**2)
           v_pred = ((pX * rX) + (pY * rY)) / r**3
           candidate_list.append(v_pred)
        predicted_voltages.append(candidate_list)

     rmse_final = []
     for z in range(0 ,len(predicted_voltages)):
        sq_errored_sum = 0
        for y in range(0,electrode):
          sq_errored_sum = sq_errored_sum + ((potential_new[y] - predicted_voltages[z][y]) **2 ) / electrode
        error = np.sqrt(sq_errored_sum)
        rmse_final.append(error)

     least_error = np.argmin(rmse_final)
     location = tuple(candidates[least_error])

     if location in result:
         result[location] += 1
     else:
         result[location] =1

print(result)             




               

          









