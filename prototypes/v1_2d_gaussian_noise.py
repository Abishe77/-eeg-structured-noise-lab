import matplotlib.pyplot as plt
import numpy as np

radius = 5 # radius of the circle
#The electrodes that will be placed on the head
electrode = int(input("Enter the number of electrodes: "))
#To evenly space the placement of the electrode
spacing = 360/electrode
theta = 0
source_x = 2 #first placement of x
source_y = 3 #first placement of y
plt.scatter(source_x,source_y)
count = 0
x = []
y = []
x_placing = []
y_placing = []
distance = []
k = 2.5 #a fixed physical parameter
voltage = []

for i in range(0 , 361):
    x.append(radius * np.cos(i * (np.pi/180)))
    y.append(radius * np.sin(i * (np.pi/180)))

while theta < 360:
    x_placing.append(radius * np.cos(theta * (np.pi/180)))
    y_placing.append(radius * np.sin(theta * (np.pi/180)))
    theta = theta + spacing

while count!= electrode:
    dist_x = (x_placing[count] - source_x)**2
    dist_y = (y_placing[count] - source_y)**2
    distance.append(np.sqrt(dist_x + dist_y))
    count = count + 1 

for distances in distance:
    voltage.append(k/distances)
#The only differnece is we add noise to perfect measurements
result ={} #To store recordings


for trial in range (0,500):
    voltage_new =[]
    for i in range(0,electrode):
      voltage_new.append(voltage[i] + np.random.normal(0,0.5)) 
#Reverse problems starts
    sample_x = []
    sample_y =[]
    final_dist = []
    for candidate_x in range(-4 , 5):
       for candidate_y in range( -4,5):
          if candidate_x**2 + candidate_y**2 < 25:
            separate_dist = []
            sample_x.append(candidate_x)
            sample_y.append(candidate_y)
            for i in range(0,electrode):
                sample_x_dist = (candidate_x - x_placing[i])**2
                sample_y_dist = (candidate_y - y_placing[i])**2
                separate_dist.append(np.sqrt(sample_x_dist + sample_y_dist))
            final_dist.append(separate_dist) 
    final_predicted_voltage =[]
    for j in range (0, len(sample_x)):
       predicted_voltage = []
       for i in range(0,electrode):
         predicted_voltage.append(k/final_dist[j][i])
       final_predicted_voltage.append(predicted_voltage) 

    mse_final = []
    for j in range(0 , len(sample_x)):
       sum_values = 0
       for i in range(0,electrode):
          sum_values = sum_values +(voltage_new[i]- final_predicted_voltage[j][i])**2
       mse_final.append(sum_values/electrode)

    best_candidate = np.argmin(mse_final)
    location = (sample_x[best_candidate] , sample_y[best_candidate])
    if location in result :
      result[location]+=1
    else:    
      result[location] = 1

print(result)
     