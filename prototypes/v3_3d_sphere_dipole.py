'''Gonna construct an absolute sphere brain model
with dynamic changes for intuition and understading 
for myself and then we'll be movung to the real EEG shi'''

import numpy as np
import matplotlib.pyplot as plt
radius = int(input("Enter the radius: "))
x = []
y = []
z = []

notSatisfied = True

while notSatisfied :
    source_x = int(input("x: "))
    source_y = int(input("y: "))
    source_z = int(input("z: "))
    if source_x**2 + source_y**2 + source_z**2 < radius**2:
        notSatisfied = False
        print(f"Source is at ({source_x},{source_y},{source_z})") 
        break
    else:
        print("Does not satisfy , choose good coordinates")

#The above three are (x,y,z) coordinates inside sphere

for phi in range(0,181):
    for theta in range( 0 ,361):
        x.append((radius * np.sin(phi * np.pi/180))* (np.cos(theta * np.pi/180))) 
        y.append((radius * np.sin(phi * np.pi/180))* (np.sin(theta * np.pi/180)))
        z.append(radius * np.cos(phi * np.pi/180))

#Now placing electrodes at fixed gaps using math intuitions
electrodes = int(input("Enter the number of electrodes: "))
z_spacing = 2 * radius / electrodes #moving downward
electrodes_x =[]
electrodes_y=[]
electrodes_z=[]
for i in range(0,electrodes):
    z_i = radius - (i+0.5) * z_spacing #for electrode i , how far down the head have I travelled
    phi_i = np.arccos(z_i/ radius) #At this height , this is how dar Ive tilted diwnward from the north pole
    theta_i = i * (137.5 * np.pi/180) #everytime I place another electrode , rotate around the head by another 137.5 degrees (this is golden angle)
    x_i = radius * np.sin(phi_i) * np.cos(theta_i)
    y_i = radius * np.sin(phi_i) * np.sin(theta_i)
    electrodes_x.append(x_i)
    electrodes_y.append(y_i)
    electrodes_z.append(z_i)

#calculating distances from electrodes to source
distances = []
grouped_xyz = []
for j in range(electrodes):
    rx = electrodes_x[j] - source_x
    ry = electrodes_y[j] - source_y
    rz = electrodes_z[j] - source_z
    ans = np.sqrt(rx**2 + ry**2 + rz**2)
    distances.append(ans)
    grouped_xyz.append([rx,ry,rz])

#doing the dipole physics
p =1
phi= int(input("Enter the angle you wanna look up/down: "))
theta = int(input("Enter the angle you wanna rotate/spin: "))
phi_rad = phi * np.pi / 180
theta_rad = theta * np.pi / 180

px = p * np.sin(phi_rad) * np.cos(theta_rad)
py = p * np.sin(phi_rad) * np.sin(theta_rad)
pz= p * np.cos(phi_rad)

dot_product = [] #to store intermediate values
for k in range(electrodes):
    res = (grouped_xyz[k][0] * px) + (grouped_xyz[k][1] * py) + (grouped_xyz[k][2] * pz)
    dot_product.append(res)

potentials = []
for l in range(electrodes):
    potentials.append(float(dot_product[l] / (distances[l]**3)))
result ={}
for trials in range(100):
  potentials_new =[]
  for potential in potentials :
    potentials_new.append(potential + np.random.normal(0,0.1))


  candidates =[]
  angle_combos =[]

  for theta_candidate in range(0,360,10):
      for phi_candidate in range(0,180,10):
          angle_combos.append([phi_candidate,theta_candidate])

  for x in np.arange(-radius , radius ,1):
      for o in range(len(angle_combos)):
          for y in np.arange(-radius,radius ,1):
              for z in np.arange(-radius , radius ,1):
                  if x**2 + y**2 + z**2 < radius**2:
                      candidates.append([x,y,z,angle_combos[o]])

  predicted_voltages =[]
  for m in range(len(candidates)):
      intermediate_dist =[]
      for n in range(electrodes):
          rX = electrodes_x[n] - candidates[m][0]
          rY = electrodes_y[n] - candidates[m][1]
          rZ = electrodes_z[n] - candidates[m][2]
          pX = p * np.sin(candidates[m][3][0] * np.pi/180) * np.cos(candidates[m][3][1] * np.pi/180)
          pY = p * np.sin(candidates[m][3][0] * np.pi/180) * np.sin(candidates[m][3][1] * np.pi/180)
          pZ = p * np.cos(candidates[m][3][0] * np.pi/180)
          r = np.sqrt(rX**2 + rY**2 + rZ**2)
          v_pred = ((pX * rX) + (pY * rY) + (pZ * rZ)) / r**3
          intermediate_dist.append(v_pred)
      predicted_voltages.append(intermediate_dist) 

  rmse_final =[]
  for t in range(len(predicted_voltages)):
      sq_sum = 0
      for b in range(electrodes):
          sq_sum = sq_sum + ((potentials_new[b] - predicted_voltages[t][b])**2)
      mean = (sq_sum / electrodes)
      rmse_final.append(np.sqrt(mean))

  least_error = np.argmin(rmse_final)   
  location = (
    candidates[least_error][0],
    candidates[least_error][1],
    candidates[least_error][2],
    tuple(candidates[least_error][3])
)
  if location in result:
      result[location]+=1
  else:
      result[location] =1

print(result)          



        

        




