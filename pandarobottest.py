# Este codigo permite calcular la cinemática directa e inversa de un robot de ejemplo

import roboticstoolbox as rtb   
import spatialmath as sm

# Este comando imprime los paramatros Denevit-Hartenberg del robot panda
robot = rtb.models.DH.Panda()
print(robot)  

# Se define T para cinematida directa e inversa
T = robot.fkine(robot.qz)  # forward kinematics
# print(T) 
T = sm.SE3(0.7, 0.2, 0.1)*sm.SE3.OA([0, 1, 0], [0, 0, -1])
# print(T)

sol = robot.ikine_LM(T)  # inverse kinematics
print(sol)
