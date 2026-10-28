# Este codigo permite calcular la cinemática directa e inversa de un robot de ejemplo
from pathlib import Path
import roboticstoolbox as rtb   
import spatialmath as sm

urdf_path = Path(__file__).parent / "hachi.urdf"

# Este comando imprime los paramatros Denevit-Hartenberg del robot Hachi
robot = rtb.ERobot.URDF(str(urdf_path))
print(robot)  

# Se define T para cinematida directa e inversa
T = robot.fkine(robot.q)  # forward kinematics
# print(T) 
T = sm.SE3(0.7, 0.2, 0.1)*sm.SE3.OA([0, 1, 0], [0, 0, -1])


print(T)

sol = robot.ikine_LM(T)  # inverse kinematics
print(sol)
