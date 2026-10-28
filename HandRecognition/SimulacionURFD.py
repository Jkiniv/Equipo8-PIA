# En este archivo se corren simulaciones de URDF
# Se busca generar una visualizacion del robot a manera de wireframe y animar un barrido (sweep) en todas las articulaciones
from pathlib import Path
import time
import numpy as np
import roboticstoolbox as rtb   
import spatialmath as sm

urdf_path = Path(__file__).parent / "hachi.urdf"

# Cargar el robot desde el URDF usando ERobot
robot = rtb.ERobot.URDF(str(urdf_path))

# Imprimir los parámetros y estructura cinemática
print(robot)

# Iniciar la ventana de visualización en modo interactivo/bloqueo de animación
# Swift es el backend moderno por defecto en roboticstoolbox-python
env = rtb.backends.Swift()
env.launch(realtime=True)
env.add(robot)

# Obtener los límites de las articulaciones para evitar sobre-extender el modelo
qlim = robot.qlim

# Generar una trayectoria de barrido (sweep) para cada articulación
steps = 150
t = np.linspace(0, 2 * np.pi, steps)

print("Iniciando simulación de barrido en las articulaciones...")

# Bucle infinito o de varias iteraciones para mostrar el movimiento de barrido
for i in range(3):  # Repetir el ciclo 3 veces
    for step in range(steps):
        # Crear una postura donde cada articulación se mueve sinusoidalmente 
        # dentro de sus límites permitidos
        q_sweep = np.zeros(robot.n)
        for j in range(robot.n):
            # Amplitud basada en los límites de cada articulación
            q_min = qlim[0, j]
            q_max = qlim[1, j]
            
            # Si los límites son infinitos o nulos por defecto, usar un rango seguro de [-1.57, 1.57]
            if np.isinf(q_min) or np.isinf(q_max) or q_min == q_max:
                q_min, q_max = -1.57, 1.57
                
            amplitude = (q_max - q_min) / 2.0
            center = (q_max + q_min) / 2.0
            
            # Desfasar cada articulación (phase shift) para que no se muevan todas al unísono exacto
            q_sweep[j] = center + amplitude * np.sin(t[step] + (j * np.pi / robot.n))

        # Asignar la configuración actual al robot en el entorno gráfico
        robot.q = q_sweep
        
        # Actualizar la escena en Swift
        env.step(0.02)

# Mantener la ventana abierta al finalizar
print("Simulación finalizada.")
env.hold()