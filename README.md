PIA de Proyecto mecatronica Equipo 8

# Prerequisitos
Se recomienda usar Visual Studio para la programación
Lenguajes utilizados:
    - Python, the goat does the heavy lifting
        - cv2
        - Mediapipe
        - Roboticstoolbox
        - Numpy
        - Spatialmath
    - Siwft, simulación se intala por parte de roboticstoolbox

# Estructura de codigo

El codigo está dividido en varias secciones, para que sea mas sencillo de modificar tiene la sigueinte estructura:
- Modulo de reconocimiento de brazo:
    Es el que determina la posición de los landmarks, usa mediapipe & cv2, contiene
    - main.py, controla ciclo de camara, timing, y renderizado
    - processor.py, corre los proceso de mediapie, actualiza la ubicaciond de coordenadas y dibuja el wireform
    - utils.py, agrupa los procesos matemáticos, formulas, normalización de distancias
- Simulaciónd de modelo
    Permite vizualizar el comportamiento que debe presentar el brazo, contiene:
    - Hachi.urdf, define la morfología del brazo
    - SimulacionURDF.py, genera la vizualización del modelo utilizando swift
- Control del brazo IRL
    Toma los datos seriales obtenidos por Modulo de reconcimiento de brazo, y permite el control de los servos del brazo
- Neural Network
    TBD

-



# Estructura de brazo robotico

El codigo está pensado para un 6DOF RR-R-RRR-E
Se utilizaron los sigueintes servos:

Junto con un (ESP32) para el control del brazo