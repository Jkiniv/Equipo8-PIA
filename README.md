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
    - detector.py, se encarga de inicializar mediapipe
    - utils.py, agrupa los procesos matemáticos, formulas, normalización de distancias
- Simulaciónd de modelo
    Permite vizualizar el comportamiento que debe presentar el brazo, contiene:
    - Hachi.urdf, define la morfología del brazo
    - SimulacionURDF.py, genera solamente vizualización del modelo utilizando swift
    - realtimesim.py, permite vizualizar el modelo con lo obtenido con mediapipe
- Control del brazo IRL
    Toma los datos seriales obtenidos por Modulo de reconcimiento de brazo, y permite el control de los servos del brazo
- Neural Network
    TBD

-



# Estructura de brazo robotico

El codigo está pensado para un 6DOF RR-R-RRR-E
Se utilizaron los sigueintes servos:

Junto con un (ESP32) para el control del brazo

# Updates a control de brazo

Esta rama mejora el rendimiento del sistema de visión e integra un método completo de calibración para controlar en tiempo real un robot URDF de siete articulaciones mediante MediaPipe y Swift.

El sistema ahora utiliza referencias corporales, Pose World Landmarks, calibración personalizada del brazo y una garra paralela de dos dedos controlada proporcionalmente con la apertura de la mano.
Cambios principales
## Optimización del procesamiento

    Se redujo la resolución utilizada para la inferencia de MediaPipe a 640 × 360, conservando la visualización de cámara a mayor resolución.
    Se mantuvo la detección de Pose y Hand en cada frame para conservar una respuesta continua.
    Se separaron los tiempos de procesamiento de Pose, Hand y postproceso.
    Se redujo el procesamiento de la mano a los landmarks necesarios para controlar la garra.
    Se mantuvo la opción de dibujar los 21 landmarks completos para diagnóstico.
    Se agregó tolerancia independiente ante pérdidas temporales del brazo, torso y mano.
    Se conserva la última posición válida durante pérdidas breves de detección.

## Asociación entre brazo y mano

    Se permite detectar hasta dos manos.
    Se selecciona la mano más cercana a la muñeca del brazo objetivo.
    Se conserva la lateralidad anatómica seleccionada mediante TARGET_ARM.
    Se evita asociar accidentalmente la mano contraria con el brazo controlado.

## Apertura de la mano

    Se implementó una medida continua de apertura utilizando:
        separación entre pulgar e índice;
        extensión del índice;
        extensión del dedo medio;
        tamaño de la palma como referencia de normalización.
    Se agregó calibración de mano cerrada con la tecla F.
    Se agregó calibración de mano abierta con la tecla O.
    Cada postura se registra utilizando 30 muestras y la mediana.
    Se agregó validación del rango mínimo de calibración.
    Se agregó suavizado exponencial de la apertura.
    Se agregó reinicio de calibración mediante la tecla R.

## Garra paralela de dos dedos

    Se reemplazó el bloque único del efector final por una garra paralela sencilla.
    Se agregó una base fija para la garra.
    Se agregaron dos dedos prismáticos independientes.
    El robot pasó de seis a siete joints, con estructura RRRRRPP.
    El joint 6 controla el dedo izquierdo.
    El joint 7 controla el dedo derecho.
    Ambos dedos se desplazan simultáneamente en direcciones opuestas.
    La apertura de cada dedo se obtiene proporcionalmente a partir de la apertura calibrada de la mano.
    Se agregó un respaldo del URDF anterior en hachi_backup.urdf.
    Se validó el movimiento de los siete joints en Swift.

## Referencia corporal del torso

    Se agregaron ambos hombros y ambas caderas como referencias corporales.
    Se obtiene el centro de hombros y el centro de caderas.
    Se construyen tres ejes corporales:
        body_right;
        body_up;
        body_forward.
    Se creó una base ortogonal ligada al torso.
    La orientación del brazo se interpreta respecto al cuerpo y no directamente respecto a la cámara.
    Se añadió tolerancia ante pérdidas temporales de la referencia del torso.
    Se añadieron líneas y ejes de diagnóstico sobre la imagen.

## Pose World Landmarks

    Los landmarks normalizados se conservan para dibujo y asociación visual.
    Los Pose World Landmarks se utilizan para los cálculos tridimensionales.
    Se añadieron coordenadas de mundo para:
        hombro;
        codo;
        muñeca;
        ambos hombros;
        ambas caderas.
    El ángulo del codo se calcula en coordenadas tridimensionales de mundo.
    La referencia matemática del torso también se construye con coordenadas de mundo.

## Calibración del brazo

    Se agregó calibración neutral con la tecla C.
    Se agregó calibración horizontal con la tecla H.
    Se agregó calibración del codo flexionado con la tecla E.
    Se agregó reinicio de calibración mediante la tecla X.
    Cada postura utiliza 30 muestras y calcula la mediana.
    Se valida un rango mínimo para el hombro y el codo.
    La postura C representa el brazo abajo y recto.
    La postura H representa el brazo horizontal y recto.
    La postura E representa el brazo superior horizontal con la muñeca arriba.
    El hombro se calibra utilizando el rango C-H.
    El codo se calibra utilizando el rango H-E.

## Cero dinámico del codo

    Se detectó que MediaPipe reporta diferentes valores residuales de flexión con el codo recto según la elevación del hombro.
    Se implementó un cero dinámico que interpola entre:
        el valor recto registrado en C;
        el valor recto registrado en H.
    Esto permite conservar aproximadamente cero grados de codo mientras el brazo recto se mueve entre la posición inferior y horizontal.
    Se añadieron zona muerta, limitación y suavizado para reducir movimientos residuales.

## Control del robot

    El joint 1 controla el plano de movimiento del hombro.
    El joint 2 controla la elevación calibrada del hombro.
    El joint 3 controla la flexión calibrada del codo.
    Los joints 4 y 5 permanecen disponibles para el control futuro de la muñeca.
    Los joints 6 y 7 controlan los dedos de la garra.
    Se añadieron signos configurables para corregir la orientación de los joints.
    Se añadió un offset configurable de 180 grados para orientar el plano del hombro hacia el lado correcto del robot.
    Se limita cada objetivo utilizando los límites articulares definidos en el URDF.
    Se conserva la última posición válida cuando se pierde el tracking.
    El plano del hombro se mantiene estable cuando el brazo está cerca de la postura inferior.

## Configuración y diagnóstico

    Se homologaron las claves de configuración necesarias para detector.py, processor.py, main.py, Calibracion.py y realtimesim.py.
    Se añadieron validaciones para detectar claves faltantes.
    Se añadieron diagnósticos de límites articulares.
    Se añadieron estados visibles de calibración y tracking.
    Se muestran tiempos de Pose, Hand, visión, Swift y frame total.
    Se verificaron los archivos mediante py_compile.

## Resultado

El sistema permite:

Detectar el torso y el brazo objetivo.
Calibrar el movimiento del brazo para una persona específica.
Mover el robot respecto a la referencia corporal.
Calibrar la apertura de la mano.
Abrir y cerrar proporcionalmente una garra paralela de dos dedos.
Mantener una respuesta fluida en tiempo real.
Compensar parcialmente inclinaciones y cambios moderados de postura.