# En este archivo se corren simulaciones de URDF.
# Genera una visualización del robot y anima un barrido
# en todas las articulaciones.
#
# Controles:
# - ENTER en la terminal: cerrar la simulación.
# - Ctrl + C: salida de emergencia.

from pathlib import Path
import threading

import numpy as np
import roboticstoolbox as rtb
import swift


# ============================================================
# CONFIGURACIÓN
# ============================================================

URDF_FILENAME = "hachi.urdf"

STEPS = 150
NUMBER_OF_CYCLES = 3
SIMULATION_STEP_SECONDS = 0.02


# ============================================================
# CONTROL DE SALIDA
# ============================================================

exit_event = threading.Event()


def wait_for_exit():
    """
    Espera que el usuario presione ENTER en la terminal.
    Se ejecuta en un hilo independiente para no detener
    la animación del robot.
    """

    input("\nPresiona ENTER en la terminal para cerrar la simulación...\n")
    exit_event.set()


# ============================================================
# CARGAR ROBOT
# ============================================================

urdf_path = Path(__file__).parent / URDF_FILENAME

if not urdf_path.is_file():
    raise FileNotFoundError(
        f"No se encontró el URDF:\n{urdf_path}"
    )

robot = rtb.ERobot.URDF(str(urdf_path))

print(robot)

if robot.n < 7:
    raise RuntimeError(
        "El URDF debe contener al menos 7 joints."
    )

qlim = robot.qlim

print("=" * 70)
print("Limites articulares")

for joint_index in range(robot.n):
    unit = "m" if joint_index in (5, 6) else "rad"
    print(
        f"Joint {joint_index + 1}: "
        f"{qlim[0, joint_index]:+.4f} a "
        f"{qlim[1, joint_index]:+.4f} {unit}"
    )

print("=" * 70)


# ============================================================
# INICIAR SWIFT
# ============================================================

env = swift.Swift()
env.launch(realtime=True)
env.add(robot)


# ============================================================
# OBTENER LÍMITES ARTICULARES
# ============================================================

steps = STEPS
t = np.linspace(0, 2 * np.pi, steps)


# ============================================================
# INICIAR HILO PARA DETECTAR ENTER
# ============================================================

exit_thread = threading.Thread(
    target=wait_for_exit,
    daemon=True
)

exit_thread.start()


# ============================================================
# SIMULACIÓN
# ============================================================

print("=" * 70)
print("Iniciando simulación de barrido en las articulaciones...")
print("Presiona ENTER en esta terminal para salir.")
print("=" * 70)

try:

    for cycle in range(NUMBER_OF_CYCLES):

        if exit_event.is_set():
            break

        print(
            f"Ejecutando ciclo "
            f"{cycle + 1}/{NUMBER_OF_CYCLES}"
        )

        for step in range(steps):

            if exit_event.is_set():
                break

            q_sweep = np.zeros(
                robot.n,
                dtype=np.float64
            )

            for joint_index in range(robot.n):

                q_min = qlim[0, joint_index]
                q_max = qlim[1, joint_index]

                if (
                    not np.isfinite(q_min)
                    or not np.isfinite(q_max)
                    or q_min == q_max
                ):
                    q_min = -1.57
                    q_max = 1.57

                amplitude = (
                    q_max - q_min
                ) / 2.0

                center = (
                    q_max + q_min
                ) / 2.0

                phase_shift = (
                    joint_index
                    * np.pi
                    / robot.n
                )

                q_sweep[joint_index] = (
                    center
                    + amplitude
                    * np.sin(
                        t[step] + phase_shift
                    )
                )

            # Asegurar que las articulaciones respeten
            # los límites definidos en el URDF.
            q_sweep = np.clip(
                q_sweep,
                qlim[0, :],
                qlim[1, :]
            )

            robot.q = q_sweep

            env.step(SIMULATION_STEP_SECONDS)

    if not exit_event.is_set():

        print("=" * 70)
        print("El barrido automático terminó.")
        print("La simulación permanecerá abierta.")
        print("Presiona ENTER en la terminal para salir.")
        print("=" * 70)

        # Mantener el simulador abierto sin usar env.hold().
        while not exit_event.is_set():
            env.step(0.05)

except KeyboardInterrupt:

    print("\nSimulación detenida con Ctrl + C.")

finally:

    print("Cerrando simulador Swift...")

    try:
        env.close()
    except Exception:
        pass

    print("Simulación cerrada correctamente.")