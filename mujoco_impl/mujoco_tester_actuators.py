import mujoco
import numpy as np

# Constants for keycodes
UP_ARROW_KEYCODE = 265
DOWN_ARROW_KEYCODE = 264
LEFT_ARROW_KEYCODE = 263
RIGHT_ARROW_KEYCODE = 262
SPACEBAR_KEYCODE = 32

# Constants for the actuator control
ACTUATOR_X_INDEX = 0
ACTUATOR_Y_INDEX = 1
ACTUATOR_INCREMENT = 0.05  # Increment value for each keypress

# Initialize paused state
paused = False
change_values = False
changes = [0, 0]


def key_callback(keycode):
    global paused
    global change_values

    # Handle pause/unpause toggle
    if keycode == SPACEBAR_KEYCODE:
        paused = not paused
    else:
        get_actuator_values_for_key(keycode)
        change_values = True


def toggle_pause():
    """Toggle the paused state of the simulation."""
    global paused


def get_actuator_values_for_key(keycode):
    """Map keypress to actuator values for controlling board tilt."""
    global changes
    if keycode == UP_ARROW_KEYCODE:
        changes = [-ACTUATOR_INCREMENT, 0]  # Increment tilt upwards on y-axis
    elif keycode == DOWN_ARROW_KEYCODE:
        changes = [ACTUATOR_INCREMENT, 0]  # Increment tilt downwards on y-axis
    elif keycode == LEFT_ARROW_KEYCODE:
        changes = [0, -ACTUATOR_INCREMENT]  # Increment tilt left on x-axis
    elif keycode == RIGHT_ARROW_KEYCODE:
        changes = [0, ACTUATOR_INCREMENT]  # Increment tilt right on x-axis



# Load the model and data globally
model = mujoco.MjModel.from_xml_path('./resources/mujoco_xml/labyrinth_wo_meshes_actuators.xml')
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        if not paused:
            if change_values:

                data.ctrl[ACTUATOR_X_INDEX] += changes[ACTUATOR_X_INDEX]  # Increment or decrement x-axis
                data.ctrl[ACTUATOR_Y_INDEX] += changes[ACTUATOR_Y_INDEX]  # Increment or decrement y-axis
                change_values = False
            mujoco.mj_step(model, data)
        viewer.sync()