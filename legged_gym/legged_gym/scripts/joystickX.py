import pygame
from time import sleep
import queue

class JoystickController:
    DEADZONE = 0.1  # Deadzone threshold

    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        # Ensure at least one joystick is connected
        if pygame.joystick.get_count() < 1:
            raise Exception("No joystick detected. Please connect a joystick.")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.last_mode = 0
        # self.cmd=None
        self.cmd_queue = queue.Queue(maxsize=10)

    def apply_deadzone(self, value):
        """Apply the deadzone to a joystick axis value."""
        if abs(value) < self.DEADZONE:
            return 0.0
        return value

    def get_joystick_data(self):
        pygame.event.pump()
        axes = [self.apply_deadzone(self.joystick.get_axis(i)) for i in range(self.joystick.get_numaxes())]
        buttons = [self.joystick.get_button(i) for i in range(self.joystick.get_numbuttons())]
        return axes, buttons

    def update_cmd(self):
        # Compute cmd based on axes
        axes, buttons = self.get_joystick_data()
        # print(f"axes: {axes}, buttons: {buttons}")
        # Determine the mode based on button presses
        mode_mapping = buttons[0:8]
        current_mode = mode_mapping.index(1) if 1 in mode_mapping else self.last_mode

        if current_mode != self.last_mode:
            self.last_mode = current_mode
            print(f"Mode: {self.last_mode}")
        self.cmd = {'vx': axes[1], 'vy': axes[0], 'dyaw': axes[3], 'mode': self.last_mode}
        print(self.cmd)
        try:
            self.cmd_queue.put_nowait(self.cmd)  # Non-blocking put
        except queue.Full:
            # print("Queue is full. Dropping oldest command.")
            self.cmd_queue.get_nowait()

    def run(self):
        while True:
            self.update_cmd()
            sleep(0.02)
                
if __name__ == "__main__":
    joystick_controller = JoystickController()
    joystick_controller.run()
