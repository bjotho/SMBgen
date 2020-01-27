import random

from . import constants as c

class GAN():
    def __init__(self):
        pass

    def generate(self, file_path):
        """Create 2D list """
        choices = ["0", "0", "1", "1", "2", "2", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "G", "B", "X", "S", "Q"]
        output = []

        for i in range(c.GEN_LENGTH - 1):
            map_row = ""
            for _ in range(c.COL_HEIGHT):
                map_row += random.choice(choices)

            output.append(map_row)

            if c.WRITE:
                with open(file_path, "a") as file:
                    file.write(map_row + "\n")

        return output