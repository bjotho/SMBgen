import random

from . import constants as c

class GAN():
    def __init__(self, do_write):
        self.write = do_write
        pass

    def generate(self, file_name):
        """Create 2D list """
        choices = [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", "G", "B", "X", "S", "Q"]
        output = []

        for i in range(c.GEN_LENGTH - 1):
            map_row = "\nGG  "
            for _ in range(c.COL_HEIGHT - 4):
                map_row += random.choice(choices)

            output.append(map_row)

            if self.write:
                with open("source/data/maps/" + file_name, "a") as file:
                    file.write(map_row)

        return output