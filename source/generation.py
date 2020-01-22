import random

class GAN():
    def __init__(self, do_write):
        self.write = do_write
        pass

    def generate(self, file_name):
        print("hello from generation")
        # Her endrer vi tekstfila

        choices = [" ", " ", " ", "G", "B", "X", "S"]
        output = []

        for i in range(4):
            map_row = "\nGG00"
            for _ in range(9):
                map_row += random.choice(choices)

            output.append(map_row)

            if self.write:
                with open("source/data/maps/" + file_name, "a") as file:
                    file.write(map_row)

        return output