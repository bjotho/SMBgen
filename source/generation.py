import random

class GAN():
    def __init__(self):
        pass

    def generate(self, file_name):
        print("hello from generation")
        # Her endrer vi tekstfila

        choices = [" ", " ", " ", "G", "B", "X", "S"]
        output = []

        for i in range(4):
            map_row = "\nBB00"
            for _ in range(9):
                map_row += random.choice(choices)

            output.append(map_row)

            with open("source/data/maps/" + file_name, "a") as file:
                file.write(map_row)

        return output