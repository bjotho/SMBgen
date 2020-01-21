import random

class GAN():
    def __init__(self):
        pass

    def generate(self, file_name):
        print("hello from generation")
        # Her endrer vi tekstfila

        choices = ["0","0","0","0","0", "X", "B"]

        for i in range(4):
            map_row = "\nBB00"
            for _ in range(9):
                map_row += random.choice(choices)

            with open("source/data/maps/" + file_name, "a") as file:
                file.write(map_row)