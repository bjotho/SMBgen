import numpy as np
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.utils.vis_utils import plot_model
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Embedding, RepeatVector, TimeDistributed
# from keras.callbacks import ModelCheckpoint

from . import constants as c


class GAN():
    def __init__(self):
        # self.memory = []
        # for _ in range(c.COL_MEMORY):
        #     self.memory.append([])
        #
        # self.step = 1

        self.tiles = ["A", "0", "1", "2", "G", "B", "X", "S", "Q"]
        # self.tokenizer = Tokenizer()
        # self.tokenizer.fit_on_texts(self.tiles)
        # self.generator = self.create_generator()

    def generate(self, file_path):
        """Create 2D list """
        output = []

        for i in range(c.GEN_LENGTH - 1):
            map_row = ""
            for _ in range(c.COL_HEIGHT - len(map_row)):
                map_row += np.random.choice(self.tiles)

            output.append(map_row)

            if c.WRITE:
                with open(file_path, "a") as file:
                    file.write(map_row + "\n")

        return output

    def create_generator(self):
        return None

    def tokenize_input(self, input):
        for column in input:
            token_list = self.tokenizer.texts_to_sequences([column])[0]
            print(token_list)

    def update_memory(self, update):
        # if column is full, shift all columns left and insert new empty column
        if len(self.memory[-1]) >= c.COL_HEIGHT:
            self.memory = self.memory[1:]
            self.memory.append([])
            self.step *= -1

        # Insert new tile in most recent column
        for i in range(update):
            self.memory[-1].append(update)