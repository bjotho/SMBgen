import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint

from . import constants as c
from .states import level_state


class GAN():
    def __init__(self):
        self.memory = []
        for _ in range(c.COL_MEMORY):
            self.memory.append([])

        self.step = 1

        # self.tiles = ["0", "1", "2", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "g", "b", "x", "s", "q"]
        self.tiles = ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "a", "g", "b", "x", "s", "q"]
        # self.tiles = ["0", "1", "2", "a", "g", "b", "x", "s", "q"]

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.tiles)
        self.generator = self.create_generator()

    def generate(self, file_path):
        """Create 2D list """
        output = []

        for i in range(c.GEN_LENGTH - 1):
            map_col = ""
            map_col_list = []
            for _ in range(c.COL_HEIGHT - len(map_col)):
                map_col_list.append(np.random.choice(self.tiles))
                map_col += map_col_list[-1]
                self.update_memory(map_col_list[-1])

            output.append(map_col)
            self.tokenize_input()

            if c.WRITE:
                with open(file_path, "a") as file:
                    file.write(map_col + "\n")

        return output

    def create_generator(self):
        pass
        # model = Sequential()
        # model.add(Embedding())

    def tokenize_input(self):
        token_list_sequence = []
        for column in self.memory:
            for i in column[::self.step]:
                token_list_sequence.append(self.tokenizer.texts_to_sequences([i])[0][0])
            if c.SNAKING:
                self.step *= -1

        # for col in self.memory:
        #     print(col)
        # print("[ ", end="", flush=True)
        # for n, i in enumerate(token_list_sequence):
        #     if not (n+1) % c.COL_HEIGHT and n != 0 and n != len(token_list_sequence)-1:
        #         print(str(i) + ",\n  ", end="", flush=True)
        #     elif n == len(token_list_sequence)-1:
        #         print(str(i) + " ]", flush=True)
        #     else:
        #         print(str(i) + ", ", end="", flush=True)


    def update_memory(self, update):
        # if column is full, shift all columns left and insert new empty column
        if len(self.memory[-1]) >= c.COL_HEIGHT:
            self.memory = self.memory[1:]
            self.memory.append([])

        # Insert new tile in last column
        self.memory[-1].append(update)