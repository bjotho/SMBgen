import numpy as np
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from tensorflow.python.keras.utils.vis_utils import plot_model
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, LSTM, Embedding, RepeatVector, TimeDistributed
# from tensorflow.python.keras.callbacks import ModelCheckpoint

from source import constants as c
from source.states import level_state


class Generator:
    def __init__(self):
        self.memory = []
        for _ in range(c.COL_MEMORY):
            self.memory.append([])

        self.step = 1

        self.tiles = [c.GOOMBA_ID, c.FLY_KOOPA, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.COIN_ID, c.SOLID_ID, c.BRICK_ID, c.BOX_ID]
        # self.tiles = [c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.SOLID_ID, c.BRICK_ID, c.BOX_ID]
        # self.tiles = c.GENERATOR_TILES

        # self.generator = self.create_generator()

    def generate(self, file_path):
        """Create 2D list """
        output = []

        for _ in range(c.GEN_LENGTH - 1):
            map_col = str(c.GROUND_ID * 2)
            map_col_list = []
            for _ in range(c.COL_HEIGHT - len(map_col)):
                map_col_list.append(np.random.choice(self.tiles))
                map_col += map_col_list[-1]
                self.update_memory(map_col_list[-1])

            output.append(map_col)

            if c.WRITE:
                with open(file_path, 'a') as file:
                    file.write(map_col + "\n")

        return output

    def create_generator(self):
        pass

    def update_memory(self, update):
        # if column is full, shift all columns left and insert new empty column
        if len(self.memory[-1]) >= c.COL_HEIGHT:
            self.memory = self.memory[1:]
            self.memory.append([])

        # Insert new tile in last column
        self.memory[-1].append(update)
