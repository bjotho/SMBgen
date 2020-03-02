import numpy as np
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from tensorflow.python.keras.utils.vis_utils import plot_model
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, LSTM, Embedding, RepeatVector, TimeDistributed
# from tensorflow.python.keras.callbacks import ModelCheckpoint

from source import constants as c
from source.states import level_state


class Generator:
    def __init__(self, gen_file_path):
        self.map_gen_file = gen_file_path
        self.step = 1
        self.memory = [c.AIR_ID for _ in range(c.GEN_MEMORY)] # range(c.GEN_MEMORY - num of tiles in gen_file)

        # self.tiles = [c.GOOMBA_ID, c.FLY_KOOPA, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.AIR_ID, c.COIN_ID, c.SOLID_ID, c.BRICK_ID, c.BOX_ID]
        # self.tiles = [c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.AIR_ID", c.SOLID_ID, c.BRICK_ID, c.BOX_ID]
        self.tiles = c.GENERATOR_TILES

        self.generator = self.create_generator()

    def generate(self):
        """Create 2D list """
        output = []

        for _ in range(c.GEN_LENGTH - 1):
            map_col = ""
            if c.INSERT_GROUND:
                map_col = str(c.GROUND_ID * 2)
            map_col_list = []
            for _ in range(c.COL_HEIGHT - len(map_col)):
                map_col_list.append(np.random.choice(self.tiles))
                map_col += map_col_list[-1]
                self.update_memory(map_col_list[-1])

            output.append(map_col)

            if c.WRITE:
                with open(self.map_gen_file, 'a') as file:
                    file.write(map_col + "\n")

        return output

    def create_generator(self):
        pass

    def update_memory(self, update):
        # if column is full, shift all columns left and insert new empty column
        if len(self.memory) >= c.GEN_MEMORY:
            self.memory = self.memory[1:]

        # Insert new tile in last column
        self.memory.append(update)
