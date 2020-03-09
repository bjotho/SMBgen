import numpy as np
from collections import deque
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, LSTM, Dense
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from tensorflow.python.keras.utils.vis_utils import plot_model
# from tensorflow.python.keras.layers import Dense, LSTM, Embedding, RepeatVector, TimeDistributed
# from tensorflow.python.keras.callbacks import ModelCheckpoint

from source import constants as c
from source.states import level_state


class Generator:
    def __init__(self, gen_file_path):
        self.map_gen_file = gen_file_path
        self.step = -1
        self.memory = []
        self.tiles_per_col = c.COL_HEIGHT - 2 if c.INSERT_GROUND else c.COL_HEIGHT
        self.tiles = c.GENERATOR_TILES + [c.AIR_ID for _ in range(50)]
        self._TILE_MAP = level_state.tokenize_tiles(self.tiles)

        self.populate_memory()
        self.generator = self.create_generator()

        self.replay_memory = deque(maxlen=c.REPLAY_MEMORY_SIZE)

        level_state.print_2d(self.memory, chop=self.tiles_per_col)

    def populate_memory(self):
        if c.INSERT_GROUND:
            for char in str(c.AIR_ID * ((c.COL_HEIGHT - 2) * c.PLATFORM_LENGTH)):
                self.memory.append(self._TILE_MAP[char])
        else:
            with open(self.map_gen_file, 'r') as f:
                for line in f:
                    self.step *= -1
                    clean_line = line.rstrip()
                    for char in clean_line[::self.step]:
                        self.memory.append(self._TILE_MAP[char])

        for _ in range(c.MEMORY_LENGTH - len(self.memory)):
            self.memory.append(self._TILE_MAP[c.AIR_ID])

    def generate(self):
        output = []

        for _ in range(c.GEN_LENGTH):
            map_col = ""
            if c.INSERT_GROUND:
                map_col = str(2 * c.SOLID_ID)
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
        vocab_size = len(self.tiles) + 1
        model = Sequential()
        model.add(Embedding(vocab_size, 5, input_length=c.MEMORY_LENGTH))
        model.add(LSTM(100, return_sequences=True))
        model.add(LSTM(100))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(vocab_size, activation='softmax'))
        print(model.summary())
        return model

    def update_replay_memory(self, generation):
        pass

    def update_memory(self, new_tile):
        # Remove oldest tile from memory if memory length limit is exceeded
        if len(self.memory) >= c.MEMORY_LENGTH:
            self.memory = self.memory[1:]

        # Insert new tile
        self.memory.append(new_tile)
