import numpy as np
import random
from collections import deque

from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical

from source import constants as c
from source.states import level_state


class Generator:
    def __init__(self, gen_file_path):
        self.map_gen_file = gen_file_path
        self.step = -1 if c.SNAKING else 1
        self.memory = []
        self.tiles_per_col = c.COL_HEIGHT - 2 if c.INSERT_GROUND else c.COL_HEIGHT
        self.gen_size = c.GEN_LENGTH * self.tiles_per_col
        self.tiles = c.GENERATOR_TILES + [c.AIR_ID for _ in range(50)]
        self._TILE_MAP = self.tokenize_tiles(c.GENERATOR_TILES)
        print("self._TILE_MAP:", self._TILE_MAP)

        self.populate_memory()
        self.generator = self.create_generator()
        self.replay_memory = deque(maxlen=c.REPLAY_MEMORY_SIZE)

    def tokenize_tiles(self, tiles: list):
        # Tokenize tiles and populate tile_map dict with (tile_id: token) pairs
        tile_map = {}
        for n, tile in enumerate(tiles):
            tile_map[tile] = n

        return tile_map

    def populate_memory(self):
        if c.INSERT_GROUND:
            for char in str(c.AIR_ID * ((c.COL_HEIGHT - 2) * c.PLATFORM_LENGTH)):
                self.memory.append(self._TILE_MAP[char])
        else:
            with open(self.map_gen_file, 'r') as f:
                for line in f:
                    if c.SNAKING:
                        self.step *= -1
                    clean_line = line.rstrip()
                    for char in clean_line[::self.step]:
                        self.memory.append(self._TILE_MAP[char])

    # returns train, inference_encoder and inference_decoder models
    def create_generator(self):
        # https://github.com/golsun/deep-RL-trading/blob/master/src/agents.py#L161 # <-- TODO - Look here :D
        model = Sequential()
        model.add(Input(shape=(c.MEMORY_LENGTH, len(c.GENERATOR_TILES))))  # model.add(Embedding(len(c.GENERATOR_TILES), 5, input_length=1))
        model.add(LSTM(c.MEMORY_LENGTH))
        model.add(Dense(len(c.GENERATOR_TILES), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=c.LEARNING_RATE), metrics=['accuracy'])
        print(model.summary())
        return model

    def train(self):
        # Only start training if we have enough transitions in replay memory
        if len(self.replay_memory) < c.MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from replay memory
        minibatch = random.sample(self.replay_memory, c.MINIBATCH_SIZE)

        # Get current states from minibatch and create list of predicted sequences
        current_states = np.array([transition[0] for transition in minibatch])
        current_predicted_sequences = []
        for state in current_states:
            generator_input = self.one_hot_encode(state, len(c.GENERATOR_TILES))
            current_predicted_sequences.append(self.generator.predict(generator_input))
        current_predicted_sequences = np.array(current_predicted_sequences)

        greedy_mode = True
        preds = np.zeros_like(np.asarray(current_states))
        for i, seqs in enumerate(current_predicted_sequences):
            for j, seq in enumerate(seqs):

                if greedy_mode:
                    # Greedy-variant
                    tile = np.argmax(seq[0])
                else:
                    # Not-greedy-variant
                    tile = np.random.choice(np.arange(len(seq[0])), p=seq[0])
                preds[i] = tile

            # print("preds:", preds)

        # Get future states from minibatch, and create new list of sequece predictions
        new_current_states = [transition[3] for transition in minibatch]
        future_predicted_sequences = []
        for new_state in new_current_states:
            future_predicted_sequences.append(self.generator.predict(self.one_hot_encode(new_state,
                                                                                         len(c.GENERATOR_TILES))))

        # print("current_predicted_sequences:", len(current_predicted_sequences[0]), current_predicted_sequences[0])

        X = []
        y = []

        # Enumerate transitions
        # for n, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
        #     new_state_value = reward + c.DISCOUNT * state_value

    def generate(self):
        output = []

        for _ in range(c.GEN_LENGTH):
            if c.SNAKING:
                self.step *= -1
            map_col = ""
            if c.INSERT_GROUND:
                map_col = str(2 * c.SOLID_ID)
            map_col_list = []
            for _ in range(self.tiles_per_col - 1):
                map_col_list.append(np.random.choice(self.tiles))
                map_col += map_col_list[-1]
                self.update_memory(self._TILE_MAP[map_col_list[-1][::self.step]])

            output.append(map_col)
            # level_state.print_2d(self.memory, chop=self.tiles_per_col)

            if c.WRITE:
                with open(self.map_gen_file, 'a') as file:
                    file.write(map_col + "\n")

        return output

    def one_hot_encode(self, seq, cardinality):
        return to_categorical([seq], num_classes=cardinality)

    def one_hot_decode(self, encoded_seq):
        return [np.argmax(vector) for vector in encoded_seq]

    # Insert a transition as a tuple of
    # (state, action, reward, new_state, state_value, done)
    def update_replay_memory(self, generation):
        start = (generation[c.GEN_LINE] * self.tiles_per_col) - c.MEMORY_LENGTH
        memory = self.get_padded_memory(start)
        if start < 0:
            start = 0
        end = start + c.MEMORY_LENGTH
        state = memory[start:end]
        start += self.gen_size
        end += self.gen_size
        new_state = memory[start:end]
        start = end - self.gen_size
        action = memory[start:end]
        transition = (np.array(state),
                      np.array(action),
                      generation[c.REWARD],
                      np.array(new_state),
                      generation[c.DONE])
        self.replay_memory.append(transition)

    def update_memory(self, new_tile):
        # Insert new tile
        self.memory.append(new_tile)

    def get_padded_memory(self, pos):
        if pos < 0:
            # Prepend padding to memory list
            padding_size = -pos
            padding = [self._TILE_MAP[c.AIR_ID] for _ in range(padding_size)]
            return padding + self.memory

        return self.memory
