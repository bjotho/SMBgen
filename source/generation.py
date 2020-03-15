import numpy as np
import random
from collections import deque

import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow_probability.python.distributions import Categorical

from source import constants as c
# from source.states import level_state


class Generator:
    def __init__(self, gen_file_path, epsilon=1.0):
        self.map_gen_file = gen_file_path
        self.epsilon = epsilon
        self.step = -1 if c.SNAKING else 1
        self.memory = []
        self.tiles_per_col = c.COL_HEIGHT - 2 if c.INSERT_GROUND else c.COL_HEIGHT
        self.gen_size = c.GEN_LENGTH * self.tiles_per_col
        self._TILE_MAP, self._CHAR_MAP = self.tokenize_tiles(c.GENERATOR_TILES)
        print("self._TILE_MAP:", self._TILE_MAP)
        print("self._CHAR_MAP:", self._CHAR_MAP)

        self.populate_memory()
        self.generator = self.create_generator()
        self.replay_memory = deque(maxlen=c.REPLAY_MEMORY_SIZE)

    def tokenize_tiles(self, tiles: list):
        # Tokenize tiles and populate tile_map dict with (tile_id: token) pairs
        # char_map dict is translation in opposite direction (token: tile_id)
        tile_map = {}
        char_map = {}
        for n, tile in enumerate(tiles):
            tile_map[tile] = n
            char_map[n] = tile

        return tile_map, char_map

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
        model.add(LSTM(c.LSTM_CELLS))
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

        # Randomly set tile choice to greedy or not greedy variant
        greedy = np.random.random() < self.epsilon

        # Get current states from minibatch and create list of predicted sequences
        current_states = np.array([transition[0] for transition in minibatch])
        current_predicted_sequences = self.predict_new_states(current_states, greedy, return_qs=True)
        current_predicted_sequences = np.array(current_predicted_sequences)
        # print("current_predicted_sequences:", current_predicted_sequences)

        # Get future states from minibatch, and create new list of sequece predictions
        new_current_states = [transition[3] for transition in minibatch]
        future_predicted_sequences = self.predict_new_states(new_current_states, greedy, return_qs=True)
        future_predicted_sequences = np.array(future_predicted_sequences)
        # print("future_predicted_sequences:", future_predicted_sequences)

        X = []
        y = []

        # Enumerate transitions
        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
            # If not a terminal state, get new qs from future states, otherwise set it to reward
            if not done:
                max_future_Qs = [np.max(qs) for qs in future_predicted_sequences[index]]
                new_Qs = [reward + c.DISCOUNT * fqs for fqs in max_future_Qs]
            else:
                new_Qs = [reward for _ in range(len(future_predicted_sequences[index]))]

            new_Qs = np.array(new_Qs)

            # Update Q values for given state
            current_qs = current_predicted_sequences[index]
            enc_action = self.one_hot_encode(action, len(c.GENERATOR_TILES))
            for n, action_n in enumerate(enc_action):
                current_qs[n][self.choose_new_tile(action_n, greedy)] = new_Qs[n]

            # Insert sliding window states into X
            generator_input = []
            for i in range(self.gen_size):
                state_slice = max(0, i - c.MEMORY_LENGTH)
                tmp_state = np.concatenate((current_state[i:], action[state_slice:i]))
                generator_input.append(self.one_hot_encode(tmp_state, len(c.GENERATOR_TILES))[0])

            # Append to training data
            X.append(generator_input)
            y.append(current_qs)

        # Fit on all transitions in minibatch
        X = np.array(X)
        y = np.array(y)
        for i in range(len(X)):
            self.generator.fit(X[i], y[i], batch_size=self.gen_size, verbose=0, shuffle=False)

    def predict_new_states(self, states, greedy, return_qs=False):
        # Loop through states and make predictions for new_states for each state
        predicted_states = []
        output = [] if return_qs else predicted_states
        for state in states:
            predicted_states.append([])
            if return_qs:
                output.append([])
            for i in range(self.gen_size):
                state_slice = max(0, len(predicted_states[-1]) - c.MEMORY_LENGTH)
                tmp_state = np.concatenate((state[i:], predicted_states[-1][state_slice:]))
                generator_input = self.one_hot_encode(tmp_state, len(c.GENERATOR_TILES))
                tile_qs = self.generator.predict(generator_input)[0]
                if return_qs:
                    output[-1].append(tile_qs)
                new_tile = self.choose_new_tile(tile_qs, greedy)
                predicted_states[-1].append(new_tile)

        return output

    def choose_new_tile(self, qs, greedy):
        if greedy:
            # Greedy-variant
            new_tile = np.argmax(qs)
        else:
            # Semi-random-variant
            dist = Categorical(probs=qs)
            n = 1e4
            empirical_prob = tf.cast(
                tf.histogram_fixed_width(
                    dist.sample(int(n)),
                    [0, len(c.GENERATOR_TILES)],
                    nbins=len(c.GENERATOR_TILES)),
                dtype=tf.float32) / n
            empirical_prob /= np.sum(empirical_prob)
            new_tile = self.weighted_tile_choice(p=empirical_prob)

        return new_tile

    def weighted_tile_choice(self, p):
        choice = np.random.random()
        p_i = 0
        i = 0
        while p_i <= choice:
            try:
                p_i += p[i]
            except IndexError:
                break

            i += 1

        return i - 1

    def generate(self):
        output = []

        # Randomly set tile choice to greedy or not greedy variant
        greedy = np.random.random() < self.epsilon

        for _ in range(c.GEN_LENGTH):
            if c.SNAKING:
                self.step *= -1
            map_col = ""
            if c.INSERT_GROUND:
                map_col = str(2 * c.SOLID_ID)
            map_col_list = []
            for _ in range(self.tiles_per_col - 1):
                start = len(self.memory) - c.MEMORY_LENGTH
                state = self.get_padded_memory(start, slice=True)
                prediction = self.generator.predict(self.one_hot_encode(state, len(c.GENERATOR_TILES)))
                new_tile = self.choose_new_tile(prediction, greedy)
                map_col_list.append(self._CHAR_MAP[new_tile])
                self.update_memory(new_tile)

            map_col_list = map_col_list[::self.step]
            for tile in map_col_list:
                map_col += tile
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
    # (state, action, reward, new_state, done)
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

    def get_padded_memory(self, pos, slice=False):
        if pos < 0:
            # Prepend padding to memory list
            padding_size = -pos
            padding = [self._TILE_MAP[c.AIR_ID] for _ in range(padding_size)]
            return padding + self.memory

        if slice:
            return self.memory[pos:]
        else:
            return self.memory
