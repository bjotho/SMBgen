import numpy as np
import random
import os
from collections import deque
try:
    import cPickle as pickle
except ImportError:
    import pickle

from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical

from source import constants as c
from source.states import level_state

dir_path = os.path.dirname(os.path.realpath(__file__))


class Generator:
    def __init__(self, gen_file_path, epsilon=1.0):
        self.map_gen_file = gen_file_path
        self.epsilon = epsilon
        self.tiles_per_col = c.COL_HEIGHT - 2 if c.INSERT_GROUND else c.COL_HEIGHT
        self.gen_size = c.GEN_LENGTH * self.tiles_per_col
        self.checkpoint_gen = os.path.join(dir_path, "checkpoints", "generator")
        self._TILE_MAP, self._CHAR_MAP = self.tokenize_tiles(c.GENERATOR_TILES)
        # print("generator._TILE_MAP:", self._TILE_MAP)
        # print("generator._CHAR_MAP:", self._CHAR_MAP)

        self.replay_memory = deque(maxlen=c.REPLAY_MEMORY_SIZE)
        loaded_model_path = self.load_model_path() if c.LOAD_GEN_MODEL else None
        callback_dir = os.path.join(dir_path, "checkpoints", "generator_graphs", f"graph_{self.start_checkpoint}")
        os.makedirs(callback_dir, exist_ok=True)

        self.generator = self.create_generator(model=loaded_model_path)
        self.tensorboard = TensorBoard(log_dir=callback_dir, histogram_freq=0, write_graph=True, write_images=False)
        print("tensorboard callback dir:", callback_dir)

        if loaded_model_path:
            self.load_replay_memory()
        print(self.generator.summary())

    def generator_startup(self):
        self.step = -1 if c.SNAKING else 1
        self.memory = []
        self.populate_memory()

    def tokenize_tiles(self, tiles: list):
        # Tokenize tiles and populate tile_map dict with (tile_id: token) pairs
        # char_map dict is translation in opposite direction (token: tile_id)
        tile_map = {}
        char_map = {}
        for n, tile in enumerate(tiles):
            tile_map[tile] = n
            char_map[n] = tile

        return tile_map, char_map

    def load_model_path(self):
        os.makedirs(self.checkpoint_gen, exist_ok=True)
        self.start_checkpoint = level_state.find_latest_checkpoint(self.checkpoint_gen)
        latest_checkpoint = None
        if self.start_checkpoint > -1:
            latest_checkpoint = os.path.join(self.checkpoint_gen, f"model_{str(self.start_checkpoint)}",
                                             f"model_{str(self.start_checkpoint)}")
        else:
            self.start_checkpoint = 0

        return latest_checkpoint

    def save_model(self, num):
        # Serialize model to JSON
        model_json = self.generator.to_json()
        model_dir_name = os.path.join(self.checkpoint_gen, f"model_{str(num)}")
        os.makedirs(model_dir_name, exist_ok=True)
        model_name = os.path.join(model_dir_name, f"model_{str(num)}")
        with open(f"{model_name}.json", 'w') as json_file:
            json_file.write(model_json)
        # Serialize weights to HDF5
        self.generator.save_weights(f"{model_name}.h5")
        print("Saved generator model:", model_name)

    def load_replay_memory(self):
        try:
            pickle_file = os.path.join(self.checkpoint_gen, f"model_{str(self.start_checkpoint)}", f"{c.REP_MEM}.pickle")
            with open(pickle_file, 'rb') as file:
                self.replay_memory = pickle.load(file)

            print("Loaded replay_memory:", pickle_file)
        except:
            pass

    def save_replay_memory(self, num):
        pickle_file = os.path.join(self.checkpoint_gen, f"model_{str(num)}", f"{c.REP_MEM}.pickle")
        with open(pickle_file, 'wb') as file:
            pickle.dump(self.replay_memory, file, protocol=pickle.HIGHEST_PROTOCOL)

        print("Saved replay_memory:", pickle_file)

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
    def create_generator(self, model=None):
        # https://github.com/golsun/deep-RL-trading/blob/master/src/agents.py#L161 # <-- TODO - Look here :D
        if model is not None:
            # Load JSON and create model
            json_file = open(f"{model}.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # Load weights into new model
            loaded_model.load_weights(f"{model}.h5")
            loaded_model.compile(loss='mse', optimizer=Adam(lr=c.LEARNING_RATE), metrics=['accuracy'])
            print("Loaded generator model:", model)
            return loaded_model

        model = Sequential()
        model.add(Input(shape=(c.MEMORY_LENGTH, len(c.GENERATOR_TILES))))
        model.add(LSTM(c.LSTM_CELLS))
        model.add(Dense(len(c.GENERATOR_TILES), activation='linear'))  # model.add(LSTM(len(c.GENERATOR_TILES), return_sequences=True))
        model.compile(loss='mse', optimizer=Adam(lr=c.LEARNING_RATE), metrics=['accuracy'])
        return model

    # Train the generator model if replay memory is large enough.
    # Return 1 if the generator model is trained, 0 otherwise
    def train(self):
        # Only start training if we have enough transitions in replay memory
        print("len(self.replay_memory)", len(self.replay_memory))
        if len(self.replay_memory) < c.MIN_REPLAY_MEMORY_SIZE:
            return 0

        print("Training on", c.MINIBATCH_SIZE, "transitions")

        # Get a minibatch of random samples from replay memory
        minibatch = random.sample(self.replay_memory, c.MINIBATCH_SIZE)

        # Get current states from minibatch and create list of predicted sequences
        current_states = np.array([transition[0] for transition in minibatch])
        current_predicted_sequences = self.predict_new_states(current_states, return_qs=True)
        current_predicted_sequences = np.array(current_predicted_sequences)
        # print("current_predicted_sequences:", current_predicted_sequences)

        # Get future states from minibatch, and create new list of sequece predictions
        new_current_states = [transition[3] for transition in minibatch]
        future_predicted_sequences = self.predict_new_states(new_current_states, return_qs=True)
        future_predicted_sequences = np.array(future_predicted_sequences)
        # print("future_predicted_sequences:", future_predicted_sequences)

        X = []
        y = []

        # Enumerate transitions
        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
            # Randomly set tile choice to greedy or not greedy variant
            greedy = np.random.random() < self.epsilon

            # If not a terminal state, get new qs from future states, otherwise set it to reward
            if not done:
                max_future_Qs = [np.max(qs) for qs in future_predicted_sequences[index]]
                new_Qs = [reward + c.DISCOUNT * fqs for fqs in max_future_Qs]
            else:
                new_Qs = [reward for _ in range(len(future_predicted_sequences[index]))]

            new_Qs = np.array(new_Qs)

            # Update Q values for given state
            current_qs = current_predicted_sequences[index]
            enc_action = self.one_hot_encode(action)
            for n, action_n in enumerate(enc_action):
                current_qs[n][self.choose_new_tile(action_n, greedy)] = new_Qs[n]

            # Insert sliding window states into X
            generator_input = []
            for i in range(self.gen_size):
                state_slice = max(0, i - c.MEMORY_LENGTH)
                tmp_state = np.concatenate((current_state[i:], action[state_slice:i]))
                generator_input.append(self.one_hot_encode(tmp_state))

            # Append to training data
            X.append(generator_input)
            y.append(current_qs)

        # Fit on all transitions in minibatch
        X = np.array(X)
        y = np.array(y)
        for i in range(len(X)):
            self.generator.fit(X[i], y[i], batch_size=self.gen_size, verbose=0, shuffle=False, callbacks=[self.tensorboard])

        return 1

    def predict_new_states(self, states, return_qs=False):
        # Loop through states and make predictions for new_states for each state
        predicted_states = []
        output = [] if return_qs else predicted_states
        for state in states:
            # Randomly set tile choice to greedy or not greedy variant
            greedy = np.random.random() < self.epsilon

            predicted_states.append([])
            if return_qs:
                output.append([])
            for i in range(self.gen_size):
                state_slice = max(0, len(predicted_states[-1]) - c.MEMORY_LENGTH)
                tmp_state = np.concatenate((state[i:], predicted_states[-1][state_slice:]))
                one_hot = self.one_hot_encode(tmp_state)
                generator_input = np.reshape(one_hot, np.concatenate((np.array([1]), one_hot.shape)))
                tile_qs = self.generator.predict(generator_input)[0]
                if return_qs:
                    output[-1].append(tile_qs)
                new_tile = self.choose_new_tile(tile_qs, greedy)
                predicted_states[-1].append(new_tile)

        return output

    def choose_new_tile(self, qs, greedy):
        if not c.CHUNK_BASED_GREEDY:
            # Randomly set tile choice to greedy or not greedy variant
            greedy = np.random.random() < self.epsilon

        if greedy:
            # Greedy-variant
            new_tile = np.argmax(qs)
        else:
            # Random-variant
            new_tile = np.random.randint(0, len(c.GENERATOR_TILES))

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
        q_values = []

        # Randomly set tile choice to greedy or not greedy variant
        greedy = np.random.random() < self.epsilon

        # c.GEN_LENGTH: 5
        # self.tiles_per_col: 11 or 13 (c.INSERT_GROUND = False)

        for _ in range(c.GEN_LENGTH):
            q_values.append([])
            map_col = ""
            if c.INSERT_GROUND:
                map_col = str(2 * c.SOLID_ID)
            map_col_list = []
            for _ in range(self.tiles_per_col):
                if not c.RANDOM_GEN:
                    start = len(self.memory) - c.MEMORY_LENGTH
                    #print("start", start)
                    state = self.get_padded_memory(start, slice=True)
                    #print("state", state)
                    one_hot = self.one_hot_encode(state)
                    #print("one_ho", one_hot)
                    generator_input = np.reshape(one_hot, np.concatenate((np.array([1]), one_hot.shape)))
                    #print("np.concatenate((np.array([1]), one_hot.shape))", np.concatenate((np.array([1]), one_hot.shape)))
                    #print("generator_input", generator_input)
                    prediction = self.generator.predict(generator_input)[0]
                    # self.generator.predict(generator_input) er et array som inneholder arrayet
                    # med prediction, derfor [0]
                    #print("prediction", prediction)
                    new_tile = self.choose_new_tile(prediction, greedy)
                    #print("new_tile", new_tile)
                    map_col_list.append(self._CHAR_MAP[new_tile])
                    q_values[-1].append(str("%.2f" % prediction[new_tile]))
                    self.update_memory(new_tile)
                else:
                    map_col_list.append(np.random.choice(c.GENERATOR_TILES))
                    self.update_memory(self._TILE_MAP[map_col_list[-1]])

            #print("q_values before step", q_values)
            map_col_list = map_col_list[::self.step]
            q_values[-1] = q_values[-1][::self.step]

            #print("q_values after step", q_values)

            for tile in map_col_list:
                map_col += tile
            output.append(map_col)
            # level_state.print_2d(self.memory, chop=self.tiles_per_col)

            if c.WRITE:
                with open(self.map_gen_file, 'a') as file:
                    file.write(map_col + "\n")

        return output, q_values

    def one_hot_encode(self, seq, cardinality=len(c.GENERATOR_TILES)):
        return to_categorical([seq], num_classes=cardinality)[0]

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
            padding_size = np.abs(pos)
            padding = [self._TILE_MAP[c.AIR_ID] for _ in range(padding_size)]
            return padding + self.memory

        if slice:
            return self.memory[pos:]
        else:
            return self.memory
