import numpy as np
from collections import deque

from tensorflow.python.keras.layers import Input, LSTM, Dense
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
# from tensorflow.python.keras.layers import Dense, LSTM, Embedding, RepeatVector, TimeDistributed

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
        self._TILE_MAP = level_state.tokenize_tiles(c.GENERATOR_TILES)

        self.populate_memory()
        self.generator, self.inference_encoder, self.inference_decoder = self.create_generator(c.N_FEATURES,
                                                                                           c.N_FEATURES, 100)
        self.generator.compile(optimizer=Adam(lr=c.LEARNING_RATE),
                               loss='categorical_crossentropy', metrics=['accuracy'])
        self.replay_memory = deque(maxlen=c.REPLAY_MEMORY_SIZE)

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
    def create_generator(self, n_input, n_output, n_units):
        """
        Args:
            n_input: Cardinality of input sequence (num of words/characters)
            n_output: Cardinality of output sequence (num of words/characters)
            n_units: Number of cells in encoder/decoder models
        """
        # define training encoder
        encoder_inputs = Input(shape=(None, n_input))
        encoder = LSTM(n_units, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        # define training decoder
        decoder_inputs = Input(shape=(None, n_output))
        decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(n_output, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # define inference encoder
        encoder_model = Model(encoder_inputs, encoder_states)
        # define inference decoder
        decoder_state_input_h = Input(shape=(n_units,))
        decoder_state_input_c = Input(shape=(n_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        # return all models
        return model, encoder_model, decoder_model

    def train(self):
        # Only start training if we have enough transitions in replay memory
        # if len(self.replay_memory) < c.MIN_REPLAY_MEMORY_SIZE:
        #     return

        # Get a minibatch of random samples from replay memory
        minibatch = np.random.sample(self.replay_memory, c.MINIBATCH_SIZE)

        # Get current states from minibatch and create list of predicted sequences
        current_states = [self.one_hot_encode(transition[0], len(self._TILE_MAP)) for transition in minibatch]
        current_predicted_sequences = []
        for state in current_states:
            current_predicted_sequences.append(self.predict_sequence(self.inference_encoder, self.inference_decoder,
                                                                     state, self.gen_size, len(self._TILE_MAP)))

        # Get future states from minibatch, and create new list of sequece predictions
        new_current_states = [self.one_hot_encode(transition[3], len(self._TILE_MAP)) for transition in minibatch]
        future_predicted_sequences = []
        for state in new_current_states:
            future_predicted_sequences.append(self.predict_sequence(self.inference_encoder, self.inference_decoder,
                                                                    state, self.gen_size, len(self._TILE_MAP)))

        X = []
        y = []

        # Enumerate transitions
        for n, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
            # If not a terminal state, get new predicted sequence from future states, otherwise set it to 0
            break

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

    # generate target given source sequence
    def predict_sequence(self, infenc, infdec, source, n_steps, cardinality):
        """
        Args:
            infenc: Encoder model for prediction for new source sequence
            infdec: Decoder model for prediction for new source sequence
            source: Encoded source sequence
            n_steps: Time steps in target sequence
            cardinality: Cardinality of output sequence (num of words/characters)
        """
        # encode
        state = infenc.predict(source)
        # start of sequence input
        target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
        # collect predictions
        output = []
        for t in range(n_steps):
            # predict next char
            yhat, h, c = infdec.predict([target_seq] + state)
            # store prediction
            output.append(yhat[0, 0, :])
            # update state
            state = [h, c]
            # update target sequence
            target_seq = yhat
        return np.array(output)

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
        transition = (np.array(state), np.array(action), generation[c.REWARD], np.array(new_state), generation[c.DONE])
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
