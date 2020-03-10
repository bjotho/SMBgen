import numpy as np
from collections import deque

from tensorflow.python.keras.layers import Input, LSTM, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.np_utils import to_categorical
# from tensorflow.python.keras.layers import Dense, LSTM, Embedding, RepeatVector, TimeDistributed

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
        self.train, self.inference_encoder, self.inference_decoder = self.create_generator(c.N_FEATURES,
                                                                                           c.N_FEATURES, 100)

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

    def generate(self):
        output = []

        for _ in range(c.GEN_LENGTH):
            self.step *= -1
            map_col = ""
            if c.INSERT_GROUND:
                map_col = str(2 * c.SOLID_ID)
            map_col_list = []
            for _ in range(self.tiles_per_col):
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

    def update_replay_memory(self, generation):
        state_size = c.GEN_LENGTH * self.tiles_per_col
        start = (c.PLATFORM_LENGTH * self.tiles_per_col) + (generation[c.GEN_LINE] * self.tiles_per_col) - 1
        end = start + state_size
        state = self.memory[start:end]
        start += state_size
        end += state_size
        action = self.memory[start:end]
        transition = (state, action, generation[c.REWARD], action, generation[c.DONE])

        self.replay_memory.append(transition)

    def update_memory(self, new_tile):
        # Insert new tile
        self.memory.append(new_tile)

    def get_memory_portion(self):
        if c.MEMORY_LENGTH >= len(self.memory):
            # Prepend padding to memory list
            padding_size = c.MEMORY_LENGTH - (len(self.memory) + 1)
            padding = [c.AIR_ID for _ in range(padding_size)]
            return padding + self.memory

        # Extract the latest c.MEMORY_LENGTH tiles from the memory
        return self.memory[:c.MEMORY_LENGTH]