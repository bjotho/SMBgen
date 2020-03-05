from tensorflow.python.keras.layers import Activation, Dense, Conv2D
import tensorflow as tf
import numpy as np
import math

class GaussianNoise(Activation):            #originally nn.module, is Activation the alternative?
    def _init_(self, in_features, out_features, sigma0=0.5): #def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        standarddev = np.std(in_features, out_features, sigma0)
        self.noise_weight = tf.keras.layers.GaussianNoise(standarddev)
        self.noise_bias =
        self.noise_std = sigma0/math.sqrt(self.in_features)
        '''self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.noisy_weight = nn.Parameter(
            torch.Tensor(out_features, in_features))
        self.noisy_bias = nn.Parameter(torch.Tensor(out_features))
        self.noise_std = sigma0 / math.sqrt(self.in_features)'''

        self.reset_parameters()
        self.register_noise()

    def register_noise(self):
        in_noise = tf.float32(self.in_features)
        out_noise = tf.float32(self.out_features)
        noise = tf.float32(self.out_features, self.in_features)
        self.register_buffer('in_noise, in_noise')
        self.register_buffer('out_noise, out_noise')
        self.register_buffer('noise, noise')        #what is the alternative in Tensorflow?

        ''' in_noise = torch.FloatTensor(self.in_features)
        out_noise = torch.FloatTensor(self.out_features)
        noise = torch.FloatTensor(self.out_features, self.in_features)
        self.register_buffer('in_noise', in_noise)
        self.register_buffer('out_noise', out_noise)
        self.register_buffer('noise', noise) '''

    def sample_noise(self):
        self.in_noise.normal_(0, self.noise_std)
        self.out_noise.normal_(0, self.noise_std)
        self.noise = torch.mm(
            self.out_noise.view(-1, 1), self.in_noise.view(1, -1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.noisy_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.noisy_bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        Note: noise will be updated if x is not volatile
        """
        normal_y = nn.functional.linear(x, self.weight, self.bias)
        if self.training:
            # update the noise once per update
            self.sample_noise()

        noisy_weight = self.noisy_weight * self.noise
        noisy_bias = self.noisy_bias * self.out_noise
        noisy_y = nn.functional.linear(x, noisy_weight, noisy_bias)
        return noisy_y + normal_y

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) + ')'



class Densing():





class curiosity():


