from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
import scipy as sp
import numpy as np

parallel_scan = jax.lax.associative_scan


# Parallel scan operations
@jax.vmap
def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def matrix_init(key, shape, dtype=jnp.float32, normalization=1):
    return jax.random.normal(key=key, shape=shape, dtype=dtype) / normalization

def identity_init(key, shape, dtype=jnp.float32):
    return jnp.eye(shape[0], dtype=dtype)


def fixed_glorot(key, shape, dtype=jnp.float32, n=128):
    gamma = jnp.log(n / (2*jnp.pi * (jnp.log(n)**2)))
    fix_term = (1 + jnp.sqrt(gamma / (4*n)) + (jnp.euler_gamma + jnp.pi / jnp.sqrt(6)) / np.sqrt(4*gamma*n))**(-2)

    return jax.random.normal(key=key, shape=shape, dtype=dtype) * jnp.sqrt((1/n) * fix_term)


def nu_init(key, shape, r_min, r_max, dtype=jnp.float32):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5*np.log(u*(r_max**2-r_min**2) + r_min**2))

def fixed_init_A(key, shape, r_min, r_max, dtype=jnp.float32):
    n = shape[0]
    gamma = np.log(n / (2*np.pi * (np.log(n)**2)))
    fix_term = (1 + np.sqrt(gamma / (4*n)) + (np.euler_gamma + np.pi / np.sqrt(6)) / np.sqrt(4*gamma*n))**(-2)
    A = np.random.normal(size=shape) * np.sqrt((1/n) * fix_term)
    T, Q = sp.linalg.schur(A, output='complex')
    A_diag = jnp.ones_like(np.diag(T))
    A_diag = jnp.diag(T) * A_diag
    return A_diag

def fixed_init_for_lru_theta(A_diag):
    print("Init theta with fixed A_diag")
    theta = jnp.log(jnp.angle(A_diag) + jnp.pi) # for positive angles
    return theta

def fixed_init_for_lru_nu(A_diag):
    print("Init nu with fixed A_diag")
    nu = jnp.log(-jnp.log(jnp.abs(A_diag)))
    return nu

def theta_init(key, shape, max_phase, dtype=jnp.float32):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class LRU(nn.Module):
    """
    LRU module in charge of the recurrent processing.
    Implementation following the one of Orvieto et al. 2023.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions
    r_min: float = 0.0  # smallest lambda norm
    r_max: float = 1.0  # largest lambda norm
    max_phase: float = 6.28  # max phase lambda
    bidirectional: bool = False  # whether to use bidirectional processing
    A_init: str = "glorot"  # initialization method for Ax§

    def setup(self):
        if self.A_init == "fixed":
            print("Using fixed initialization for A_diag")
            self.A_diag = self.param(
                "A_diag",
                fixed_init_A,
                (self.d_hidden, self.d_hidden),
                self.r_min,
                self.r_max,
            )
            self.theta_log = fixed_init_for_lru_theta(self.A_diag)
            self.nu_log = fixed_init_for_lru_nu(self.A_diag)
        elif self.A_init == "halved_glorot":
            print("Using halved glorot initialization for A_diag then diagonalized")
            normalization = np.sqrt(2 * self.d_hidden)
            self.A_diag = matrix_init_diag(jax.legacy_prng_key, shape=(self.d_hidden, self.d_hidden),
                                           normalization=normalization)
            self.theta_log = fixed_init_for_lru_theta(self.A_diag)
            self.nu_log = fixed_init_for_lru_nu(self.A_diag)
        else:
            self.theta_log = self.param(
                "theta_log", partial(theta_init, max_phase=self.max_phase), (self.d_hidden,)
            )
            self.nu_log = self.param(
                "nu_log", partial(nu_init, r_min=self.r_min, r_max=self.r_max), (self.d_hidden,)
            )
        self.gamma_log = self.param("gamma_log", gamma_log_init, (self.nu_log, self.theta_log))

        # Glorot initialized Input/Output projection matrices
        self.B_re = self.param(
            "B_re",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.B_im = self.param(
            "B_im",
            partial(matrix_init, normalization=jnp.sqrt(2 * self.d_model)),
            (self.d_hidden, self.d_model),
        )
        self.C_re = self.param(
            "C_re",
            partial(matrix_init, normalization=jnp.sqrt(2*self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.C_im = self.param(
            "C_im",
            partial(matrix_init, normalization=jnp.sqrt(2*self.d_hidden)),
            (self.d_model, self.d_hidden),
        )
        self.D = self.param("D", matrix_init, (self.d_model,))


    def __call__(self, inputs):
        """Forward pass of a LRU: h_t+1 = lambda * h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
        diag_lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(jnp.exp(self.gamma_log), axis=-1)
        C = self.C_re + 1j * self.C_im

        Lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)
        Bu_elements = jax.vmap(lambda u: B_norm @ u)(inputs)
        # Compute hidden states
        _, hidden_states = parallel_scan(binary_operator_diag, (Lambda_elements, Bu_elements))
        # Use them to compute the output of the module
        outputs = jax.vmap(lambda h, x: (C @ h).real + self.D * x)(hidden_states, inputs)

        return outputs


class SequenceLayer(nn.Module):
    """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""

    lru: LRU  # lru module
    d_model: int  # model size
    dropout: float = 0.0  # dropout probability
    norm: str = "layer"  # which normalization to use
    training: bool = True  # in training mode (dropout in trainign mode only)

    def setup(self):
        """Initializes the ssm, layer norm and dropout"""
        self.seq = self.lru()
        self.out1 = nn.Dense(self.d_model)
        self.out2 = nn.Dense(self.d_model)
        if self.norm in ["layer"]:
            self.normalization = nn.LayerNorm()
        else:
            self.normalization = nn.BatchNorm(
                use_running_average=not self.training, axis_name="batch"
            )
        self.drop = nn.Dropout(self.dropout, broadcast_dims=[0], deterministic=not self.training)

    def __call__(self, inputs):
        x = self.normalization(inputs)  # pre normalization
        x = self.seq(x)  # call LRU
        x = self.drop(nn.gelu(x))
        x = nn.gelu(x)
        x = self.out1(x) * jax.nn.sigmoid(self.out2(x))  # GLU
        x = self.drop(x)
        return inputs + x  # skip connection


class StackedEncoderModel(nn.Module):
    """Encoder containing several SequenceLayer"""

    lru: LRU
    d_model: int
    n_layers: int
    dropout: float = 0.0
    training: bool = True
    norm: str = "batch"

    def setup(self):
        self.encoder = nn.Dense(self.d_model)
        self.layers = [
            SequenceLayer(
                lru=self.lru,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                norm=self.norm,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, inputs):
        x = self.encoder(inputs)  # embed input in latent space
        for layer in self.layers:
            x = layer(x)  # apply each layer
        return x


class ClassificationModel(nn.Module):
    """Stacked encoder with pooling and softmax"""

    lru: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    dropout: float = 0.0
    training: bool = True
    pooling: str = "mean"  # pooling mode
    norm: str = "batch"  # type of normaliztion
    multidim: int = 1  # number of outputs

    def setup(self):
        self.encoder = StackedEncoderModel(
            lru=self.lru,
            d_model=self.d_model,
            n_layers=self.n_layers,
            dropout=self.dropout,
            training=self.training,
            norm=self.norm,
        )
        self.decoder = nn.Dense(self.d_output * self.multidim)

    def __call__(self, x):
        x = self.encoder(x)
        if self.pooling in ["mean"]:
            x = jnp.mean(x, axis=0)  # mean pooling across time
        elif self.pooling in ["last"]:
            x = x[-1]  # just take last
        elif self.pooling in ["none"]:
            x = x  # do not pool at all
        x = self.decoder(x)
        if self.multidim > 1:
            x = x.reshape(-1, self.d_output, self.multidim)
        return nn.log_softmax(x, axis=-1)



class RNN(nn.Module):
    """
    RNN module in charge of the recurrent processing.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions
    r_min: float = 0.0  # smallest lambda norm
    r_max: float = 1.0  # largest lambda norm
    max_phase: float = 6.28  # max phase lambda
    A_init: str = "glorot"
    bidirectional: bool = False

    def setup(self):
    # Glorot initialized Input/Output projection matrices
        if self.A_init == "halved_glorot":
            a_matrix_init = partial(
                matrix_init, normalization=jnp.sqrt(2*self.d_hidden)
            )
        elif self.A_init == "glorot":
            a_matrix_init = partial(
                matrix_init, normalization=jnp.sqrt(self.d_hidden)
            )
        elif self.A_init == "fixed":
            print("Fixed initialization")
            a_matrix_init = partial(
                fixed_glorot,
                n=self.d_hidden,
            )
        elif self.A_init == "identity":
            a_matrix_init = partial(identity_init)
        else:
            raise ValueError(f"Unknown initialization {self.A_init}")

        self.A = self.param(
            "A",
            a_matrix_init,
            (self.d_hidden, self.d_hidden),
        )
        self.B = self.param(
            "B",
            partial(matrix_init, normalization=jnp.sqrt(self.d_model)),
            (self.d_hidden, self.d_model),
        )
        if self.bidirectional:
            self.C1 = self.param(
                "C1",
                partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
                (self.d_model, self.d_hidden),
            )
            self.C2 = self.param(
                "C2",
                partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
                (self.d_model, self.d_hidden),
            )
            self.C = jnp.concatenate((self.C1, self.C2), axis=-1)
        else:
            self.C = self.param(
                "C",
                partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
                (self.d_model, self.d_hidden),
            )
        self.D = self.param("D", matrix_init, (self.d_model,))

    def __call__(self, inputs):
        """Forward pass of a RNN: h_t+1 = A h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
        Bu_elements = jax.vmap(lambda u: self.B @ u)(inputs)
        h0 = jnp.zeros((self.d_hidden,))

        # Define the scan function
        def scan_fn(h, x):
            new_h = self.A @ h + x
            return new_h, new_h  # Return updated state and store it

        # Apply scan over the sequence
        _, hidden_states = jax.lax.scan(scan_fn, h0, Bu_elements)
        if self.bidirectional:
            _, hidden_states_rev = jax.lax.scan(scan_fn, h0, Bu_elements, reverse=True)
            hidden_states = jnp.concatenate((hidden_states, hidden_states_rev), axis=-1)
        # Use them to compute the output of the module
        outputs = jax.vmap(lambda h, x: (self.C @ h) + self.D * x)(hidden_states, inputs)

        return outputs



def matrix_init_diag(key, shape, dtype=jnp.float32, normalization=1):
    if type(shape) is int:
        shape = (shape, shape)
    A = np.random.normal(size=shape) / normalization
    T, Q = sp.linalg.schur(A, output='complex')
    A_diag = jnp.ones_like(np.diag(T))
    A_diag = jnp.diag(T) * A_diag
    return A_diag

def matrix_uniform_complex_diag(key, shape, dtype=jnp.float32, normalization=1):
    r = jax.random.uniform(key=key, shape=(shape[0],))
    theta = jax.random.uniform(key=key, shape=(shape[0],)) * 2 * jnp.pi
    A = jnp.array(r * jnp.exp(1j * theta), dtype=dtype)
    return A

def fixed_glorot_diag(key, shape, dtype=jnp.float32, n=128):
    gamma = np.log(n / (2*np.pi * (np.log(n)**2)))
    fix_term = (1 + np.sqrt(gamma / (4*n)) + (np.euler_gamma + np.pi / np.sqrt(6)) / np.sqrt(4*gamma*n))**(-2)
    A = np.random.normal(size=shape) * np.sqrt((1/n) * fix_term)
    T, Q = sp.linalg.schur(A, output='complex')
    A_diag = jnp.ones_like(np.diag(T))
    A_diag = jnp.diag(T) * A_diag
    return A_diag

def identity_diag(key, shape, dtype=jnp.float32):
    return jnp.ones(shape[0])


class DiagRNN(nn.Module):
    """
    DiagRNN module in charge of the recurrent processing.
    """

    d_hidden: int  # hidden state dimension
    d_model: int  # input and output dimensions
    r_min: float = 0.0  # smallest lambda norm
    r_max: float = 1.0  # largest lambda norm
    max_phase: float = 6.28  # max phase lambda
    A_init: str = "glorot"
    bidirectional: bool = False

    def setup(self):
        # Glorot initialized Input/Output projection matrices
        if self.A_init == "halved_glorot":
            a_matrix_init = partial(
                matrix_init_diag, normalization=np.sqrt(2*self.d_hidden), dtype=jnp.complex64
            )
        elif self.A_init == "glorot":
            a_matrix_init = partial(
                matrix_init_diag, normalization=np.sqrt(self.d_hidden), dtype=jnp.complex64
            )
        elif self.A_init == "fixed":
            print("Fixed initialization")
            a_matrix_init = partial(
                fixed_glorot_diag,
                n=self.d_hidden,
                dtype=jnp.complex64
            )
        elif self.A_init == "identity":
            a_matrix_init = partial(identity_diag)
        elif self.A_init == "uniform":
            a_matrix_init = partial(
                matrix_uniform_complex_diag,
                dtype=jnp.complex64
            )
        else:
            raise ValueError(f"Unknown initialization {self.A_init}")

        self.A = self.param(
            "A",
            a_matrix_init,
            (self.d_hidden, self.d_hidden),
        )
        # Glorot initialized Input/Output projection matrices
        self.B = self.param(
            "B",
            partial(matrix_init, normalization=jnp.sqrt(self.d_model), dtype=jnp.complex64),
            (self.d_hidden, self.d_model),
        )
        if self.bidirectional:
            self.C1 = self.param(
                "C1",
                partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
                (self.d_model, self.d_hidden),
            )
            self.C2 = self.param(
                "C2",
                partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
                (self.d_model, self.d_hidden),
            )
            self.C = jnp.concatenate((self.C1, self.C2), axis=-1)
        else:
            self.C = self.param(
                "C",
                partial(matrix_init, normalization=jnp.sqrt(self.d_hidden)),
                (self.d_model, self.d_hidden),
            )
        self.D = self.param("D", matrix_init, (self.d_model,))

    def __call__(self, inputs):
        """Forward pass of a RNN: h_t+1 = A h_t + B x_t+1, y_t = Re[C h_t + D x_t]"""
        B = self.B
        Bu_elements = jax.vmap(lambda u: B @ u)(inputs)
        h0 = jnp.zeros((self.d_hidden,))

        # Define the scan function
        def scan_fn(h, x):
            new_h = self.A * h + x
            return new_h  # Return updated state and store it

        As = jnp.repeat(self.A[None, ...], inputs.shape[0], axis=0)

        # Apply scan over the sequence
        _, hidden_states = parallel_scan(binary_operator_diag, (As, Bu_elements))
        if self.bidirectional:
            _, hidden_states_rev = parallel_scan(binary_operator_diag, (As, Bu_elements), reverse=True)
            hidden_states = jnp.concatenate((hidden_states, hidden_states_rev), axis=-1)
        # Use them to compute the output of the module
        C = self.C
        outputs = jax.vmap(lambda h, x: (C @ h).real + self.D * x)(hidden_states, inputs)

        return outputs
















# Here we call vmap to parallelize across a batch of input sequences
BatchClassificationModel = nn.vmap(
    ClassificationModel,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "batch_stats": None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True},
    axis_name="batch",
)
