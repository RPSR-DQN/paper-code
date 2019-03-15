import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn

from rpsp.policy.replay import ReplayBuffer


def build_network(state_shape, num_actions):
    """Builds a network with input size state_shape & num_actions output.
        
    As problem is discrete, we predict the Q-value for all possible actions.
    """

    input_shape = (None, ) + state_shape
    W_init = lasagne.init.GlorotUniform()
    nonlin = lasagne.nonlinearities.rectify

    network = nn.InputLayer(input_shape, input_var=None)
    network = nn.DenseLayer(network,50, W=W_init, nonlinearity=nonlin)
    network = nn.DenseLayer(network, num_actions, W=W_init, nonlinearity=None)
    return network


class Agent(object):
    """Implements an agent that follows deep Q-learning policy."""

    def __init__(self, state_shape, num_actions, epsilon=1.0, epsilon_min=0.1,
                 epsilon_iter=100000, discount=0.99, lrate=1e-4,
                 batch_size=100, q_update_iter=1000, capacity=50000):

        if not isinstance(state_shape, tuple):
            raise AssertionError('state_shape must be of type <tuple>.')
        elif len(state_shape) == 0:
            raise AssertionError('No state space dimensions provided.')
        elif num_actions == 0:
            raise ValueError('Number of actions must be > 0.')
        elif epsilon_min is not None:
            assert epsilon_min < epsilon, 'Epsilon(min) must be < epsilon(max).'
        elif capacity < batch_size:
            raise ValueError('Replay capacity must be > batch_size.')

        self.state_shape = state_shape
        self.num_actions = num_actions
        self.q_network = build_network(state_shape, num_actions)
        self.q_targets = build_network(state_shape, num_actions)
        self.epsilon = epsilon
        self.epsilon_max = epsilon # How greedy the policy is
        self.epsilon_min = epsilon_min
        self.epsilon_iter = float(epsilon_iter)
        self.discount = discount
        self.lr = lrate
        self.batch_size = batch_size # How many samples to draw from buffer
        self.q_update_iter = q_update_iter # Update the q_target every C iter
        self.step = 0
        self.replay_buffer = ReplayBuffer(capacity, state_shape)

        # Build training and sampling functions
        s0_sym = nn.get_all_layers(self.q_network)[0].input_var
        s1_sym = nn.get_all_layers(self.q_targets)[0].input_var
        a_sym = T.icol('actions') #(n, 1)
        r_sym = T.col('rewards')
        t_sym = T.col('terminal_state')
        sym_vars = [s0_sym, a_sym, r_sym, s1_sym, t_sym]

        # Training phase uses non-deterministic mapping
        loss = T.sum(self._build_loss(*sym_vars, deterministic=False))
        params = nn.get_all_params(self.q_network, trainable=True)
        updates = lasagne.updates.adam(loss, params, self.lr, beta1=0.9)

        self.train_fn = theano.function(sym_vars, loss, updates=updates)

        # Build function for sampling from DQN
        pred = nn.get_output(self.q_network, deterministic=True)
        self.pred_fn = theano.function([s0_sym], pred)

    def _build_loss(self, s0_sym, a_sym, r_sym, s1_sym, t_sym, deterministic=False):
        """Builds the loss for the DQN Agent.

        The loss is the squared error between the current Q-values, and
        Q-values predicted by the target network. The target Q-values are
        dependent on whether or not the agent has reached the terminal state;

        y_t = r_t if terminal, else r_t + max_{a'} gamma * Q^(s_{t+1}, a')

        Parameters
        ----------
        s0_sym: symbolic variable for current state
        a_sym: symbolic variable for current action
        r_sym: symbolic variable for current reward
        s1_sym: symbolic variable for next state
        t_sym: symbolic variable denoting whether next state is terminal
        """

        # Prepare target Q-values; build a mask using t_sym to denote whether
        # to use the 'terminal' reward (t_sym=1) or discounted reward (t_sym=0)
        q_targets = nn.get_output(self.q_targets, deterministic=deterministic)
        q_targets = self.discount * T.max(q_targets, axis=1, keepdims=True)
        q_targets = r_sym + (1. - t_sym) * q_targets

        # Q-Function for current state
        q_pred = nn.get_output(self.q_network, deterministic=deterministic)

        # The replay buffer holds (state, action, reward, next_state) tuples.
        # We compute the Q-value of the current state, and use a mask created
        # via the previous action action chosen to zero-out all other values.
        action_mask = T.eye(self.num_actions)[a_sym.flatten()]

        # Convert a sparse Q-matrix to a column vector
        q_pred = T.sum(q_pred * action_mask, axis=1, keepdims=True)

        return T.sqr(q_targets - q_pred)

    def choose_action(self, state):
        """Returns an action for the agent to perform in the environment.

        Return a random action with p < self.epsilon, or sample the best
        action from the Q-function.
        """
        state=np.array([state])
        state = state[np.newaxis].astype(theano.config.floatX)
        # With probability e, select a random action a_t
        if np.random.uniform(0.0, 1.0, size=1) < self.epsilon:
            return np.random.randint(0, self.num_actions, size=1)[0]
        return np.argmax(self.pred_fn(state))

    def update_buffer(self, s0, a, r, s1, terminal):
        self.replay_buffer.update(s0, a, r, s1, terminal)

    def update_policy(self):
        """Updates Q-networks using replay memory data + performing SGD"""

        minibatch = self.replay_buffer.sample(self.batch_size)
        self.train_fn(*minibatch)

        # Every few steps in an episode we update target network weights
        if self.step == self.q_update_iter:
            weights = nn.get_all_param_values(self.q_network)
            nn.set_all_param_values(self.q_targets, weights)
        self.step = self.step + 1 if self.step != self.q_update_iter else 0

        # Linearily anneal epsilon
        if self.epsilon_min is not None:
            diff = self.epsilon_max - self.epsilon_min
            curr_eps = self.epsilon - diff / self.epsilon_iter
            self.epsilon = np.maximum(self.epsilon_min, curr_eps)
