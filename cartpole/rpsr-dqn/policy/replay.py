import numpy as np

class ReplayBuffer(object):
    """Implements the replay buffer for training a DQN.

    The structure of the buffer is like a revolving memory system.
    When a new sample is to be stored, it either gets appended to the end of
    the buffer, or the oldest sample gets replaced.
    """
    def __init__(self, capacity, state_shape, s0=None, a=None, r=None, 
                 s1=None, terminals=None, rng=None):

        state_dim = (capacity, ) + state_shape

        self.capacity = capacity
        self.rng = rng or np.random.RandomState(1234)
        self.s0 = s0 or np.zeros(state_dim, dtype=np.float32)
        self.a = a or np.zeros((capacity, 1), dtype=np.int32)
        self.r = r or np.zeros((capacity, 1), dtype=np.float32)
        self.s1 = s1 or np.zeros(state_dim, dtype=np.float32)
        self.terminals = terminals or np.zeros((capacity, 1), dtype=np.float32)

        self.step = 0
        self.cycle = 0 # Whether we've hit the cap on memory size


    def update(self, s0, a, r, s1, terminal):
        """Replace current memory at <step> with new experience."""

        self.s0[self.step] = s0
        self.a[self.step] = a
        self.r[self.step] = r
        self.s1[self.step] = s1
        self.terminals[self.step] = terminal

        self.step += 1
        if self.step >= self.capacity:
            self.step = 0
            self.cycle += 1


    def sample(self, batch_size):
        """Draws N random samples without replacement as training items."""

        # We'll draw only from samples we've collected, not an empty buffer
        max_idx = self.capacity if self.cycle > 0 else self.step
        max_idx = np.maximum(max_idx, batch_size)

        batch = np.random.choice(np.arange(max_idx), batch_size, False)
        return (self.s0[batch], self.a[batch], self.r[batch], self.s1[batch],
                self.terminals[batch])
