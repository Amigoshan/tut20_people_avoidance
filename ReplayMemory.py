import numpy as np
from os.path import join
import pickle as pkl

class ReplayMemory(object):
    """
    ReplayMemory keeps track of the environment dynamic.
    We store all the transitions (s(t), action, s(t+1), reward, done).
    The replay memory allows us to efficiently sample minibatches from it, and generate the correct state representation
    (w.r.t the number of previous frames needed).
    """
    def __init__(self, size, sample_shape, extra_sample_shape=None, history_length=1):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        #ipdb.set_trace()
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.bool)

        self._savedir = './memory'

        self._extra_state_shape = extra_sample_shape
        if extra_sample_shape is not None:
            self._extra_states = np.zeros((size,) + extra_sample_shape, dtype=np.float32)
        else:
            self._extra_states = None

    def __len__(self):
        """ Returns the number of items currently present in the memory
        Returns: Int >= 0
        """
        return self._count

    def append(self, state, action, reward, done, extra_state=None):
        """ Appends the specified transition to the memory.

        Attributes:
            state (Tensor[sample_shape]): The state to append
            action (int): An integer representing the action done
            reward (float): An integer representing the reward received for doing this action
            done (bool): A boolean specifying if this state is a terminal (episode has finished)
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        if extra_state is not None and self._extra_states is not None:
            self._extra_states[self._pos] = extra_state

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ Generate size random integers mapping indices in the memory.
            The returned indices can be retrieved using #get_state().
            See the method #minibatch() if you want to retrieve samples directly.

        Attributes:
            size (int): The minibatch size

        Returns:
             Indexes of the sampled states ([int])
        """

        # Local variable access is faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def save_memory(self, prefix=''):
        #ipdb.set_trace()
        new_dict = {
            'pos': self._pos,
            'count': self._count,
            'max_size': self._max_size,
            'history_length': self._history_length,
            'state_shape': self._state_shape,
            'extra_state_shape': self._extra_state_shape,
        }
        np.save(join(self._savedir,prefix+'states.npy'), self._states)
        np.save(join(self._savedir,prefix+'actions.npy'), self._actions)
        np.save(join(self._savedir,prefix+'rewards.npy'), self._rewards)
        np.save(join(self._savedir,prefix+'terminals.npy'), self._terminals)
        pkl.dump(new_dict, open(join(self._savedir,prefix+'memory.pkl'),'wb'))

        if self._extra_states is not None:
            np.save(join(self._savedir,prefix+'extra_states.npy'), self._extra_states)


    def load_memory(self, prefix=''):
        new_dict = pkl.load(open(join(self._savedir, prefix+'memory.pkl'),'rb'))
        #ipdb.set_trace()
        self._pos = new_dict['pos']
        self._count = new_dict['count']
        self._max_size = new_dict['max_size']
        self._history_length = new_dict['history_length']

        self._state_shape = new_dict['state_shape']
        self._extra_state_shape = new_dict['extra_state_shape']

        self._states = np.load(join(self._savedir,prefix+'states.npy'))#new_dict['states']
        self._actions = np.load(join(self._savedir,prefix+'actions.npy'))#new_dict['actions']
        self._rewards = np.load(join(self._savedir,prefix+'rewards.npy'))#new_dict['rewards']
        self._terminals = np.load(join(self._savedir,prefix+'terminals.npy'))#new_dict['terminals']
        del new_dict
        
        if self._extra_state_shape is not None:
            self._extra_states = np.load(join(self._savedir,prefix+'extra_states.npy'))

        print('load memory from.. {} {}'.format(self._states.shape, self._extra_states.shape))

    def minibatch(self, size):
        """ Generate a minibatch with the number of samples specified by the size parameter.

        Attributes:
            size (int): Minibatch size

        Returns:
            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]
        """
        assert(size<=self._count)

        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        if self._extra_states is not None:
            extra_pre_states = np.array([self._extra_states[index%self._count] for index in indexes], dtype=np.float32)
            extra_post_states = np.array([self._extra_states[(index+1)%self._count] for index in indexes], dtype=np.float32)

            pre_states = (pre_states, extra_pre_states)
            post_states = (post_states, extra_post_states)

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        Return the specified state with the replay memory. A state consists of
        the last `history_length` perceptions.

        Attributes:
            index (int): State's index

        Returns:
            State at specified index (Tensor[history_length, input_shape...])
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        # history_length = self._history_length

        return self._states[index]
        # # If index > history_length, take from a slice
        # if index >= history_length:
        #     return self._states[(index - (history_length - 1)):index + 1, ...]
        # else:
        #     indexes = np.arange(index - history_length + 1, index + 1)
        #     return self._states.take(indexes, mode='wrap', axis=0)