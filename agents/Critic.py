from keras import backend as K
from keras import layers, models, optimizers


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=24, activation='relu')(states)  # ACS changed from 32
        net_states = layers.Dense(units=48, activation='relu')(net_states)  # ACS changed from 64
        # ACS added another layer
        net_states = layers.Dense(units=24, activation='relu')(net_states) # ACS changed from 32

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=24, activation='relu')(actions)      # ACS changed from 32
        net_actions = layers.Dense(units=48, activation='relu')(net_actions)  # ACS changed from 64
        # ACS added another layer
        net_actions = layers.Dense(units=24, activation='relu')(net_actions)  # ACS changed from 32

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        #self.model.summary()
        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)