from tensorflow.keras.activations import sigmoid, relu


class Config:
    def __init__(self, **kwargs):
        self.c1 = kwargs.get('c1', 1.0)
        self.c2 = kwargs.get('c2', 1.0)
        self.c3 = kwargs.get('c3', 0.4)

        self.output_activation = kwargs.get('output_activation', sigmoid)
        self.hidden_activation = kwargs.get('hidden_activation', relu)

        self.genome_distance_threshold = kwargs.get('genome_distance_threshold', 3.0)
        self.kill_worst = kwargs.get('kill_worst', 0.2)

        self.toggle_probability = kwargs.get('toggle_probability', 0.01)
        self.shift_probability = kwargs.get('shift_probability', 0.03)
        self.random_probability = kwargs.get('random_probability', 0.01)
        self.connection_probability = kwargs.get('connection_probability', 0.05)
        self.node_probability = kwargs.get('node_probability', 0.03)

        self.weight_randomization_factor = kwargs.get('weight_randomization_factor', 1.)
        self.weight_shift_factor = kwargs.get('weight_shift_factor', 0.2)
        self.similar_fitness_range = kwargs.get('similar_fitness_range', 0.04)
        self.max_add_random_connection_tries = kwargs.get('max_add_random_connection_tries', 100)
        self.input_layer_nb = kwargs.get('input_layer_nb', 0)
        self.output_layer_nb = kwargs.get('output_layer_nb', 256)