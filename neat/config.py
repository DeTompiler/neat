from tensorflow.keras.activations import sigmoid, relu


class Config:
    def __init__(self, **kwargs):
        self.excess_distance_coefficient = kwargs.get('excess_distance_coefficient', 1.0)
        self.disjoint_distance_coefficient = kwargs.get('disjoint_distance_coefficient', 1.0)
        self.weights_distance_coefficient = kwargs.get('weights_distance_coefficient', 0.4)

        self.output_activation = kwargs.get('output_activation', sigmoid)
        self.hidden_activation = kwargs.get('hidden_activation', relu)

        self.genome_distance_threshold = kwargs.get('genome_distance_threshold', 3.0)
        self.survivors = kwargs.get('survivors', 0.2)
        self.min_species_population = kwargs.get('min_species_population', 2)
        self.max_stagnation = kwargs.get('max_stagnation', 15)
        self.reset_on_extinction = kwargs.get('reset_on_extinction', True)

        self.toggle_conn_prob = kwargs.get('toggle_conn_prob', 0.01)
        self.shift_weight_prob = kwargs.get('shift_weight_prob', 0.03)
        self.random_weight_prob = kwargs.get('random_weight_prob', 0.01)
        self.add_conn_prob = kwargs.get('add_conn_prob', 0.05)
        self.add_node_prob = kwargs.get('add_node_prob', 0.03)

        self.weight_rand_factor = kwargs.get('weight_rand_factor', 1.)
        self.weight_shift_factor = kwargs.get('weight_shift_factor', 0.2)
        self.similar_fitness_range = kwargs.get('similar_fitness_range', 0.04)
        self.add_conn_tries = kwargs.get('add_conn_tries', 100)
        self.input_layer_nb = kwargs.get('input_layer_nb', 0)
        self.output_layer_nb = kwargs.get('output_layer_nb', 256)