pub mod layer;

use layer::Layer;

#[derive(Clone)]
pub struct Net<'a> {
    layers: Vec<Layer>,
    f_activation: &'a dyn Fn(f64) -> f64,
}

impl<'a> Net<'a> {
    /// Creates a new [`Net`] with randomized weights and biases.
    pub fn new(layer_sizes: &[usize], f_activation: &'a dyn Fn(f64) -> f64) -> Self {
        let num_layers = layer_sizes.len() - 1;
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]))
        }

        Net {
            layers,
            f_activation,
        }
    }

    pub fn calculate(&self, inputs: &[f64]) -> Vec<f64> {
        let mut result = Vec::from(inputs);

        for layer in self.layers.iter() {
            result = layer.calculate(&result, self.f_activation)
        }

        result
    }

    /// returns the index of the highest output.
    ///
    /// # Panics
    ///
    /// Panics if any input is NaN or if the number of inputs is less than the net was initialized with.
    pub fn classify(&self, inputs: &[f64]) -> usize {
        self.calculate(inputs)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    }

    pub fn set_weight(&mut self, layer: usize, input: usize, output: usize, weight: f64) {
        self.layers[layer].set_weight(input, output, weight);
    }

    pub fn set_bias(&mut self, layer: usize, input: usize, bias: f64) {
        self.layers[layer].set_bias(input, bias);
    }

    pub fn randomize_weights(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.randomize_weights();
        }
    }
}

pub fn step(input: f64) -> f64 {
    if input > 0. {
        1.
    } else {
        0.
    }
}

pub fn sigmoid(input: f64) -> f64 {
    1. / (1. + (-input).exp())
}

pub fn sigmoid_step(input: f64) -> f64 {
    sigmoid(input).round()
}

pub fn hyperbolic_tangent(input: f64) -> f64 {
    let e2w = (2. * input).exp();
    (e2w - 1.) / (e2w + 1.)
}
