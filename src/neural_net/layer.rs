use neural_net::vec2d::Vec2d;
use rand::prelude::*;

#[derive(Clone)]
pub struct Layer {
    pub nodes_in: usize,
    pub nodes_out: usize,
    weights: Vec2d<f64>,
    biases: Vec<f64>,
}

impl Layer {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let mut layer = Layer {
            nodes_in: inputs,
            nodes_out: outputs,
            weights: Vec2d::new(vec![0.; inputs * outputs], inputs, outputs),
            biases: vec![0.; inputs],
        };

        layer.randomize_weights();

        layer
    }

    pub fn set_weight(&mut self, input: usize, output: usize, weight: f64) {
        self.weights.set(input, output, weight);
    }

    pub fn set_bias(&mut self, input: usize, bias: f64) {
        self.biases[input] = bias;
    }

    pub fn calculate(&self, inputs: &Vec<f64>, f_activation: &dyn Fn(f64) -> f64) -> Vec<f64> {
        let mut weighted_inputs = Vec::with_capacity(self.nodes_out);

        for node_out in 0..self.nodes_out {
            let mut weighted_input = self.biases[node_out];
            for node_in in 0..self.nodes_in {
                weighted_input += inputs[node_in] * self.weights.index(node_in, node_out);
            }
            weighted_inputs[node_out] = f_activation(weighted_input);
        }

        weighted_inputs
    }

    pub fn randomize_weights(&mut self) {
        let mut rng = rand::thread_rng();
        for node_in in 0..self.nodes_in {
            self.biases[node_in] = rng.gen_range(-1.0..1.0);
            for node_out in 0..self.nodes_out {
                self.weights
                    .set(node_in, node_out, rng.gen_range(-1.0..1.0));
            }
        }
    }
}
