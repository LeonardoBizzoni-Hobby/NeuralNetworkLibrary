use nalgebra::DMatrix;
use rand_distr::{Distribution, Normal};

use crate::nn_error::{NeuralNetworkError, WeightError};

#[derive(Debug)]
#[allow(dead_code)]
pub struct NeuralNetwork {
    weights: Vec<DMatrix<f64>>,

    learning_rate: f64,

    number_of_input_nodes: usize,
    number_of_output_nodes: usize,

    hlayer_node_count: Vec<usize>,
}

#[allow(dead_code)]
impl NeuralNetwork {
    pub fn new(
        number_of_input_nodes: usize,
        number_of_output_nodes: usize,
        learning_rate: f64,
        hlayer_node_count: Vec<usize>,
    ) -> Result<Self, NeuralNetworkError> {
        if number_of_input_nodes == 0 {
            return Err(NeuralNetworkError::InvalidNumberOfInputNodes);
        } else if number_of_output_nodes == 0 {
            return Err(NeuralNetworkError::InvalidNumberOfOutputNodes);
        }

        // Weights from input to first hidden (or output) layer
        let mut weights: Vec<DMatrix<f64>> = vec![DMatrix::from_vec(
            number_of_input_nodes,
            1,
            NeuralNetwork::generate_random_weights(number_of_input_nodes)?,
        )];

        // Weights from hidden to next hidden (or output)
        for node_count in hlayer_node_count.iter() {
            weights.push(DMatrix::from_vec(
                *node_count,
                1,
                NeuralNetwork::generate_random_weights(*node_count)?,
            ));
        }

        Ok(Self {
            weights,
            number_of_input_nodes,
            number_of_output_nodes,
            learning_rate,
            hlayer_node_count,
        })
    }

    // pub fn train(&mut self, inputs: &[f64], target: f64) {
    //     todo!()
    // }

    // pub fn compute(&self, inputs: &[f64]) -> Result<f64, NNError> {
    //     todo!()
    // }

    // fn activation(&self, sum: f64) -> f64 {
    //     todo!()
    // }

    fn generate_random_weights(n: usize) -> Result<Vec<f64>, NeuralNetworkError> {
        let random = match Normal::new(0.0, (n as f64).powf(-0.5)) {
            Ok(value) => value,
            Err(_) => {
                return Err(NeuralNetworkError::WeightError(
                    WeightError::InvalidVariance(n as f64),
                ))
            }
        };
        let mut weights: Vec<f64> = vec![0.0; n];

        for weight in weights.iter_mut() {
            *weight = random.sample(&mut rand::thread_rng());
        }

        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nn_with_invalid_number_of_inputs() {
        let nn = NeuralNetwork::new(0, 10, 42.0, vec![]);
        assert!(nn.is_err())
    }

    #[test]
    fn nn_without_outputs() {
        let nn = NeuralNetwork::new(1, 0, 42.0, vec![]);
        assert!(nn.is_err())
    }

    #[test]
    fn nn_invalid_hidden_layer_node_count() {
        let nn = NeuralNetwork::new(1, 1, 42.0, vec![0]);
        assert!(nn.is_err())
    }

    #[test]
    fn valid_nn_definition() {
        // NN with 1 input, no hidden and 1 output
        let nn = NeuralNetwork::new(1, 1, 42.0, vec![]);
        assert!(nn.is_ok())
    }
    #[test]
    fn valid_nn_with_hidden_layers_definition() {
        // NN with 1 input, 4 hidden layers each with 3 nodes, and 1 output
        let nn = NeuralNetwork::new(1, 1, 42.0, vec![3, 3, 3, 3]);

        assert!(nn.is_ok());
        assert_eq!(nn.unwrap().weights.len(), 5);
    }
}
