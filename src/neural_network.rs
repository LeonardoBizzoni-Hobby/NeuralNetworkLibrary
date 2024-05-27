use nalgebra::DMatrix;
use rand_distr::{Distribution, Normal};

use crate::nn_error::{NeuralNetworkError, WeightError};

#[derive(Debug)]
#[allow(dead_code)]
pub struct NeuralNetwork {
    weights: Vec<DMatrix<f64>>,

    learning_rate: f64,
    node_count_per_layer: Vec<usize>,
}

#[allow(dead_code)]
impl NeuralNetwork {
    pub fn new(
        learning_rate: f64,
        node_count_per_layer: Vec<usize>,
    ) -> Result<Self, NeuralNetworkError> {
        if node_count_per_layer.len() <= 0 {
            return Err(NeuralNetworkError::InvalidNumberOfInputNodes);
        } else if node_count_per_layer.len() <= 1 {
            return Err(NeuralNetworkError::InvalidNumberOfOutputNodes);
        }

        // Weights from input to first hidden (or output) layer
        let mut weights: Vec<DMatrix<f64>> = vec![];

        for (i, node_count) in node_count_per_layer[..node_count_per_layer.len() - 1]
            .iter()
            .enumerate()
        {
            if *node_count <= 0 || node_count_per_layer[i + 1] <= 0 {
                return Err(NeuralNetworkError::InvalidNumberOfNodes);
            }

            let random = match Normal::new(0.0, (*node_count as f64).powf(-0.5)) {
                Ok(value) => value,
                Err(_) => {
                    return Err(NeuralNetworkError::WeightError(
                        WeightError::InvalidVariance(*node_count as f64),
                    ))
                }
            };

            weights.push(DMatrix::from_fn(
                node_count_per_layer[i + 1], // nrows = node_count in the next layer
                *node_count,                 // ncols = node_count in the current layer
                |_, _| -> f64 { random.sample(&mut rand::thread_rng()) },
            ));
        }

        Ok(Self {
            weights,
            learning_rate,
            node_count_per_layer,
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
        let nn = NeuralNetwork::new(42.0, vec![0, 10]);
        assert!(nn.is_err())
    }

    #[test]
    fn nn_without_outputs() {
        let nn = NeuralNetwork::new(42.0, vec![1, 0]);
        assert!(nn.is_err())
    }

    #[test]
    fn nn_invalid_hidden_layer_node_count() {
        let nn = NeuralNetwork::new(42.0, vec![1, 0, 1]);
        assert!(nn.is_err())
    }

    #[test]
    fn valid_nn_definition() {
        // NN with 1 input, no hidden and 1 output
        let nn = NeuralNetwork::new(42.0, vec![1, 1]);
        assert!(nn.is_ok())
    }
    #[test]
    fn valid_nn_with_hidden_layers_definition() {
        // NN with 1 input, 4 hidden layers each with 3 nodes, and 1 output
        let nn = NeuralNetwork::new(42.0, vec![1, 3, 3, 3, 3, 1]);

        assert!(nn.is_ok());
        assert_eq!(nn.unwrap().weights.len(), 5);
    }
}
