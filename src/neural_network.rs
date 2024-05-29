use std::{
    cell::RefCell,
    iter::{Peekable, Rev},
    rc::Rc,
    slice::{Iter, IterMut},
};

use nalgebra::DMatrix;
use rand_distr::{Distribution, Normal};

use crate::nn_error::{NeuralNetworkError, WeightError};

#[derive(Debug)]
#[allow(dead_code)]
pub struct NeuralNetwork {
    weights: Rc<RefCell<Vec<DMatrix<f64>>>>,

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

        let weights: Rc<RefCell<Vec<DMatrix<f64>>>> = Rc::new(RefCell::new(vec![]));
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

            weights.borrow_mut().push(DMatrix::from_fn(
                node_count_per_layer[i + 1], // nrows = node_count in the next layer
                *node_count,                 // ncols = node_count in the current layer
                |_, _| -> f64 { random.sample(&mut rand::thread_rng()) },
            ));
        }

        #[cfg(debug_assertions)]
        for (i, weight) in weights.borrow().iter().enumerate() {
            println!("Weight {i}-{}: {weight}", i + 1);
        }

        Ok(Self {
            weights,
            learning_rate,
            node_count_per_layer,
        })
    }

    pub fn train(&mut self, input: &[f64], target: &[f64]) -> Result<(), NeuralNetworkError> {
        if target.len() != *self.node_count_per_layer.last().unwrap() {
            return Err(NeuralNetworkError::InsufficientTargetsData {
                expected: self.node_count_per_layer[0],
                found: input.len(),
            });
        }

        // Creates 2 column matrices
        let input = DMatrix::from_column_slice(self.node_count_per_layer[0], 1, input);
        let target = DMatrix::from_column_slice(self.node_count_per_layer[0], 1, target);

        let outputs = self.compute_query(self.weights.borrow().iter(), vec![input]);
        let output_error = target - outputs.last().unwrap();

        {
            let weights = Rc::clone(&self.weights);
            self.adjust_weights(
                output_error,
                weights.borrow_mut().iter_mut().rev(),
                outputs.iter().rev().peekable(),
            );
        }

        Ok(())
    }

    fn adjust_weights(
        &mut self,
        next_error: DMatrix<f64>,
        mut weights: Rev<IterMut<DMatrix<f64>>>,
        mut outputs: Peekable<Rev<Iter<DMatrix<f64>>>>,
    ) {
        if let (Some(weight), Some(next_output)) = (weights.next(), outputs.next()) {
            *weight += self.learning_rate
                * next_error
                    .zip_map(&next_output, |a, b| a * b)
                    .zip_map(&next_output.map(|x| 1.0 - x), |a, b| a * b)
                * (outputs.peek().unwrap().transpose());

            self.adjust_weights(weight.transpose() * next_error, weights, outputs);
        }
    }

    pub fn query(&mut self, inputs: &[f64]) -> Result<Vec<f64>, NeuralNetworkError> {
        if inputs.len() != self.node_count_per_layer[0] {
            return Err(NeuralNetworkError::InsufficientInputData {
                expected: self.node_count_per_layer[0],
                found: inputs.len(),
            });
        }

        Ok(self
            .compute_query(
                self.weights.borrow().iter(),
                vec![DMatrix::from_column_slice(
                    self.node_count_per_layer[0],
                    1,
                    inputs,
                )],
            )
            .last()
            .unwrap()
            .column(0)
            .as_slice()
            .into())
    }

    fn compute_query(
        &self,
        mut weight_iter: Iter<DMatrix<f64>>,
        mut per_layer_output: Vec<DMatrix<f64>>,
    ) -> Vec<DMatrix<f64>> {
        match weight_iter.next() {
            Some(weight) => {
                let next_input: DMatrix<f64> =
                    NeuralNetwork::activation_function(weight * per_layer_output.last().unwrap());
                per_layer_output.push(next_input);

                self.compute_query(weight_iter, per_layer_output)
            }
            None => per_layer_output,
        }
    }

    fn activation_function(mut x: DMatrix<f64>) -> DMatrix<f64> {
        // Sigmoid function applied to each element in the matrix (column vector)
        for xi in x.iter_mut() {
            *xi = 1.0 / (1.0 + std::f64::consts::E.powf(-*xi));
        }

        x
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
        assert_eq!(nn.unwrap().weights.borrow().len(), 5);
    }
}
