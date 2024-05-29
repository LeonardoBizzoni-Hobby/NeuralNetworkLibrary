#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuralNetworkError {
    InvalidNumberOfOutputNodes,
    InvalidNumberOfInputNodes,
    InvalidNumberOfNodes,

    WeightError(WeightError),
    InsufficientInputData { expected: usize, found: usize },
    InsufficientTargetsData { expected: usize, found: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightError {
    InvalidVariance(f64),
}
