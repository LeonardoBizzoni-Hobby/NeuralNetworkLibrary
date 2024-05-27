#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeuralNetworkError {
    InvalidNumberOfOutputNodes,
    InvalidNumberOfInputNodes,
    InvalidNumberOfNodes,

    WeightError(WeightError),
    InsufficientInputData,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightError {
    InvalidVariance(f64),
}
