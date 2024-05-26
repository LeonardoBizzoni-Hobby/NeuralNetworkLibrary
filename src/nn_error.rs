#[derive(Debug)]
pub enum NeuralNetworkError {
    InvalidNumberOfOutputNodes,
    InvalidNumberOfInputNodes,

    WeightError(WeightError),
}

#[derive(Debug)]
pub enum WeightError {
    InvalidVariance(f64),
}
