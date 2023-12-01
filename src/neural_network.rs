pub type InOut = f32;

pub type Activation = fn(InOut) -> InOut;

/* LAYER */

pub fn layer_forward<const I: usize, const O: usize>(
    inputs: &[InOut; I],
    weights: &[[InOut; I]; O],
    biases: &[InOut; O],
    activations: &[Activation; O]
) -> [InOut; O] {
    let mut outputs = [InOut::default(); O];

    for i in 0..O {
        outputs[i] = node_forward(&inputs, &weights[i], biases[i], activations[i])
    }

    outputs
}

pub fn layer_backward<const I: usize, const O: usize>(
    inputs: &[InOut; I],
    weights: &[[InOut; I]; O],
    activations_d: &[Activation; O],
    outputs: &[InOut; O],
    output_errors: &[InOut; O],
) -> ([InOut; I], [[InOut; I]; O], [InOut; O]) {
    let mut input_errors = [InOut::default(); I];
    let mut weight_gradients = [[InOut::default(); I]; O];
    let mut bias_gradients = [InOut::default(); O];

    for i in 0..O {
        let (node_input_errors, node_weight_gradients, node_bias_gradient) =
            node_backward(inputs, &weights[i], activations_d[i], outputs[i], output_errors[i]);
        
        for j in 0..I {
            input_errors[j] += node_input_errors[j];
            weight_gradients[i][j] = node_weight_gradients[j];
        }
            
        bias_gradients[i] = node_bias_gradient;
    }

    (input_errors, weight_gradients, bias_gradients)
}

/* NODE */

pub fn node_forward<const I: usize>(
    inputs: &[InOut; I],
    weights: &[InOut; I],
    bias: InOut,
    activation: Activation
) -> InOut {
    let weighted_sum: InOut = inputs.iter().zip(weights).map(|(&x, &y)| x * y).sum();
    let total_input = weighted_sum + bias;
    activation(total_input)
}

pub fn node_backward<const I: usize>(
    inputs: &[InOut; I],
    weights: &[InOut; I],
    activation_d: Activation,
    output: InOut,
    output_error: InOut,
) -> ([InOut; I], [InOut; I], InOut) {
    let mut input_errors = [InOut::default(); I];
    let mut weight_gradients = [InOut::default(); I];

    let activation_gradient = activation_d(output);
    let bias_gradient = activation_gradient * output_error;

    for i in 0..I {
        input_errors[i] = weights[i] * activation_gradient * output_error;
        weight_gradients[i] = inputs[i] * activation_gradient * output_error;
    }

    (input_errors, weight_gradients, bias_gradient)
}

/* ACTIVATION */

pub fn relu(value: InOut) -> InOut {
    value.max(InOut::default())
}

pub fn relu_d(value: InOut) -> InOut {
    if value > InOut::default() { 1.0 as InOut } else { InOut::default() }
}

pub fn fast_tanh(value: InOut) -> InOut {
    value / (1.0 + value.abs())
}

pub fn fast_tanh_d(value: InOut) -> InOut {
    1.0 / (1.0 + value.abs()).powi(2)
}

/* SOFTMAX LAYER */

pub fn softmax_forward<const IO: usize>(inputs: &[InOut; IO]) -> [InOut; IO] {
    let exp_sum: InOut = inputs.iter().map(|&x| InOut::exp(x)).sum();
    let mut outputs = [InOut::default(); IO];

    for (index, &in_out) in inputs.iter().enumerate() {
        outputs[index] = InOut::exp(in_out) / exp_sum;
    }

    outputs
}

pub fn softmax_backward<const IO: usize>(
    outputs: &[InOut; IO],
    output_errors: InOut // Cross Entropy Loss
) -> [InOut; IO] {
    let mut input_errors = [InOut::default(); IO];

    for i in 0..IO {
        let sum_term: InOut = outputs.iter().sum();
        input_errors[i] = outputs[i] * (output_errors - sum_term);
    }

    input_errors
}

/* CROSS ENTROPY LOSS */

pub fn cross_entropy_loss<const O: usize>(predictions: &[InOut; O], targets: &[InOut; O]) -> InOut {
    -targets
        .iter()
        .zip(predictions)
        .map(|(target, prediction)| target * prediction.log2())
        .sum::<InOut>()
        / predictions.len() as InOut
}