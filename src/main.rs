mod neural_network;
mod mnist;

use macroquad::{window::next_frame, miniquad::KeyCode};
use neural_network::*;
use mnist::*;
use rand::Rng;

const OUTPUTS: usize = 10;
const NODES: usize = 5;

const DATA_SIZE: usize = 1000;
const BATCH_SIZE: usize = 10;

const LEARNING_RATE: InOut = 0.05 as InOut;

const GROUND_TRUTHS: [[InOut; OUTPUTS]; OUTPUTS] = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
];

fn random_weights<const I: usize, const O: usize>() -> [[InOut; I]; O] {
    let mut layer: [[f32; I]; O] = [[InOut::default(); I]; O];

    for output in layer.iter_mut() {
        for input in output.iter_mut() {
            *input = rand::thread_rng().gen_range(-1.0..1.0) as InOut;
        }
    }

    layer
}

fn random_biases<const O: usize>() -> [InOut; O] {
    let mut layer = [InOut::default(); O];

    for output in layer.iter_mut() {
        *output = rand::random::<InOut>();
    }

    layer
}

#[macroquad::main("ML")]
async fn main() {
    let (train_images, train_labels) = mnist::read::<DATA_SIZE>().expect("Could not read train data");

    println!("Train data loaded...");

    let mut hidden_weights = random_weights::<IMAGE_SIZE, NODES>();
    let mut hidden_biases = random_biases::<NODES>();
    let mut output_weights = random_weights::<NODES, OUTPUTS>();
    let mut output_biases = random_biases::<OUTPUTS>();

    for i in (0..train_images.len()).step_by(BATCH_SIZE) {
        let end = (i + BATCH_SIZE).min(train_images.len());

        // Batch processing not implemented
        for (image, label) in train_images[i..end].iter().zip(train_labels[i..end].iter()) {
            let hidden_layer_output = layer_forward(
                &image,
                &hidden_weights,
                &hidden_biases,
                &[relu; NODES],
            );
    
            let output_layer_output = layer_forward(
                &hidden_layer_output,
                &output_weights,
                &output_biases,
                &[fast_tanh; OUTPUTS],
            );

            let predictions = softmax_forward(&output_layer_output);

            let loss = cross_entropy_loss(&predictions, &GROUND_TRUTHS[*label as usize]);

            println!("Prediction: {:?}", predictions);
            println!("Label: {:?}", GROUND_TRUTHS[*label as usize]);
            println!("Loss: {loss}\n");

            let predictions_error = softmax_backward(&predictions, loss);

            let (
                output_layer_input_errors,
                output_weights_gradients,
                output_bias_gradients
            ) = layer_backward(
                &hidden_layer_output, 
                &output_weights, 
                &[fast_tanh_d; OUTPUTS], 
                &output_layer_output, 
                &predictions_error
            );

            let (
                _hidden_layer_input_errors,
                hidden_weights_gradients,
                hidden_bias_gradients
            ) = layer_backward(
                &image, 
                &hidden_weights, 
                &[relu_d; NODES], 
                &hidden_layer_output, 
                &output_layer_input_errors
            );

            // Update output weights
            for (weights, gradients) in output_weights.iter_mut().zip(&output_weights_gradients) {
                for (weight, gradient) in weights.iter_mut().zip(gradients) {
                    *weight -= LEARNING_RATE * gradient
                }
            }

            // Update output biases
            for (bias, gradient) in output_biases.iter_mut().zip(&output_bias_gradients) {
                *bias -= LEARNING_RATE * gradient
            }

            // Update hidden weights
            for (weights, gradients) in hidden_weights.iter_mut().zip(&hidden_weights_gradients) {
                for (weight, gradient) in weights.iter_mut().zip(gradients) {
                    *weight -= LEARNING_RATE * gradient
                }
            }

            // Update hidden biases
            for (bias, gradient) in hidden_biases.iter_mut().zip(&hidden_bias_gradients) {
                *bias -= LEARNING_RATE * gradient
            }
        }
    }

    println!("DONE!");

    // Hidden layer
    // Output layer

    /*
    loop {
        if macroquad::input::is_key_down(KeyCode::Enter) { break }
        next_frame().await
    }
    */
}