#Genral Parameters
TRAIN_FOR = 120
WINDOW = 100
PREDICT_FOR = 10

# Hyperparameters
epochs = 100
train_for = 1000  # Number of training samples
predict_for = 200  # Number of predictions
input_chunk_length = 50  # Input sequence length for the model
output_chunk_length = 1  # Output sequence length for the model
d_model = 64  # Dimensionality of model input
nhead = 2  # Number of attention heads
num_encoder_layers = 5  # Number of layers in encoder
num_decoder_layers = 5  # Number of layers in decoder
dim_feedforward = 512  # Feedforward layer size
dropout = 0.1  # Dropout rate
activation = 'relu'  # Activation function
