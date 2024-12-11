# Transformer from Scratch for Machine Translation

This was done as part of the coursework for CS7.501 Advanced NLP.


The code has been writted modularly (in the required format) and hence is easily interpretable.

The arguments required to run the files are as follows:

test.py: script to evaluate our model
run test.py --help to to get the below:
parser.add_argument('--d_model', type=int, default=512, help='Dimensionality of the model')
parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the model')
parser.add_argument('--d_ff', type=int, default=2048, help='Dimensionality of the feed-forward layer')
parser.add_argument('--max_seq_length', type=int, default=100, help='Maximum sequence length for input')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
parser.add_argument('--wandb', type=bool, default=False, help='Use Weights & Biases for logging')


similar format for the training script train.py


Due to lack of onedrive space, not including the final checkpoint, however running train.py with the default params and 30 epochs gives the final model.