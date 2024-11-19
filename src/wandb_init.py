# Initialize WandB for tracking

# The article doesn't specify a single on_threshold value for all appliances. Instead, it mentions different
# on-thresholds for various appliances in Table 1. For example:

# - Fridge: 50W
# - Washer: 20W
# - Microwave: 200W
# - Dishwasher: 10W
# - Kettle: 2000W

# These values are specific to each appliance and are used to determine when an appliance is considered to be in the
# "on" state. In the context of the 'kettle' appliance that was used in the example, the correct on_threshold should be
# 2000W, not 50W. To correct this, we should modify the WandB configuration in the runner script to use the appropriate
# on_threshold for each appliance:


config = {
    "appliance": "kettle",  # The selected appliance must be the same for training and testing !!
    "on_threshold": 2000,
    "max_power": 3100,
    "min_on_duration": 12,  # in seconds
    "min_off_duration": 0,  # in seconds
    "loss": "bert4nilm_loss",  # The BERT4NILM custom loss is called from inside the model
    "window_size": 128,  # for UK Dale, 10 time steps mean 1 minute
    "batch_size": 128,
    "hidden_size": 256,
    "num_heads": 2,
    "n_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-4,
    "epochs": 10,
    "optimizer": "adam",
    "tau": 1.0,
    "lambda_val": 1,  # inside the loss function
    "masking_portion": 0.25,
    "output_size": 1,
    "conv_kernel_size": 3,
    "deconv_kernel_size": 4,
    "conv_activation": "gelu",
    "dense_activation": "relu",
    "ff_dim": 256,  # Feed-forward network dimension
    "layer_norm_epsilon": 1e-2,
    "kernel_initializer": "glorot_uniform",
    "bias_initializer": "zeros",
    "kernel_regularizer": None,  # Options: None, 'l1', 'l2', 'l1_l2'
    "bias_regularizer": None,  # Options: None, 'l1', 'l2', 'l1_l2'
}
