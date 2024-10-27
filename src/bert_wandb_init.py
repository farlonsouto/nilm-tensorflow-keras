import wandb

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

wandb.init(
    project="nilm_bert_transformer",
    config={
        "appliance": "kettle",
        "loss": "nde_loss",
        # "loss": "bert4nilm_loss",
        "on_threshold": 2000,
        "window_size": 64,
        "batch_size": 128,
        "head_size": 64,
        "num_heads": 2,
        "n_layers": 1,
        "dropout": 0.1,
        "learning_rate": 1e-5,
        "epochs": 2,
        "optimizer": "adam",
        "tau": 1.0,
        "lambda_val": 0.1,
        "masking_portion": 0.2,
        "output_size": 1,
        "conv_kernel_size": 8,
        "deconv_kernel_size": 8,
        "embedding_dim": 64,
        "pooling_type": "max",  # Options: 'max', 'average'
        "conv_activation": "relu",
        "dense_activation": "tanh",
        "conv_filters": 32,  # separate from head_size
        "ff_dim": 128,  # Feed-forward network dimension
        "layer_norm_epsilon": 1e-6,
        "kernel_initializer": "glorot_uniform",
        "bias_initializer": "zeros",
        "kernel_regularizer": None,  # Options: None, 'l1', 'l2', 'l1_l2'
        "bias_regularizer": None,  # Options: None, 'l1', 'l2', 'l1_l2'
    }
)

# Retrieve the configuration from WandB
wandb_config = wandb.config