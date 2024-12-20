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

    # Appliance and Dataset specific
    "appliance": "kettle",  # The selected appliance must be the same for training and testing !!
    "on_threshold": 2000,
    "max_power": 3200,

    # Training
    "batch_size": 32,
    "epochs": 3,
    "learning_rate": 1e-4,
    "optimizer": "adam",
    "loss": "huber",  # "mse" or "huber" seems to make no difference
    "lambda_val": 1.0,  # inside the loss function
    "num_features": 1,  # The aggregated power readings, AC type; hour, minute, second; appliance status, etc

    # Input
    "window_size": 480,  # for UK Dale, 10 time steps mean 1 minute
    "mlm_mask": True,  # MLM masking for BERT
    "masking_portion": 0.25,
    "window_stride": 60,
    "add_artificial_activations": False,
    "balance_enabled": False,
    "normalize_aggregated": False,  # min-max, squeezes between 0 and 1
    "normalize_appliance": False,  # min-max, squeezes between 0 and 1
    "standardize_aggregated": False,  # z-score, Uses mean and std: x = (x - x_mean) / x_std
    "standardize_appliance": False,  # z-score, Uses mean and std: y = (y - y_mean) / y_std

    # 1D Convolution layer
    "conv_kernel_size": 5,
    "conv_strides": 1,  # to be fixed in 1
    "conv_padding": 2,
    "conv_activation": "relu",  # preferably ReLU

    # Transformer
    "hidden_size": 256,  # It is also the hidden size of the LSTMs in the seq2seq
    "num_heads": 2,
    "n_layers": 3,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-6,  # Original value is 1e-6
    "dense_activation": "gelu",  # Originally GELU
    "ff_dim": 512,  # Feed-forward Network: 4x the hidden size is the recommended.

    # Deconvolution layer
    "deconv_kernel_size": 4,
    "deconv_strides": 2,
    "deconv_padding": 1,
    "deconv_activation": "relu",

    # Dimension (number of features) in the output layer
    "output_size": 1,
}


def for_model_appliance(model_name, appliance_name) -> dict:
    # Initialize the configuration dictionary
    config["appliance"] = appliance_name
    config["model"] = model_name

    # Set the appliance-specific configuration
    if appliance_name == "kettle":
        config.update({
            "lambda_val": 1.0,
            "appliance_max_power": 3200,
            "on_threshold": 2000,
            "min_on_duration": 0,
            "min_off_duration": 12,
        })
    elif appliance_name == "fridge":
        config.update({
            "lambda_val": 1e-6,
            "appliance_max_power": 400,
            "on_threshold": 50,
            "min_on_duration": 60,
            "min_off_duration": 12,
        })
    elif appliance_name == "washer":
        config.update({
            "lambda_val": 1e-2,
            "appliance_max_power": 2500,
            "on_threshold": 20,
            "min_on_duration": 1800,
            "min_off_duration": 160,
        })
    elif appliance_name == "microwave":
        config.update({
            "lambda_val": 1.0,
            "appliance_max_power": 3000,
            "on_threshold": 200,
            "min_on_duration": 12,
            "min_off_duration": 30,
        })
    elif appliance_name == "dish washer":
        config.update({
            "lambda_val": 1.0,
            "appliance_max_power": 2500,
            "on_threshold": 10,
            "min_on_duration": 1800,
            "min_off_duration": 1800,
        })
    else:
        raise ValueError(f"Unknown appliance: {appliance_name}")

    return config
