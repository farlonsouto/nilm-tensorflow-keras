import sys


def get_args():
    # Defaults
    valid_models = ['bert', 'seq2seq', 'seq2p']
    valid_appliances = ['kettle', 'fridge', 'microwave', 'dish washer']
    valid_kernel_regularizers = ['none', 'l1l2']

    args = sys.argv

    if len(args) < 4:
        print("❌ Not enough arguments.")
        sys.exit(1)

    model = args[1].lower()
    appliance = args[2].lower()
    kernel_regularizer = args[3].lower()

    # Set defaults and validate
    if model not in valid_models:
        print(f"⚠️ Invalid model '{model}'. Use seq2p, seq2seq or bert.")
        show_usage()
        sys.exit(1)

    if appliance not in valid_appliances:
        print(f"⚠️ Invalid appliance '{appliance}'. Use kettle, fridge, microwave or 'dish washer'.")
        show_usage()
        sys.exit(1)

    if kernel_regularizer not in valid_kernel_regularizers:
        print(f"⚠️ Invalid kernel_regularizer '{kernel_regularizer}'. Use none or l1l2.")
        show_usage()
        sys.exit(1)

    if kernel_regularizer == 'l1l2':
        kernel_regularizer = 'l1_l2'
    else:
        kernel_regularizer = None

    # Check if training or testing
    is_continuation = False
    masked = False

    if len(args) == 6:  # Training mode
        if args[4].lower() not in ["continue", "first"]:
            print("⚠️  Invalid value for continuation flag. Use 'continue' or 'first'.")
            show_usage()
            sys.exit(1)
        if args[5].lower() not in ["masked", "unmasked"]:
            print("⚠️  Invalid value for masking flag. Use 'masked' or 'unmasked'.")
            show_usage()
            sys.exit(1)

        is_continuation = args[4].lower() == "continue"
        masked = args[5].lower() == "masked"

    elif len(args) > 4:
        print("⚠️  Too many arguments for test mode. Only provide model, appliance, and kernel_regularizer.")
        show_usage()
        sys.exit(1)

    return model, appliance, kernel_regularizer, is_continuation, masked


def show_usage():
    print("Syntax (train): python train.py <model> <appliance> <kernel_regularizer> <is_continuation> <masked>")
    print("Usage (train): python train.py seq2p kettle none continue unmasked")
    print("Usage (train): python train.py bert kettle none first masked")
    print("⚠️ Only the bert model handles masked. Recommended: Use unmasked for all the other models.")
    print("⚠️ Any model can continue training from the last checkpoint with using the 'continue' flag.")
    print("⚠️ Using the flag 'first' will start a training from the beginning, overwriting any existing checkpoints.")
    print("Syntax (test):  python test.py <model> <appliance> <kernel_regularizer>")
    print("Usage (test):  python test.py bert fridge l1l2")
