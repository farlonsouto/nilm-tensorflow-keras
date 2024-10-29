import os
import pandas as pd

# Define the path to the main directory where you extracted the UK-DALE dataset
data_dir = '../../datasets/ukdale'

# Expected columns in each .dat file
EXPECTED_COLUMNS = ['unix_timestamp', 'value']


def load_labels(house_path):
    """
    Load the labels.dat file to get a mapping of channels to appliance names.
    """
    labels_path = os.path.join(house_path, 'labels.dat')
    labels = {}

    if os.path.exists(labels_path):
        labels_df = pd.read_csv(labels_path, sep=' ', header=None, names=['channel', 'appliance_name'])
        labels = dict(labels_df.values)

        print("\nAppliance Labels:")
        print(labels_df.to_string(index=False, header=False))
        print("=" * 20)

    return labels


# Iterate over each house directory
for house_dir in ["house_3"]:  # sorted(os.listdir(data_dir)):
    house_path = os.path.join(data_dir, house_dir)
    if os.path.isdir(house_path):
        print(f"\nHouse: {house_dir}")
        print("=" * 20)

        # Load the appliance labels for this house
        appliance_labels = load_labels(house_path)

        # Iterate over each .dat file in the house directory
        for file_name in sorted(os.listdir(house_path)):
            file_path = os.path.join(house_path, file_name)
            if file_name.endswith('.dat') and file_name != 'labels.dat':
                # Extract the channel number from the file name
                channel_number = file_name.replace('.dat', '').replace('channel_', '')
                appliance_name = appliance_labels.get(int(channel_number), f"Channel {channel_number}")

                # Read the .dat file
                try:
                    df = pd.read_csv(file_path, sep=' ', header=None, names=EXPECTED_COLUMNS)
                    # Convert UNIX timestamp to datetime
                    df['datetime'] = pd.to_datetime(df['unix_timestamp'], unit='s')
                    df.set_index('datetime', inplace=True)

                    # Drop the original timestamp column to focus on the datetime index
                    df.drop(columns=['unix_timestamp'], inplace=True)

                    # Print appliance/channel information
                    print(f"Appliance: {appliance_name}")
                    print(f"Data range: {df.index.min()} to {df.index.max()}")
                    print(f"Number of records: {len(df)}")
                    print(f"Sample data:\n{df.head()}")
                    print("-" * 20)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
