import numpy as np
import pandas as pd
from nilmtk import DataSet
from tensorflow.keras.utils import Sequence


class TimeSeries:
    """
    Encapsulates UK Dale dataset handling intelligence with focus on loading the data from a single HDF5 format file.
    """

    def __init__(self, dataset: DataSet, training_buildings: list, test_buildings: list, window_size: int,
                 batch_size: int, appliance: str, on_threshold: int = 2000):
        self.training_buildings = training_buildings
        self.test_buildings = test_buildings
        self.dataset = dataset
        self.window_size = window_size
        self.batch_size = batch_size
        self.appliance = appliance
        self.mean_power = None
        self.std_power = None
        self._compute_normalization_params()

    def _compute_normalization_params(self):
        all_train_mains_power = []

        for building in self.training_buildings:
            train_elec = self.dataset.buildings[building].elec
            train_mains = train_elec.mains()

            mains_data_frame = train_mains.load(sample_period=6)

            for train_mains_df in mains_data_frame:
                if not train_mains_df.empty:
                    if 'power' in train_mains_df.columns:
                        mains_power = train_mains_df['power']
                    elif ('power', 'active') in train_mains_df.columns:
                        mains_power = train_mains_df[('power', 'active')]
                    else:
                        mains_power = train_mains_df.iloc[:, 0]
                    all_train_mains_power.append(mains_power)

        if all_train_mains_power:
            combined_mains_power = pd.concat(all_train_mains_power, axis=0)
            self.mean_power = combined_mains_power.mean()
            self.std_power = combined_mains_power.std()
            print(f"Mean power: {self.mean_power}")
            print(f"Std power: {self.std_power}")

            if self.mean_power.isnull().any() or self.std_power.isnull().any():
                raise ValueError("Normalization parameters contain NaN values. Check your data preprocessing steps.")
        else:
            raise ValueError("No training data available for normalization.")

    def getTrainingDataGenerator(self):
        return TimeSeriesDataGenerator(self.dataset, self.training_buildings, self.appliance, self.mean_power,
                                       self.std_power, self.window_size, self.batch_size, is_training=True)

    def getTestDataGenerator(self):
        return TimeSeriesDataGenerator(self.dataset, self.test_buildings, self.appliance, self.mean_power,
                                       self.std_power, self.window_size, self.batch_size, is_training=False)


class TimeSeriesDataGenerator(Sequence):
    def __init__(self, dataset, buildings, appliance, mean_power, std_power, window_size, batch_size, is_training=True):
        self.dataset = dataset
        self.buildings = buildings
        self.appliance = appliance
        self.mean_power = mean_power
        self.std_power = std_power
        self.window_size = window_size
        self.batch_size = batch_size
        self.is_training = is_training
        self.data_generator = self._data_generator()
        self.total_samples = self._count_samples()

    def _data_generator(self):
        chunk_size = 1000000  # Adjust as needed
        for building in self.buildings:
            elec = self.dataset.buildings[building].elec
            mains = elec.mains()
            appliance = elec[self.appliance]
            for mains_df, appliance_df in zip(mains.load(chunksize=chunk_size), appliance.load(chunksize=chunk_size)):
                mains_power, appliance_power = self._process_data(mains_df, appliance_df)
                for i in range(0, len(mains_power) - self.window_size + 1, self.window_size):
                    yield mains_power[i:i + self.window_size], appliance_power[i:i + self.window_size]

    def _count_samples(self):
        total_samples = 0
        for building in self.buildings:
            elec = self.dataset.buildings[building].elec
            mains = elec.mains()
            mains_length, _ = next(mains.load()).shape
            total_samples += mains_length
        return total_samples // self.window_size

    def __len__(self):
        return self.total_samples // self.batch_size

    def __getitem__(self, index):
        batch_X, batch_y = [], []
        for _ in range(self.batch_size):
            try:
                X, y = next(self.data_generator)
                batch_X.append(X)
                batch_y.append(y)
            except StopIteration:
                self.data_generator = self._data_generator()  # Reset generator
                X, y = next(self.data_generator)
                batch_X.append(X)
                batch_y.append(y)

        return np.array(batch_X), np.array(batch_y)

    def _process_data(self, mains_df, appliance_df):
        # Extract power readings
        if 'power' in mains_df.columns:
            mains_power = mains_df['power']
        elif ('power', 'active') in mains_df.columns:
            mains_power = mains_df[('power', 'active')]
        else:
            mains_power = mains_df.iloc[:, 0]  # Select the first column if 'power' is not present

        appliance_power = appliance_df[('power', 'active')]
        if appliance_power.empty:
            appliance_power = appliance_df[('power', 'apparent')]

        # Resample to a 6-second interval
        # mains_power = mains_power.resample('6S').mean()
        # ppliance_power = appliance_power.resample('6S').mean()

        # Align the series by their indices
        mains_power, appliance_power = mains_power.align(appliance_power, join='inner', axis=0)

        # Fill missing values
        mains_power = mains_power.fillna(method='ffill').fillna(method='bfill')
        appliance_power = appliance_power.fillna(0)

        # Normalize mains power
        mains_power = (mains_power - self.mean_power) / (self.std_power + 1e-8)

        # Convert to numpy arrays
        mains_power = mains_power.values.reshape(-1, 1)
        appliance_power = appliance_power.values.reshape(-1, 1)

        return mains_power, appliance_power
