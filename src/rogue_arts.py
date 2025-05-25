import matplotlib.pyplot as plt
import numpy as np


class HarryPlotter:
    def __init__(self, model, test_data_generator, model_type='seq2seq'):
        """
        Initializes the Plotter class.

        Args:
            model (tf.keras.Model): The trained model for inference.
            test_data_generator: A generator that provides test data (inputs and ground truth).
            model_type (str): Type of model - 'seq2seq' or 'seq2p' (sequence-to-point).
        """
        self.model = model
        self.test_data_generator = test_data_generator
        self.model_type = model_type
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line_gt, = self.ax.plot([], [], label="Ground Truth", color="blue", linewidth=1.5)
        self.line_pred, = self.ax.plot([], [], label="Prediction", color="red", linewidth=1.5)

        # For seq2p, add scatter plot for point predictions
        if self.model_type == 'seq2p':
            self.scatter_pred = self.ax.scatter([], [], c='red', s=30, alpha=0.7, label="Predicted Points", zorder=5)

        self.ax.legend()
        self.ax.set_title(f"Real-Time Predictions vs. Ground Truth ({model_type.upper()})")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Power (W)")
        self.ax.grid(True, alpha=0.3)

        self.ground_truth = []
        self.predictions = []
        self.time_steps = []

    def update_plot(self, x, y):
        """
        Updates the plot with new predictions and ground truth values.

        Args:
            x: Input batch from the test data generator.
            y: Ground truth batch corresponding to `x`.
        """
        # Perform prediction on the current batch
        pred = self.model.predict(x, verbose=0)  # Set verbose=0 to reduce output clutter

        if self.model_type == 'seq2p':
            # For seq2p: y should be single points, pred should be single points
            batch_size = len(y)

            # Handle the case where y might be sequences (for visualization)
            if len(y.shape) > 1 and y.shape[1] > 1:
                # If y is still a sequence, we'll plot the full sequence as ground truth
                # but only the midpoint as the target for the prediction
                current_time = len(self.ground_truth)

                for i in range(batch_size):
                    # Add the full ground truth sequence
                    gt_sequence = y[i].flatten()
                    seq_len = len(gt_sequence)

                    # Extend ground truth with the sequence
                    self.ground_truth.extend(gt_sequence)

                    # For predictions, we need to align them with the midpoint of each sequence
                    midpoint_idx = current_time + seq_len // 2

                    # Extend predictions with NaN for non-midpoint positions
                    pred_sequence = [np.nan] * seq_len
                    pred_sequence[seq_len // 2] = pred[i, 0] if pred.shape[1] == 1 else pred[i]
                    self.predictions.extend(pred_sequence)

                    current_time += seq_len
            else:
                # If y is already single points (as expected for seq2p)
                self.ground_truth.extend(y.flatten())
                self.predictions.extend(pred.flatten())
        else:
            # For seq2seq: both y and pred should be sequences
            self.ground_truth.extend(y.flatten())
            self.predictions.extend(pred.flatten())

        # Update time steps
        self.time_steps = list(range(len(self.ground_truth)))

        # Update the plot data
        self.line_gt.set_data(self.time_steps, self.ground_truth)

        if self.model_type == 'seq2p':
            # For seq2p, show continuous ground truth and discrete predictions
            pred_clean = np.array(self.predictions)
            valid_mask = ~np.isnan(pred_clean)
            valid_times = np.array(self.time_steps)[valid_mask]
            valid_preds = pred_clean[valid_mask]

            self.line_pred.set_data(valid_times, valid_preds)

            # Update scatter plot for point predictions
            if len(valid_times) > 0:
                self.scatter_pred.set_offsets(np.column_stack([valid_times, valid_preds]))
        else:
            self.line_pred.set_data(self.time_steps, self.predictions)

        # Adjust the plot limits dynamically
        self.ax.relim()
        self.ax.autoscale_view()

        # Add some padding to y-axis
        if len(self.ground_truth) > 0:
            y_min, y_max = min(self.ground_truth), max(self.ground_truth)
            if len([p for p in self.predictions if not np.isnan(p)]) > 0:
                pred_clean = [p for p in self.predictions if not np.isnan(p)]
                y_min = min(y_min, min(pred_clean))
                y_max = max(y_max, max(pred_clean))

            y_range = y_max - y_min
            self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # Redraw the plot
        plt.pause(0.01)

    def run(self):
        """
        Executes the real-time plotting during inference.
        """
        print(f"Starting real-time plotting for {self.model_type.upper()} model...")
        try:
            for x, y in self.test_data_generator:
                self.update_plot(x, y)
        except KeyboardInterrupt:
            print("\nPlotting interrupted by user.")
        except Exception as e:
            print(f"Error during plotting: {e}")

        print("Inference completed. Close the plot window to exit.")
        plt.show()

    def save_plot(self, filename="prediction_results.png"):
        """
        Saves the current plot to a file.

        Args:
            filename (str): Name of the file to save the plot.
        """
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")