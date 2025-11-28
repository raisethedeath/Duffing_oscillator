import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class Visualizer:
    @staticmethod
    def plot_trajectory(csv_path):
        """
        Read CSV and plot trajectory
        """
        data = pd.read_csv(csv_path)
        plt.figure(figsize=(10, 4))
        plt.plot(data['time'], data['x'], lw=1)
        plt.title(f"Trajectory: {os.path.basename(csv_path)}")
        plt.xlabel("Time")
        plt.ylabel("Displacement (x)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_steady_parameter(omega, window_cycles, csv_path):
        """
        Read CSV and plot steady parameter
        """
        df = pd.read_csv(csv_path)
        t = df['time'].values
        x = df['x'].values
        dt = t[1] - t[0]
        period = 2 * np.pi / omega
        points_per_cycle = int(period / dt)
        
        if points_per_cycle == 0: return 0
        
        # Calculate cycle difference error E(t) = |x(t) - x(t-T)|
        # We start calculating from the 2nd cycle
        max_idx = len(x)
        errors = []
        times = []
        
        for i in range(points_per_cycle, max_idx):
            diff = abs(x[i] - x[i - points_per_cycle])
            errors.append(diff)
            times.append(t[i])
            
        errors = np.array(errors)
        times = np.array(times)

        # Use a sliding window to check if the error remains below the threshold
        # 在5个周期内计算平均
        window_size = points_per_cycle * window_cycles
        average_err = []

        for i in range(0, len(errors) - window_size):
            window_data = errors[i : i + window_size]
            average_err.append(np.mean(window_data))

        plt.figure(figsize=(10, 4))
        plt.plot(times, errors, lw=1)
        plt.xlabel("Time")
        plt.ylabel("$E(t)=\|x(t)-x(t-T)\|$")
        plt.grid(True)
        plt.tight_layout()
        
        plt.figure(figsize=(10, 4))
        plt.plot(times[0:len(errors) - window_size], average_err, lw=1)
        plt.xlabel("Time")
        plt.ylabel("$E(t)=\|x(t)-x(t-T)\|$")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_heatmap(x_vals, y_vals, z_vals, xlabel, ylabel, zlabel, title):
        """
        Plot heatmap (used for steady state time analysis)
        """

        plt.figure(figsize=(8, 6))
        # Convert 1D arrays to grid
        X, Y = np.meshgrid(np.unique(x_vals), np.unique(y_vals))
        Z = z_vals.reshape(len(np.unique(y_vals)), len(np.unique(x_vals)))
        
        cp = plt.contourf(X, Y, Z, cmap='viridis', levels=20)
        plt.colorbar(cp, label=zlabel)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_hysteresis(forward_data, backward_data):
        """
        Plot hysteresis loop
        """

        plt.figure(figsize=(8, 5))
        plt.plot(forward_data['omega'], forward_data['amplitude'], 'b-o', label='Forward Sweep', markersize=3)
        plt.plot(backward_data['omega'], backward_data['amplitude'], 'r-o', label='Backward Sweep', markersize=3)
        plt.xlabel("Frequency ($\omega$)")
        plt.ylabel("Amplitude")
        plt.title("Frequency Response & Hysteresis")
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_poincare(poincare_data):
        """
        Poincaré Section
        """

        plt.figure(figsize=(6, 6))
        plt.scatter(poincare_data['x'], poincare_data['v'], s=0.5, c='k')
        plt.xlabel("x")
        plt.ylabel("v")
        plt.title("Poincaré Section")
        plt.grid(True)
        plt.show()

class Analyzer:
    @staticmethod
    def calculate_steady_time(df, omega, threshold=1e-1, window_cycles=5):
        """
        Quantitatively determine the time to reach steady state.
        Method: Compare the trajectory difference between the current cycle and the previous cycle.
        If the variance of the difference is below the threshold for N consecutive cycles, steady state is considered reached.
        """
        t = df['time'].values
        x = df['x'].values
        dt = t[1] - t[0]
        period = 2 * np.pi / omega
        points_per_cycle = int(period / dt)
        
        if points_per_cycle == 0: return 0
        
        # Calculate cycle difference error E(t) = |x(t) - x(t-T)|
        # We start calculating from the 2nd cycle
        max_idx = len(x)
        errors = []
        times = []
        
        for i in range(points_per_cycle, max_idx):
            diff = abs(x[i] - x[i - points_per_cycle])
            errors.append(diff)
            times.append(t[i])
            
        errors = np.array(errors)
        times = np.array(times)
        
        # Use a sliding window to check if the error remains below the threshold
        window_size = points_per_cycle * window_cycles
        
        for i in range(0, len(errors) - window_size):
            window_data = errors[i : i + window_size]
            if np.mean(window_data) < threshold:
                return times[i] # Return the time point when steady state is reached
        
        return t[-1] # If steady state is not reached, return the maximum time

    @staticmethod
    def get_poincare_points(df, omega):
        """Extract Poincaré section points (sample once every T = 2pi/omega)"""
        t = df['time'].values
        period = 2 * np.pi / omega
        
        # Find indices of time points close to integer multiples of the period
        indices = []
        for n in range(1, int(t[-1] / period)):
            target_t = n * period
            idx = (np.abs(t - target_t)).argmin()
            indices.append(idx)
            
        return df.iloc[indices]
