import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import ipywidgets as widgets
from ipywidgets import interact

# =========================================
# STEP 1: Load Dataset (Keep CSV in same folder)
# =========================================
data = pd.read_csv("LBM_Simulated_Data.csv")  # Change path if needed

X = data[["Gas_Pressure", "Cutting_Speed", "Laser_Power", "Pulse_Frequency"]]
y = data["Kerf_Width"]

# =========================================
# STEP 2: Train ML Model
# =========================================
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# =========================================
# STEP 3: Feature Importance Plot
# =========================================
perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

plt.figure(figsize=(7, 4))
plt.bar(X.columns, perm_importance.importances_mean, color="#1f77b4", edgecolor="black")
plt.title("Parameter Sensitivity (Permutation Importance)", fontsize=14)
plt.ylabel("Importance Score", fontsize=12)
plt.xticks(rotation=20)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# =========================================
# STEP 4: Interactive Prediction with 4 Dynamic Graphs
# =========================================
def predict_and_plot_all(gas_pressure, cutting_speed, laser_power, pulse_frequency):
    # Predict current kerf width
    input_data = pd.DataFrame([[gas_pressure, cutting_speed, laser_power, pulse_frequency]],
                              columns=["Gas_Pressure", "Cutting_Speed", "Laser_Power", "Pulse_Frequency"])
    pred = rf.predict(input_data)[0]
    print(f"🔧 Predicted Kerf Width: {pred:.4f} mm")
    
    # Ranges for smooth curves
    gas_range = np.linspace(4, 12, 50)
    speed_range = np.linspace(2500, 4500, 50)
    power_range = np.linspace(2400, 3200, 50)
    freq_range = np.linspace(6000, 10000, 50)

    # Predict for each parameter while keeping others fixed
    pred_gas = rf.predict(pd.DataFrame({"Gas_Pressure": gas_range,
                                        "Cutting_Speed": cutting_speed,
                                        "Laser_Power": laser_power,
                                        "Pulse_Frequency": pulse_frequency}))
    pred_speed = rf.predict(pd.DataFrame({"Gas_Pressure": gas_pressure,
                                          "Cutting_Speed": speed_range,
                                          "Laser_Power": laser_power,
                                          "Pulse_Frequency": pulse_frequency}))
    pred_power = rf.predict(pd.DataFrame({"Gas_Pressure": gas_pressure,
                                          "Cutting_Speed": cutting_speed,
                                          "Laser_Power": power_range,
                                          "Pulse_Frequency": pulse_frequency}))
    pred_freq = rf.predict(pd.DataFrame({"Gas_Pressure": gas_pressure,
                                         "Cutting_Speed": cutting_speed,
                                         "Laser_Power": laser_power,
                                         "Pulse_Frequency": freq_range}))
    
    # Plot with better styling
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Kerf Width Sensitivity Analysis", fontsize=16, fontweight='bold')

    # Style function to avoid repetition
    def style_ax(ax, x, y, xlabel, current_value):
        ax.plot(x, y, color="#1f77b4", linewidth=2)
        ax.axvline(current_value, color="red", linestyle="--", label=f"Selected: {current_value:.2f}")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Kerf Width (mm)", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend()

    style_ax(axs[0, 0], gas_range, pred_gas, "Gas Pressure (bar)", gas_pressure)
    style_ax(axs[0, 1], speed_range, pred_speed, "Cutting Speed (mm/min)", cutting_speed)
    style_ax(axs[1, 0], power_range, pred_power, "Laser Power (W)", laser_power)
    style_ax(axs[1, 1], freq_range, pred_freq, "Pulse Frequency (Hz)", pulse_frequency)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

# Interactive sliders
interact(
    predict_and_plot_all,
    gas_pressure=widgets.FloatSlider(min=4, max=12, step=0.1, value=8, description='Gas Pressure (bar)'),
    cutting_speed=widgets.IntSlider(min=2500, max=4500, step=50, value=3500, description='Cut Speed (mm/min)'),
    laser_power=widgets.IntSlider(min=2400, max=3200, step=50, value=2800, description='Laser Power (W)'),
    pulse_frequency=widgets.IntSlider(min=6000, max=10000, step=100, value=8000, description='Pulse Freq (Hz)')
);
