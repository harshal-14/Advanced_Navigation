import os
from plot import create_overlay_plots, rmse_plots, rmse
from ins_gnss import INSGNSS, load_data
import numpy as np

def process_trajectory(data, model_params):
    """Process the trajectory data with given UKF parameters."""
    os.makedirs("./output_imgs/", exist_ok=True)
    counter = 100
    
    for model_type, measurement_noise, prediction_noise in model_params:
        for attempt in range(counter):
            ukf = INSGNSS(model_type=model_type, measurement_noise_scale=measurement_noise, pred_noise_scale=prediction_noise)
            try:
                # Run the UKF for the given data, with the first data point as the initial state
                estimates, gt, haversines = ukf.run(data)

                # Save the plots   
                save_plots(estimates[1:], gt[1:], haversines[1:], [d.time for d in data][2:], model_type)
                print(f"Success for {model_type} after {attempt + 1} attempts")
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {model_type}, error: {e}")
        else:
            print(f"Failure to complete after {counter} attempts for {model_type}")

def save_plots(estimates, gt, haversines, times, model_type):
    """Save plots generated from the UKF results."""
    rmse_figure = rmse_plots(gt, estimates, data)
    rmse_figure.savefig(f"./output_imgs/{model_type}_rmse.png")
    
    position_fig, haversine_fig, lat_lon_fig = create_overlay_plots(gt, estimates, haversines, times)
    position_fig.savefig(f"./output_imgs/{model_type}_position.png")
    haversine_fig.savefig(f"./output_imgs/{model_type}_haversine.png")
    lat_lon_fig.savefig(f"./output_imgs/{model_type}_latlon.png")

data = load_data("/home/lucifer/WPI/Spring_courses/Advanced_Navigation/INS-GNSS/trajectory_data.csv")
model_params = [
    ("FeedBack", 2e-3, 5e-5),
    ("FeedForward", 1e-9, 0.18)
]

process_trajectory(data, model_params)
