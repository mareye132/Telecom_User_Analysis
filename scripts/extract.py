import os
import yaml

mlruns_dir = 'C:/Users/user/Desktop/Github/TelecomUserAnalysis/mlruns'  # Path to your mlruns directory

for experiment in os.listdir(mlruns_dir):
    experiment_path = os.path.join(mlruns_dir, experiment)
    if os.path.isdir(experiment_path):
        for run_id in os.listdir(experiment_path):
            run_path = os.path.join(experiment_path, run_id)
            if os.path.isdir(run_path):
                # Read metadata
                meta_file = os.path.join(run_path, 'meta.yaml')
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, 'r') as file:
                            meta = yaml.safe_load(file)
                            start_time = meta.get('start_time')
                            end_time = meta.get('end_time')
                    except Exception as e:
                        print(f"Error reading metadata for run {run_id}: {e}")
                        start_time = end_time = "N/A"
                
                # Read parameters
                params_file = os.path.join(run_path, 'params')
                if os.path.exists(params_file):
                    if os.path.isfile(params_file):
                        try:
                            with open(params_file, 'r') as file:
                                params = yaml.safe_load(file)
                        except Exception as e:
                            print(f"Error reading parameters for run {run_id}: {e}")
                            params = "Error"
                    else:
                        params = "Not a file"
                else:
                    params = "File not found"
                
                # Read metrics
                metrics_file = os.path.join(run_path, 'metrics')
                if os.path.exists(metrics_file):
                    if os.path.isfile(metrics_file):
                        try:
                            with open(metrics_file, 'r') as file:
                                metrics = yaml.safe_load(file)
                        except Exception as e:
                            print(f"Error reading metrics for run {run_id}: {e}")
                            metrics = "Error"
                    else:
                        metrics = "Not a file"
                else:
                    metrics = "File not found"
                
                # Read artifacts
                artifacts_dir = os.path.join(run_path, 'artifacts')
                artifacts = []
                if os.path.exists(artifacts_dir):
                    if os.path.isdir(artifacts_dir):
                        artifacts = os.listdir(artifacts_dir)
                    else:
                        artifacts.append("Artifacts not a directory")

                print(f"Run ID: {run_id}")
                print(f"Start Time: {start_time}")
                print(f"End Time: {end_time}")
                print(f"Parameters: {params}")
                print(f"Metrics: {metrics}")
                print(f"Artifacts: {artifacts}")
                print('-' * 40)
