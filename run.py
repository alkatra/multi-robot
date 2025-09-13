
import sim as simulation  
import csv
import os
import time

ROBOT_COUNTS_TO_TEST = [2, 4, 6, 8, 10]  # placeholder
OUTPUT_FILENAME = "simulation_metrics_log.csv"

def main():

    run_batch_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"Starting experiment batch: {run_batch_id}")

    file_exists = os.path.isfile(OUTPUT_FILENAME)
    
    headers = [
        'run_batch_id', 'num_robots_tested', 'robot_density', 'success_rate',
        'avg_speed_successful', 'makespan_frames', 'sum_completion_times_frames',
        'min_inter_robot_dist', 'total_collisions', 'total_replans',
        'deadlock_count'
    ]

    with open(OUTPUT_FILENAME, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        
        if not file_exists:
            writer.writeheader()

        for count in ROBOT_COUNTS_TO_TEST:
            print(f"\n--- Starting simulation for {count} robots ---")
            
            try:
                metrics = simulation.run_simulation(num_robots=count, save_video=True)
                
                if "error" in metrics:
                    print(f"Could not run simulation for {count} robots. Reason: {metrics['error']}")
                    continue

                row_data = {
                    'run_batch_id': run_batch_id,
                    'num_robots_tested': metrics.get('num_robots', count),
                    'robot_density': f"{metrics.get('robot_density', 0):.4f}",
                    'success_rate': f"{metrics.get('success_rate', 0):.2f}",
                    'avg_speed_successful': f"{metrics.get('avg_speed_successful', 0):.3f}",
                    'makespan_frames': metrics.get('makespan_frames', -1),
                    'sum_completion_times_frames': metrics.get('sum_completion_times_frames', -1),
                    'min_inter_robot_dist': f"{metrics.get('min_inter_robot_dist', -1):.3f}",
                    'total_collisions': metrics.get('total_collisions', -1),
                    'total_replans': metrics.get('total_replans', -1),
                    'deadlock_count': metrics.get('deadlock_count', -1)
                }
                
                writer.writerow(row_data)
                csvfile.flush()
                
                print(f"--- Finished simulation for {count} robots. Metrics saved. ---")

            except Exception as e:
                print(f"!!! An unexpected error occurred during simulation for {count} robots: {e} !!!")

    print(f"\nExperiment batch '{run_batch_id}' complete. Results saved to '{OUTPUT_FILENAME}'.")

if __name__ == "__main__":
    main()
