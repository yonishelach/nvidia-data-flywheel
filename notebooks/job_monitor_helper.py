import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
from IPython.display import clear_output

def format_runtime(seconds):
    """Format runtime in seconds to a human-readable string."""
    if seconds is None:
        return "-"
    minutes, seconds = divmod(seconds, 60)
    if minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    return f"{int(seconds)}s"

def create_results_table(job_data):
    """Create a pandas DataFrame from job data."""
    rows = []
    for nim in job_data["nims"]:
        model_name = nim["model_name"]
        for eval in nim["evaluations"]:
            all_scores = eval["scores"]
            
            row = {
                "Model": model_name,
                "Eval Type": eval["eval_type"].upper(),
                "Percent Done": eval["progress"],
                "Runtime": format_runtime(eval["runtime_seconds"]),
                "Status": "Completed" if eval["finished_at"] else "Running",
                "Started": datetime.fromisoformat(eval["started_at"]).strftime("%H:%M:%S"),
                "Finished": datetime.fromisoformat(eval["finished_at"]).strftime("%H:%M:%S") if eval["finished_at"] else "-"
            }
            
            if "function_name" in all_scores:
                row["Function Name Accuracy"] = all_scores["function_name"]
            
            if "function_name_and_args_accuracy" in all_scores:
                row["Function + Args Accuracy"] = all_scores["function_name_and_args_accuracy"]
                
            if "tool_calling_correctness" in all_scores:
                row["Tool Calling Correctness (LLM-Judge)"] = all_scores["tool_calling_correctness"]
            
            # Add any other scores with formatted names
            for score_name, score_value in all_scores.items():
                if score_name not in ["function_name", "tool_calling_correctness", "similarity", "function_name_and_args_accuracy"]:
                    formatted_name = score_name.replace("_", " ").title()
                    row[formatted_name] = score_value
            
            rows.append(row)
    
    if not rows:
        return pd.DataFrame(columns=["Model", "Eval Type", "Function Name Accuracy", "Tool Calling Correctness (LLM-Judge)", "Similarity (LLM-Judge)", "Percent Done", "Runtime", "Status", "Started", "Finished"])
    
    df = pd.DataFrame(rows)
    return df.sort_values(["Model", "Eval Type"])

def create_customization_table(job_data):
    """Create a pandas DataFrame from customization data."""
    customizations = []
    for nim in job_data["nims"]:
        model_name = nim["model_name"]
        for custom in nim["customizations"]:
            customizations.append({
                "Model": model_name,
                "Started": datetime.fromisoformat(custom["started_at"]).strftime("%H:%M:%S"),
                "Epochs Completed": custom["epochs_completed"],
                "Steps Completed": custom["steps_completed"],
                "Finished": datetime.fromisoformat(custom["finished_at"]).strftime("%H:%M:%S") if custom["finished_at"] else "-",
                "Status": "Completed" if custom["finished_at"] else "Running",
                "Runtime": format_runtime(custom["runtime_seconds"]),
                "Percent Done": custom["progress"],
            })
   
    if not customizations:
        customizations = pd.DataFrame(columns=["Model", "Started", "Epochs Completed", "Steps Completed", "Finished", "Runtime", "Percent Done"])
    customizations = pd.DataFrame(customizations)
    return customizations.sort_values(["Model"])

def monitor_job(job_id):
    """Monitor a job and display its progress in a table."""
    print(f"Monitoring job {job_id}...")
    print("Press Ctrl+C to stop monitoring")
    
    while True:
        try:
            clear_output(wait=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            job_data = get_job_status(job_id)
            results_df = create_results_table(job_data)
            customizations_df = create_customization_table(job_data)
            clear_output(wait=True)
            print(f"Job Status: {job_data['status']}")
            print(f"Total Records: {job_data['num_records']}")
            print(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
            print("\nResults:")
            display(results_df)
            print("\nCustomizations:")
            display(customizations_df)
            display(job_data)

            # Plot 1: Evaluation Scores
            ax.set_title("Evalulation Results", fontsize=14)
            if not results_df.empty:
                pivot_df = results_df.pivot(index="Model", columns="Eval Type", values="Tool Calling Correctness (LLM-Judge)").fillna(0)
                pivot_df.plot(kind='bar', ax=ax)
                ax.set_ylabel("Eval Metrics")
                ax.set_ylim(0, 1)
                ax.legend(title="Eval Type")
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, "No Evaluation Data", ha='center', va='center')

            plt.tight_layout()
            plt.show()                        
            time.sleep(POLL_INTERVAL)

            # Check if job is completed or failed
            if job_data['status'] in ['completed', 'failed']:
                # print(f"\nJob monitoring complete! Final status: {job_data['status']}")
                if job_data['status'] == 'failed':
                    print("Job failed - check error details above")
                    if job_data.get('error'):
                        print(f"Error: {job_data['error']}")
                else:
                    print("Job completed successfully!")
                break

        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            break