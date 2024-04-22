import subprocess
import threading


def run_batch_file():
    batch_file = "GeneDriveMosquitoes-IH240126_windows.bat"  # Replace with the path to your batch file
    try:
        subprocess.check_call([batch_file], shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running batch file: {e}")

    print("Batch file completed.")

# Number of times to run the batch file in parallel
num_parallel_runs = 10

# Create a list to store thread objects
threads = []

# Start the batch file in parallel threads
for _ in range(num_parallel_runs):
    thread = threading.Thread(target=run_batch_file)
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All batch files have completed.")
