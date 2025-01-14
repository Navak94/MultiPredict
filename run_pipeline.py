import subprocess

def run_pipeline():
    # Define the paths to your script files
    linear_regression_script = "linear_regression_sklearn.py"
    tensorflow_nn_script = "neural_nets_tensorflow.py"
    sentiment_analysis_script = "sentiment_analysis.py"

    try:
        # Step 1: Run the Linear Regression Model
        print("Running Linear Regression Model...")
        subprocess.run(["python", linear_regression_script], check=True)
        print("Linear Regression Model completed successfully.\n")

        # Step 2: Run the tensorflow_nn Model
        print("Running tensorflow_nn Model...")
        subprocess.run(["python", tensorflow_nn_script], check=True)
        print("tensorflow_nn completed successfully.\n")

         # Step 3: Run Sentiment Analysis
        print("Running Sentiment Analysis...")
        subprocess.run(["python", sentiment_analysis_script], check=True)
        print("Sentiment Analysis completed successfully.\n")

        # Step 4: Run Language Model

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    run_pipeline()