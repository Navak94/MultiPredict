import subprocess

def run_pipeline():
    # Define the paths to your script files
    linear_regression_script = "linear_regression_sklearn.py"
    tensorflow_nn_script = "neural_nets_tensorflow.py"
    sentiment_analysis_script = "language_model_sentiment_analysis.py"
    language_model_standalone = "language_model_standalone.py"
    format_csv = "organize_info.py"
    try:

        # Step 1: Run Sentiment Analysis
        print("Running Sentiment Analysis GPT...")
        subprocess.run(["python", sentiment_analysis_script], check=True)
        print("Sentiment Analysis completed successfully.\n")

        # Step 2: Run Sentiment Analysis
        print("Running GPT...")
        subprocess.run(["python", language_model_standalone], check=True)
        print("GPT Analysis completed successfully.\n")

        # Step 3: Run the Linear Regression Model
        print("Running Linear Regression Model...")
        subprocess.run(["python", linear_regression_script], check=True)
        print("Linear Regression Model completed successfully.\n")

        # Step 4: Run the tensorflow_nn Model
        print("Running tensorflow_nn Model...")
        subprocess.run(["python", tensorflow_nn_script], check=True)
        print("tensorflow_nn completed successfully.\n")

        # Step 5: Run the tensorflow_nn Model
        print("formatting the info...")
        subprocess.run(["python", format_csv], check=True)
        print("csv formatted\n")


    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    run_pipeline()