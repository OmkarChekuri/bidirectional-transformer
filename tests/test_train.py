import os
import subprocess
import pytest

def test_training_script_runs_without_error():
    """
    Tests that the main training script runs to completion without crashing.
    This is a basic smoke test to validate the pipeline.
    """
    # Command to run the training script
    command = ['python', 'src/main.py']

    try:
        # Run the command and capture output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # Check if there are any errors in the stderr output
        assert "Error" not in result.stderr
        assert "Traceback" not in result.stderr
        # A more advanced test would check for a "Training Complete" message
        assert "Training Complete" in result.stdout
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Training script failed with an error: {e.stderr}")
    finally:
        # Clean up dummy files created by the script
        if os.path.exists('english.txt'):
            os.remove('english.txt')
        if os.path.exists('sanskrit.txt'):
            os.remove('sanskrit.txt')