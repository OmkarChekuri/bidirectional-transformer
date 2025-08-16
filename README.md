# Bidirectional Language Translation with a Transformer

This project implements a character-level Transformer model for bidirectional language translation between English and Sanskrit. The core innovation is a single model that can translate in both directions by leveraging a unified vocabulary and special **direction tokens**. The project also demonstrates a robust MLOps approach for handling large datasets and ensuring reproducibility.

## Project Overview

The Transformer is trained from scratch on a small, aligned dataset of English and Sanskrit sentences. It learns to translate from a source to a target language by recognizing a special direction token prepended to the input sequence (e.g., `[<eng_to_sanskrit>]`).

## Folder Structure

```
.
├── configs/
│   └── config.yaml          # Hyperparameters and model settings
├── src/
│   └── main.py              # Main training and inference script
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── .github/                 # GitHub Actions CI/CD workflows
│   └── workflows/
│       └── ci.yml
└── jenkins/                 # Jenkins Pipeline file for Continuous Deployment
    └── Jenkinsfile


```

## Setup and Installation

### Prerequisites

* Python 3.8+

* A CUDA-enabled GPU (optional, but recommended)

### Local Environment Setup

1. **Navigate to your projects directory:**

   ```
   cd /path/to/MLPractice/GithubProjects/bidirectional-transformer
   
   
   ```

2. **Activate your virtual environment:**

   ```
   source ../sklearn-pytorch-env-py312/bin/activate
   
   
   ```

3. **Install dependencies:**

   ```
   pip install -r requirements.txt
   
   
   ```

### Running the Training Script

The `main.py` script will automatically create dummy data files, train the model, and perform a few test translations.

```
python src/main.py


```

### Solving the Unicode Encoding Error

You may encounter a `UnicodeEncodeError` when running the script in a terminal (like PowerShell) that doesn't natively support UTF-8. This is a display issue, not a bug in your code.

**Solution:** Set the `PYTHONIOENCODING` environment variable to `UTF-8` before running the script.

```
# In your PowerShell terminal
$env:PYTHONIOENCODING="UTF-8"
python src/main.py


```

This ensures that the console can correctly display Unicode characters like the Sanskrit text.

## MLOps & Reproducibility

To ensure a professional and reproducible workflow, this project is built with MLOps best practices.

### Dockerization

A `Dockerfile` is included to containerize the entire environment. This guarantees that the project runs consistently across different machines.

```
# Build the Docker image
docker build -t bidirectional-transformer:latest .

# Run the training script inside the container
docker run --gpus all bidirectional-transformer:latest python3.10 src/main.py


```

### CI/CD with GitHub Actions

The `.github/workflows/ci.yml` file defines a CI pipeline to automatically run linting and test the training script on every push, ensuring code quality and functionality.

### Continuous Deployment (Jenkins)

The `jenkins/Jenkinsfile` provides a deployment script for a Jenkins server. This automates the process of building and pushing the Docker image for production deployment.
