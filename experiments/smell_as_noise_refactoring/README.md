# Code Smell as Semantic Noise: Refactoring Threshold Experiment

This experiment investigates whether code smells (e.g., poor variable naming) behave like "semantic noise" in diffusion language models. 

## Hypothesis
If code smells are akin to noise, then injecting random noise (masking) into the code and letting the model "denoise" it should preferentially fix the smell (restore it to a clean state) compared to already clean code.

We expect to see:
1.  **Lower Stability for Smelly Code**: Smelly tokens should be less likely to be preserved than clean tokens under the same noise level.
2.  **Refactoring Sweet Spot**: There should be an optimal noise level (e.g., 30% masking) where the model is most likely to refactor the smell into a better name.

## Directory Structure
- `run_experiment.py`: Main script to run the experiment.
- `analyze_results.py`: Script to analyze the results and generate plots.
- `utils.py`: Helper functions for model loading and data processing.
- `results/`: Directory where results and plots will be saved.

## Usage

1.  **Setup Environment**:
    Ensure you have `torch`, `transformers`, `pandas`, `seaborn`, `matplotlib` installed.
    You also need the `DiffuCoder` model weights (or `DreamCoder`).

2.  **Run Experiment**:
    ```bash
    python run_experiment.py
    ```
    This will:
    - Load the test dataset (`data/test_filtered_1024.csv`).
    - For each sample, create a "Smelly" version (inject bad names) and a "Clean" version (ground truth).
    - Apply random masking at various levels (10%, 30%, ..., 90%).
    - Run the diffusion model to fill in the masks.
    - Record whether the variable name was preserved (Stability) or changed to the ground truth (Refactoring).
    - Save results to `results/retention_rate_*.csv`.

3.  **Analyze Results**:
    ```bash
    python analyze_results.py
    ```
    This will:
    - Load the latest CSV from `results/`.
    - Plot the **Stability Curve** (Retention Rate vs. Noise Level).
    - Plot the **Refactoring Curve** (Refactoring Success vs. Noise Level).
    - Save plots to `results/*.png`.

## Configuration
You can modify `run_experiment.py` to change:
- `NOISE_LEVELS`: The range of noise to test.
- `TRIALS_PER_LEVEL`: Number of random masks per level (higher = more statistical significance).
- `LIMIT_SAMPLES`: Number of samples to run (set to `None` for full dataset).
