# Real-Time Semantic Segmentation + Vision-based Heading Controller

## Overview
This repository contains the codebase for a student team project combining a semantic segmentation module with a vision-based heading controller. The structure is designed to support independent development of components, which can then be tested together in a central pipeline.

## Repository Structure
- `src/segmentation/`: Core logic for semantic segmentation (training, inference). You can add multiple model architectures inside the `src/segmentation/models/` directory for experimentation.
- `src/heading_controller/`: Core logic for the heading controller (training, inference). You can add multiple model architectures inside the `src/heading_controller/models/` directory for experimentation.
- `src/pipeline/`: Integration layer connecting both modules.
- `notebooks/`: Contains `main.ipynb` for central experimentation and testing.
- `data/`: Datasets (raw and processed) should be placed here.
- `checkpoints/`: Saved model weights and training checkpoints.
- `results/`: Output figures, logs, and evaluation metrics.
- `docs/`: Internal planning and documentation notes.

## Team Workflow
- Work on core models and logic independently in the `src/` directory.
- Avoid writing complex logic directly inside notebooks. Instead, import from `src/`.
- Use `notebooks/main.ipynb` to import and experiment with modules together.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Download and prepare the dataset by running: `python src/pipeline/prepare_dataset.py`
3. Use `notebooks/main.ipynb` or the training scripts in `src/` to begin experiments.
