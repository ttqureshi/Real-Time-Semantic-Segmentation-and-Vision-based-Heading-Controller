# Internal Team Plan

## Objective
Develop and integrate a semantic segmentation model with a vision-based heading controller.

## Suggested Role Split
- **Member 1**: Semantic segmentation models (architecture, training, inference)
- **Member 2**: Heading controller models (architecture, training, inference)
- **Member 3**: Pipeline integration and end-to-end inference flow (`src/pipeline`)
- **Member 4**: Notebook orchestration (`main.ipynb`), results tracking, and documentation

## Collaboration Rules
- **Core logic in `src/`**: Avoid writing complex logic directly inside notebooks. Place it in Python scripts and import it.
- **Commit frequently**: Commit often with clear messages.
- **Organization**: Keep outputs, figures, and checkpoints neatly organized in their designated folders.
- **Shared Files**: Coordinate before changing shared files like `main.ipynb` or `run_pipeline.py`.

## Immediate Next Steps
1. Finalize the dataset format and distribution.
2. Assign initial baseline models for each task.
3. Implement the first training and inference skeletons.
4. Connect both modules through the integration pipeline.
