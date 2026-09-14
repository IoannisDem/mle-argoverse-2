# Next frame prediction - Small world model

## Project Objective

This project develops an action-conditioned world model for autonomous driving. Given a short sequence of camera frames, the ego vehicle's current state, and a planned action, the model predicts the next camera frame.

The goal is to learn how driving actions affect the visual scene, enabling model-based planning and simulated rollouts without repeatedly interacting with the driving simulator. Driving episodes are collected from MetaDrive, and the model is trained end-to-end to predict frame-to-frame changes using a residual prediction approach.

## Key Scripts

- `src/data_creation/generate_synthetic_data.py` — runs MetaDrive simulations and saves camera frames, vehicle states, actions, and episode metadata.
- `src/data_creation/loader.py` — loads episode data and creates sliding-window training samples from consecutive frames.
- `src/train/train_baseline.py` — prepares the datasets, builds the dataloaders, and trains the baseline model.
- `src/models/baseline.py` — defines the action-conditioned model that encodes frame history, state, and action to predict the next frame.

## Reference

For an example of well-structured, production-quality code, see the
[photo retrieval project](https://github.com/IoannisDem/photo_retrieval).
