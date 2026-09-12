# mle-argoverse-2

PYTHONPATH=src python src/train/train_baseline.py \
  --data-dir data/raw/traffic_0.15_accident_0_steps_1000 \
  --epochs 10 --batch-size 8 --num-workers 4 \
  --checkpoint-dir checkpoints/baseline_residual


## Objective

Learn an action-conditioned world model of a driving scene: given a short history of
forward-facing camera frames, the current ego state, and the action the ego vehicle is
about to take, predict the next camera frame.

The model answers "what will the camera see next if I take this action?", which is the
building block for model-based planning and for imagining rollouts without touching the
simulator.

## Data

Episodes are collected from MetaDrive with an IDM policy driving the ego vehicle
(`src/data_creation/config.yaml` controls resolution, traffic density, accident
probability, and episode length). Each episode is stored as its own directory under
`data/raw/<spec>/episode_XXX/`:

- `images.npy` — `[num_steps, height, width, 3]` RGB frames from the ego camera
- `states.npy` — `[num_steps, state_dim]` MetaDrive
  [state observation](https://github.com/metadriverse/metadrive/blob/main/metadrive/obs/state_obs.py#L30)
  (speed, steering, heading, lane geometry, ...)
- `actions.npy` — `[num_steps, 2]` steering and throttle/brake applied at that step
- `meta.json` — episode metadata, including `num_steps`

A training datapoint is a sliding window over one episode: `window_size` consecutive
frames as history, the state and action at the last history frame as the condition, and
the following frame as the target. Windows never cross episode boundaries, and the
train/validation split is done at the episode level so that no episode contributes to
both splits.

The dataset is loaded lazily (`EpisodeFrameWindowDataset_V2`): the index only holds file
paths and window offsets, and frames are read on demand through memory-mapped
`np.load`, so full episodes are never held in RAM. `EpisodeFrameWindowDataset_V1` is the
earlier eager variant kept for comparison;
`local_script_tests/loader_memory_usage.py` profiles both across worker counts for
iteration time and peak memory.

## Approach

The model (`src/models/baseline.py`) is an encoder → condition → decoder pipeline built
from small composable modules rather than one monolithic network:

1. `FrameEncoder` — strided convolutions with group norm, pooled to a fixed grid and
   projected to a per-frame embedding. Applied to every frame in the history, and its
   per-stage feature maps for the last frame are kept as decoder skips.
2. `TemporalEncoder` — a GRU over the sequence of frame embeddings, summarising the
   recent past (and therefore the current motion) into a single latent.
3. `ConditionEncoder` — an MLP over the concatenated ego state and next action.
4. `FrameDecoder` — the history latent and condition latent are concatenated and decoded
   by progressive upsampling and convolutions, fusing the encoder skips so texture does
   not have to pass through the latent bottleneck.

The decoder predicts a **residual** that is added to the last observed frame rather than
the frame itself, and its output convolution is zero-initialised. A zero output is
therefore exactly the copy-last-frame prediction, so the model starts at that baseline
and spends its capacity on what changes between frames instead of re-synthesising the
static scene every step. This is standard practice in action-conditioned prediction
(residual anchoring in
[Diffusion Transformer World-Action Model for AV Scene Prediction](https://arxiv.org/abs/2606.12987),
delta tokens in [DeltaWorld](https://arxiv.org/abs/2604.04913), motion/content
decomposition in [MCnet](https://arxiv.org/abs/1706.08033)). Predictions are returned
unclamped; clamp to [0, 1] before rendering or scoring.

Conditioning inputs are cleaned up before they reach the network: recorded actions are
clipped to [-1, 1] (the raw policy output is unbounded, but MetaDrive clips before
applying it, so only the clipped value drove the simulator), and state vectors are
standardised per dimension using statistics fitted on the training episodes only, with
zero-variance dimensions mapped to zero rather than amplified.

## Loss

In words: the loss is the mean absolute error, averaged over every pixel, colour channel
and sample in the batch, between the predicted next frame and the true next frame. All
three submodules are trained jointly end-to-end from this single reconstruction term —
no staged pretraining and no auxiliary regulariser, since the reconstruction target
itself prevents representation collapse. Because the prediction is the last observed
frame plus a predicted residual, minimising it is the same as supervising the residual
against the true frame-to-frame difference: the model is penalised only for getting the
*change* wrong, not for failing to redraw the static scene.

Write a window of $k$ observed frames as $x_{t-k+1}, \dots, x_t$ with
$x_i \in [0, 1]^{C \times H \times W}$, the ego state at the last observed frame as
$s_t$, and the action applied there as $a_t$. The network $\Delta_\theta$ predicts a
residual and the prediction is

$$\hat{x}_{t+1} = x_t + \Delta_\theta\left(x_{t-k+1:t},\ s_t,\ a_t\right),
\qquad \Delta_\theta \in [-1, 1]^{C \times H \times W}$$

and the training objective over a batch $\mathcal{B}$ is

$$\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|\,C H W} \sum_{b \in \mathcal{B}}
\sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W}
\left| \hat{x}^{(b)}_{t+1}[c, i, j] - x^{(b)}_{t+1}[c, i, j] \right|$$

Substituting the residual parameterisation makes the equivalence explicit:

$$\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|\,C H W} \sum_{b \in \mathcal{B}}
\left\| \Delta_\theta\left(x^{(b)}_{t-k+1:t},\ s^{(b)}_t,\ a^{(b)}_t\right)
- \underbrace{\left(x^{(b)}_{t+1} - x^{(b)}_{t}\right)}_{\text{true frame difference}}
\right\|_1$$

So $\Delta_\theta = 0$ gives exactly the copy-last-frame baseline, whose loss is the
mean absolute inter-frame difference of the data. On this dataset that is 0.0164 on the
validation episodes, which is the number any trained model has to beat.

The absolute value (L1) rather than a square (L2) is deliberate: its optimum is the
per-pixel median rather than the mean of plausible futures, so it blurs somewhat less
under uncertainty.

## Training

`src/train/train_model.py` holds a model-agnostic loop (AdamW, per-epoch train and
validation passes, early stopping, best/last checkpoints, `loss_history.json`, optional
Weights & Biases logging). `src/train/train_baseline.py` wires the baseline model and
the dataloaders into it with the L1 objective above (`nn.L1Loss`).

```bash
python3 src/train/train_baseline.py \
  --data-dir data/raw/traffic_0.15_accident_0_steps_1000 \
  --epochs 1 --batch-size 8
```

## Evaluation

Every epoch scores the held-out episodes with `src/train/metrics.py` and reports, next to
the training loss: L1, the L1 of the copy-last-frame baseline on the same windows, the
**skill** ratio between them (below 1.0 means the model beats doing nothing), PSNR, and
SSIM. The skill ratio is the number to watch — an absolute L1 is uninterpretable here
because consecutive frames at 10 Hz are already nearly identical, so copying the last
frame is a strong predictor.

Note that PSNR and SSIM are distortion metrics and will prefer a blurry average over a
sharp but slightly misaligned prediction; judging sharpness needs a distribution metric
such as FID/KID, which is not implemented yet.

Qualitatively, side-by-side grids of last input frame / ground-truth next frame /
prediction:

```bash
python3 src/train/visualize_baseline.py \
  --checkpoint checkpoints/baseline/checkpoint_best.pt \
  --data-dir data/raw/traffic_0.15_accident_0_steps_1000
```

The main risk to watch is blur: under uncertainty a pixel-space L1/L2 objective hedges
by averaging plausible futures. If predictions look washed out, the planned mitigations
are a perceptual loss term and stronger action conditioning (tiling the action across
spatial feature maps at multiple scales instead of injecting it once at the bottleneck).

## Layout

```
src/data_creation/   MetaDrive episode collection, dataset, and dataloaders
src/models/          baseline world-model modules
src/train/           generic training loop, baseline entrypoint, visualisation
local_script_tests/  dataloader throughput and memory profiling
notebooks/           data inspection and output rendering
tests/               dataset/loader tests
```
