# nav

Dataloaders and a JEPA-style world model for TD-MPC, trained on recorded RGB, depth and 2D lidar
alongside commanded velocity.

## Data

Raw session data lives under `/data/model-training`: a `;`-separated nav-command CSV
(`timestamp_us`, `navigation_status`, `desired_linear_velocity`, `desired_angular_velocity`, ...),
one `oakd_back/images_0..3/` JPEG shard set, and four `lidar2d_{front,back}_{left,right}/*_points.csv`
files. Timestamps are int64 microseconds everywhere. Depth is expected under `depth_0..3/` shard
dirs with the same filename convention, once recorded on the training machine — the repo assumes
16-bit PNGs in millimetres at 640x400 (see `depth_loader.py`).

A "stretch" is a contiguous run of `navigation_status == "navigating"` rows; `chronological_split`
holds out whole trailing stretches for validation so train/val never interleave in time.

Lidar directories are named `lidar2d_*` but the `frame_id` inside each CSV (and the calibration key
in `frame_transformer.py`) is `lidar_*` — always key off `frame_id`, never the path.

## World model

Latent `z` is 128-d: 64 RGB + 32 depth + 32 lidar, concatenated. Encoders:

- **RGB** — frozen ImageNet ResNet-18 trunk, pooled, concatenated with a learned per-camera
  embedding, then a trainable linear head to 64-d.
- **Depth** — small conv net over (depth, validity-mask) to 32-d.
- **Lidar** — the four scans are merged into one robot-frame occupancy grid (±6m, 0.1m cells,
  120x120) and encoded with a small conv net to 32-d.

A latent MLP (`nav.models.dynamics.LatentDynamics`) takes `(z_t, action_t)` and predicts a residual
onto `z_t` to get `z_{t+1}`.

Training is self-supervised (`nav.models.world_model.JepaWorldModel`): online encoders produce
`z_t`, an EMA copy (stop-gradient, `tau=0.99`) produces the `z_{t+1}` target, and the dynamics MLP
is trained to bridge them with a smooth-L1 loss. Watch two numbers in the logs, not just the loss:

- `val_loss / val_identity_loss` — `identity_loss` is `z_t` vs `z_{t+1}` directly. Consecutive
  latents 200ms apart are similar, so a model that learned nothing can still post a good-looking
  raw loss; only the ratio against this baseline says anything.
- `val_z_std` — mean per-dimension std of `z_t` across the batch. Falling toward zero means the
  encoders are collapsing.

### Anchors and alignment

Training samples are anchored on a synthetic 200ms grid seeded at each stretch start (not on raw
command timestamps, which jitter). For each anchor `t`, every modality is read back by its own
latency before `t` (RGB 130ms, depth 150ms, lidar 100ms) and matched to the nearest real reading
within a tolerance; samples where any modality misses tolerance are dropped. The lagged sensor
reading is not required to be in-stretch itself — it's the observation the controller actually
acted on at `t`, and requiring it would discard real data for no physical reason.

Commanded action is the mean of `desired_linear_velocity` / `desired_angular_velocity` over
`[t, t+200ms)` (2 commands at the nav command's ~10Hz), normalized per-component using train-split
statistics computed after the chronological split.

## Running

```
uv run python scripts/preprocess_lidar.py     # one-time: rasterizes lidar to derived/lidar_bev*.npy
uv run python -m nav.data_module              # sanity-check shapes and sample counts
uv run python scripts/train_world_model.py
```
