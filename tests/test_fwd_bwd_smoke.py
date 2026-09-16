from nav.data_module import WorldModelDataModule
from nav.models.world_model import JepaWorldModel

data_dir = "/data/tdmpc/suyash"
csv_path = f"{data_dir}/sains-ladbroke-grove_karter-01_2026-05-21_03-35-26_ll_navigation_command_ll_navigation_command.csv"
bev_path = f"{data_dir}/derived/lidar_bev.npy"
bev_ts_path = f"{data_dir}/derived/lidar_bev_ts.npy"

dm = WorldModelDataModule(data_dir, csv_path, bev_path, bev_ts_path, cameras=("oakd_front","oakd_back"), batch_size=8, num_workers=2)
dm.setup()
batch = next(iter(dm.train_dataloader()))

model = JepaWorldModel(n_cameras=2).cuda()
batch = {k: v.cuda() for k, v in batch.items()}
metrics = model.shared_step(batch)
print({k: v.item() for k, v in metrics.items()})
loss = metrics["loss"]
loss.backward()
print("backward ok, loss=", loss.item())