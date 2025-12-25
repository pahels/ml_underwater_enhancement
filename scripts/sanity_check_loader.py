# scripts/sanity_check_loader.py
from datasets.uieb.dataloader import UIEBConfig, make_uieb_loader

cfg = UIEBConfig(root="datasets/uieb", split="train", size=256, limit=16)
loader = make_uieb_loader(cfg, batch_size=4, num_workers=0, shuffle=True)

raw, gt, names = next(iter(loader))
print("raw:", raw.shape, raw.min().item(), raw.max().item())
print("gt :", gt.shape, gt.min().item(), gt.max().item())
print("names:", list(names)[:3])
