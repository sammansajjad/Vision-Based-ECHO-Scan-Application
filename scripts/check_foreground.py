import torch, numpy as np
from cv_pipeline.camus_dataset import CAMUSDataset
from cv_pipeline.models.unet import UNet

ck = torch.load("outputs/checkpoints/unet_4ch_best_full_validation.pth",
                map_location="cpu", weights_only=True)
m = UNet(); m.load_state_dict(ck["model_state_dict"]); m.eval()
ds = CAMUSDataset(split="validation", view="4CH", phases=("ED", "ES"))
fg, gt = [], []
for i in range(len(ds)):
    s = ds[i]
    x = torch.from_numpy(s["image"])[None, None]
    with torch.no_grad():
        p = (torch.sigmoid(m(x)) >= 0.5).numpy()[0, 0]
    fg.append(p.mean()); gt.append(s["mask"].mean())
fg, gt = np.array(fg), np.array(gt)
print("Predicted FG fraction mean/min/max:", fg.mean(), fg.min(), fg.max())
print("Ground-truth fraction mean/min/max:", gt.mean(), gt.min(), gt.max())
for i in range(4):
    print(ds[i]["patient_id"], ds[i]["phase"], round(float(fg[i]), 3), round(float(gt[i]), 3))