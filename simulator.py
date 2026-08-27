import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import joblib

# ---------------- Model ----------------

class ParamsToImage(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(n_features,128),
            nn.ReLU(),
            nn.Linear(128,128*5*7),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64,32,4,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32,16,4,2,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16,8,4,2,1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8,3,3,padding=1)
        )

    def forward(self,x):

        x=self.fc(x)
        x=x.view(-1,128,5,7)
        x=self.deconv(x)

        x=torch.nn.functional.interpolate(
            x,
            size=(150,200),
            mode="bilinear",
            align_corners=False
        )

        return torch.sigmoid(x)


# ---------------- Load ----------------

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=ParamsToImage(7)

model.load_state_dict(
    torch.load(
        r"D:\New folder\blackhole\Notebooks\blackhole_generator.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

scaler=joblib.load(r"D:\New folder\blackhole\Notebooks\blackhole_scaler.pkl")

# ---------------- Dataset ----------------

df=pd.read_csv(r"D:\New folder\blackhole\test_dataset\labels.csv")

df=df.drop(columns=["num_extra_bodies"])

features=df.drop(columns=["filename"])

FEATURES=list(features.columns)

mins=features.min()
maxs=features.max()

values=[]

for col in FEATURES:
    values.append((mins[col]+maxs[col])/2)

# indices

IDX_MASS=FEATURES.index("bh_mass_kg")
IDX_RADIUS=FEATURES.index("schwarzschild_radius_m")
IDX_AZ=FEATURES.index("camera_azimuth_rad")
IDX_EL=FEATURES.index("camera_elevation_rad")
IDX_DIST=FEATURES.index("camera_distance_m")
IDX_R1=FEATURES.index("disk_r1_m")
IDX_R2=FEATURES.index("disk_r2_m")

# camera starts facing front

values[IDX_AZ]=0

# ---------------- Prediction ----------------

def predict():

    x=np.array([values],dtype=np.float32)

    x=scaler.transform(x)

    x=torch.tensor(x,dtype=torch.float32,device=device)

    with torch.no_grad():
        img=model(x)[0]

    return img.cpu().permute(1,2,0).numpy()

# ---------------- Figure ----------------

fig=plt.figure(figsize=(12,8))

ax=plt.axes([0.05,0.18,0.55,0.75])

image=ax.imshow(predict())

ax.axis("off")

# ---------------- Sliders ----------------

slider_mass=Slider(
    plt.axes([0.68,0.22,0.25,0.03]),
    "Mass",
    mins["bh_mass_kg"],
    maxs["bh_mass_kg"],
    valinit=values[IDX_MASS],
    valfmt="%.2e"
)

slider_el=Slider(
    plt.axes([0.68,0.17,0.25,0.03]),
    "Elevation",
    mins["camera_elevation_rad"],
    maxs["camera_elevation_rad"],
    valinit=values[IDX_EL]
)

slider_r1=Slider(
    plt.axes([0.68,0.12,0.25,0.03]),
    "Disk Inner",
    mins["disk_r1_m"],
    maxs["disk_r1_m"],
    valinit=values[IDX_R1],
    valfmt="%.2e"
)

slider_r2=Slider(
    plt.axes([0.68,0.07,0.25,0.03]),
    "Disk Outer",
    mins["disk_r2_m"],
    maxs["disk_r2_m"],
    valinit=values[IDX_R2],
    valfmt="%.2e"
)

# ---------------- Update ----------------

def update(val=None):

    values[IDX_MASS]=slider_mass.val
    values[IDX_EL]=slider_el.val
    values[IDX_R1]=slider_r1.val
    values[IDX_R2]=slider_r2.val

    image.set_data(predict())

    fig.canvas.draw_idle()

slider_mass.on_changed(update)
slider_el.on_changed(update)
slider_r1.on_changed(update)
slider_r2.on_changed(update)

# ---------------- Automatic Rotation ----------------

def animate(frame):

    values[IDX_AZ]+=0.02

    if values[IDX_AZ]>maxs["camera_azimuth_rad"]:
        values[IDX_AZ]=mins["camera_azimuth_rad"]

    image.set_data(predict())

    return [image]

ani=FuncAnimation(
    fig,
    animate,
    interval=40,
    blit=False
)

plt.show()