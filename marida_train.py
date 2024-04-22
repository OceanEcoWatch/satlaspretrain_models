import rasterio
import torch
import torch.nn

import satlaspretrain_models
from datasets.marida import MARIDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 1
criterion = torch.nn.CrossEntropyLoss()
val_step = 1

marida_train = MARIDA("data/MARIDA", split="train")
marida_val = MARIDA("data/MARIDA", split="val")

weights_manager = satlaspretrain_models.Weights()
model = weights_manager.get_pretrained_model(
    "Sentinel1_SwinB_MI",
    head=satlaspretrain_models.Head.SEGMENT,
    num_categories=16,
    device="cpu",
)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(num_epochs):
    print("Starting Epoch...", epoch)

    for i in range(len(marida_train)):
        with rasterio.open(marida_train.images[i]) as src:
            transform = src.transform
            crs = src.crs.copy()
            meta = src.meta.copy()
        print(meta)
        data = marida_train[i].copy()
        image = data["image"]
        mask = data["mask"]
        print("Image shape = ", image.shape)
        print("Mask shape = ", mask.shape)

        output, loss = model(image)
        print("Train Loss = ", loss)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Validation.
    if epoch % val_step == 0:
        model.eval()

        for i in range(len(marida_val)):
            with rasterio.open(marida_val.images[i]) as src:
                transform = src.transform
                crs = src.crs

            val_data = marida_val[i]
            val_image = val_data["image"]
            val_target = val_data["mask"]
            val_output, val_loss = model(val_image)

            val_accuracy = (
                (val_output.argmax(dim=1) == val_target).float().mean().item()
            )
            print("Validation accuracy = ", val_accuracy)
