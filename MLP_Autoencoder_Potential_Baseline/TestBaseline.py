import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Baseline_Dataset import Load_Set
from Autoencoder_Class import Autoencoder

# Load Testing Data
smallSetImg, smallSetMsk = Load_Set(overfit_set = True)
img = smallSetImg[9]
mask = smallSetMsk[9]

model = Autoencoder()
model.load_state_dict(torch.load('ModelBatch10_0.0001.pt'))
model.eval()
predict = model(img, 1).detach().squeeze(0)
loss_fnc = nn.BCEWithLogitsLoss()
loss = loss_fnc(predict, mask)
print('Loss: ', loss)

# Transform images to numpy
img = img.numpy()
mask = mask.numpy()
predict = predict.numpy()

# Display Image and Predicted Mask
plt.figure(figsize=(3, 1))
plt.subplot(1, 3, 1)
plt.imshow(np.transpose(img, (1, 2, 0)))
plt.subplot(1, 3, 2)
plt.imshow(np.transpose(mask, (1, 2, 0)))
plt.subplot(1, 3, 3)
plt.imshow(np.transpose(predict, (1, 2, 0)))
plt.show()

