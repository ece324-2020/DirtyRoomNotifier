import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device, writer, global_step):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    incr = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch',
              leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            print('GOT IMAGES', len(imgs), imgs[0].shape)
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()
            pbar.update()
            incr += 1

            step = int(str(global_step) + str(incr))
            writer.add_images('valid_images', imgs[:, :3, :, :], step)
            writer.add_images('valid_masks/true', true_masks, step)
            writer.add_images('valid_masks/pred',
                              torch.sigmoid(mask_pred) > 0.5, step)

    net.train()
    return tot / n_val
