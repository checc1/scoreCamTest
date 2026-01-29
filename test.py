# pip install importlib_resources
import numpy as np
from utils import *
from cam.scorecam import *


test_data_path = "/Users/francescoaldoventurelli/Downloads/test_dataset.pt"
datatest = torch.load(test_data_path, weights_only=False)

model = torch.load("model_16_features_trained_20_epochs", weights_only=False, map_location='cpu')
model_dict = dict(type='ModifiedResNet18', arch=model, layer_name='features')
print(model_dict)
input_image = datatest[11][0]
input_ = apply_transforms(input_image)
if torch.cuda.is_available():
  input_ = input_.cuda()
predicted_class = model(input_).max(1)[-1]

model_scorecam = ScoreCAM(model_dict)

scorecam_map = model_scorecam(input_)
#basic_visualize(input_.cpu(), scorecam_map.type(torch.FloatTensor).cpu(),save_path='modifiedResNet18.png')

def show_cam_on_image_custom(input_tensor, heatmap, alpha=0.3):
    if input_tensor.ndim == 4:  # [B, C, H, W]
        input_tensor = input_tensor[0]
    input_tensor = input_tensor.detach().cpu().float()
    if torch.is_tensor(heatmap):
        heatmap = heatmap.detach().cpu().float()

    if hasattr(heatmap, "ndim"):
        if heatmap.ndim == 4:  # [B, C, H, W]
            heatmap = heatmap[0]
        if heatmap.ndim == 3:  # [C, H, W] or [H, W, C]
            # If it's channel-first, take first channel
            if heatmap.shape[0] in (1, 3) and heatmap.shape[0] != heatmap.shape[-1]:
                heatmap = heatmap[0]
            else:
                heatmap = heatmap[..., 0] if heatmap.shape[-1] == 1 else heatmap.mean(axis=-1)

    heatmap = np.copy(heatmap)
    heatmap = np.squeeze(heatmap)
    heatmap = np.clip(heatmap, 0, None)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    image = transforms.ToPILImage()(input_tensor)
    image_np = np.asarray(image).astype(np.float32) / 255.0
    cmap = plt.get_cmap("jet")

    colored = cmap(heatmap)[..., :3].astype(np.float32)

    overlay = (1 - alpha) * image_np + alpha * colored
    overlay = np.clip(overlay, 0, 1)

    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

show_cam_on_image_custom(input_image, scorecam_map)