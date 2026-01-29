import torch
import numpy as np
from torchvision import transforms
from gradcam import GradCAMpp
import matplotlib.pyplot as plt
import torch.nn.functional as F


beta=0.7; imgClass=6; idx=2
test_data_path = "/Users/francescoaldoventurelli/Downloads/test_dataset.pt"
datatest = torch.load(test_data_path, weights_only=False)

cam_dict = dict()
model = torch.load("model_16_features_trained_20_epochs", weights_only=False, map_location='cpu')
model_dict = dict(type='ModifiedResNet18', arch=model, layer_name='reduce')

with np.load(f"/Users/francescoaldoventurelli/conda/scoreCamTest/newFeatureExtracted/class{imgClass}/Img_class_{imgClass}_idx_{idx}_beta_{beta}.npz") as data:
    inputTensor = data["input_tensor"]

#inputTensor = np.permute_dims(inputTensor, (2,1,0))
inputTensor = torch.unsqueeze(torch.from_numpy(inputTensor), 0)
resnet_model_dict = dict(type='resnet', arch=model, layer_name='reduce', input_size=(224, 224))
resnet_gradcampp = GradCAMpp(resnet_model_dict, True)

torch_img = inputTensor
torch_img = F.interpolate(torch_img, size=(224, 224), mode='bilinear', align_corners=False)

cam_dict['resnetModified18'] = resnet_gradcampp
mask_pp = resnet_gradcampp(torch_img)[0]


def show_cam_on_image_custom(input_tensor, heatmap, alpha=0.45):
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
    input_tensor = input_tensor.squeeze(0).detach().cpu().numpy()
    input_tensor = np.permute_dims(input_tensor, (1, 2, 0))
    image = transforms.ToPILImage()(input_tensor)
    image_np = np.asarray(image) / 255.0
    cmap = plt.get_cmap("jet")

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    colored = cmap(heatmap)[..., :3]

    overlay = (1 - alpha) * image_np + alpha * colored
    overlay = np.clip(overlay, 0, 1)

    plt.imshow(overlay)
    plt.axis("off")
    plt.savefig(f"/Users/francescoaldoventurelli/Downloads/horseGradCam++{idx}.png")
    plt.show()

show_cam_on_image_custom(inputTensor, mask_pp)