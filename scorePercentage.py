from matplotlib import pyplot as plt
import numpy as np
import torch
from gradcam import GradCAMpp
import torch.nn.functional as func
import os


def visualize():
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1); plt.imshow(img_np); plt.title("Original"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(masked_img); plt.title("Masked (soft)"); plt.axis("off")
    plt.tight_layout()
    plt.show()

def PercentageDrop(img: np.ndarray, mask: np.ndarray):

    with torch.no_grad():
        output1 = model(torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2))

    with torch.no_grad():
        output2 = model(torch.from_numpy(mask).unsqueeze(0).permute(0, 3, 1, 2))
    probs1, probs2 = torch.softmax(output1, dim=1), torch.softmax(output2, dim=1)
    cls = torch.argmax(probs1, dim=1).item()

    conf1 = probs1[0, cls].item()
    conf2 = probs2[0, cls].item()

    drop = (conf1 - conf2) / (conf1 + 1e-12) * 100.0
    return drop


path = "/Users/francescoaldoventurelli/qml/gradcamPP/newFeatureExtracted"
test_data_path = "/Users/francescoaldoventurelli/Downloads/test_dataset.pt"
datatest = torch.load(test_data_path, weights_only=False)
imgClass = 9; idx = os.listdir(os.path.join(path, f"class{imgClass}")); beta=0.7
model = torch.load("/Users/francescoaldoventurelli/qml/gradcamPP/model_16_features_trained_20_epochs", weights_only=False, map_location=torch.device('cpu'))
print(idx)
dropList = []
for x in idx:
    with np.load(os.path.join(path, f"class{imgClass}", x), allow_pickle=True) as data:
        inputTensor = data["input_tensor"]
        inputTensor = torch.unsqueeze(torch.from_numpy(inputTensor), 0)

    cam_dict = dict()
    masked_input = inputTensor.clone()
    resnet_model_dict = dict(type='resnet', arch=model, layer_name='reduce', input_size=(224, 224))
    resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
    torch_img = func.interpolate(inputTensor, size=(224, 224), mode='bilinear', align_corners=False)

    cam_dict['resnetModified18'] = resnet_gradcampp
    mask_pp = resnet_gradcampp(torch_img)[0]

    img = func.interpolate(inputTensor, size=(224, 224), mode='bilinear', align_corners=False)
    cam = mask_pp
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()

    cam = np.squeeze(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
    img_np = img[0].permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    cam_3 = np.repeat(cam[:, :, None], 3, axis=2)
    masked_img = img_np * cam_3
    masked_img = np.clip(masked_img, 0, 1)


    p = PercentageDrop(img_np, masked_img)
    dropList.append(p)

dropList = np.array(dropList)
avg = np.mean(dropList)
print(avg)