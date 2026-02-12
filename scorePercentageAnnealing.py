from matplotlib import pyplot as plt
import numpy as np
from torchvision import transforms
import torch.nn.functional as func
from PIL import Image
import torch
from qutip_class import SpinOperator
from src.feature_extraction_utils import FeatureExtractor
from collections import Counter
import os


beta = 0.7; F = 16; tau=100

def PercentageDrop(img: np.ndarray, mask: np.ndarray):

    with torch.no_grad():
        output1 = model(torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2))

    with torch.no_grad():
        output2 = model(torch.from_numpy(mask).unsqueeze(0).permute(0, 3, 1, 2))
    probs1, probs2 = torch.softmax(output1, dim=1), torch.softmax(output2, dim=1)
    cls = torch.argmax(probs1, dim=1).item()

    conf1 = probs1[0, cls].item()
    conf2 = probs2[0, cls].item()

    drop = max(0.0, (conf1 - conf2) / conf1)
    drop *= 100
    return drop

def bitstring_basis_msb(n_qubits: int) -> np.ndarray:
    num = 2**n_qubits
    arr = np.array([list(np.binary_repr(i, width=n_qubits)) for i in range(num)], dtype=int)
    return arr

def bitstring_basis_lsb(n_qubits: int) -> np.ndarray:
    msb = bitstring_basis_msb(n_qubits)
    return msb[:, ::-1]


#probs = coeff / coeff.sum()

def show_cam_on_image_custom(input_tensor, heatmap, alpha=0.6):
    image = transforms.ToPILImage()(input_tensor)
    heatmap = heatmap[0, 0]
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size, resample=Image.BILINEAR)
    heatmap = np.array(heatmap) / 255.0
    colormap = plt.cm.jet(heatmap)[..., :3]
    image_np = np.array(image) / 255.0
    overlay = (1 - alpha) * colormap + alpha * image_np
    overlay = np.clip(overlay, 0, 1)

    plt.imshow(overlay)
    plt.axis('off')
    plt.savefig("/Users/francescoaldoventurelli/Downloads/explMap.pdf")
    #plt.savefig(f"/Users/francescoaldoventurelli/Downloads/horse-feature5.svg")
    plt.show()


def grad_cam(features, gradients, subset_features=None):
    """
    Generate Grad-CAM heatmap from features and gradients.
    :param features: Output features from the target layer.
    :param gradients: Gradients from the backward pass.
    :return: Heatmap as a numpy array.
    """

    avg_gradients = gradients.mean(dim=(2, 3), keepdim=True)  # shape: [1, C, 1, 1]
    if subset_features is not (None):
        gradcam = torch.relu(avg_gradients[:, subset_features] * features[:, subset_features]).sum(dim=1,
                                                                                                   keepdim=True)  # shape: [1, 1, H, W]
        alpha = avg_gradients.squeeze()[subset_features]
    else:
        gradcam = torch.relu(avg_gradients[:, :] * features[:, :]).sum(dim=1, keepdim=True)  # shape: [1, 1, H, W]
        alpha = avg_gradients

    gradcam -= gradcam.min()
    gradcam /= (gradcam.max() - gradcam.min() + 1e-8)

    return gradcam.detach().cpu().numpy(), alpha.squeeze().detach().numpy()  # Convert to numpy array for visualization


path = "/Users/francescoaldoventurelli/conda/scoreCamTest/newFeatureExtracted"
imgClass = 9; idx = os.listdir(os.path.join(path, f"class{imgClass}"))
model = torch.load("/Users/francescoaldoventurelli/conda/scoreCamTest/Score-CAM/model_16_features_trained_20_epochs", weights_only=False, map_location=torch.device('cpu'))
dropList = []

for x in idx:

    with np.load(os.path.join(path, f"class{imgClass}", x), allow_pickle=True) as data:
        inputTensor = data["input_tensor"]
        nonzero_indices = data["nonzero_indices"]
        cosine_similarity = data["coupling"]
        alpha = data["linear"]

        idx_num = int(x.split("_idx_")[1].split("_beta_")[0])

        evState = np.load(
            f"/Users/francescoaldoventurelli/qml/projet_one/quantumStateEvolution/tau{tau}/class{imgClass}/states/"
            f"evState_idx_{idx_num}_class{imgClass}_beta_{beta}model{F}.npy"
        )

        spectrum = np.load(
            f"/Users/francescoaldoventurelli/qml/projet_one/quantumStateEvolution/tau{tau}/class{imgClass}/spectrums/"
            f"energies_idx_{idx_num}_class{imgClass}_beta_{beta}model{F}.npy"
        )

        Probs = np.load(
            f"/Users/francescoaldoventurelli/qml/projet_one/quantumStateEvolution/tau{tau}/class{imgClass}/probs/"
            f"probs_idx_{idx_num}_class{imgClass}_beta_{beta}model{F}.npy"
        )

    n_qubits = nonzero_indices.shape[0]
    inp = torch.from_numpy(inputTensor).float()
    if inp.shape[-1] == 3:
        inp = inp.permute(2, 0, 1)

    #inputTensorUnsqueezed = torch.unsqueeze(torch.from_numpy(inputTensor), 0)

    ham_zz = 0.
    for i in range(n_qubits):
        for j in range(n_qubits):
            ham_zz += SpinOperator([('pz', i, 'pz', j)], coupling=[cosine_similarity[i, j]], size=n_qubits,
                                   verbose=0).qutip_op

    ham_ext_z = 0.
    for i in range(n_qubits):
        ham_ext_z += SpinOperator([('pz', i)], coupling=[-alpha[i]], size=n_qubits, verbose=0).qutip_op

    ham_qubo = (1 - beta) * ham_zz + beta * ham_ext_z
    energies = ham_qubo.diag().real
    index = np.argsort(energies)


    extractor = FeatureExtractor()
    extractor.set_model(model)
    extractor.set_target_layer(model.reduce[-1])
    extractor.set_input_tensor(torch.tensor(inputTensor), see_picture=False)
    extractor.register_hooks()

    shots=2**n_qubits

    ### analytical -> The plot should tell analytically what you'd expect from the state evolution
    P = np.abs(evState)**2
    P = P / P.sum()

    ## Statistical: I sample a bitstring from the distribution n_shot times and I
    # would expect same behavior
    samples = np.random.choice(np.arange(len(P)), size=shots, p=P/np.sum(P))
    count=0
    counter = Counter(samples.tolist())

    gs_idx = index[0]

    allSol = {
        "counts": counter,
        "probs": {k: v / len(samples) for k, v in counter.items()}
    }


    sampleGs = gs_idx
    #pGs = allSol["probs"].get(gs_idx, 0.0)
    bitstringGs = nonzero_indices[
        np.nonzero(bitstring_basis_msb(n_qubits)[sampleGs])[0]
    ]

    counter = Counter(samples.tolist())
    sampleMostFrequent = counter.most_common(1)[0][0]
    bitstringMostFrequent = nonzero_indices[np.nonzero(bitstring_basis_msb(n_qubits)[sampleMostFrequent])[0]]
    #print("Gs", bitstringGs)
    #print("Most", bitstringMostFrequent)

    features, gradients, pred = extractor.extract_features()
    gradcam, _ = grad_cam(
        features=features.cpu(),
        gradients=gradients.cpu(),
        subset_features=bitstringMostFrequent,
    )
    inputTensorUnsqueezed = inp.unsqueeze(0)
    img = func.interpolate(inputTensorUnsqueezed, size=(224, 224), mode='bilinear', align_corners=False)
    cam = gradcam
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()

    cam = np.squeeze(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
    img_np = img[0].permute(1, 2, 0).detach().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    cam = np.array(
        Image.fromarray(np.uint8(255 * cam)).resize((img_np.shape[1], img_np.shape[0]), resample=Image.BILINEAR),
        dtype=np.float32
    ) / 255.0

    cam_3 = np.repeat(cam[:, :, None], 3, axis=2)
    masked_img = img_np * cam_3
    masked_img = np.clip(masked_img, 0, 1)
    p = PercentageDrop(img_np, masked_img)
    dropList.append(p)

dropList = np.array(dropList)
avg = np.mean(dropList)

print(avg)