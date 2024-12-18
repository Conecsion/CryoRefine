import torch

def rotate2D(img: torch.Tensor, angleDegree: torch.Tensor) -> torch.Tensor:
    '''
    Rotate the input img by angleDegree anti-clockwise
    img shape (N, C, H, W)
    angleDegree shape (N,)
    return shape (N, C, H, W)
    '''
    angleRadian = torch.deg2rad(angleDegree)
    batchSize = img.shape[0]

    # Construct the affine matrices with shape (N, 2, 3)
    affineMatrices = torch.zeros(batchSize, 2, 3)
    affineMatrices[:, 0, 0] = torch.cos(angleRadian)
    affineMatrices[:, 0, 1] = -torch.sin(angleRadian)
    affineMatrices[:, 1, 0] = torch.sin(angleRadian)
    affineMatrices[:, 1, 1] = torch.cos(angleRadian)
    # No translation is applied
    affineMatrices[:, 0, 2] = 0
    affineMatrices[:, 1, 2] = 0

    rotatedGrid = torch.nn.functional.affine_grid(affineMatrices, img.shape)
    rotatedImg = torch.nn.functional.grid_sample(img, rotatedGrid, mode='bilinear')

    return rotatedImg

def addNoise(img:torch.Tensor, noiseRatio:float):
    """
    Add Gaussian noise to the img
    img shape (N, C, H, W)
    noiseRatio: the ratio of the noise to the original image
    return shape (N, C, H, W)
    """
    noise = torch.randn_like(img)
    return torch.nn.InstanceNorm2d(num_features=1)(img + noiseRatio * noise)

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    #     fig, axes = plt.subplots(1,3)
    img = cv2.imread('java.png', cv2.IMREAD_GRAYSCALE)
    #     originalImg = img
    #     axes[0].imshow(originalImg, cmap='gray')
    #
    imgTensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    rotatedImg = rotate2D(imgTensor, torch.tensor(70.,))

    fig, axes = plt.subplots(2)
    axes[0].imshow(img, cmap='gray')
    axes[1].imshow(rotatedImg.squeeze(0).squeeze(0).numpy(), cmap='gray')
    plt.show()
