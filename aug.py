import torchvision.transforms as transforms

# training data augmentation에 사용할 transform 정의
transform_input = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(180),
    transforms.RandomResizedCrop(256, scale=(0.75, 1.25)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # target에 적용 X
    transforms.GaussianBlur(kernel_size=5),  # target에 적용 X
    transforms.ToTensor()  # 이미지를 텐서로 나타냄
])
transform_target = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(180),
    transforms.RandomResizedCrop(256, scale=(0.75, 1.25)),
    transforms.ToTensor()  # 이미지를 텐서로 나타냄
])