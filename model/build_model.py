from model.model_fed import CNN, LeNet
from model.model_res import ResNet18, ResNet34, ResNet50
import torchvision.models as models
import torch.nn as nn

def build_model(args):
    """
    Builds the model architecture for FedSpectralGate.
    Note: For CIFAR datasets, ResNet architectures are highly recommended
    as they maintain a better signal-to-noise ratio in the frequency domain.
    """
    
    if args.model == 'cnn':
        # Simple CNN for lightweight testing
        netglob = CNN(args=args).to(args.device)
        
    elif args.model == 'lenet':
        netglob = LeNet().to(args.device)
        
    elif args.model == 'resnet18':
        # Ideal for CIFAR-10 FedSpectralGate experiments
        netglob = ResNet18(num_classes=args.num_classes)
        netglob = netglob.to(args.device)
        
    elif args.model == 'resnet34':
        # Better for CIFAR-100 where classification is harder
        netglob = ResNet34(num_classes=args.num_classes)
        netglob = netglob.to(args.device)
        
    elif args.model == 'resnet50':
        netglob = ResNet50(num_classes=args.num_classes)
        netglob = netglob.to(args.device)
          
    elif args.model == 'resnext':
        netglob = models.resnext50_32x4d(num_classes=args.num_classes)
        netglob = netglob.to(args.device)

    elif args.model == 'vgg16':
        # VGG models often have high spectral noise in FC layers
        netglob = models.vgg16(num_classes=args.num_classes)
        netglob = netglob.to(args.device)

    else:
        exit(f'Error: unrecognized model {args.model}.')

    return netglob