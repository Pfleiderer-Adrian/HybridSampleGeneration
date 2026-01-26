from models.VAE_ConvNeXt_2D import ConvNeXtVAE2D, Config as ConvNeXtVAE2D_Config
from models.VAE_ConvNeXt_3D import ConvNeXtVAE3D, Config as ConvNeXtVAE3D_Config
from models.VAE_ResNet_2D import ResNetVAE2D, Config as ResNetVAE2D_Config
from models.VAE_ResNet_3D import ResNetVAE3D, Config as ResNetVAE3D_Config

def model_loader(model_name, params):
    model = None
    if model_name == "VAE_ResNet_3D":
        model = ResNetVAE3D(ResNetVAE3D_Config(**params))
    elif model_name == "VAE_ResNet_2D":
        model = ResNetVAE2D(ResNetVAE2D_Config(**params))
    elif model_name == "VAE_ConvNeXt_3D":
        model = ConvNeXtVAE3D(ConvNeXtVAE3D_Config(**params))
    elif model_name == "VAE_ConvNeXt_2D":
        model = ConvNeXtVAE2D(ConvNeXtVAE2D_Config(**params))  
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model