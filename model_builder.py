# model_builder.py
import torch.nn as nn
import timm
import torch

def build_model(pretrained=True, dropout_rate=0.5, num_classes=8):
    model = timm.create_model('tiny_vit_21m_512.dist_in22k_ft_in1k', pretrained=pretrained)

    # Determine which attribute holds the patch embedding layer.
    if hasattr(model.patch_embed, 'proj'):
        attr_name = 'proj'
    elif hasattr(model.patch_embed, 'projection'):
        attr_name = 'projection'
    elif hasattr(model.patch_embed, 'conv'):
        attr_name = 'conv'
    elif hasattr(model.patch_embed, 'conv1'):
        attr_name = 'conv1'
    else:
        raise AttributeError("Could not find patch embedding layer.")

    # Retrieve the original layer
    orig_layer = getattr(model.patch_embed, attr_name)
    # If layer is a ConvNorm (wrapper), get underlying conv layer.
    if hasattr(orig_layer, 'conv'):
        conv_layer = orig_layer.conv
    else:
        conv_layer = orig_layer

    # Get weight shape from the underlying conv layer.
    C_out, _, k, k = conv_layer.weight.shape

    # Create new weight tensor for 4 channels.
    new_weight = torch.zeros(C_out, 4, k, k)
    new_weight[:, :3, :, :] = conv_layer.weight
    new_weight[:, 3:, :, :] = 0.0
    
    # Create a new Conv2d layer that accepts 4 channels.
    new_conv = nn.Conv2d(4, C_out, kernel_size=k, stride=conv_layer.stride, padding=conv_layer.padding)
    new_conv.weight.data = new_weight
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data

    # update the underlying conv layer inside the wrapper if applicable
    if hasattr(orig_layer, 'conv'):
        orig_layer.conv = new_conv
    else:
        # else replace the whole layer.
        setattr(model.patch_embed, attr_name, new_conv)

    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # Reduces spatial dimensions to (batch_size, 576, 1, 1)
        nn.Flatten(),             # Flattens to (batch_size, 576)
        nn.Dropout(dropout_rate),          # added dropout of 0.3, may need to tweak
        nn.Linear(model.head.in_features, num_classes)  # Final classification layer
    )

    return model
