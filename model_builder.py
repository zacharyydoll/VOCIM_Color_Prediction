import torch.nn as nn
import timm
import torch
from config import (
    use_heatmap_mask, model_used, use_glan,
    glan_hidden_dim, glan_num_layers, glan_dropout,
    num_classes
)
from transformers import ResNetConfig, ResNetForImageClassification
from color_gnn import ColorGNN, extract_frame_id
import torch.nn.functional as F
from collections import defaultdict

def build_model(pretrained=True, dropout_rate=0.5, num_classes=8, input_channels=3):
    if model_used.lower() == "resnet": 
        if use_heatmap_mask:
            # load default configs, then set num_channels to 4 (for the mask)
            config = ResNetConfig.from_pretrained('microsoft/resnet-50')
            config.num_channels = 4
            model = ResNetForImageClassification.from_pretrained(
                'microsoft/resnet-50',
                config=config,
                ignore_mismatched_sizes=True
            )
        else:
            model = ResNetForImageClassification.from_pretrained(
                'microsoft/resnet-50',
                ignore_mismatched_sizes=True
            )
        
        model.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        )
        return model

    elif model_used.lower() == "tinyvit":
        model = timm.create_model('tiny_vit_21m_512.dist_in22k_ft_in1k', pretrained=pretrained)

        if use_heatmap_mask:
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

            # retrieve original layer
            orig_layer = getattr(model.patch_embed, attr_name)
            # If layer is a ConvNorm (wrapper), get underlying conv layer.
            if hasattr(orig_layer, 'conv'):
                conv_layer = orig_layer.conv
            else:
                conv_layer = orig_layer

            # Get weight shape from the underlying conv layer.
            C_out, _, k, k = conv_layer.weight.shape

            # create new weight tensor for 4 channels.
            new_weight = torch.zeros(C_out, 4, k, k)
            new_weight[:, :3, :, :] = conv_layer.weight
            new_weight[:, 3:, :, :] = 0.0
            
            # create new Conv2d layer that accepts 4 channels.
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

        # Replace the head with our custom head
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),            
            nn.Dropout(dropout_rate),         
            nn.Linear(in_features, num_classes) 
        )

        # Create ColorGNN for post-processing during inference
        if use_glan:
            model.color_gnn = ColorGNN(
                num_colors=num_classes,
                hidden_dim=glan_hidden_dim,
                num_layers=glan_num_layers,
                dropout=glan_dropout
            )
            
            # Store original forward method
            original_forward = model.forward
            
            def new_forward(self, x, image_paths=None, use_gnn=False):
                # Get logits from TinyViT
                logits = original_forward(x)
                
                # During training, return raw logits
                if not use_gnn or image_paths is None:
                    return logits
                
                # During inference, apply GNN post-processing
                probs = F.softmax(logits, dim=1)
                
                # Group probabilities by frame ID
                frame_probs = {}
                for i, path in enumerate(image_paths):
                    frame_id = extract_frame_id(path)
                    if frame_id not in frame_probs:
                        frame_probs[frame_id] = []
                    frame_probs[frame_id].append((i, probs[i]))
                
                # Process each frame's birds together
                for frame_id, bird_probs in frame_probs.items():
                    # Sort by original index to maintain order
                    bird_probs.sort(key=lambda x: x[0])
                    indices, probs_list = zip(*bird_probs)
                    
                    # Stack probabilities for this frame
                    frame_probs_tensor = torch.stack(probs_list)
                    
                    # Get GNN assignments for this frame
                    assignments = self.color_gnn(frame_probs_tensor)
                    
                    # Update logits based on GNN assignments
                    for idx, assigned_color in zip(indices, assignments):
                        # Set assigned class to high value, others to low value
                        logits[idx] = -1000.0  # Reset all logits to very negative
                        logits[idx, assigned_color] = 1000.0  # Set assigned class to very positive
                
                return logits
            
            # Bind the new forward method to the model
            model.forward = new_forward.__get__(model)
            
        return model
    
    else: 
        raise ValueError(f"Unknown model type specified in config: {model_used}")
    
