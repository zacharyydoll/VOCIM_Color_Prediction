import torch.nn as nn
import timm
import torch
from config import (
    use_heatmap_mask, model_used, use_glan,
    glan_hidden_dim, glan_num_layers
)
from transformers import ResNetConfig, ResNetForImageClassification
from glan import GLAN
from torch_geometric.data import Data, Batch

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

        if use_glan:
            # Get the embedding dimension from the first attention block in stage 1
            # Stage 0 uses MBConv blocks, stage 1 onwards use attention blocks
            embed_dim = model.stages[1].blocks[0].attn.qkv.in_features
            
            # Create GLAN module
            glan = GLAN(
                node_dim=embed_dim,
                edge_dim=embed_dim,
                hidden_dim=glan_hidden_dim,
                num_layers=glan_num_layers
            )
            
            # Store GLAN in the model
            model.glan = glan
            
            # Modify forward_features to include GLAN
            original_forward_features = model.forward_features
            
            def new_forward_features(x):
                # Get features from TinyViT
                features = original_forward_features(x)
                
                # Convert features to graph format
                B, C, H, W = features.shape
                features = features.flatten(2).transpose(1, 2)  # B, N, C
                
                # Create graph data for each image in batch
                graph_data_list = []
                for b in range(B):
                    # Create fully connected graph
                    num_nodes = H * W
                    edge_index = []
                    edge_attr = []
                    for i in range(num_nodes):
                        for j in range(num_nodes):
                            edge_index.append([i, j])
                            # Compute edge attributes based on feature similarity
                            edge_attr.append(torch.cosine_similarity(
                                features[b, i].unsqueeze(0),
                                features[b, j].unsqueeze(0)
                            ))
                    # Move tensors to the same device as input
                    edge_index = torch.tensor(edge_index, dtype=torch.long, device=x.device).t()
                    edge_attr = torch.tensor(edge_attr, dtype=torch.float32, device=x.device)
                    
                    # Create graph data
                    graph_data = Data(
                        x=features[b],  # N x C
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        batch=torch.zeros(num_nodes, dtype=torch.long, device=x.device)
                    )
                    graph_data_list.append(graph_data)
                
                # Process through GLAN
                batch = Batch.from_data_list(graph_data_list)
                graph_features = model.glan(batch)
                
                # Reshape back to original format
                graph_features = graph_features.view(B, H, W, -1).permute(0, 3, 1, 2)  # B, C, H, W
                return graph_features
            
            model.forward_features = new_forward_features

        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),            
            nn.Dropout(dropout_rate),         
            nn.Linear(model.head.in_features, num_classes) 
        )
        return model 
    
    else: 
        raise ValueError(f"Unknown model type specified in config: {model_used}")
    
