from PIL import Image
import torch
import torch.nn as nn
import timm 
from torch.optim import AdamW
from model import Trainer
from dataset import ImageDataset
from dataloader import get_eval_dataloder, get_train_dataloder
from config import batch_size, num_epochs, dropout_rate, learning_rate, weight_decay, scheduler_factor, scheduler_patience, num_classes, model_name
from model_builder import build_model


def main(eval_json_data, img_dir = '/mydata/vocim/zachary/data/cropped'):
    eval_batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating model...")
    model = build_model(pretrained=True, dropout_rate=dropout_rate, num_classes=num_classes)
    model = model.to(device)
    print("Model created.")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    eval_loader = get_eval_dataloder(eval_json_data, img_dir, batch_size=eval_batch_size, num_workers=0)
    #print("Number of evaluation batches: ", len(eval_loader))

    trainer = Trainer(model=model, loss=criterion, optimizer=optimizer, device=device)
    loaded_acc = trainer.load_model(ckpt='/mydata/vocim/zachary/color_prediction/ResNet50_no_mask/top_colorid_best_model.pth')
    print(f"Loaded checkpoint with best accuracy: {loaded_acc}")

    #for debugging 
    #for i, sample in enumerate(eval_loader):
    #    print(f"Batch {i} contains {sample['image'].size(0)} samples.")
    #    break

    print("Evaluating model...")
    trainer.evaluate(eval_loader, json_filename='output_top.pkl')
    print("Evaluation complete.")

if __name__=="__main__":
    eval_json_data='/mydata/vocim/zachary/color_prediction/data/mult_bkpk_sub_test_set.json' # replace with test set 
    main(eval_json_data = eval_json_data)