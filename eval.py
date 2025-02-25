from PIL import Image
import torch
from torch.optim import AdamW
from transformers import ResNetForImageClassification
from model import Trainer
from dataset import ImageDataset
from dataloader import get_eval_dataloder, get_train_dataloder

def main(eval_json_data, img_dir = '/mydata/vocim/shared/KeypointAnnotations'):
    # train_json_data = 'data/top_train_cls_mmdet_vocim.json'
    # eval_json_data = 'data/top_val_cls_mmdet_vocim.json'
    # img_dir = '/mydata/vocim/xiaoran/scripts/mmpose/data/vocim/images'
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')
    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=2048, out_features=8, bias=True))
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    #train_loader = get_train_dataloder(train_json_data, img_dir, batch_size=batch_size)
    eval_loader = get_eval_dataloder(eval_json_data, img_dir, batch_size=batch_size)

    trainer = Trainer(model = model, loss = criterion, optimizer = optimizer, device = device)
    trainer.load_model(ckpt='top_colorid_best_model.pth')
    trainer.evaluate(eval_loader, json_filename='output_top.pkl')

if __name__=="__main__":
    eval_json_data='data/newdata_cls_test_vidsplit.json'
    main(eval_json_data = eval_json_data)




