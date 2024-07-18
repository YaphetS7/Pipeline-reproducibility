import torch
import torch.nn as nn
import torch.optim as optim
from random import randint

from torchvision import transforms
import torchvision

import numpy as np

import random
import os

from torch.utils.data import Dataset, DataLoader

from torch.optim.lr_scheduler import ReduceLROnPlateau
import imgaug.augmenters as iaa
import imgaug as ia
from config import path2save, SEED, LOADER_SEED, batch_size, epochs



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ia.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    iaa.SomeOf((0, 5),
    [
        iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),
        sometimes(iaa.AdditiveLaplaceNoise(scale=(0, 0.2*255))),
        iaa.CoarseDropout((0.0, 0.05), size_percent=(0.25, 0.35)),
        iaa.JpegCompression(compression=(50, 75)),
        sometimes(iaa.SaltAndPepper(0.1, per_channel=True)),
        iaa.MotionBlur(k=15),
        sometimes(iaa.GaussianBlur(sigma=(0.0, 3.0))),
        sometimes(iaa.RemoveSaturation(0.25)),
        sometimes(iaa.AddToHueAndSaturation((-10, 10), per_channel=True)),
        sometimes(iaa.LogContrast(gain=(0.6, 1.4)))
    ]),
    iaa.size.PadToSquare(),
    iaa.Rotate((0, 20)),
    iaa.size.Resize((224, 224))
])


data_transforms = {
    'train': transforms.Compose([
        seq.augment_image,
        transforms.ToPILImage(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        iaa.size.PadToSquare().augment_image,
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),   
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ]),
}





def create_model(n_classes):
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(1280, n_classes)
    
    return model



class DatasetCustom(Dataset):
    def __init__(self, transform=None):
    
        self.transform = transform

        
    def __len__(self):
        return 100

    def __getitem__(self, idx):
        label = randint(0, 1)
        face = np.random.rand(224, 224, 3).astype(np.uint8)
        if self.transform:
            face = self.transform(face)
            
        return (face, label)
    
def train():
    
    model.train()

    total_loss = 0.0

    total_preds=[]

    for step,batch in enumerate(train_loader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_loader)))
        batch = [r.to(device) for r in batch]
        
        images, labels = batch

        model.zero_grad() 
        
        preds = model(images)
        
        loss = criterion(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()
        preds=preds.detach().cpu().numpy()
        total_preds.append(preds)

    avg_loss = total_loss / len(train_loader)

    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

def evaluate(loader):

    print("\nEvaluating...")

    model.eval()

    total_loss = 0.0

    total_preds = []

    for step,batch in enumerate(loader):
    
        if step % 50 == 0 and not step == 0:
                
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(loader)))

        batch = [t.to(device) for t in batch]
        
        images, labels = batch

        with torch.no_grad():
    
            preds = model(images)

            loss = criterion(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

    avg_loss = total_loss / len(loader) 

    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds



if __name__ == '__main__':

    print(torch.version.cuda)
    setup_seed(SEED)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)    

    train_set = DatasetCustom(data_transforms['train'])
    val = DatasetCustom(data_transforms['test'])
    test = DatasetCustom(data_transforms['test'])




    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, worker_init_fn=seed_worker, 
            generator=torch.Generator().manual_seed(LOADER_SEED))
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(LOADER_SEED))
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(LOADER_SEED))


    model = create_model(2).to(device)



    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=2,verbose=True)
    criterion = nn.CrossEntropyLoss()





    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    valid_losses=[]
    test_losses=[]


    #for each epoch
    for epoch in range(epochs):
        
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
        train_loss, _ = train()

        valid_loss, _ = evaluate(val_loader)
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
        torch.save(model.state_dict(), f'{path2save}/1_mobnet_v2_' + str(epoch) + '.pt')

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        
            
        test_loss, preds = evaluate(test_loader)
        test_losses.append(test_loss)
        
        
            
        preds = np.argmax(preds, axis = 1)
        
        

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

    print(f'\nBest valid loss: {best_valid_loss:.3f}')