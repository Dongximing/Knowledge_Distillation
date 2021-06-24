import torch
import tqdm
from tqdm import tqdm


def binary_accuracy(preds, y):

    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc
def train_fc(data_loader, device, model,optimizer, criterion,scheduler):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for bi,d in tqdm(enumerate(data_loader),total = len(data_loader)):
        ids = d['text']
        lengths = d['length']
        targets = d['target']
        ids = ids.to(device, dtype=torch.long)
        lengths = lengths.to(device, dtype=torch.int)
        targets = targets.to(device, dtype=torch.float).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(ids,lengths)
        loss = criterion(outputs, targets)
        acc = binary_accuracy(outputs, targets)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    scheduler.step()
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

def eval_fc(valid_loader, model, device, criterion):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for bi, d in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            ids = d['text']
            lengths = d['length']
            targets = d['target']

            ids = ids.to(device, dtype=torch.long)
            lengths = lengths.to(device, dtype=torch.int)
            targets = targets.to(device, dtype=torch.float).unsqueeze(1)

            outputs = model(ids, lengths)
            # print(outputs.shape)
            # print(targets.shape)
            loss = criterion(outputs, targets)



            acc = binary_accuracy(outputs, targets)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(valid_loader), epoch_acc / len(valid_loader)
