import torch
import tqdm
from tqdm import tqdm


# def binary_accuracy(preds, y):
#
#     rounded_preds = torch.round(torch.sigmoid(preds))
#     correct = (rounded_preds == y).float()
#     acc = correct.sum() / len(correct)
#     return acc
def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc
def train_fc(data_loader, device, model,optimizer, criterion,scheduler):
    model.train()
    a = 0.5
    epoch_loss = 0
    epoch_acc = 0
    for bi,d in tqdm(enumerate(data_loader),total = len(data_loader)):

        ids = d['text']
        lengths = d['length']
        targets = d['target']
        ids = ids.to(device, dtype=torch.long)

        lengths = lengths.to(device, dtype=torch.int)
        targets = targets.to(device, dtype=torch.long)
        optimizer.zero_grad()
       
        outputs = model(ids,lengths)

        loss= criterion(outputs, targets)

        acc = categorical_accuracy(outputs, targets)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    scheduler.step()
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)

def train_kd_fc(data_loader, device, bert_model, model,optimizer, criterion,criterion_kd,scheduler):
    model.train()
    a = 0.5
    epoch_loss = 0
    epoch_acc = 0
    for bi,d in tqdm(enumerate(data_loader),total = len(data_loader)):
        bert_id = d['ids']
        bert_mask = d['mask']
        ids = d['text']
        lengths = d['length']
        targets = d['target']
        ids = ids.to(device, dtype=torch.long)
        bert_id = bert_id.to(device, dtype=torch.long)
        bert_mask = bert_mask.to(device, dtype=torch.long)

        lengths = lengths.to(device, dtype=torch.int)
        targets = targets.to(device, dtype=torch.long)
        optimizer.zero_grad()
        with torch.no_grad():
            bert_output = bert_model(bert_id,bert_mask)

        outputs = model(ids,lengths)
        loss_soft =criterion_kd(outputs,bert_output)
        loss_hard = criterion(outputs, targets)
        loss = loss_hard*a + (1-a)*loss_soft
        acc = categorical_accuracy(outputs, targets)
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
            targets = targets.to(device, dtype=torch.long)

            outputs = model(ids, lengths)
            # print(outputs.shape)
            # print(targets.shape)
            loss = criterion(outputs, targets)



            acc = categorical_accuracy(outputs, targets)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(valid_loader), epoch_acc / len(valid_loader)
