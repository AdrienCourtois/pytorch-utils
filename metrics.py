import torch 

@torch.no_grad()
def compute_batch_accuracy(y_hat, y):
    if y.dim() == 4:
        if y.size(1) != 1:
            y = y.argmax(1)
        else:
            y = y.squeeze(1)
    
    if y_hat.size(1) != 1:
        y_hat = y_hat.argmax(1)
    else:
        y_hat = y_hat.squeeze(1)
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0

    accuracy = (y == y_hat).float().view(y.size(0), -1).mean(1)

    return accuracy


@torch.no_grad()
def compute_batch_iou(y_hat, y, eps=1e-6):
    if y.dim() == 4:
        if y.size(1) == 2:
            y = y.argmax(1)
        elif y.size(1) == 1:
            y = y.squeeze(1)
        else:
            raise Exception('IOU is not defined for non-binary ground truths.')
    
    if y_hat.dim() == 4:
        if y_hat.size(1) == 2:
            y_hat = y_hat.argmax(1)
        elif y_hat.size(1) == 1:
            y_hat = y_hat.squeeze(1)
            y_hat[y_hat > 0.5] = 1
            y_hat[y_hat <= 0.5] = 0
    
    y, y_hat = map(lambda x: x.bool(), [y, y_hat])
    
    intersection = (y_hat & y).float().sum((1, 2))
    union = (y_hat | y).float().sum((1, 2))
    
    iou = intersection / (union + eps)

    return iou


@torch.no_grad()
def compute_batch_l1(y_hat, y):
    return (y_hat - y).abs().mean((1,2,3))