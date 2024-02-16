import numpy as np
import torch

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot

def accuracy(y_pred, y_true, nb_old, class_increments):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    acc_total = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for classes in class_increments:
        idxes = np.where(
            np.logical_and(y_true >= classes[0], y_true <= classes[1])
        )[0]
        label = "{}-{}".format(
            str(classes[0]).rjust(2, "0"), str(classes[1]).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    return acc_total,all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)
