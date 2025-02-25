import torch


def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename='checkpoint.pth'):
    """
    Save a checkpoint of the model and optimizer state.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        loss (float): The current loss value.
        filename (str): The file path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved to {filename}')

def save_best_model(model, accuracy, best_accuracy, filename='best_model.pth'):
    """
    Save the model if the current accuracy is better than the best accuracy.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        accuracy (float): The current accuracy.
        best_accuracy (float): The best accuracy so far.
        filename (str): The file path to save the best model.
    """
    if accuracy > best_accuracy:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'accuracy': accuracy
        }
        torch.save(checkpoint, filename)
        print(f'Best model saved to {filename} with accuracy {accuracy:.4f}')
        return accuracy
    return best_accuracy

def load_checkpoint(model, filename='best_model.pth'):
    """
    Load the best model from the checkpoint.

    Args:
        model (torch.nn.Module): The PyTorch model to load into.
        filename (str): The file path to load the best model from.

    Returns:
        accuracy (float): The accuracy of the best model.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    accuracy = checkpoint['accuracy']
    print(f'Best model loaded from {filename} with accuracy {accuracy:.4f}')
    return model, accuracy