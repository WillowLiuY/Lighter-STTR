import torch.utils.data as data
from dataset.scared import ScaredDataset

def create_data_loader(args):
    '''
    Create data loaders for custom datasets (e.g., Scared)

    :param args: arg parser object
    :return: data loaders for training, validation and test sets.
    '''
    if not args.data_path:
        raise ValueError('Data path must be specified.')
    data_path = args.data_path
    
    # Dataset options
    if args.dataset.lower() == 'scared':
        train_set = ScaredDataset(data_path, 'train')
        val_set = ScaredDataset(data_path, args.validation)
        test_set = ScaredDataset(data_path, 'test')
    elif args.dataset.lower() == 'scared_toy':
        train_set = ScaredDataset(data_path, 'train')
        val_set = ScaredDataset(data_path, 'validation')
        test_set = ScaredDataset(data_path, 'validation')
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    
    # Create data loaders
    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True) #pin_memory for faster GPU data loading
    
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True)
    
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True)

    return train_loader, val_loader, test_loader
