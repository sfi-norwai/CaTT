import torch
from tqdm import tqdm
from src.datasets.harth import * 
from src.models.attention_model import *
import src.config, src.utils, src.models, src.hunt_data
from src.losses.contrastive import MARGIN_LOSS, LS_MARGIN_LOSS
from src.loader.dataloader import SequentialRandomSampler
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
from torch.utils.data import Subset
import wandb
import argparse




def check_next_two_elements_tensor_single(tensor, thresh):
    """
    Compute the cosine similarity between two vectors. and 
    return 0 and 1 depending on the threshold
    
    Args:
    float: tensor
    float: threshold
    
    Returns:
    int: 0 or 1 in the size of the tensor.
    """
    tensor2 = tensor.reshape(-1, tensor.size(-1))
    tensor_normalized = F.normalize(tensor2, dim=-1, p=2)

    tensor_normalized2 = torch.roll(tensor_normalized, shifts=-1, dims=0)
    
    # Calculate the cosine similarity matrix
    similarities = torch.matmul(tensor_normalized, tensor_normalized2.mT)
    

    positives = torch.diagonal(similarities)
    positives[positives >= thresh] = 0
    positives[positives <= thresh] = 1
    
    
    return positives[:-1]

def check_next_two_elements_tensor_double(tensor, thresh):
    """
    Compute the cosine similarity between two vectors. and 
    return 0 and 1 depending on the threshold
    
    Args:
    float: tensor
    float: threshold
    
    Returns:
    int: 0 or 1 in the size of the tensor.
    """
    tensor2 = tensor.reshape(-1, tensor.size(-1))
    tensor_normalized = F.normalize(tensor2, dim=-1, p=2)

    tensor_normalized2 = torch.roll(tensor_normalized, shifts=-1, dims=0)
    
    # Calculate the cosine similarity matrix
    similarities = torch.matmul(tensor_normalized, tensor_normalized2.mT)
    

    positives = torch.diagonal(similarities)
    positives[positives >= thresh] = 0
    positives[positives <= thresh] = 1
    
    
    return positives


def main(ssl_loader, valid_loader):

    attn_model = TransformerEncoderNetwork(args['feature_dim'])
    # Define loss function and optimizer

    if args['loss'] == 'MARGIN_LOSS':
        margin_loss = MARGIN_LOSS(temperature=args['temperature'], margin=args['margin'])
        next_element_function = check_next_two_elements_tensor_single

    elif args['loss'] == 'LS_MARGIN_LOSS':
        margin_loss = LS_MARGIN_LOSS(temperature=args['temperature'], margin=args['margin'])
        next_element_function = check_next_two_elements_tensor_double

    else:
        raise ValueError(f"Unsupported loss function: {args['loss']}")

    thresh = args['margin_thresh']

    args['lr'] = float(args['lr'])

    if args['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    elif args['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    elif args['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    elif args['optimizer'] == 'Adadelta':
        optimizer = torch.optim.Adadelta(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    else:
        optimizer = torch.optim.Adam(attn_model.parameters(), lr=args['lr'])  # Example optimizer

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.PATIENCE)



    # Wandb setup
    if config.WANDB:
        ds_name = os.path.realpath(ds_path).split('/')[-1]
        proj_name = 'Hunt_cl_TRAIN_' + config.PROJ_NAME + ds_name
        wandb_logger = WandbLogger(project=proj_name)
        
        # Initialize Wandb
        wandb.init(project=proj_name)
        wandb.watch(attn_model, log='all', log_freq=100)

        # Update Wandb config
        wandb.config.update(ds_args)
        wandb.config.update(args)
        wandb.config.update({
            'Algorithm': 'ONLY MARGIN LOSS',
            'Dataset': 'HUNT',
            'Train_DS_size': len(train_ds),
            'Batch_Size': args["batch_size"],
            'Epochs': args["epochs"],
            'Patience': config.PATIENCE

        })

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_model.to(device)

    # Training loop with best model saving
    best_val_loss = float('inf')    

    # Training and validation loop
    num_epochs = args['epochs']
    for epoch in range(num_epochs):
        # Training phase
        attn_model.train()  # Set the model to training mode
        train_running_loss = 0.0
        
        for batch_idx, time_series in enumerate(tqdm(ssl_loader)):
            time_series = time_series.to(device)
            
            sim_vector = next_element_function(time_series, thresh)

            
             # Forward pass
            features = attn_model(time_series)

            # Flatten features to have dimensions [batch_size * sequence_length, feature dim]
            features = features.reshape(-1, features.size(-1))
        
            # Compute training loss
            train_loss = margin_loss(features, sim_vector)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Update training statistics
            train_running_loss += train_loss.item() * time_series.size(0)

            # Log training loss to Wandb
            if config.WANDB and batch_idx % 10 == 0:
                wandb.log({'Train Loss': train_running_loss/(batch_idx + 1), 'Epoch': epoch})

        train_epoch_loss = train_running_loss / len(ssl_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}")

        if epoch % config.VALID_INTERVAL == 0:
        
            attn_model.eval()  # Set the model to evaluation mode
            valid_running_loss = 0
            for time_series in tqdm(valid_loader):
                time_series = time_series.to(device)

                sim_vector = next_element_function(time_series, thresh)
            
                # Forward pass
                features = attn_model(time_series)

                # Flatten features to have dimensions [batch_size * sequence_length, feature dim]
                features = features.reshape(-1, features.size(-1))
            
                # Compute training loss
                valid_loss = margin_loss(features, sim_vector)

                # Update training statistics
                valid_running_loss += valid_loss.item() * time_series.size(0)

            scheduler.step(valid_running_loss)
            valid_epoch_loss = valid_running_loss / len(valid_loader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {valid_epoch_loss:.4f}")

            if config.WANDB:
                wandb.log({'Validation Loss': valid_epoch_loss, 'Epoch': epoch})

            # Optionally save the model every config.SAVE_INTERVAL epochs
            if (epoch + 1) % config.SAVE_INTERVAL == 0 and valid_epoch_loss < best_val_loss:
                best_val_loss = valid_epoch_loss
                torch.save(attn_model.state_dict(), f'models/model_epoch_{epoch + 1}.pth')

    if config.WANDB:
        wandb.finish()

    print("Training and validation complete.")
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start ML training.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='configs/harconfig.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default='data/FAKE_HUNT2')
    args = parser.parse_args()
    config_path = args.params_path
    # Read config
    config = src.config.Config(config_path)
    ds_path = args.dataset_path
    
    # Log in to Wandb
    wandb.login(key=config.WANDB_KEY)

    for ds_args in src.utils.grid_search(config.DATASET_ARGS):
        # Iterate over all model configs if given
        for args in src.utils.grid_search(config.ALGORITHM_ARGS):
            ######### Train with given args ##########
            # Create the dataset
            dataset = src.hunt_data.get_dataset(
                    dataset_name=config.DATASET,
                    dataset_args=ds_args,
                    root_dir=ds_path,
                    num_classes=config.num_classes,
                    label_map=config.label_index,
                    replace_classes=config.replace_classes,
                    config_path=config.CONFIG_PATH,
                    name_label_map=config.class_name_label_map,
                    number_subject = args['number_subject'],
                    number_sample = args['number_sample'],
                    total_subjects = args['total_subjects']
                )
            
            valid_split = config.VALID_SPLIT
        
            valid_amount = int(np.floor(len(dataset)*valid_split))
            train_amount = len(dataset) - valid_amount
            
            train_indices = list(range(train_amount))
            valid_indices = list(range(train_amount, train_amount + valid_amount))
            
            # Create subsets
            train_ds = Subset(dataset, train_indices)
            valid_ds = Subset(dataset, valid_indices)
            
            train_loader = torch.utils.data.DataLoader(
                dataset=train_ds,
                batch_size=args['batch_size'],
                sampler=SequentialRandomSampler(train_ds, args['batch_size']),
                num_workers=config.NUM_WORKERS,
            )
            
            valid_loader = torch.utils.data.DataLoader(
                dataset=valid_ds,
                batch_size=args['batch_size'],
                sampler=SequentialRandomSampler(valid_ds, args['batch_size']),
                shuffle = False,
                num_workers=config.NUM_WORKERS,
            )

    main(train_loader, valid_loader)