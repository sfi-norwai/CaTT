import torch
from tqdm import tqdm
from src.datasets.harth import * 
from src.models.attention_model import *
import torch
import src.config, src.utils, src.models, src.hunt_data
from src.losses.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from src.loader.dataloader import SequentialRandomSampler, FlattenedDataset, STFTDataset, SLEEPDataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import wandb
import argparse
import random
import math


from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import src.config, src.utils, src.models, src.data
import torchvision
from torch.utils.data import random_split

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_model(seed):
    set_seed(seed)
    model = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features'])
    return model

def load_balanced_dataset(dataset, class_counts):
    # Initialize dictionary to store indices of each class
    class_indices = {label: [] for label in class_counts.keys()}
    
    # Populate class_indices with indices of each class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label = label.item()  # Ensure the label is a scalar
        if label in class_indices:
            class_indices[label].append(idx)

    # Ensure each class has the required number of instances
    balanced_indices = []
    for label, count in class_counts.items():
        if len(class_indices[label]) >= count:
            balanced_indices.extend(random.sample(class_indices[label], count))
        else:
            raise ValueError(f"Not enough instances of class {label} to satisfy the requested count")

    # Create a subset of the dataset with the balanced indices
    balanced_subset = Subset(dataset, balanced_indices)
    return balanced_subset

# Function to apply t-SNE and visualize the results
def visualize_tsne(images, labels, class_names, model):

    # Evaluate the model and get features
    model.eval()
    with torch.no_grad():
        model_features = model(images)

    # Standardize the data before applying t-SNE
    scaler = StandardScaler()
    tsne = TSNE(n_components=2, init='random', learning_rate='auto')

    # Standardize model features before applying t-SNE
    standardized_model_features = scaler.fit_transform(model_features.view(-1, model_features.size(-1)).cpu().numpy())

    # Apply t-SNE to model features
    reduced_features_model = tsne.fit_transform(standardized_model_features)

    # Plot the results
    plt.figure(figsize=(10, 8))
    for i in range(len(class_names)):
        indices = labels == i
        plt.scatter(reduced_features_model[indices, 0], reduced_features_model[indices, 1], label=class_names[i])

    plt.title('t-SNE Visualization of Vanilla CL Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()

    return plt, model_features

def main(train_loader, valid_loader, valid_balanced_dataloader, seed):

    attn_model = initialize_model(seed)
    # Define loss function and optimizer

    
    # Wandb setup
    if config.WANDB:
        ds_name = os.path.realpath(ds_path).split('/')[-1]
        proj_name = 'Visual_CL' + ds_name + str(seed)
        run_name = 'ds_path'

        wandb_logger = WandbLogger(project=proj_name)
        
        # Initialize Wandb
        wandb.init(project=proj_name, name=run_name)
        wandb.watch(attn_model, log='all', log_freq=100)

        # Update Wandb config
        wandb.config.update(ds_args)
        wandb.config.update(args)
        wandb.config.update({
            'Algorithm': f'{run_name}',
            'Dataset': f'{ds_name}',
            'Train_DS_size': len(train_ds),
            'Batch_Size': args["batch_size"],
            'Epochs': args["epochs"],
            'Patience': config.PATIENCE,
            'Seed': seed

        })
        wandb.run.name = run_name
        wandb.run.save()

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn_model.to(device)

    attn_model.load_state_dict(torch.load(f'models/my_models/harth42_cpc_model_epoch_500.pth'))
    for batch in valid_balanced_dataloader:
        images, labeli = batch
        images = images.view(-1, 1, images.shape[-1]).to(device)

    tsne_plot, time_features = visualize_tsne(images, labeli, class_dict, attn_model)
    
    if config.WANDB:
        wandb.log({
            't-SNE': wandb.Image(tsne_plot)
            })
            
    tsne_plot.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start Vanilla CL training.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='configs/harthconfig.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default='data/harth')
    parser.add_argument('-s', '--seed_value', required=False, type=int,
                        help='seed value.', default=42)
    parser.add_argument('-f', '--feature_value', required=False, type=int,
                        help='feature value.', default=156)
    
    args = parser.parse_args()
    config_path = args.params_path
    # Read config
    config = src.config.Config(config_path)
    ds_path = args.dataset_path
    seed = int(args.seed_value)
    feature = int(args.feature_value)
    # Log in to Wandb
    wandb.login(key=config.WANDB_KEY)

    # Set all seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    
    for ds_args in src.utils.grid_search(config.DATASET_ARGS):
        # Iterate over all model configs if given
        for args in src.utils.grid_search(config.ALGORITHM_ARGS):
             # Create the dataset
            if config.DATASET == 'STFT':
                dataset = src.data.get_dataset(
                        dataset_name=config.DATASET,
                        dataset_args=ds_args,
                        root_dir=ds_path,
                        num_classes=config.num_classes,
                        label_map=config.label_index,
                        replace_classes=config.replace_classes,
                        config_path=config.CONFIG_PATH,
                        name_label_map=config.class_name_label_map
                    )
                
            elif config.DATASET == 'ECG':
                dataset = STFTDataset(
                        data_path=ds_path,
                        n_fft = ds_args['n_fft'],
                        seq_length=ds_args['seq_length'],
                        class_to_exclude=ds_args['class_to_exclude'],
                        hop_length=ds_args['hop_length'],
                        win_length=ds_args['win_length'],
                        num_labels=ds_args['num_labels']
                    )

            elif config.DATASET == 'SLEEPEEG':
                dataset = SLEEPDataset(ds_path, seq_length=ds_args['seq_length'])

            else:
                raise ValueError(f"Unsupported DATASET: {config.DATASET}")
            
            valid_split = config.VALID_SPLIT
            
            valid_amount = int(np.floor(len(dataset)*valid_split))
            train_amount = len(dataset) - valid_amount
            
            train_indices = list(range(train_amount))
            valid_indices = list(range(train_amount, train_amount + valid_amount))
            
            # Create subsets
            train_ds = Subset(dataset, train_indices)
            valid_ds = Subset(dataset, valid_indices)
        
            
    
    desired_count_per_class = config.class_count
    class_dict = config.class_dict

    flattened_data = FlattenedDataset(valid_ds)

    # Load balanced dataset

    balanced_dataset = load_balanced_dataset(flattened_data, desired_count_per_class)
    valid_balanced_dataloader = DataLoader(balanced_dataset, batch_size=config.display_batch, shuffle=False, num_workers=config.NUM_WORKERS)


    main(valid_balanced_dataloader, seed, feature)