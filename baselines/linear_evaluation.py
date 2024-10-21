import torch
from tqdm import tqdm
from src.datasets.harth import * 
from src.models.attention_model import *
import torch
import src.config, src.utils, src.models, src.hunt_data, src.data
from src.losses.contrastive import LS_HATCL_LOSS, HATCL_LOSS
from src.loader.dataloader import SequentialRandomSampler, FlattenedDataset, STFTDataset, SLEEPDataset, KpiDataset
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import wandb
import argparse
import random
import math
from sklearn.metrics import precision_score, recall_score, f1_score


from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

def is_backbone_frozen(model):
    frozen = True
    for param in model.parameters():
        if param.requires_grad:
            frozen = False
            break
    return frozen

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

def main(train_loader, valid_loader, algorithms, seeds, num_epochs):
    diction_algs = {}
        
    
    for alg in algorithms:
        seed_acc = []
        seed_f1 = []
        seed_prec = []
        seed_recall = []
        
        for seed in seeds:
            seed = int(seed)
            set_seed(seed)
            frozen_backbone = FeatureProjector(input_size=args['feature_dim'], output_size=args['out_features'])
            frozen_backbone.load_state_dict(torch.load(f'models/harth{seed}_{alg}_model_epoch_500.pth',
                                            map_location=torch.device('cpu')))
            
            
            num_activities = len(class_dict)
            mine_model = LinearEvaluation(frozen_backbone, num_classes=num_activities)
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(mine_model.parameters(), lr=0.001)  # Example optimizer

            # Move model to device
            device = torch.device("cpu" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
            mine_model.to(device)

            # Training and validation loop
            num_epochs = num_epochs
            for epoch in tqdm(range(num_epochs)):
                # Training phase
                mine_model.train()  # Set the model to training mode
                train_running_loss = 0.0
                train_correct_predictions = 0
                train_total_samples = 0

                all_preds = []
                all_labels = []

                for time_series, labels in (train_loader):
                    time_series = time_series.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    features = mine_model(time_series)
                    # Flatten y_hat to have dimensions [batch_size * sequence_length, num_classes]
                    y_hat_flat = features.reshape(-1, features.size(-1))

                    # Reshape y to have dimensions [batch_size * sequence_length]
                    labels_flat = labels.view(-1)

                    # Compute training loss
                    train_loss = criterion(y_hat_flat, labels_flat)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    train_loss.backward()
                    optimizer.step()

                    # Update training statistics
                    train_running_loss += train_loss.item() * time_series.size(0)

                    _, predicted = torch.max(y_hat_flat, 1)
                    train_correct_predictions += (predicted == labels_flat).sum().item()

                    #Store the labels for future computation of F1-score
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels_flat.cpu().numpy())

                    train_total_samples += labels_flat.size(0)

                # Calculate average training loss and accuracy for the epoch
                train_epoch_loss = train_running_loss / len(train_loader.dataset)
                train_epoch_accuracy = 100*train_correct_predictions / train_total_samples

                f1 = f1_score(all_labels, all_preds,average='weighted')

#                 print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f},\
#                       Train Accuracy: {train_epoch_accuracy:.2f}%, F1-score: {f1:.4f}")
                
            # Validation phase
            mine_model.eval()  # Set the model to evaluation mode
            val_running_loss = 0.0
            val_correct_predictions = 0
            val_total_samples = 0

            with torch.no_grad():
                val_preds = []
                val_labels = []
                for time_series, labels in (valid_loader):
                    time_series = time_series.to(device)
                    labels = labels.to(device)

                    # Forward pass
                    features = mine_model(time_series)

                    # Flatten y_hat to have dimensions [batch_size * sequence_length, num_classes]
                    y_hat_flat = features.reshape(-1, features.size(-1))

                    # Reshape y to have dimensions [batch_size * sequence_length]
                    labels_flat = labels.view(-1)

                    # Compute validation loss
                    val_loss = criterion(y_hat_flat, labels_flat)

                    # Update validation statistics
                    val_running_loss += val_loss.item() * time_series.size(0)

                    _, predicted = torch.max(y_hat_flat, 1)
                    val_correct_predictions += (predicted == labels_flat).sum().item()
                    val_total_samples += labels_flat.size(0)

                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels_flat.cpu().numpy())

            # Calculate average validation loss and accuracy for the epoch
            val_epoch_loss = val_running_loss / len(valid_loader.dataset)
            val_epoch_accuracy = 100*val_correct_predictions / val_total_samples

            # Precision and recall using sklearn
            precision = precision_score(val_labels, val_preds, average='macro')
            recall = recall_score(val_labels, val_preds, average='macro')

            f1 = f1_score(val_labels, val_preds, average='weighted')
#             print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_epoch_loss:.4f},\
#                   Val Accuracy: {val_epoch_accuracy:.2f}%, F1-score: {f1:.2f},\
#                   Precision: {precision:.2f}, Recall: {recall:.2f}")
            
            
            seed_acc.append((round(val_epoch_accuracy,2)))
            seed_f1.append((round(f1,2)))
            seed_prec.append((round(precision,2)))
            seed_recall.append((round(recall,2)))
            
            
        diction_algs[f'{alg}'] = [(round(np.mean(seed_acc),2), round(np.std(seed_acc),2)),
                                   (round(np.mean(seed_f1),2), round(np.std(seed_f1),2)),
                                   (round(np.mean(seed_prec),2), round(np.std(seed_prec),2)),
                                   (round(np.mean(seed_recall),2), round(np.std(seed_recall),2))]
        
    print(diction_algs)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Start Linear Evaluation.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='configs/harthconfig.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default='data/harth')
    parser.add_argument('-a', '--algorithm', required=False, type=str,
                        help='algorithm.', default=['dynacl'])
    parser.add_argument('-e', '--num_epochs', required=False, type=int,
                        help='number epochs.', default=5)
    parser.add_argument('-s', '--seed_values', required=False, type=int,
                        help='seed value.', default=[42])
    args = parser.parse_args()
    config_path = args.params_path

    # Read config
    config = src.config.Config(config_path)
    ds_path = args.dataset_path
    seed = args.seed_values
    algorithms = args.algorithm
    num_epochs = int(args.num_epochs)
    
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
                
            elif config.DATASET == 'KPI':
                dataset = KpiDataset(
                        data_path=ds_path,
                        n_fft = ds_args['n_fft'],
                        seq_length=ds_args['seq_length'],
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
        
            train_loader = torch.utils.data.DataLoader(
                dataset=train_ds,
                batch_size=args['batch_size'],
                # sampler=SequentialRandomSampler(train_ds, args['batch_size']),
                shuffle = True,
                num_workers=config.NUM_WORKERS,
                drop_last = True,
            )
            
            valid_loader = torch.utils.data.DataLoader(
                dataset=valid_ds,
                batch_size=args['batch_size'],
                # sampler=SequentialRandomSampler(valid_ds, args['batch_size']),
                shuffle = False,
                num_workers=config.NUM_WORKERS,
                drop_last = True,
            )
    
    desired_count_per_class = config.class_count
    class_dict = config.class_dict

    flattened_data = FlattenedDataset(valid_ds)

    # Load balanced dataset

    balanced_dataset = load_balanced_dataset(flattened_data, desired_count_per_class)
    valid_balanced_dataloader = DataLoader(balanced_dataset, batch_size=config.display_batch, shuffle=False, num_workers=config.NUM_WORKERS)


    main(train_loader, valid_loader, algorithms, seed, num_epochs)