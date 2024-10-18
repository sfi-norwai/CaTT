import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.datasets.harth import * 
from src.models.attention_model import *

def main():
    # create dataloader
    data_folder = 'data/harth'

    batch_size = 64
    window_size = 599

    custom_dataset = HarthDownstream(data_folder, window_size)

    # Split the training dataset into train and validation sets
    train_size = int(0.8 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

    # Create DataLoaders for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # Define loss function and optimizer
    a_model = TransformerEncoderNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(a_model.parameters(), lr=0.001)  # Example optimizer

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_model.to(device)

    # Training and validation loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Training phase
        a_model.train()  # Set the model to training mode
        train_running_loss = 0.0
        train_correct_predictions = 0
        train_total_samples = 0
        
        for time_series, labels in tqdm(train_loader):
            time_series = time_series.to(device)
            labels = labels.to(device)
            
            # Forward pass
            features = a_model(time_series)
            features = features.reshape(-1, features.shape[-1])
            labels = labels.reshape(-1, labels.shape[-1]).squeeze()

            # Compute training loss
            train_loss = criterion(features, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Update training statistics
            train_running_loss += train_loss.item() * time_series.size(0)
            
            _, predicted = torch.max(features, 1)
            train_correct_predictions += (predicted == labels).sum().item()
            
            train_total_samples += labels.size(0)
        
        # Calculate average training loss and accuracy for the epoch
        train_epoch_loss = train_running_loss / len(train_loader.dataset)
        train_epoch_accuracy = 100*train_correct_predictions / train_total_samples
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Train Accuracy: {train_epoch_accuracy:.2f}%")

        # Validation phase
        a_model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for time_series, labels in val_loader:
                time_series = time_series.to(device)
                labels = labels.to(device)

                # Forward pass
                features = a_model(time_series)
                features = features.reshape(-1, features.shape[-1])
                labels = labels.reshape(-1, labels.shape[-1]).squeeze()

                # Compute validation loss
                val_loss = criterion(features, labels)

                # Update validation statistics
                val_running_loss += val_loss.item() * time_series.size(0)
                _, predicted = torch.max(features, 1)
                val_correct_predictions += (predicted == labels).sum().item()
                val_total_samples += labels.size(0)
        
        # Calculate average validation loss and accuracy for the epoch
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_accuracy = 100*val_correct_predictions / val_total_samples
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.2f}%")

    print("Training and validation complete.")
    
if __name__ == "__main__":
    main()