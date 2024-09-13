import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Load the ResNet model with pre-trained weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.train()
model.to(device)

# Step 2: Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adjust learning rate as needed

# Step 3: Load adversarial images and labels
class AdversarialDataset(Dataset):
    def __init__(self, adv_images, labels):
        self.adv_images = adv_images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.adv_images[idx], self.labels[idx]

saved_data = torch.load('results/pgd_attack_results_eps_0.2.pt') 
adv_images = saved_data['adv_images'] 
labels = saved_data['labels'] 
adv_dataset = AdversarialDataset(adv_images, labels)
adv_loader = DataLoader(adv_dataset, batch_size=64, shuffle=True)

# Step 4: Fine-tune the model on adversarial images
epochs = 5  # Number of epochs for fine-tuning, adjust based on training data size and performance
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in adv_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels) 

        loss.backward()  # Backward pass (calculate gradients)
        optimizer.step()  # Update model parameters

        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(adv_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Step 5: Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_resnet.pth')
print("Model saved as fine_tuned_resnet.pth")
