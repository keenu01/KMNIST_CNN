import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CNN
import ssl


ssl._create_default_https_context = ssl._create_unverified_context


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5
batch_size = 64
learning_rate = 0.001


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = CNN(1, 200, 10).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

 
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


torch.save(model.state_dict(), "kmnist_cnn.pth")



import matplotlib.pyplot as plt
import numpy as np


kmnist_classes = ['o', 'ki', 'su', 'tsu', 'na', 'ha', 'ma', 'ya', 're', 'wo']


model.eval()
images_to_show = []
labels_to_show = []
preds_to_show = []
correct_flags = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

      
        images_to_show.extend(images.cpu())
        labels_to_show.extend(labels.cpu())
        preds_to_show.extend(predicted.cpu())
        correct_flags.extend((predicted == labels).cpu())

        if len(images_to_show) >= 25:  # Only show 25 images
            break


plt.figure(figsize=(10, 10))
for i in range(25):
    img = images_to_show[i].squeeze(0).numpy()
    label = kmnist_classes[labels_to_show[i]]
    pred = kmnist_classes[preds_to_show[i]]
    correct = correct_flags[i]

    plt.subplot(5, 5, i + 1)
    plt.imshow(img, cmap='gray')
    title_color = 'green' if correct else 'red'
    plt.title(f"Pred: {pred}\nTrue: {label}", color=title_color, fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.show()
