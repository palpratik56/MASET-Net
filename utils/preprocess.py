


data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),transforms.ToTensor(),
    ])
# Initialize datasets
# dataset = CVCDataset(image_folder, mask_folder,transform=data_transforms)
# dataset = KVASDataset(image_folder, mask_folder,transform=data_transforms)
dataset = ISICDataset(image_folder, mask_folder,transform=data_transforms)
print("Number of Training Images:", len(os.listdir(image_folder)))
print("Number of Training Masks:", len(os.listdir(mask_folder)))
# Split the dataset into train, validation, and test sets
dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
bs = 16
n_w=4
# Adjust batch size for multi-GPU
bs = bs * torch.cuda.device_count()

# Create data loaders with multi-processing support
train_loader = DataLoader(train_dataset, batch_size=bs, 
               shuffle=True,num_workers=n_w,pin_memory=True,
               drop_last=True  # Ensures consistent batch sizes across GPUs
               )

val_loader = DataLoader(val_dataset, batch_size=bs, 
             shuffle=False,num_workers=n_w,pin_memory=True,
             drop_last=True)

test_loader = DataLoader(test_dataset, batch_size=bs, 
              shuffle=False,num_workers=n_w,pin_memory=True,
              drop_last=True)

print(f"Total training images: {len(train_dataset)}")
print(f"Total validation images: {len(val_dataset)}")
print(f"Total test images: {len(test_dataset)}")

# Define the loss function and optimizer
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.95, weight_decay=0.0001)
# optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
# optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
optimizer = optim.Adagrad(model.parameters(), lr=0.0001, weight_decay=0.0001)
# optimizer = optim.RMSprop(model.parameters(),lr=0.0001,momentum=0.9, weight_decay=0.0001)
criterion = nn.BCEWithLogitsLoss()
# Define the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
