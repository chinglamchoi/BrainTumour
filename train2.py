import torch
import torch.utils.data as data
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
import model
import torch.nn as nn
import torch.optim as optim
#torch.backends.cudnn.enabled = False

class_counts = [425.0, 856.0, 558.0]
train_label_file = np.load('/home/summervisitor2/BrainTumorMRI/trainlabels1.npy')
weights_per_class = [1839.0/class_counts[i] for i in range(3)]
weights = [weights_per_class[train_label_file[i]] for i in range(1839)]
sampler = data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), 1839)

class BrainTumour(data.Dataset):

    def __init__(self, data_dir, indices_file, label_file, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])):
        self.data_dir = data_dir
        self.indices_file = np.load(indices_file)
        self.label_file = label_file
        self.transform = transform

    def __len__(self):
        return len(self.label_file)

    def __getitem__(self, index):
        label = self.label_file[index]
        data_name = self.data_dir + str(str(self.indices_file[index]+1) + ".png")
        data = io.imread(data_name)
        img = self.transform(data)
        return (img, label)

trainset = BrainTumour(data_dir='/home/summervisitor2/BrainTumorMRI/imgs/train/', indices_file='/home/summervisitor2/BrainTumorMRI/trainindices.npy', label_file=train_label_file)
test_label_file = np.load('/home/summervisitor2/BrainTumorMRI/testlabels1.npy')
testset = BrainTumour(data_dir='/home/summervisitor2/BrainTumorMRI/imgs/test/', indices_file='/home/summervisitor2/BrainTumorMRI/testindices.npy', label_file=test_label_file)

trainloader = data.DataLoader(trainset, batch_size=16, shuffle=False, sampler=sampler, num_workers=2) #115
testloader = data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)


# Define model
net = model.run_cnn()
#save_number = input("save file number: ")
save_number = "_newer"
#a = input("gpu number: ")
a = "2"
a = "cuda:" + a
device = torch.device(a if torch.cuda.is_available() else "cpu")
if device != "cpu":
    gpuu = True
    net.to(device)
else:
    gpuu = False

criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights_per_class)).to(device)
#criterion = nn.CrossEntropyLoss().to(device)
#optimizer = optim.SGD(net.parameters(), lr=alpha, momentum=0.9, weight_decay=1e-06)
optimizer = optim.Adam(net.parameters())
epochs, best_acc, last_improvement, threshhold = 200, 0.0, 0, 0.99

checkpoint = torch.load("/home/summervisitor2/BrainTumorMRI/models/tumour_adam_newer.pt")
net.load_state_dict(checkpoint["net"])
optimizer.load_state_dict(checkpoint["optimizer"])
epoch = checkpoint["epoch"]
alpha = checkpoint["alpha"]
best_acc = checkpoint["acc"]
for param_group in optimizer.param_groups:
    param_group["lr"] = alpha



# Training & Testing
for epoch in range(1, epochs+1):
    # Train 1 epoch
    net = net.train()
    for images, labels in trainloader:
        images, labels = (images.to(device), labels.to(device)) if gpuu else (images, labels)
        optimizer.zero_grad()
        outputs = net(images).to(device) if gpuu else net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Test
    net = net.eval()
    total, correct = 0,0
    for images,labels in testloader:
        images, labels = (images.to(device), labels.to(device)) if gpuu else (images, labels)
        outputs = net(images).to(device) if gpuu else net(images)
        _, predicted = torch.max(outputs.data, 1) ######
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    # Accuracy
    acc = correct/total
    print("[%i]:"%(epoch), "Test Accuracy: %.4f"%(acc))

    # Early exit at 99% accuracy
    if acc >= threshhold:
        print(" -> Early-exiting: We reached our target accuracy of 99.0%")
        break
    
    if acc > best_acc:
        # save weights:
        state = {
            "net":net.state_dict(),
            "acc": acc,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "alpha": alpha
            }
        torch.save(state, str("./models/tumour_adam" + save_number + ".pt"))
        print(" -> New best accuracy! Saving model out to tumour_sgd" + save_number + ".pt")
        best_acc, last_improvement = acc, epoch
    
    # Learning rate decay to dampen oscillations
    if epoch - last_improvement >= 20:
        alpha /= 10
        for param_group in optimizer.param_groups:
            param_group["lr"] = alpha
        print(" -> Haven't improved in a while, dropping learning rate to %f!"%(alpha))
        last_improvement = epoch

    if epoch - last_improvement >= 30:
        print(" -> We're calling this converged.")
        break
