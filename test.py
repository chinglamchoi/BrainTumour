import torch
import torch.utils.data as data
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import model
#torch.backends.cudnn.enabled = False


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

test_label_file = np.load('/home/summervisitor2/BrainTumorMRI/testlabels1.npy')
testset = BrainTumour(data_dir='/home/summervisitor2/BrainTumorMRI/imgs/test/', indices_file='/home/summervisitor2/BrainTumorMRI/testindices.npy', label_file=test_label_file)

testloader = data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)


# Define model
net = model.run_cnn()
checkpoint = torch.load("models/tumour_adam_newer.pt")
net.load_state_dict(checkpoint["net"])

a = "0"
a = "cuda:" + a
device = torch.device(a if torch.cuda.is_available() else "cpu")
if device != "cpu":
    gpuu = True
    net.to(device)
else:
    gpuu = False


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
print("Test Accuracy: %.4f"%(acc))
