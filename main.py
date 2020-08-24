from torch import nn
from models.ResNet50 import ResNet, bottleneck
from models.ResNet18 import ResNet, baseBlock
from models.vgg_16 import VGGNet
from models.inception_main import GoogLeNet
import torchvision.transforms as transforms
import torch
import torch.nn
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.utils.data as data
from sklearn.metrics import confusion_matrix


def train(epochs, learningRate, device, net, trainData, validData):
    loss_function = nn.CrossEntropyLoss()  # calculates how far off the classifications are from reality
    optimizer = optim.Adam(net.parameters(), lr=learningRate)

    trainLossValues = []
    trainEpochValues = []

    validLossValues = []
    validEpochValues = []

    for epoch in range(1, epochs + 1):  # number of full passes over the data
        for data in trainData:
            features, targets = data[0].to(device), data[1].to(device)
            net.zero_grad()  # sets gradients to 0 before calculating loss
            output = net(features)
            loss = loss_function(output, targets)
            loss.backward()
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
        print("Training Loss: ", loss)
        trainLossValues.append(loss)
        trainEpochValues.append(epoch)

        for data in validData:
            features, targets = data[0].to(device), data[1].to(device)
            net.zero_grad()  # sets gradients to 0 before calculating loss
            output = net(features)
            loss = loss_function(output, targets)
            loss.backward()
            optimizer.step()  # attempt to optimize weights to account for loss/gradients
        print("Validation Loss: ", loss)
        validLossValues.append(loss)
        validEpochValues.append(epoch)

    plt.plot(trainEpochValues, trainLossValues, label="Train")
    plt.plot(validEpochValues, validLossValues, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per Epoch")
    plt.show()


def trainInception(epochs, learningRate, device, net, trainData, validData):
    loss_function = nn.CrossEntropyLoss()  # calculates how far off the classifications are from reality
    optimizer1 = optim.Adam(net.parameters(), lr=learningRate)
    optimizer2 = optim.Adam(net.parameters(), lr=learningRate)
    optimizer3 = optim.Adam(net.parameters(), lr=learningRate)

    trainLossValues = []
    trainEpochValues = []

    validLossValues = []
    validEpochValues = []

    aux1_loss_values = []
    aux2_loss_values = []

    for epoch in range(epochs):  # number of full passes over the data
        for data in trainData:
            features, targets = data[0].to(device), data[1].to(device)
            net.zero_grad()  # sets gradients to 0 before calculating loss
            output1, output2, output3 = net(
                features)  # obtaining prediction of the auxiliary classifiers and final classifier
            loss_aux1 = loss_function(output1, targets)
            optimizer1.step()  # attempt to optimize weights to account for loss/gradients
            loss_aux2 = loss_function(output2, targets)
            optimizer2.step()
            loss = loss_function(output3, targets)
            total_loss = loss + (loss_aux1 + loss_aux2) * 0.3
            total_loss.backward()
            optimizer3.step()

        trainLossValues.append(total_loss)
        trainEpochValues.append(epoch)
        aux1_loss_values.append(loss_aux1)
        aux2_loss_values.append(loss_aux2)

        for data in validData:
            features, targets = data[0].to(device), data[1].to(device)
            net.zero_grad()  # sets gradients to 0 before calculating loss
            output1, output2, output3 = net(features)
            loss_aux1 = loss_function(output1, targets)
            optimizer1.step()  # attempt to optimize weights to account for loss/gradients
            loss_aux2 = loss_function(output2, targets)
            optimizer2.step()
            loss = loss_function(output3, targets)
            total_loss = loss + (loss_aux1 + loss_aux2) * 0.3
            total_loss.backward()
            optimizer3.step()
        validLossValues.append(total_loss)
        validEpochValues.append(epoch)

    plt.plot(trainEpochValues, aux1_loss_values, label="learning rate")
    plt.xlabel("epoch")
    plt.ylabel("aux1 loss")
    plt.title("epoch vs aux1 loss curve")
    plt.show()

    plt.plot(trainEpochValues, aux2_loss_values, label="learning rate")
    plt.xlabel("epoch")
    plt.ylabel("aux2 loss")
    plt.title("epoch vs aux2 loss curve")
    plt.show()

    plt.plot(trainEpochValues, trainLossValues, label="Train")
    plt.plot(validEpochValues, validLossValues, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per Epoch")
    plt.show()


def test(device, net, testData):
    correct = 0
    total = 0

    with torch.no_grad():
        predictedValues = []
        actualValues = []

        for data in testData:
            features, targets = data[0].to(device), data[1].to(device)
            output = net(features)
            for idx, i in enumerate(output):
                if torch.argmax(i) == torch.tensor(0):
                    predictedValues.append(0)
                if torch.argmax(i) == torch.tensor(1):
                    predictedValues.append(1)

                if targets[idx] == torch.tensor(0):
                    actualValues.append(0)
                if targets[idx] == torch.tensor(1):
                    actualValues.append(1)

                if torch.argmax(i) == targets[idx]:
                    correct += 1
                total += 1

    # print evaluation stats
    print("\nOverall Accuracy", round(correct / total, 3) * 100, "%")
    print("\nConfusion Matrix:",
          "\n", confusion_matrix(actualValues, predictedValues))

    tn, fp, fn, tp = confusion_matrix(actualValues, predictedValues).ravel()
    precision = tp / (tp + fp)
    print("\nPrecision:", round(precision, 3))

    recall = tp / (tp + fn)
    print("Recall:", round(recall, 3))

    f1 = 2 * ((precision * recall) / (precision + recall))
    print("F1:", round(f1, 3))


def main():
    # CNN Variables
    epochs = 20
    learningRate = .001
    batchSize = 16

    # file paths
    trainingDataPath = "./Images/train/"
    testDataPath = "./Images/test/"
    validateDataPath = "./Images/validate"

    # randomly transform training images
    transformTrainingImage = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(.7, 1)),
        transforms.RandomRotation(degrees=45),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # normal images
    transformImage = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Generate data loaders
    trainingData = torchvision.datasets.ImageFolder(root=trainingDataPath, transform=transformTrainingImage)
    trainingDataLoader = data.DataLoader(trainingData, batch_size=batchSize, shuffle=True)
    testingData = torchvision.datasets.ImageFolder(root=testDataPath, transform=transformImage)
    testingDataLoader = data.DataLoader(testingData, batch_size=batchSize, shuffle=True)
    validData = torchvision.datasets.ImageFolder(root=validateDataPath, transform=transformImage)
    validDataLoader = data.DataLoader(validData, batch_size=batchSize, shuffle=True)

    # initialise nets
    ResNet18 = ResNet(baseBlock, [2, 2, 2, 2])
    ResNet50 = ResNet(bottleneck, [3, 4, 6, 3])
    VGG16 = VGGNet()
    Inception = GoogLeNet(True, True)

    #  check for CUDA device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Device: ", device)

    # user input for training or testing
    modeSelect = int(input("Training or Testing Model? (input number to select Mode)"
                           "\n1- Train"
                           "\n2- Test"
                           "\nInput: "))

    # user input for which CNN model to use
    model_use = int(input("\nWhich Model to train? (input number to use model)"
                          "\n1 - ResNet18"
                          "\n2 - ResNet50"
                          "\n3 - Inception"
                          "\n4 - VGG16"
                          "\nInput: "))

    # set up appropriate model
    if model_use == 1:
        print("Model - ResNet18")
        ResNet18.to(device)
        net = ResNet18
        path = "./results/ResNet18.pt"

    elif model_use == 2:
        print("Model - ResNet50")
        ResNet50.to(device)
        net = ResNet50
        path = "./results/ResNet50.pt"

    elif model_use == 3:
        print("Model - Inception")
        path = "./results/Inception.pt"
        if modeSelect == 1:
            Inception.to(device)
            net = Inception
        if modeSelect == 2:
            Inception.to(device)
            net = Inception
            net.aux_class = False
            net.training_enable = False
            net.aux1 = net.aux2 = None

    elif model_use == 4:
        print("Model - VGG16")
        VGG16.to(device)
        net = VGG16
        path = "./results/VGG.pt"

    # complete training or testing code
    if modeSelect == 1:
        if model_use != 3:
            train(epochs=epochs, learningRate=learningRate, device=device, net=net, trainData=trainingDataLoader,
                  validData=validDataLoader)
        else:
            trainInception(epochs=epochs, learningRate=learningRate, device=device, net=net,
                           trainData=trainingDataLoader,
                           validData=validDataLoader)

        save = input("Save Model?(y/n)")
        if save.lower() == "y":
            print(path)
            torch.save(net.state_dict(), path)

    if modeSelect == 2:
        net.load_state_dict(torch.load(path))
        net.eval()
        test(device=device, net=net, testData=testingDataLoader)


if __name__ == '__main__':
    main()
