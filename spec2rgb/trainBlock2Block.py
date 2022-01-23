from locale import normalize
from unittest.mock import patch
from __init__ import *
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batchsize", default=64)
parser.add_argument("-ep", "--epochs", default=100)
parser.add_argument("-lr", "--lrate", default=0.0002)
parser.add_argument("-omn", "--outmodelname", default="ConvNN")
parser.add_argument("-tr", "--train", default=False)
parser.add_argument("-dev", "--device", default="cpu")
args = parser.parse_args()

print('-bs',args.batchsize)
print('-tr',args.train)



dSet = PanagiaDataset(
    inTrainImagePath = parentdir + "\\" + "data\jesus.h5", 
    outTrainImagePath = parentdir + "\\" + "data\jesus.png",
    inTestImagePath = parentdir + "\\" + "data\panagia.h5", 
    outTestImagePath = parentdir + "\\" + "data\panagia.png",
    h5TrainShape = [31, 46, 2048],
    h5TestShape = [21, 33, 2048]
)


dataLoader = DataLoader(
    dataset = dSet, 
    batch_size = int(args.batchsize), 
    shuffle = True)

testLoader = DataLoader(
    dataset = dSet, 
    batch_size = 1, 
    shuffle = False)

model = ConvNN(
    in_channels=2048, 
    out_channels=3).to(args.device)


optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=args.lrate)


loss_fn = nn.L1Loss()

display_step = 50

if(bool(args.train)):

    print('---training---')  
    
    for epoch in range(int(args.epochs)):
        dSet.setTrain(True)

        epoch_loss = 0

        for batch_index, (patch_in, patch_real) in enumerate(dataLoader):

            patch_in = patch_in['img'].to(args.device)
            patch_real = patch_real['img'].to(args.device)

            patch_out = model(patch_in)

            loss = loss_fn(patch_out, patch_real)

            epoch_loss += (loss * 100) / int(args.epochs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
        print('epoch:',epoch, epoch_loss)
        print('---testing---')
        dSet.setTrain(False)
        
        for batch_index, (patch_in, patch_real) in enumerate(testLoader):
                
            patch_out = model(patch_in['img'].to(str(args.device)))
            dSet.reconstructFromPatches(patch_out.detach(), patch_real['r'], patch_real['c'])
            
            
        dSet.showRecImage()