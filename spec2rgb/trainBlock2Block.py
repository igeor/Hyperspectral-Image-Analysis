from __init__ import *
import torchvision.transforms.functional as TF

parser = argparse.ArgumentParser()
parser.add_argument("-ps", "--patchsize", default="(10,14)")
parser.add_argument("-bs", "--batchsize", default=64)
parser.add_argument("-ep", "--epochs", default=100)
parser.add_argument("-lr", "--lrate", default=0.0002)
parser.add_argument("-omn", "--outmodelname", default="ConvNN")
parser.add_argument("-tr", "--train", default=False)
parser.add_argument("-dev", "--device", default="cpu")
args = parser.parse_args()

patch_r, patch_c = tuple(int(s.replace("(","").replace(")","")) for s in args.patchsize.split(','))

# # Panagia Train - Jesus Test
# dSet = PanagiaDataset(
#     inTrainImagePath = parentdir + "\\" + "data\panagia.h5", 
#     outTrainImagePath = parentdir + "\\" + "data\panagia.png",
#     inTestImagePath = parentdir + "\\" + "data\jesus.h5", 
#     outTestImagePath = parentdir + "\\" + "data\jesus.png",
#     h5TrainShape = [21, 33, 840],
#     h5TestShape = [31, 46, 840],
#     r=patch_r, c=patch_c
# )

#Jesus Train - Panagia Test
dSet = PanagiaDataset(
    inTrainImagePath = parentdir + "\\" + "data\panagia.h5", 
    outTrainImagePath = parentdir + "\\" + "data\panagia.png",
    inTestImagePath = parentdir + "\\" + "data\jesus.h5", 
    outTestImagePath = parentdir + "\\" + "data\jesus.png",
    h5TrainShape = [21, 33, 840],
    h5TestShape = [31, 46, 840],
    r=patch_r, c=patch_c
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
    in_channels=840, 
    out_channels=3).to(args.device)


optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=float(args.lrate))


loss_L1 = nn.L1Loss()
loss_L2 = nn.MSELoss()

display_step = 50

if(bool(args.train)):

    #======== TRAINING ========#     
    for epoch in range(int(args.epochs)):
        dSet.setTrain(True)
        epoch_loss = 0
        for batch_index, (patch_in, patch_real) in enumerate(dataLoader):
            patch_in, patch_real = patch_in['img'], patch_real['img']
            
            patch_in = patch_in.to(args.device)
            patch_real = patch_real.to(args.device)

            patch_out = model(patch_in)

            lossL1 = loss_L1(patch_out, patch_real) 

            epoch_loss +=  lossL1 #+ lossL2 
            
            optimizer.zero_grad()
            lossL1.backward()
            optimizer.step()
            
        print('epoch:',epoch, epoch_loss)

        #======== VALIDATING ========#     
        dSet.setTrain(False)
        for batch_index, (patch_in, patch_real) in enumerate(testLoader):
            with torch.no_grad():
                patch_out = model(patch_in['img'].to(str(args.device)))
            dSet.reconstructFromPatches(patch_out.detach(), patch_real['r'], patch_real['c'])
            
            
        #dSet.saveRecImage('./results/block2block/epoch' + str(epoch) + '.png')
        dSet.saveRecImage('./results/block2block/currEpoch.png')