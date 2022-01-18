from __init__ import *

parser = argparse.ArgumentParser()
parser.add_argument("-bs", "--batchsize", default=64)
parser.add_argument("-ep", "--epochs", default=100)
parser.add_argument("-lr", "--lrate", default=0.0002)
parser.add_argument("-omn", "--outmodelname", default="ConvNN")
parser.add_argument("-dev", "--device", default="cpu")
args = parser.parse_args()



X_train = PanagiaDataset(dir = parentdir + "\\" + "data\jesus.h5")


dataLoader = DataLoader(
    dataset=X_train, 
    batch_size=args.batchsize, 
    shuffle=True)


model = ConvNN(
    in_channels=2048, 
    out_channels=3).to(args.device)


optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=args.lrate)


loss_fn = nn.L1Loss()

    
    
for epoch in range(int(args.epochs)):
    
    total_loss = 0

    for i, x_in in enumerate(dataLoader):
        
        y_out = model(x_in.to(args.device))
        y_real = torch.tensor(torch.rand(y_out.shape)).to(args.device)
        print(y_out.shape, y_real.shape)
        loss = loss_fn(y_out, y_real)
        total_loss += loss * 100

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
