import torch
import numpy as np

#see https://medium.com/@aakashns/linear-regression-with-pytorch-3dde91d60b50

def fc_grad_desc(X,y,w=None,num_epochs=1000,batch_size=5,bias=False,lr=1e-6,loss_cutoff=0.05,verbose=False):
    '''
    Function that takes a set of input activations, FC and target activations to estimate the optimal FC to predict the target activations within the AF framework.
    It uses linear regression and gradient descent to move 'further away' from the empirical measurements towards the best FC to fit the problem
    
    inputs:
        X: numpy array (obs x feature) [design matrix]
        y: numpy array (obs x target feature) [target]
        w: numpy array (feature x 1) FC weights to "start" the regression
        num_epochs: number of iterations of the fitting algo
        batch_size: size of cross fold validation (def=5)
        bias: whether regression model as an intercept (default=False)
        lr : 
    '''
    # create torch dataset
    # organise things in a pytorchian way
    inputs = torch.from_numpy(X)
    targets = torch.from_numpy(y)
    ds = torch.utils.data.TensorDataset(inputs, targets)

    # Define data loader
    dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True)

    # Define model
    model = torch.nn.Linear(inputs.shape[1],targets.shape[0],bias=bias)

    # set initial model weights to FC
    if w is not None:
        weights = torch.from_numpy(w)
        weights.requires_grad = True
        model.weight.data = weights

    # Define loss function
    #loss_fn = torch.nn.functional.l1_loss
    loss_fn = torch.nn.functional.mse_loss
    
    # Define optimizer
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    ## Start the fitting process
    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb,yb in dl:

            # 1. Generate predictions
            pred = model(xb)

            # 2. Calculate loss
            loss = loss_fn(pred, yb)

            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients
            opt.step()

            # 5. Reset the gradients to zero
            opt.zero_grad()
            
            if loss.item() < loss_cutoff:
                if verbose:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
                model_weights = model.weight.data.numpy().copy()
                return model,model_weights

    if verbose:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    model_weights = model.weight.data.numpy().copy()
    return model,model_weights