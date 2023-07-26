import torch
import torch.nn

class Sport_pred_1GRU_3(torch.nn.Module):
    """
        Prediction model with one GRU and output prediction for team.
        Input:   x, time series for team
        Output:  result, probabilities for w/l/d for team
    """
    
    def __init__(self,n_features, hidden, num_classes, num_layers = 1, drop = 0):
        super(Sport_pred_1GRU_3, self).__init__()
        self.n_features = n_features 
        self.num_classes = num_classes # number of classes (win, draw, lose)
        self.n_hidden = hidden # number of hidden states
        self.n_layers = num_layers # number of LSTM layers (stacked)
        self.drop = drop
        
        # two separate lstms to account for every teams history
        self.l_lstm1 = torch.nn.GRU(input_size = n_features, 
                             hidden_size = self.n_hidden,
                             num_layers = self.n_layers,
                             batch_first = True, 
                             dropout = self.drop) # dropout tate
        

        # classic neural net to process outcomes
        self.l_linear1 = torch.nn.Linear(self.n_hidden, self.num_classes)


    def forward(self, x):
        # convert input to fit the model
        x = x.to(torch.float32)
        x = torch.nan_to_num(x, nan = 0.0)
        
        # run data through lstm and yield output
        lstm_out1,_ = self.l_lstm1(x)
        out = lstm_out1[:,-1,:]
        
        # run lstm output through nn to predict outcome
        result = self.l_linear1(out)

        return result
    
    ##############################################################################################################################
    
    
    
class Sport_pred_2GRU_1(torch.nn.Module):
    """
        Prediction model with two parallel GRUs and output prediction for home team.
        Input:   x, time series for home team
                 y, time series for away team
        Output:  result, probabilities for w/l/d for home team
    """
    def __init__(self,n_features, hidden, num_classes, num_layers = 1, drop = 0):
        super(Sport_pred_2GRU_1, self).__init__()
        self.n_features = n_features 
        self.num_classes = num_classes # number of classes (win, draw, lose)
        self.n_hidden = hidden # number of hidden states
        self.n_layers = num_layers # nu amber of LSTM layers (stacked)
        self.drop = drop # dropout rate
        
        # two separate lstms to account for every teams history
        self.l_lstm1 = torch.nn.GRU(input_size = n_features, 
                             hidden_size = self.n_hidden,
                             num_layers = self.n_layers,
                             batch_first = True, 
                             dropout = self.drop)
        self.l_lstm2 = torch.nn.GRU(input_size = n_features, 
                             hidden_size = self.n_hidden,
                             num_layers = self.n_layers,
                             batch_first = True,
                             dropout = self.drop)

        # classic neural net to process outcomes
        self.l_linear1 = torch.nn.Linear(2 * self.n_hidden, 2 * self.n_hidden)
        self.l_linear2 = torch.nn.Linear(2 * self.n_hidden, self.num_classes)
        
        
        self.soft = torch.nn.Softmax()
        self.relu = torch.nn.ReLU()
        self.sigm = torch.nn.Sigmoid()


    def forward(self, x, y):
        # convert input to fit the model
        x = x.to(torch.float32)
        x = torch.nan_to_num(x, nan = 0.0)
        y = y.to(torch.float32)
        y = torch.nan_to_num(y, nan = 0.0)
        
        
        # run data through lstm and yield output
        lstm_out1,_ = self.l_lstm1(x)
        lstm_out2,_ = self.l_lstm2(y)
        
        out1 = lstm_out1[:,-1,:]
        out2 = lstm_out2[:,-1,:]
        out = torch.cat([out1,out2], dim = 1)
        
        # run lstm output through nn to predict outcome
        result = self.sigm(self.l_linear1(out))
        result = self.l_linear2(result)

        return result