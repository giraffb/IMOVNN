import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import copy

class IMOVNN(nn.Module):
    
    def __init__(
        self,
        input_dims, 
        network_settings, 
        cuda = False
    ):
        super(IMOVNN, self).__init__()
        self.device = torch.device("cuda" if cuda == True else "cpu")
        self.M                = len(input_dims['x_dim_set'])
        self.x_dim_set = {}
        for m in range(self.M):
            self.x_dim_set[m] = input_dims['x_dim_set'][m]
        
        self.dim_enc          = network_settings['dim_enc']        #encoder hidden nodes
        self.num_layers_enc   = network_settings['num_layers_enc'] #encoder layers

        self.z_dim            = input_dims['z_dim']           
        #self.steps_per_batch  = input_dims['steps_per_batch']
        
        self.dim_specificpre  = network_settings['dim_specificpre']      #predictor hidden nodes
        self.num_layers_specificpre      = network_settings['num_layers_specificpre'] #predictor layers
        
        self.dim_jointpre    = network_settings['dim_joint_pre']      #predictor hidden nodes
        self.num_layers_jointpre    = network_settings['num_layers_jointpre'] #predictor layers
        self.y_dim            = input_dims['y_dim']
        self.dropout   = network_settings['dropout'] 
        #self.reg_scale        = network_settings['reg_scale']   #regularization
        
        self.F = network_settings['F']
        self.start_temp = network_settings['start_temp']
        self.min_temp = network_settings['min_temp']
        self.ITERATION = network_settings['ITERATION']
        self.rate_temp = math.exp(math.log(self.min_temp / self.start_temp) / (self.ITERATION))
        
        # module
        self.feature_selective_layer = {}
        self.specific_encoder = {}
        self.specific_predictor = {}
        for m in range(self.M):
            self.feature_selective_layer[m] = feature_selective_layer(input_dim = self.x_dim_set[m], output_dim = self.F, start_temp = self.start_temp, 
                                                                      min_temp = self.min_temp, alpha = self.rate_temp)
            self.add_module(f"feature_selective_layer_{m}", self.feature_selective_layer[m])
            self.specific_encoder[m] = specific_encoder(input_dim = self.F, output_dim = 2*self.z_dim, num_layers = self.num_layers_enc, 
                                                        hidden_dim = self.dim_enc, dropout = self.dropout)
            self.add_module(f"specific_encoder_{m}", self.specific_encoder[m])
            self.specific_predictor[m] = predictor(input_dim = self.z_dim, output_dim = self.y_dim, num_layers = self.num_layers_specificpre, 
                                                        hidden_dim = self.dim_specificpre, dropout = self.dropout)
            self.add_module(f"specific_predictor{m}", self.specific_predictor[m])
            
        self.joint_encoder = joint_encoder()
        self.joint_predictor = predictor(input_dim = self.z_dim, output_dim = self.y_dim, num_layers = self.num_layers_jointpre, 
                                                        hidden_dim = self.dim_jointpre, dropout = self.dropout)
       
    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, x_set, mask):
        mu_set, logvar_set, u_set, z_set, y_set = {},{},{},{},{}
        for m in range(self.M):
            u_set[m] = self.feature_selective_layer[m](x_set[m])
            mu_set[m], logvar_set[m] = self.specific_encoder[m](u_set[m])
            z_set[m] = self.reparametrize(mu_set[m], logvar_set[m])
        mu_z, logvar_z = self.joint_encoder(mask, mu_set, logvar_set)
        z = self.reparametrize(mu_z, logvar_z)
    
        # specific omics predict
        for m in range(self.M):
            y_set[m] = self.specific_predictor[m](z_set[m])
    
        # joint omics predict
        y = self.joint_predictor(z)
        
        # test predict
        #y_10 = torch.empty(0, y.shape[0], y.shape[1])
        #for i in range(10):
        #    z_pre = self.reparametrize(mu_z, logvar_z)
        #    y_pre = self.joint_predictor(z_pre)
        #    y_pre = y_pre.unsqueeze(0)
        #    #y_10 = torch.cat([y_10, y_pre], dim=0)
            
        return y_set, y, mu_set, logvar_set, mu_z, logvar_z
    
    def div(self, x, y):
        return torch.div(x, y + 1e-8)
        
    def log(self, x):
        return torch.log(x + 1e-8)
        
    def loss_y(self, y_true, y_pre):
        tmp_loss = -torch.sum(y_true * self.log(y_pre), dim=-1)
        return tmp_loss
        
    def loss_function(self, mask, y_true, y_set, y, mu_set, logvar_set, mu_z, logvar_z, alpha, beta):
        #mask = torch.from_numpy(mask)
        ds = torch.distributions
        qz = ds.Normal(mu_z, torch.sqrt(torch.exp(logvar_z)))
        prior_z  = ds.Normal(0.0, 1.0)
        LOSS_KL = torch.mean(torch.sum(ds.kl_divergence(qz, prior_z), dim=-1))
        LOSS_PRE = torch.mean(self.loss_y(y_true, y))
            
        LOSS_JOINT = LOSS_PRE + beta*LOSS_KL
            
        LOSS_PRE_set  = []
        LOSS_KL_set = []
        for m in range(self.M):
            qz_set, prior_z_set = {},{}
            qz_set[m] = ds.Normal(mu_set[m], torch.sqrt(torch.exp(logvar_set[m])))
            prior_z_set[m] = ds.Normal(0.0, 1.0)
            tmp_pre = self.loss_y(y_true, y_set[m])
            tmp_kl = torch.sum(ds.kl_divergence(qz_set[m], prior_z_set[m]), dim=-1)
            
            LOSS_PRE_set += [self.div(torch.sum(mask[:,m]*tmp_pre), torch.sum(mask[:,m]))]
            LOSS_KL_set += [self.div(torch.sum(mask[:,m]*tmp_kl), torch.sum(mask[:,m]))]
            
        LOSS_PRE_set  = torch.stack(LOSS_PRE_set, dim=0)
        LOSS_KL_set = torch.stack(LOSS_KL_set, dim=0)
            
        LOSS_PRE_set_all = torch.sum(LOSS_PRE_set)
        LOSS_KL_set_all = torch.sum(LOSS_KL_set)
            
        LOSS_MARGINAL = LOSS_PRE_set_all + beta*LOSS_KL_set_all
            
        LOSS_TOTAL = LOSS_JOINT\
                    + alpha*(LOSS_MARGINAL)
                    
        return LOSS_TOTAL, LOSS_PRE, LOSS_KL, LOSS_PRE_set_all, LOSS_KL_set_all, LOSS_PRE_set, LOSS_KL_set
        
    def train_model(self, train_x_set, train_y, tr_mask, test_X_set, test_y, te_mask, alpha, beta, l_rate):
        self.train()
        
        select_layers0 = torch.nn.ModuleList([self.feature_selective_layer_0])
        select_layers0_params = list(map(id, select_layers0.parameters()))
        select_layers1 = torch.nn.ModuleList([self.feature_selective_layer_1])
        select_layers1_params = list(map(id, select_layers1.parameters()))
        base_params = filter(lambda p: id(p) not in select_layers0_params and id(p) not in select_layers1_params, self.parameters())
        optimizer = optim.Adam([{"params": self.feature_selective_layer_0.parameters(), "lr": 10*l_rate},
                                {"params": self.feature_selective_layer_1.parameters(), "lr": 10*l_rate},
                              {"params": base_params}],
                             lr=l_rate)
        maxacc = 0
        prob = {}
        for epoch in range(self.ITERATION):
            epoch_LOSS_TOTAL = 0
            epoch_LOSS_PRE = 0
            epoch_LOSS_KL = 0
            epoch_LOSS_PRE_set_all = 0
            epoch_LOSS_KL_set_all = 0
                
            if epoch == 0:
                for m in range(self.M):
                    train_x_set[m] = torch.from_numpy(train_x_set[m]).to(self.device)
                    test_X_set[m] = torch.from_numpy(test_X_set[m]).to(self.device)
                train_y = torch.from_numpy(train_y).to(self.device)
                test_y = torch.from_numpy(test_y).to(self.device)
                tr_mask = torch.from_numpy(tr_mask).to(self.device)  
                te_mask = torch.from_numpy(te_mask).to(self.device)
        
            optimizer.zero_grad()
                
            y_set, y, mu_set, logvar_set, mu_z, logvar_z = self(train_x_set, tr_mask)
            epoch_LOSS_TOTAL, epoch_LOSS_PRE, epoch_LOSS_KL, epoch_LOSS_PRE_set_all, epoch_LOSS_KL_set_all, LOSS_PRE_set, LOSS_KL_set = self.loss_function(tr_mask, train_y, y_set, y, mu_set, logvar_set, mu_z, logvar_z, alpha, beta)
            epoch_LOSS_TOTAL.backward()
            optimizer.step()
                
            if (epoch+1)%500 == 0:
                print( "{:05d}: TRAIN| LT={:.3f} LP={:.3f} LKL={:.3f} LPS={:.3f} LKLS={:.3f} | ".format(
            epoch+1, epoch_LOSS_TOTAL, epoch_LOSS_PRE, epoch_LOSS_KL, epoch_LOSS_PRE_set_all, epoch_LOSS_KL_set_all))
                #acc = torch.mean((torch.argmax(train_y, dim=1) == torch.argmax(y, dim=1)).float())
                #print('train_acc', acc)
            if (epoch+1)%100 == 0:
                print('temp:',self.feature_selective_layer[0].temp.data)
                for m in range(self.M):
                    prob[m] = torch.mean(torch.max(F.softmax(self.feature_selective_layer[m].logits.data, dim = -1), dim = -1)[0])
                    print(f'mean max of probabilities {m}:',prob[m])
                print("Train ACC: {:.4f}".format(accuracy_score(train_y.argmax(1).cpu().numpy(), y.argmax(1).cpu().numpy())))
                    #print('grad:', getattr(self, f'feature_selective_layer_{m}').logits.grad.sum())
                #for name, param in self.named_parameters():
                #    print(name, param.data)
                acc, f1, auc = self.predict_acc(test_X_set, test_y, te_mask)
                #perform = acc + f1 + auc
                if all(prob[m] > 0.9 for m in range(self.M)) and (acc > maxacc or (acc == maxacc and f1 + auc > maxf1 + maxauc)) : 
                    maxacc = acc
                    maxf1 = f1
                    maxauc = auc
                    model_save = copy.deepcopy(self)
            
            #for group in optimizer.param_groups:
            #    for param in group['params']:
            #        print(f"Parameter: {param}, Learning Rate: {group['lr']}")
        print('acc:',maxacc)
        print('f1:',maxf1)
        print('auc:',maxauc)
        return model_save, maxacc, maxf1, maxauc
    
    def predict_acc(self, test_x_set, test_y, mask):
        self.eval()
        with torch.no_grad():
            #for m in range(self.M):
            #    test_x_set[m] = torch.from_numpy( test_x_set[m]).to(self.device)
            #test_y = torch.from_numpy(test_y).to(self.device)
            y_set, y, mu_set, logvar_set, mu_z, logvar_z = self(test_x_set, mask)
            acc = accuracy_score(test_y.argmax(1).cpu().numpy(), y.argmax(1).cpu().numpy())
            f1 = f1_score(test_y.argmax(1).cpu().numpy(), y.argmax(1).cpu().numpy())
            auc = roc_auc_score(test_y.cpu().numpy(), y.cpu().numpy())
            print("Test ACC: {:.4f}".format(acc))
            print("Test F1: {:.4f}".format(f1))
            print("Test AUC: {:.4f}".format(auc))
        return acc, f1, auc
        
class feature_selective_layer(nn.Module):
    
    def __init__(self, input_dim, output_dim, start_temp=10.0, min_temp=0.1, alpha=0.99999):
        super(feature_selective_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.alpha = alpha
        
        self.temp = nn.Parameter(torch.Tensor([self.start_temp]), requires_grad=False)
        self.logits = nn.Parameter(torch.Tensor(self.output_dim, self.input_dim), requires_grad=True)
        xavier_normal_(self.logits)
        
    def forward(self, X):
        uniform = torch.rand(self.logits.shape) * (1.0 - 1e-8) + 1e-8
        gumbel = -torch.log(-torch.log(uniform)).to(torch.device("cuda")) # avoid numerical instability
        self.temp.data = torch.max(self.temp * self.alpha, torch.tensor(self.min_temp))
        noisy_logits = (self.logits + gumbel) / self.temp
        samples = F.softmax(noisy_logits)
        Y = torch.matmul(X.float(), samples.T)
        return Y

class specific_encoder(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, dropout):
        super(specific_encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = dropout))
            
            for i in range(num_layers-2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p = dropout))
            
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        self.output = self.layers(x)
        self.mu = self.output[:, :self.output_dim//2] 
        self.logvar = self.output[:, self.output_dim//2:]
        return self.mu, self.logvar
        
class predictor(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, dropout):
        super(predictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.Softmax())
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p = dropout))
            
            for i in range(num_layers-2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p = dropout))
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            layers.append(nn.Softmax())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.layers(x)
        return output
    
class joint_encoder(nn.Module):
    
    def __init__(self):
        super(joint_encoder, self).__init__()
        self.epsilon = 1e-8

    def div(self, x, y):
        return torch.div(x, y + self.epsilon)
    
    def log(self, x):
        return torch.log(x + self.epsilon)

    def forward(self, mask, mu_set, logvar_set):
        #mask = torch.from_numpy(mask)
        tmp = 1.
        for m in range(len(mu_set)):
            tmp += mask[:, m].reshape(-1, 1) * self.div(1., torch.exp(logvar_set[m]))
        joint_var = self.div(1., tmp)
        joint_logvar = self.log(joint_var)

        tmp = 0.
        for m in range(len(mu_set)):
            tmp += mask[:, m].reshape(-1, 1) * self.div(1., torch.exp(logvar_set[m])) * mu_set[m]
        joint_mu = joint_var * tmp

        return joint_mu.float(), joint_logvar.float()