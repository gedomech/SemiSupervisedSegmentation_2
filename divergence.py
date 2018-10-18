import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import warnings
warnings.filterwarnings('ignore')

#
torch.random.manual_seed(4)
# loss = nn.KLDivLoss(size_average=False,reduce=False)
# batch_size = 1
# output1= torch.randn(batch_size, 2)
# probs1 = F.softmax(output1,1)
# log_probs1 = torch.log(probs1)
# probs2 = F.softmax(torch.randn(batch_size, 2), 1)
# # probs2 = probs1
# result = loss(log_probs1, probs2) / batch_size
# print(result)
# print()



prob1 = F.softmax(torch.rand(1,2,1,1),1)
# prob2 =  F.softmax(torch.rand(1,2,256,256),1)
# prob2 = copy.deepcopy(prob1)
prob3 = F.softmax(torch.rand(1, 2, 1, 1),1)
# prob3 = copy.deepcopy(prob1)

ensemble_probs = torch.cat([prob1,prob3],0)
distribution_number= ensemble_probs.shape[0]


Mixture_dist = ensemble_probs.mean(0,keepdim=True).expand(distribution_number,ensemble_probs.shape[1],ensemble_probs.shape[2],ensemble_probs.shape[3])



Kl_loss = nn.KLDivLoss(reduce=True,size_average=False)
JS_Div_loss = Kl_loss(torch.log(ensemble_probs), Mixture_dist)

## for this, using ln
JS_by_hand = 0.5*(prob1[0][0][0][0]*torch.log(prob1[0][0][0][0]/ensemble_probs[0][0][0][0])\
             +prob1[0][1][0][0]*torch.log(prob1[0][1][0][0]/ensemble_probs[0][1][0][0]) \
             + prob3[0][0][0][0] * torch.log(prob3[0][0][0][0] / ensemble_probs[0][0][0][0]) \
             +prob3[0][1][0][0]*torch.log(prob3[0][1][0][0]/ensemble_probs[0][1][0][0]))

print('implemented JS by pytorch: ',JS_Div_loss,' , implemented by hands:', JS_by_hand)

# for this, using log2
import dit,numpy as np
from dit.divergences import jensen_shannon_divergence
X = dit.ScalarDistribution(['0', '1'], prob1.numpy().ravel())
Y = dit.ScalarDistribution(['0', '1'], prob3.numpy().ravel())
print('JS-div in log2:',jensen_shannon_divergence([X, Y]),' , in ln: ', jensen_shannon_divergence([X, Y])/np.log(2)*np.log(np.e))

## image examples:


prob1 = F.softmax(torch.rand(1,2,256,256),1)
# prob2 =  F.softmax(torch.rand(1,2,256,256),1)
prob2 = copy.deepcopy(prob1)
prob3 = F.softmax(torch.rand(1, 2, 256, 256),1)
# prob3 = copy.deepcopy(prob1)

ensemble_probs = torch.cat([prob1,prob2, prob3],0)
distribution_number= ensemble_probs.shape[0]


Mixture_dist = ensemble_probs.mean(0,keepdim=True).expand(distribution_number,ensemble_probs.shape[1],ensemble_probs.shape[2],ensemble_probs.shape[3])



Kl_loss = nn.KLDivLoss(reduce=True,size_average=False)
JS_Div_loss = Kl_loss(torch.log(ensemble_probs), Mixture_dist)

print(JS_Div_loss)