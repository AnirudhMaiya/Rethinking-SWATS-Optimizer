import torch
import math
from torch.optim.optimizer import Optimizer

class SwatsLocal(Optimizer):
    def __init__(self, params,names_of_layers, lr=1e-3,div_lr_decay = 1, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(names_of_layers = names_of_layers,lr=lr,div_lr_decay = div_lr_decay, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(SwatsLocal, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SwatsLocal, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p,name_is in zip(group['params'],group['names_of_layers']):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Swats Local does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['shouldIDoSGD'] = 0
                    state['SGDLr'] = torch.tensor([0],dtype = torch.float64)
                    state['SGDMom'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            
                beta1, beta2 = group['betas']
                state['step'] += 1
                sgd_lr = state['SGDLr']

                if(state['shouldIDoSGD']):
                  sgdmom = state['SGDMom']
                  sgdmom.mul_(beta1).add_(grad)
                  p.add_(sgdmom, alpha = -1 * sgd_lr.item() * (1 - beta1) * (1 / group['div_lr_decay']))
                  continue

                else:
                  exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                  
                  bias_correction1 = 1 - beta1 ** state['step']
                  bias_correction2 = 1 - beta2 ** state['step']
                
                  exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                  exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                  denom = (exp_avg_sq.sqrt()).add_(group['eps']) / (math.sqrt(bias_correction2))

                  step_size_adam = group['lr'] / bias_correction1

                  
                  p_k = torch.div(exp_avg,denom) * -1 * step_size_adam
                  if(torch.dot(p_k.reshape(-1,),grad.reshape(-1,)) != 0):
                    num_is = torch.dot(p_k.reshape(-1,),p_k.reshape(-1,))
                    den_is = -1 * torch.dot(p_k.reshape(-1,),grad.reshape(-1,))
                    gamma_k = num_is/den_is
                    sgd_lr.mul_(beta2).add_(gamma_k,alpha = 1-beta2)
                    if(state['step'] > 1 and (abs((sgd_lr/bias_correction2)-gamma_k) <  1e-5 )):
                      state['shouldIDoSGD'] = 1
                      sgd_lr.div_(bias_correction2)
                      print('Switching to SGD for layer %s at %d steps and Lr for this layer is %f'%(name_is,state['step'],state['SGDLr']))
                  p.add_(p_k)
        return loss