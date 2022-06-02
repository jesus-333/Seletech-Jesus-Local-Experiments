"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Imports

import torch
from torch import nn

#%% Training Cycle

def advanceEpochV1(vae, device, dataloader, optimizer, spectra_section, is_train = True, alpha = 1, beta = 1):
    """
    Function used to advance one epoch of training in training script 1 and 2 (Only VAE).
    Can be used both for single mems and double mems
    """
    
    if(is_train): vae.train()
    else: vae.eval()

    global length_mems_1, length_mems_2
    
    # Track variable
    tot_vae_loss = 0
    tot_recon_loss = 0
    tot_kl_loss = 0
    
    for sample_data_batch in dataloader:
        # Move data and vae to device
        x = sample_data_batch.to(device)
        vae.to(device)
        
        if(is_train): # Train step (keep track of the gradient)
            # Zeros past gradients
            optimizer.zero_grad()
            
            # VAE works
            if(spectra_section == 'full'):
              x1 = x[:, 0:length_mems_1]
              x2 = x[:, (- 1 - length_mems_2):-1]
              x_r_1, log_var_r_1, x_r_2, log_var_r_2, mu_z, log_var_z = vae(x1, x2)

              x_r = torch.cat((x_r_1, x_r_2), 1)
              log_var_r = torch.cat((log_var_r_1, log_var_r_2), 0)
              x = torch.cat((x1,x2), 1)
            elif(spectra_section == 'lower' or spectra_section == 'upper'):
              x_r, log_var_r, mu_z, log_var_z = vae(x)
            
            # Evaluate loss
            vae_loss, recon_loss, kl_loss = VAE_loss(x, x_r, log_var_r, mu_z, log_var_z, alpha, beta)


            # Backward/Optimization pass
            vae_loss.backward()
            optimizer.step()
        else: # Test step (don't need the gradient)
            with torch.no_grad():
              if(spectra_section == 'full'):
                x1 = x[:, 0:length_mems_1]
                x2 = x[:, (- 1 - length_mems_2):-1]
                x_r_1, log_var_r_1, x_r_2, log_var_r_2, mu_z, log_var_z = vae(x1, x2)

                x_r = torch.cat((x_r_1, x_r_2), 1)
                log_var_r = torch.cat((log_var_r_1, log_var_r_2), 0)
                x = torch.cat((x1,x2), 1)
              elif(spectra_section == 'lower' or spectra_section == 'upper'):
                x_r, log_var_r, mu_z, log_var_z = vae(x)

              vae_loss, recon_loss, kl_loss = VAE_loss(x, x_r, log_var_r, mu_z, log_var_z, alpha, beta)
            
        # Compute the total loss
        tot_vae_loss += vae_loss
        tot_recon_loss += recon_loss
        tot_kl_loss += kl_loss
            
        
    return tot_vae_loss, tot_recon_loss, tot_kl_loss


def advanceEpochV2(vae, device, dataloader, optimizer, is_train = True, alpha = 1, beta = 1):
    """
    Function used to advance one epoch of training in training script 3 (Only VAE).
    Can be used only for double mems
    """
    
    if(is_train): vae.train()
    else: vae.eval()

    length_mems_1 = 300
    length_mems_2 = 400
    
    # Track variable
    tot_vae_loss = 0
    tot_recon_loss = 0
    tot_kl_loss = 0
    
    for sample_data_batch in dataloader:
        # Move data and vae to device
        x = sample_data_batch.to(device)
        vae.to(device)
        
        if(is_train): # Train step (keep track of the gradient)
            # Zeros past gradients
            optimizer.zero_grad()
            
            # VAE works
            x1 = x[..., 0:length_mems_1]
            x2 = x[..., (- 1 - length_mems_2):-1]
            x_r_1, log_var_r_1, x_r_2, log_var_r_2, mu_z, log_var_z = vae(x1, x2)

            x_r = torch.cat((x_r_1, x_r_2), -1)
            log_var_r = torch.cat((log_var_r_1, log_var_r_2), -1)
            x = torch.cat((x1,x2), -1) # N.b. the original x is long 702 sample not 700.

            # Evaluate loss
            vae_loss, recon_loss, kl_loss = VAE_loss(x, x_r, log_var_r, mu_z, log_var_z, alpha, beta)


            # Backward/Optimization pass
            vae_loss.backward()
            optimizer.step()
        else: # Test step (don't need the gradient)
            with torch.no_grad():
                x1 = x[..., 0:length_mems_1]
                x2 = x[..., (- 1 - length_mems_2):-1]
                x_r_1, log_var_r_1, x_r_2, log_var_r_2, mu_z, log_var_z = vae(x1, x2)
        
                x_r = torch.cat((x_r_1, x_r_2), -1)
                log_var_r = torch.cat((log_var_r_1, log_var_r_2), -1)
                x = torch.cat((x1,x2), -1) # N.b. the original x is long 702 sample not 700.
        
                vae_loss, recon_loss, kl_loss = VAE_loss(x, x_r, log_var_r, mu_z, log_var_z, alpha, beta)
            
        # Compute the total loss
        tot_vae_loss += vae_loss
        tot_recon_loss += recon_loss
        tot_kl_loss += kl_loss
            
        
    return tot_vae_loss, tot_recon_loss, tot_kl_loss


def advanceEpochV3(model, device, dataloader, optimizer, is_train, double_mems = True):
    """
    Used in training script 5 (Only autencoder).
    """
    
    if(is_train): model.train()
    else: model.eval()
    
    length_mems_1 = 300
    length_mems_2 = 400
    
    # Track variable
    tot_recon_loss = 0
    
    loss_function = nn.MSELoss()
    
    for sample_data_batch in dataloader:
        x = sample_data_batch.to(device)
        model.to(device)
        
        if(is_train): # Executed during training
            if(double_mems):
                x1 = x[:, ..., 0:length_mems_1]
                x2 = x[:, ..., (- 1 - length_mems_2):-1]
                
                x_r_1, x_r_2, z = model(x1, x2)
                x_r = torch.cat((x_r_1, x_r_2), -1)
                
                # N.b. the original x is long 702 sample not 700.
                x = torch.cat((x1,x2), -1)
            else:
                x_r, z = model(x)
            
        else: # Executed during testing
            with torch.no_grad(): # Deactivate the tracking of the gradient
                if(double_mems):
                    x1 = x[:, ..., 0:length_mems_1]
                    x2 = x[:, ..., (- 1 - length_mems_2):-1]
                    
                    x_r_1, x_r_2, z = model(x1, x2)
                    x_r = torch.cat((x_r_1, x_r_2), -1)
                    
                    # N.b. the original x is long 702 sample not 700.
                    x = torch.cat((x1,x2), -1)
                else:
                    x_r, z = model(x)
        
        # Compute loss
        recon_loss = loss_function(x, x_r)
        
        # If is training update model weights
        if(is_train):
            recon_loss.backward()
            optimizer.step()
        
        # Compute the total loss
        tot_recon_loss += recon_loss
        
        
    return tot_recon_loss



#%% Loss functions

def VAE_loss(x, x_r, log_var_r, mu_q, log_var_q, alpha = 1, beta = 1):
    """
    Loss of the VAE. 
    It return the reconstruction loss between x and x_r and the Kullback between a standard normal distribution and the ones defined by sigma and log_var
    It also return the sum of the two.
    The hyperparameter alpha multiply the reconstruction loss.
    The hyperparameter beta multiply the KL loss.
    """
    
    # Kullback-Leibler Divergence
    # N.b. Due to implementation reasons I pass to the function the STANDARD DEVIATION, i.e. the NON-SQUARED VALUE
    # When the variance is needed inside the function the sigmas are eventually squared
    sigma_p = torch.ones(log_var_q.shape).to(log_var_q.device) # Standard deviation of the target standard distribution
    mu_p = torch.zeros(mu_q.shape).to(mu_q.device) # Mean of the target gaussian distribution
    sigma_q = torch.sqrt(torch.exp(log_var_q)) # standard deviation obtained from the VAE
    # kl_loss = KL_Loss(sigma_p, mu_p, sigma_q, mu_q).mean()
    
    # kl_loss = KL_Loss(sigma_q, mu_q, sigma_p, mu_p).mean() # TODO REMOVE
    kl_loss =  torch.mean(-0.5 * torch.sum(1 + log_var_q - mu_q ** 2 - log_var_q.exp(), dim = 1), dim = 0) # TODO Remove
    
  
    # Reconstruction loss 
    sigma_r = torch.sqrt(torch.exp(log_var_r))
    recon_loss = advance_recon_loss(x, x_r, sigma_r).mean()
    
    vae_loss = recon_loss * alpha + kl_loss * beta
    # print(float(kl_loss), float(recon_loss), float(vae_loss))
    return vae_loss, recon_loss * alpha, kl_loss * beta


def advance_recon_loss(x, x_r, std_r):
    """
    Advance versione of the recontruction loss for the VAE when the output distribution is gaussian.
    Instead of the simple L2 loss we use the log-likelihood formula so we can also encode the variance in the output of the decoder.
    Input parameters:
      x = Original data
      x_r = mean of the reconstructed output
      std_r = standard deviation of the reconstructed output. This is a scalar value.
    
    More info: 
    https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
    https://arxiv.org/pdf/2006.13202.pdf
    """
    
    total_loss = 0
    
    # MSE part
    mse_core = torch.pow((x - x_r), 2).sum(1)/ x.shape[1]
    mse_scale = (x[0].shape[0]/(2 * torch.pow(std_r, 2)))
    total_loss += (mse_core * mse_scale)
    
    # Variance part
    # total_loss += x[0].shape[0] * torch.log(std_r).mean()
    
    return total_loss

    
def KL_Loss(sigma_p, mu_p, sigma_q, mu_q):
    """
    General function for a KL loss with specified the paramters of two gaussian distributions p and q
    The parameter must be sigma (standard deviation) and mu (mean).
    The order of the parameter must be the following: sigma_p, mu_p, sigma_q, mu_q
    """
    
    tmp_el_1 = torch.log(sigma_q/sigma_p)
    
    tmp_el_2_num = torch.pow(sigma_q, 2) + torch.pow((mu_q - mu_p), 2)
    tmp_el_2_den = 2 * torch.pow(sigma_p, 2)
    tmp_el_2 = tmp_el_2_num / tmp_el_2_den
    
    kl_loss = - (tmp_el_1  - tmp_el_2 + 0.5)
    
    # P.s. The sigmas and mus have length equals to the hinner space dimension. So the final shape is [n_sample_in_batch, hidden_sapce_dimension]
    return kl_loss.sum(dim = 1)

# - - - -  - - - -  - - - -  - - - -  - - - - 
# TODO Future stuff

def classifierLoss(predict_label, true_label):
    classifier_loss_criterion = torch.nn.NLLLoss()
    # classifier_loss_criterion = torch.nn.CrossEntropyLoss()
    
    return classifier_loss_criterion(predict_label, true_label)


def VAE_and_classifier_loss(x, x_r, mu, log_var, true_label, predict_label, alpha = 1, beta = 1):
    # VAE loss (reconstruction + kullback)
    vae_loss, recon_loss, kl_loss = VAE_loss(x, x_r, mu, log_var, alpha)
       
    # Classifier (discriminator) loss
    classifier_loss = classifierLoss(predict_label, true_label) 
    
    # Total loss
    total_loss = vae_loss + classifier_loss * beta
    
    return total_loss, recon_loss, kl_loss, classifier_loss * beta

