#!/usr/bin/env python

######################################################################################################
######################################################################################################
######################################################################################################

################################
########### Figures ############
################################

######################################################################################################

import math
import numpy             as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

######################################################################################################

for Image_index in range(0,245):
    
    Loss_Y1 = np.loadtxt('Ground_Truth.txt').T.reshape(245,-1)
    Loss_Y2 = np.loadtxt('Model.txt'       ).T.reshape(245,-1)
    
    Loss_X  = np.array(list(range(0,len(Loss_Y1[0]))))/100-20
    
    fig, ax = plt.subplots( dpi=200 )
    
    ax.plot(Loss_X, Loss_Y1[Image_index,:], 'r-'  , label='DFT'         )
    ax.plot(Loss_X, Loss_Y2[Image_index,:], 'b--' , label='NN NA Model' )
    
    ax.legend( loc=1 , prop={'size':10} ).draw_frame(False)
    
    ax.set( xlabel='Energy (eV)' , ylabel='Density of States (Arb. Unit)' , title='DFT vs NN NA Model' )
    
    ax.set_xlim( np.min(Loss_X)  , np.max(Loss_X) )
    
    ax.set_ylim( int(np.min(Loss_Y1))-0.1  , (math.ceil(np.max((Loss_Y1[Image_index,:],Loss_Y2[Image_index,:]))*10)+0.1)/10 )
    
    plt.savefig('DFT_vs_NN_NA_Model_Image_' + str(Image_index).zfill(3) + '.png')
    
######################################################################################################
    
Loss_Y = np.loadtxt('Log.txt')

Loss_X = list(range(0,len(Loss_Y)))

fig, ax = plt.subplots( dpi=200 )

ax.plot(Loss_X, Loss_Y[:,0], 'b-', label='Training'  )
ax.plot(Loss_X, Loss_Y[:,1], 'y-', label='Validation')
ax.plot(Loss_X, Loss_Y[:,2], 'r-', label='Test'      )

ax.legend( loc=1 , prop={'size':10} ).draw_frame(False)

ax.set( xlabel='Epoch' , ylabel='RMSE (Arb. Unit)' , title='Epochs vs Training Loss' )

ax.set_yscale('log')

ax.set_xlim( -100  , (math.ceil(len(Loss_X)/100)+1)*100 )

ax.set_ylim( int(np.min(Loss_Y)/2*100)/100  , math.ceil(np.max(Loss_Y)*2*100)/100 )

plt.savefig('Epochs_vs_Training_Loss.png')
