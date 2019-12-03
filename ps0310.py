# Generates the plots for PS0210 for PHYS414/514 

import numpy as np
import customfunctions as F
import scipy.special as SP
import matplotlib.pyplot as plt # this is a plotting package
import timeit

def mytests():
    #plt.ion() #turns interactive mode on, try what it foes

    # a simple plot of the custom implementation of Lambert W
    x = np.arange(-1/np.e+1e-8,1,1e-6)
    code_to_test = """
import numpy as np
import customfunctions as F
x = np.concatenate([np.arange(-1/np.e+1e-8,-1e-6,1e-6),
                        1.1**np.arange(-50,400,1)])
F.lambertw_custom(x,'newton')
"""
    print(timeit.timeit(code_to_test,number=10))
    
    plt.plot(x,F.lambertw_custom(x),x,np.real(SP.lambertw(x)),'r--')
    plt.gca().legend(('custom','builtin'), loc='lower right')
    plt.xlabel('x')
    plt.ylabel('Lambert W')
    plt.grid(True)
#    plt.savefig('lambertW_narrow.pdf')
    
    # a second figure with subplots and some comparison.
    # argument range covers the negative part, the part near 0,
    # and very large values to ensure the custom implementation works
    # everywhere
    x = np.concatenate([np.arange(-1/np.e+1e-8,-1e-6,1e-6),
                        1.5**np.arange(-20,100,1)])
    plt.figure()
    plt.subplot(211)
    # plot the custom and builtin together. why real in builtin?
    plt.plot(x, F.lambertw_custom(x), x, np.real(SP.lambertw(x)), 'r--') 
    plt.xscale('symlog', linthreshx=1e-3) #some unusual axis scaling
    plt.xticks([], [])
    plt.gca().legend(('custom','builtin'), loc='lower right')
    #plt.xlabel('x')
    plt.ylabel('Lambert W')
    plt.title('Comparison of Lambert W Functions')
    plt.grid(True)
    plt.subplot(212)
    plt.plot(x,F.lambertw_custom(x)-np.real(SP.lambertw(x)))
    plt.xscale('symlog', linthreshx=1e-3) #some unusual axis scaling
    #plt.yscale('symlog', linthreshx=1e-3)
    plt.xlabel('x')
    plt.ylabel('Custom-Builtin')
    plt.grid(True)
#    plt.savefig('lambertW_wide.pdf')
    plt.show()
    
if __name__ == '__main__':
    mytests()