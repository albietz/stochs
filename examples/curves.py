import math
import pickle
import numpy as np
import matplotlib.pyplot as plt

smoothing_wsize = 10  # must be odd for savgol_filter
smooth = False

def smoothing(epochs, accs):
    accs = accs.copy()
    ws = smoothing_wsize
    for i in range(accs.shape[1]):
        accs[ws//2:-ws//2,i] = np.convolve(accs[:,i], np.ones(ws)/ws, 'same')[ws//2:-ws//2]
#         accs[:,i] = savgol_filter(accs[:,i], smoothing_wsize, 1)
    return epochs[ws//2:-ws//2], accs[ws//2:-ws//2,:]
    
def algo_label(name, lr):
    s = ''
    if name.startswith('miso'):
        s += 'S-MISO'
    elif name.startswith('saga'):
        s += 'N-SAGA'
    else:
        s += 'SGD'

    if '_avg' in name:
        s += '-AVG'
    if '_nonu' in name:
        s += '-NU'
    if lr:
        s += ' $\eta = {}$'.format(lr)
    return s

def plot_test_acc(res, step=1, last=-10):
    accs = np.array(res['test_accs'])
    epochs = res['epochs']
    if smooth:
        epochs, accs = smoothing(epochs, accs)

    plt.figure(figsize=(10, 7))
    for i in range(accs.shape[1]):
        p = res['params'][i]
        plt.plot(epochs[:last:step], accs[:last:step,i],
                 label='{}{}'.format(p['name'], '_lr' + str(p['lr']) if 'lr' in p else ''))

    plt.title('$\mu = {}$'.format(res['params'][0]['lmbda']))
    plt.xlabel('epoch')
    plt.ylabel('test accuracy')
    plt.legend(loc='lower right')
    
def plot_loss(res, ty='train', log=False, step=1, last=-10, legend=True, ylabel=None,
              small=True, fname=None, title='STL-10 ckn', filter_fn=None):
    accs = np.array(res['{}_losses'.format(ty)])
    if 'regs' in res and ty == 'train':
        accs += np.array(res['regs'])
    epochs = res['epochs']
    if smooth:
        epochs, accs = smoothing(epochs, accs)

#     best = accs[:,2].min()
    best = accs[:,:].min()
#     if fname is None:
#         plt.figure(figsize=(10, 7))
#     else:
    if small:
        plt.figure(figsize=(4,2.2))
    else:
        plt.figure(figsize=(10,7))


    for i in range(accs.shape[1]):
        p = res['params'][i]
        if filter_fn and filter_fn(p):
            print('skipping', p['name'], p['lr'])
            continue
        if log:
            plt.semilogy(epochs[:last:step], accs[:last:step,i] - best,
                         label=algo_label(p['name'], p.get('lr')),
                         linewidth=1.5)
        else:
            plt.plot(epochs[:last:step],
                     accs[:last:step,i], label=algo_label(p['name'], p.get('lr')))

    plt.title(title)
    plt.xlabel('epochs')
    if ty == 'train':
        plt.ylabel(ylabel or 'f - f*')
    else:
        plt.ylabel('{} {}loss'.format(ty, 'excess ' if log else ''))
    if legend:
        plt.legend(loc='upper right', fontsize=7)
    if fname is not None:
        plt.savefig(fname, format='pdf', bbox_inches='tight', pad_inches=0)

def plot_all(res, log=False, step=1, last=None):
    plot_test_acc(res, step=step, last=last)
    plot_loss(res, ty='train', log=log, step=step, last=last)
    plot_loss(res, ty='test', log=log, step=step, last=last)
