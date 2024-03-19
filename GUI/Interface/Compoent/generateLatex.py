import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['font.weight'] = 200
def renderLaTeX(a, path, usetex=True, dpi=400, fontsize=20):
    acopy = a
    plt.figure(figsize=(0.3,0.3))
    if usetex: #使用真正的 LaTeX 渲染
        try:
            a = '$\\displaystyle ' + a.strip('$') + ' $'
            plt.text(-0.3,0.9, a, fontsize=fontsize, usetex=usetex)#
        except:
            usetex = False
    
    if not usetex:
        a = acopy
        a = a.strip('$')
        a = '\n'.join([' $ '+_+' $ ' for _ in a.split('\\\\')])
        plt.text(-0.3,0.9, a, fontsize=fontsize, usetex=usetex)
        
    plt.ylim(0,1)
    plt.xlim(0,6)
    plt.axis('off') #隐藏坐标系
    plt.savefig(path, dpi=dpi, bbox_inches ='tight')
    plt.close() #释放内存
renderLaTeX("k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)",'./picture')