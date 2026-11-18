import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#from torchvision import datasets, transforms
#from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(42)
while True:
    try:
        mode=int(input("Mode?"))
        break
    except:
        pass
    
if mode==0:
    testx=[[1.],[3.],[5.],[7.],[9.]]
    testy=[[1.],[3.],[2.],[5.],[4.]]
    d1= torch.tensor(testx,dtype=torch.float32, requires_grad=False)
    d2= torch.tensor(testy,dtype=torch.float32, requires_grad=False)
    maxx=abs(d1).max().item()
    d1=d1/maxx
    x2=torch.cat([d1**5,d1**4,d1**3,d1**2,d1**1], dim=1)
    w=torch.tensor([[1.],[1.],[1.],[1.],[1.]], dtype=torch.float32, requires_grad=True)
    b=torch.tensor(0., dtype=torch.float32, requires_grad=True)

    mlp=nn.Sequential(nn.Linear(5,10),nn.ReLU(),nn.Linear(10,1))
#mlp.parameters()
    opt = torch.optim.AdamW([w,b], lr=0.05)
    xx=[]
    yy=[]
    for i in range(1001):
        loss1=(x2@w+b-d2).pow(2).sum()
        loss1.backward()
        opt.step()
        opt.zero_grad()
        print(f"\n---{i} th---")
        print(f"Loss: {loss1.item():.10f}")
        xx.append(i)
        yy.append(float(f"{loss1.item():.10f}"))
    print("Expected function")
    fx=(f"{w[0][0]/(maxx**5)}x^5+{w[1][0]/(maxx**4)}x^4\n+{w[2][0]/(maxx**3)}x^3+{w[3][0]/(maxx**2)}x^2\n+{w[4][0]/maxx}x+{b}")
    print(fx)
    anyx = [xi/40 for xi in range(0,401)]
  
    print(anyx)
    f = lambda xj: (w[0][0].item()/(maxx**5))*xj**5+(w[1][0].item()/(maxx**4))*xj**4+(w[2][0].item()/(maxx**3))*xj**3+(w[3][0].item()/(maxx**2))*xj**2+(w[4][0].item()/maxx)*xj+b.item()
    fanyx= [f(aa) for aa in anyx]
    plt.figure(1)
    plt.plot(anyx, fanyx, label=f"")
    px=[testx[aaa1][0] for aaa1 in range(len(testx))]
    py =[testy[aaa2][0] for aaa2 in range(len(testy))]
    plt.scatter(px, py, color='red', zorder=100, label='test')
    plt.title(fx, fontsize=16, wrap=True, family='unifont')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
if mode==1:
    
    data1=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float32)
    AND = torch.tensor([0.,0.,0.,1.])
    XOR = torch.tensor([0.,1.,1.,0.])


    print(data1, w)
    print(data1@w)

    #x1=torch.cat([d1,d1,d1,d1],dim=1)
    #x2=torch.cat([d1**1,d1**2, d1**3, d1**4], dim=1)

    #print(x2)

    w=torch.tensor([1.,0.], dtype=torch.float32, requires_grad=True)
    #w1=torch.tensor(-1., dtype=torch.float32, requires_grad=True)
    b=torch.tensor(0., dtype=torch.float32, requires_grad=True)
    #b1=torch.tensor(5., dtype=torch.float32, requires_grad=True)
    print(data1, w)
    print(data1@w)
    xx=[]
    yy=[]
    yw=[]
    yb=[]
    mlp=nn.Sequential(nn.Linear(5,10),nn.ReLU(),nn.Linear(10,1))
    #mlp.parameters()
    opt = torch.optim.AdamW([w,b], lr=0.05)
    for i in range(1001):
        #loss1=(torch.matmul(x2,w)+b-d2).pow(2).sum()
        #loss1=(data1@w+b-d2).pow(2).sum()
        loss1 = F.binary_cross_entropy_with_logits(data1@w+b, AND)
        loss1.backward()
        #print(f"權重梯度: {w.grad}")
        #print(f"偏置梯度: {b.grad}")
        #yw.append(w.grad)
        #yb.append(b.grad)
        opt.step()
        opt.zero_grad()
        print(f"\n--- 第 {i} 輪訓練後 ---")
        print(f"當前錯誤率 (Loss): {loss1.item():.10f}")
        xx.append(i)
        yy.append(float(f"{loss1.item():.10f}"))
    #print(f"{ b.item()}+{w[0][0].item()/maxx}*x+{w[1][0].item()/(maxx**2)}*x^2+{w[2][0].item()/(maxx**3)}*x^3+{w[3][0].item()/(maxx**4)}*x^4")
    
    #print(opt)
    #print(d1)
def train(y, times=100):
    pass
plt.figure(2)
plt.plot(xx,yy,linestyle='-', label='loss1', color='blue')
#plt.plot(xx,yw,linestyle='-', label='w grad', color='red')
#plt.plot(xx,yb,linestyle='-', label='b grad', color='green')

plt.xlabel("Number of opt trial")
plt.ylabel("Loss1")
plt.title("test")
plt.legend()
plt.grid(True)
if mode == 0:
    plt.figure(1)
plt.show()