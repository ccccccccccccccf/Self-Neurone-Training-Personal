# import torch
# import math
# import os
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
import pandas as pd
# from scipy import ndimage, datasets
# from keras.datasets import mnist
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Load MNIST without TensorFlow/Keras
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_data= datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

#from torchvision import datasets, transforms
#from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
fig = plt.figure()
torch.manual_seed(67)
np.random.seed(67)
while True:
    try:
        mode=int(input("Mode?"))
        break
    except:
        pass
def main():
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
    
if mode==0:
    testx=[[1.],[3.],[5.],[7.],[9.]]
    testy=[[1.],[2.],[2.],[4.],[5.]]
    d1= torch.tensor(testx,dtype=torch.float32, requires_grad=False)
    d2= torch.tensor(testy,dtype=torch.float32, requires_grad=False)
    maxx=abs(d1).max().item()
    d1=d1/maxx
    x2=torch.cat([d1**5,d1**4,d1**3,d1**2,d1**1], dim=1)
    w=torch.tensor([[1.],[1.],[1.],[1.],[1.]], dtype=torch.float32, requires_grad=True)
    b=torch.tensor(0., dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD([w,b], lr=0.05)
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
    #plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    main()

if mode == 1:
    csvdata=pd.read_csv("ai_generated_data_to_test.csv")
    # print(list(csvdata.columns))
    # print(csvdata.iloc[:,[0,1,2,3,4,6]])
    presetdata=csvdata.iloc[:,[0,1,2,3,4]].values
    outputdata=csvdata.iloc[:,5].values.reshape(-1, 1)
    print(csvdata.iloc[:,5].values)
    print(presetdata)
    print(outputdata)
    # print(csvdata.values, "\n\n")
    # print(csvdata.iloc[1:3],"\n\n")
    # print(csvdata.iloc[1:3,0],"\n\n")
    # print(csvdata.iloc[1:3,1],"\n\n")
    # print(csvdata.iloc[2],"\n\n")
    # print(csvdata.iloc[0:2],"\n\n")
    # print(csvdata.iloc[1:5])
    #mlp1=nn.Sequential(nn.Linear(5,3200),nn.GELU(),nn.Linear(3200,16),nn.GELU(),nn.Linear(16,6),nn.GELU(),nn.Linear(6,1))
    mlp1=nn.Sequential(nn.Linear(5,16),nn.Linear(16,1))
    csvdatadata=torch.tensor(presetdata, dtype=torch.float32)/100
    csvdataoutput=torch.tensor(outputdata, dtype=torch.float32)/100
    optcsv1=torch.optim.Adam(mlp1.parameters(), lr=0.01)#, momentum=0.9)
    #print(csvdatadata)
    #print(csvdataoutput)
    for i in range(2001):
        losscsv1=(mlp1(csvdatadata)-csvdataoutput).pow(2).mean()
        losscsv1.backward()
        optcsv1.step()
        optcsv1.zero_grad()
        if i%200==0:
            print(f"{i} th, loss:{losscsv1}")
    inq="abcdef"
    while True:
        try:
            inq=[float(input(f"{k}?")) for k in ["STR", "AGI", "INT", "STA", "LUCK"]]
            inqtensor=torch.tensor(inq, dtype=torch.float32)/100
        except ValueError:
            print("passed")
            continue
        print(f"{mlp1(inqtensor).item()*100:.3f}")
    
    
if mode==2:
    img0=train_data.data.numpy()[3]
    img1=torch.tensor(train_data.data.numpy()[20856], dtype=torch.int32)
    #print(img1)
    #imgtemp = ndimage.binary_fill_holes(np.array([[1,0], [0, 1]]))
    img3= np.random.random((500,500))
    f1 = lambda x: (math.floor(2*x/8000)-2*math.floor(x/8000)) %255
    img4 = np.array([f1(i) for i in range(500*500)]).reshape(500,500)
    #img3withfilter = torchvision.transforms.functional.gaussian_blur(img3,kernel_size=(67,0))
    #img4withfilter =torchvision.transforms.functional.gaussian_blur(img4, kernel_size=0.1)
    #labeled_array, num_features = ndimage.label(img1)
    #labeled_array basically make any isolated 1 into acccending index order
    #--plt1=fig.add_subplot(221)
    #--plt2=fig.add_subplot(222)
    #plt3=fig.add_subplot(223)
    #plt4=fig.add_subplot(224)
    #--plt1.imshow(img0, cmap="grey")
    #plt2.imshow(img3withfilter)
    #--plt2.imshow(img1, cmap="grey")
    #plt4.imshow(datasets.ascent(), cmap="grey")
    sampledata, sampleanswer = train_data.data.numpy(), train_data.targets.numpy()
    chunk=DataLoader(train_data, batch_size=128,shuffle=True)
    #traintensor=torch.tensor(sampledata, dtype=torch.float32)/255
    #trainans=torch.tensor(sampleanswer, dtype=torch.long)
    
    mlp2=nn.Sequential(nn.Flatten(),nn.Linear(28*28, 128),nn.ReLU(), nn.Linear(128,64),nn.ReLU(),nn.Linear(64,10))#nn.Softmax(dim=1))
    opt1=torch.optim.Adam((mlp2.parameters()), lr=0.001)
    for i in range(10):
        for traintensor, trainans in chunk:
            #trainans_expand = F.one_hot(trainans, num_classes=10).float()
            loss2=nn.CrossEntropyLoss()(mlp2(traintensor), trainans)
            loss2.backward()
            opt1.step()
            opt1.zero_grad()
        if True:
            print(f"---{i} th---")
            print(loss2)
    while True:
        try:
            intput1=int(input("no. of category(0-9999)"))
            chosenone=test_data.data.numpy()[intput1:intput1+1]
            plt.figure(1)
            plt.imshow(chosenone[0], cmap="grey")
            plt.axis("off")
            aaaa=(torch.tensor(chosenone, dtype=torch.float32)/255)
            mlp2.eval()
            with torch.no_grad():
                print(f"Predicted: {torch.argmax(mlp2(aaaa), dim=1).item()}, expected: {test_data.targets.numpy()[intput1]}")
            plt.show()
        except ValueError:
            print("qefwgernht")
        except IndexError:
            print("hfjtewrgty4rqv35 35y15 y3y5 13 gy5biequiu gv g4g4g4c1ug4cv iurgv grvui gr3")
    #plt.show()
def train(y, times=100):
    pass

    
    
    
# if mode==1:
    
#     data1=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=torch.float32)
#     AND = torch.tensor([0.,0.,0.,1.])
#     XOR = torch.tensor([0.,1.,1.,0.])


#     print(data1, w)
#     print(data1@w)

#     #x1=torch.cat([d1,d1,d1,d1],dim=1)
#     #x2=torch.cat([d1**1,d1**2, d1**3, d1**4], dim=1)

#     #print(x2)

#     w=torch.tensor([1.,0.], dtype=torch.float32, requires_grad=True)
#     #w1=torch.tensor(-1., dtype=torch.float32, requires_grad=True)
#     b=torch.tensor(0., dtype=torch.float32, requires_grad=True)
#     #b1=torch.tensor(5., dtype=torch.float32, requires_grad=True)
#     print(data1, w)
#     print(data1@w)
#     xx=[]
#     yy=[]
#     yw=[]
#     yb=[]
#     mlp=nn.Sequential(nn.Linear(5,10),nn.ReLU(),nn.Linear(10,1))
#     #mlp.parameters()
#     opt = torch.optim.AdamW([w,b], lr=0.05)
#     for i in range(1001):
#         #loss1=(torch.matmul(x2,w)+b-d2).pow(2).sum()
#         #loss1=(data1@w+b-d2).pow(2).sum()
#         loss1 = F.binary_cross_entropy_with_logits(data1@w+b, AND)
#         loss1.backward()
#         #print(f"權重梯度: {w.grad}")
#         #print(f"偏置梯度: {b.grad}")
#         #yw.append(w.grad)
#         #yb.append(b.grad)
#         opt.step()
#         opt.zero_grad()
#         print(f"\n--- 第 {i} 輪訓練後 ---")
#         print(f"當前錯誤率 (Loss): {loss1.item():.10f}")
#         xx.append(i)
#         yy.append(float(f"{loss1.item():.10f}"))
    #print(f"{ b.item()}+{w[0][0].item()/maxx}*x+{w[1][0].item()/(maxx**2)}*x^2+{w[2][0].item()/(maxx**3)}*x^3+{w[3][0].item()/(maxx**4)}*x^4")
    
    #print(opt)
    #print(d1)