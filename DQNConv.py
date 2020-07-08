import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNConv(nn.Module):
    def __init__(self, nb_actions=None, nb_vel=None, omni_dir=False, input_num=2): # input 160 x 16
        super(DQNConv, self).__init__()
        self.omniDir = omni_dir
        self.conv1 = nn.Conv2d(input_num, 8, kernel_size=4, stride=2, padding=1) # 80 x 8
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1) # 40 x 4
        self.conv3 = nn.Conv2d(16, 16, kernel_size=4, stride=2, padding=1) # 20 x 2
        self.conv4 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1) # 10 x 1
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(1, 10))

        if self.omniDir:
            self.fc1 = nn.Linear(32*4+nb_vel, 32) #find this
        else:
            self.fc1 = nn.Linear(32+nb_vel, 32)
        self.fc2 = nn.Linear(32, 16) #find this
        self.head = nn.Linear(16, nb_actions)

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        #ipdb.set_trace()
        return x

    def forward(self, x, y):
        #ipdb.set_trace()
        if self.omniDir:
            x0 = self._forward_features(x[0])
            x1 = self._forward_features(x[1])
            x2 = self._forward_features(x[2])
            x3 = self._forward_features(x[3])
            xx = torch.cat((x0, x1, x2, x3, y),dim=1) # cat img and vel
        else: 
            x = self._forward_features(x)
            xx = torch.cat((x,y),dim=1)
        xx = F.relu(self.fc1(xx))
        xx = F.relu(self.fc2(xx))
        xx = self.head(xx)
        return xx

    def _forward_features_disp(self, x): 
        x = F.relu(self.conv1(x))
        disp1 = x.detach()
        x = F.relu(self.conv2(x))
        disp2 = x.detach()
        x = F.relu(self.conv3(x))
        disp3 = x.detach()
        x = F.relu(self.conv4(x))
        disp4 = x.detach()
        x = F.relu(self.conv5(x))
        disp5 = x.detach()
        #ipdb.set_trace()
        x = x.view(x.size(0), -1)
        return x, (disp1, disp2, disp3, disp4, disp5)

    def forward_disp(self, x, y): # TODO: test the omniDir case
        if self.omniDir:
            x0, disps0 = self._forward_features_disp(x[0])
            x1, disps1 = self._forward_features_disp(x[1])
            x2, disps2 = self._forward_features_disp(x[2])
            x3, disps3 = self._forward_features_disp(x[3])
            disps = []
            for (d1,d2,d3,d4) in zip(disps0,disps1,disps2,disps3):
                dcat = torch.cat((d1,d2,d3,d4),dim=1)
                disps.append(dcat)
            x = torch.cat((x0, x1, x2, x3, y),dim=1) # cat img and vel
        else:
            x, disps = self._forward_features_disp(x)
            x = torch.cat((x,y),dim=1) # cat img and vel
        x = F.relu(self.fc1(x))
        disp6 = x.detach()
        x = F.relu(self.fc2(x))
        disp7 = x.detach()
        x = self.head(x)
        return x, tuple(disps)+(disp6, disp7)


if __name__ == '__main__':
    # test one dir
    dqn = DQNConv(nb_actions=9, nb_vel=15)
    image_size = (1,2,16,160)
    inputVar = torch.rand(image_size)
    print(inputVar.size())
    inputVar2 = torch.rand((1,15))
    print(inputVar2.size())
    outputVar = dqn(inputVar, inputVar2)
    print(outputVar.size())
    print('{} {}'.format(outputVar.max(), outputVar))
    x, disps = dqn.forward_disp(inputVar, inputVar2)

    # test omniDir
    dqn = DQNConv(nb_actions=9, nb_vel=15, omni_dir=True)
    image_size = (1,2,16,160)
    inputVar = (torch.rand(image_size), torch.rand(image_size), torch.rand(image_size), torch.rand(image_size))
    inputVar2 = torch.rand((1,15))
    print(inputVar2.size())
    outputVar = dqn(inputVar, inputVar2)
    print(outputVar.size())
    print('{} {}'.format(outputVar.max(), outputVar))
    x, disps = dqn.forward_disp(inputVar, inputVar2)
    
    # import ipdb; ipdb.set_trace()

