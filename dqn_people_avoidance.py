import cv2
import torch
import torch.optim as optim
import numpy as np
np.set_printoptions(precision=2, suppress=True)

from AirsimEnv import PeopleAvoidDiscreteEnv, PeopleAvoidDiscreteOneDirEnv
# from DQNConv import DQNConv
from ReplayMemory import ReplayMemory
from arguments import get_args

from workflow import WorkFlow, TorchFlow


class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy

    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end
    """

    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps

        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        """
        Select a random action out of `num_actions` possibilities.

        Attributes:
            num_actions (int): Number of actions available
        """
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        """ Compute the epsilon parameter according to the specified step

        Attributes:
            step (int)
        """
        if step < 0:
            return 0. # no exploration if the step is negtive
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start

    def is_exploring(self, step):
        """ Commodity method indicating if the agent should explore

        Attributes:
            step (int) : Current step

        Returns:
             bool : True if exploring, False otherwise
        """
        return np.random.rand() < self._epsilon(step)


class Agent(object):
    def __init__(self, qnet, targetqnet, explorer, actionnum):
        self.qnet = qnet
        self.targetqnet = targetqnet
        self.explorer = explorer
        self.actionnum = actionnum

    def state2input(self, state):
        if state[0].shape[1]>1: # omni-dir
            inputTensor0 = (torch.Tensor(state[0][:,0,:]).cuda(), 
                            torch.Tensor(state[0][:,1,:]).cuda(),
                            torch.Tensor(state[0][:,2,:]).cuda(),
                            torch.Tensor(state[0][:,3,:]).cuda())
        else:
            inputTensor0 = torch.Tensor(state[0][:,0,:]).cuda()
        inputTensor1 = torch.Tensor(state[1]).cuda()
        return inputTensor0, inputTensor1

    def net_forward(self, net, state): # shared function by qnet and targetqnet
        '''
        state[0]: n x 1/4 x c x h x w -> 1: one-dir, 4: omni-dir
        state[1]: n x k
        '''
        inputTensor0, inputTensor1 = self.state2input(state)
        q_values = net(inputTensor0, inputTensor1) 

        return q_values

    def targetqnet_forward(self, state):
        targetnet_qvalue = self.net_forward(self.targetqnet, state)
        return targetnet_qvalue.detach().cpu().numpy()

    def qnet_forward(self, state):
        return self.net_forward(self.qnet, state)

    def get_random_action(self,):
        return self.explorer(self.actionnum)

    def get_action(self, state, step):

        if self.explorer.is_exploring(step):
            action = self.get_random_action()
            q_mean, q_std = None, None
            # print '   random action', action
        else:
            q_values = self.qnet_forward(state)
            q_values = q_values.detach().cpu().squeeze().numpy()
            #ipdb.set_trace()
            q_mean = np.mean(q_values)
            q_std = np.std(q_values)

            action = np.argmax(q_values)
            # print '    action', action

        return action, q_mean, q_std

    def get_action_vis(self, state): # display the intermediate layers
        inputTensor0, inputTensor1 = self.state2input(state)
        q_values, disps = self.qnet.forward_disp(inputTensor0, inputTensor1) 
        q_values = q_values.detach().cpu().squeeze().numpy()
        disps_np = []
        for disp in disps:
            disp = disp.detach().cpu().squeeze().numpy()
            disps_np.append(disp)
        # import ipdb;ipdb.set_trace()
        return q_values, disps_np

class QLearning():
    def __init__(self, agent, optimizer, replaymemory, gamma, batchsize):
        self.agent = agent
        self.optimizer = optimizer
        self.memory = replaymemory
        self.gamma = gamma
        self.batchsize = batchsize
        self.criterion = torch.nn.SmoothL1Loss() # for baseline

    def compute_q_targets(self, state, rewards, terminals): # TOTEST
        ret_vals = []
        #ipdb.set_trace()
        post_states_q = self.agent.targetqnet_forward(state)
        #ipdb.set_trace()
        for i, terminal in enumerate(terminals):
            if terminal:
                ret_vals.append(rewards[i])
            else:
                ret_vals.append(rewards[i] + self.gamma*np.max(post_states_q[i]))
        return torch.Tensor(ret_vals)

    def update(self,): 
        pre_states, actions, post_states, rewards, terminals = self.memory.minibatch(self.batchsize)

        q_targets = self.compute_q_targets(post_states, rewards, terminals)
    
        actions = torch.from_numpy(actions).type(torch.LongTensor).cuda()
        q_vals_pre_states =  self.agent.qnet_forward(pre_states).gather(1, actions.view(-1,1))
        loss = self.criterion(q_vals_pre_states.squeeze(-1), q_targets.cuda())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss':loss.item()}


class TrainDQNPeopleAvoidance(TorchFlow.TorchFlow):
    def __init__(self, workingDir, args, prefix = "", suffix = "", plotterType = 'Visdom'):
        super(TrainDQNPeopleAvoidance, self).__init__(workingDir, prefix, suffix, disableStreamLogger = True, plotterType = plotterType)
        self.args = args
        self.prefix = prefix
        self.saveModelName = 'dqn'
        self.frameNum = args.multi_frame

        if args.omni_dir:
            self.imgshape = (4, 1+self.frameNum, 16, 160)
        else:
            self.imgshape = (1, 1+self.frameNum, 16, 160)
        self.velshape = (15,)

        self.lr = self.args.lr
        self.period = 0.1/self.args.speedup

        self.AV['loss'].avgWidth = 1000
        self.add_accumulated_value('reward',10)
        self.add_accumulated_value('qmean', 1000)
        self.add_accumulated_value('qstd', 1000)
        self.append_plotter("loss", ['loss'], [True])
        self.append_plotter("reward", ['reward'], [True])
        self.append_plotter("qvalue", ['qmean', 'qstd'], [True, True])

        self.countTrain = 0
        self.countTest = 0

        if self.args.omni_dir:
            self.env =  PeopleAvoidDiscreteEnv(period=self.period, framenum=self.frameNum, strictsafe=args.strict_safe) # DummyEnv() # 
        else:
            self.env =  PeopleAvoidDiscreteOneDirEnv(period=self.period, strictsafe=args.strict_safe)
        self.current_state = self.env.reset()
        self.current_state = self.stateTransform(self.current_state)
        self.action_num = self.env.action_space.shape[0]
        self.episode_reward = []

        from DQNConv import DQNConv
        self.dqn = DQNConv(self.action_num, self.velshape[0], omni_dir=self.args.omni_dir, input_num=self.frameNum+1)
        self.targetdqn = DQNConv(self.action_num, self.velshape[0], omni_dir=self.args.omni_dir, input_num=self.frameNum+1)

        if self.args.load_qnet:
            dqn_model = self.args.working_dir + '/models/' + self.args.qnet_model
            self.load_model(self.dqn, dqn_model)
        self.targetdqn.load_state_dict(self.dqn.state_dict())
        self.dqn.cuda()
        self.targetdqn.cuda()

        epsilon_start, epsilon_end = float(self.args.epsilon.split('_')[0]), float(self.args.epsilon.split('_')[1])
        self.agent = Agent(self.dqn, self.targetdqn, LinearEpsilonAnnealingExplorer(epsilon_start, epsilon_end, self.args.train_step), self.action_num)
        self.memory = ReplayMemory(self.args.memory_size, self.imgshape, self.velshape)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr = self.lr, eps=1e-5) #,weight_decay=1e-5)
        self.algo = QLearning(self.agent, self.optimizer, self.memory, self.args.gamma, self.args.batch_size)

        if self.args.load_memory:
            self.memory.load_memory(prefix=self.args.memory_prefix)

    def initialize(self):
        super(TrainDQNPeopleAvoidance, self).initialize()
        # fill memory
        if not self.args.test and len(self.memory)<self.args.batch_size:
            for k in range(self.args.batch_size):
                action = self.agent.get_random_action()
                state, reward, finish, _ = self.env.step(action)
                self.current_state = self.stateTransform(state)
                self.episode_reward.append(reward)
                self.memory.append(self.current_state[0][0], action, reward, finish, self.current_state[1][0])
                if finish:
                    self.episode_reward = []

                    self.current_state = self.env.reset()
                    self.current_state = self.stateTransform(self.current_state)

        if self.args.test:
            self.droneimg = self.loadDroneImg()

        super(TrainDQNPeopleAvoidance, self).post_initialize()
        
    def stateTransform(self, state): 
        '''
        state: state['img']: 1/4 x 2 x 80 x 32
               state['vel']: 3
        return: state[0]: 1 x 1/4 x 2 x 80 x 32 --> handle the discripency between sampling (1 x 2 x 80 x 32) and training (batch x 1 x 2 x 80 x 32)
                state[1]: 1 x 30 
        '''
        state_img = state['img'].copy()
        state_img[:,0,:,:] = state_img[:,0,:,:].clip(0.2) # minimun distance 0.2m
        state_img[:,0,:,:] = 1.0/state_img[:,0,:,:] # inverse distance

        state_vel = np.tile(state['vel'], 5) # repeat the vel input
        return np.expand_dims(state_img,axis=0), state_vel.reshape((1,)+state_vel.shape)

    def train(self):
        super(TrainDQNPeopleAvoidance, self).train()
        self.countTrain += 1

        action, qmean, qstd = self.agent.get_action(self.current_state, self.countTrain)

        state, reward, finish,_ = self.env.step(action)
        self.current_state = self.stateTransform(state)
        self.episode_reward.append(reward)
        self.memory.append(self.current_state[0][0], action, reward, finish, self.current_state[1][0])

        if qmean is not None and qstd is not None:
            self.AV['qmean'].push_back(qmean, self.countTrain)
            self.AV['qstd'].push_back(qstd, self.countTrain)

        if finish:
            self.AV['reward'].push_back(sum(self.episode_reward), self.countTrain)
            self.episode_reward = []

            self.current_state = self.env.reset()
            self.current_state = self.stateTransform(self.current_state)

        #ipdb.set_trace()
        if self.countTrain % self.args.train_interval == 0:

            # import ipdb;ipdb.set_trace()
            plot_ret = self.algo.update()     
            for item in plot_ret.keys():
                self.AV[item].push_back(plot_ret[item], self.countTrain)

        if ( self.countTrain % self.args.plot_interval == 0):
            self.plot_accumulated_values()
            losslogstr = self.get_log_str()
            self.logger.info("%s #%d - %s lr: %.6f"  % (self.args.exp_prefix[:-1], 
                self.countTrain, losslogstr, self.lr))

        if ( self.countTrain % self.args.snapshot == 0 ):
            self.write_accumulated_values()
            self.draw_accumulated_values()
            self.save_model(self.dqn, self.saveModelName+'_'+str(self.countTrain))
            self.memory.save_memory(prefix=self.prefix)

        if self.countTrain % self.args.target_update_interval == 0:
            self.targetdqn.load_state_dict(self.dqn.state_dict())
            print('===> Update target_net...')
            print('')


    def test(self, ):
        
        self.countTest += 1

        action, qmean, qstd = self.agent.get_action(self.current_state, -1)
        state, reward, finish, _ = self.env.step(action)
        self.current_state = self.stateTransform(state)
        self.episode_reward.append(reward)
        self.memory.append(self.current_state[0][0], action, reward, finish, self.current_state[1][0])
        self.visualizeAction(action)

        print('')
        print('{}, action {}'.format(self.countTest, action))
        if finish:
            print('reward for this episode: {}'.format(sum(self.episode_reward)))
            self.episode_reward = []

            self.current_state = self.env.reset()
            self.current_state = self.stateTransform(self.current_state)

            return True

        return False

    def action2coord(self, action, centerpt=(25,25), linelen=20):
        if action==0:
            endpt = centerpt
            return endpt
        # action = (action+5)%8 + 1 # adjust the visulization direction
        if action==1:
            endpt = (centerpt[0], centerpt[1]-linelen)
        elif action==2:
            endpt = (centerpt[0]+linelen, centerpt[1]-linelen)
        elif action==3:
            endpt = (centerpt[0]+linelen, centerpt[1])
        elif action==4:
            endpt = (centerpt[0]+linelen, centerpt[1]+linelen)
        elif action==5:
            endpt = (centerpt[0], centerpt[1]+linelen)
        elif action==6:
            endpt = (centerpt[0]-linelen, centerpt[1]+linelen)
        elif action==7:
            endpt = (centerpt[0]-linelen, centerpt[1])
        elif action==8:
            endpt = (centerpt[0]-linelen, centerpt[1]-linelen)
        else:
            print('action error..')
        return endpt

    def visualizeAction(self, action):
        imgsize = 50
        startpt = (int(imgsize/2),int(imgsize/2))
        endpt = self.action2coord(action,startpt)

        img=np.ones((imgsize,imgsize)).astype(np.uint8)*255
        cv2.imshow('action',cv2.arrowedLine(img,startpt,endpt,color=(0,0,255),thickness=2))
        cv2.waitKey(2)

    def activation2imgs(self, disps):
        activationlist = []
        shapelist = []
        for k,disp in enumerate(disps):
            scale = k/2+1
            disp_norm = (disp/disp.std()).clip(0,1)
            if len(disp_norm.shape)==1: # FC layers
                disp_norm = disp_norm.reshape(disp_norm.shape[-1],1)
                disp_norm = cv2.resize(disp_norm, (0,0), fx=4, fy=4, interpolation=3)
            else:
                if len(disp_norm.shape)==2:
                    disp_norm = np.expand_dims(disp_norm,axis=1)
                # add gray strip
                disp_norm = np.concatenate((disp_norm,np.ones_like(disp_norm)[:,0:int(4/scale+1),:]*0.9),axis=1)
                disp_norm = disp_norm.reshape(-1,disp_norm.shape[-1])
                disp_norm = cv2.resize(disp_norm, (0,0), fx=scale, fy=scale, interpolation=3)
            # imshow
            activationlist.append(disp_norm)
            shapelist.append(disp_norm.shape)
        return activationlist, shapelist

    def state2img(self, state):
        imgshape = state.shape # camnum x frames x h x w
        smallspace = np.ones((10,imgshape[-1]),dtype=np.float32) * 0.9
        # imgh = imgshape[2]*imgshape[0] + 10 * (imgshape[0]-1)
        # imgdisp = np.zeros((imgh, imgshape[3]),dtype=np.uint8)
        imgdisp = np.ones((0, imgshape[-1]), dtype=np.float32)
        for k in range(imgshape[0]):
            depthdisp = np.clip(state[k,0,:,:]*2.0/256.0,0,1)
            imgdisp = np.concatenate((imgdisp,depthdisp,smallspace),axis=0)
            for w in range(1, imgshape[1]):
                imgdisp = np.concatenate((imgdisp,state[k,w,:,:],smallspace),axis=0)
            imgdisp = np.concatenate((imgdisp,smallspace),axis=0)
        # cv2.imshow('input',imgdisp)
        # cv2.waitKey(1)
        return imgdisp

    def qvalue2img(self, qvalues):
        imgsize = 200
        startpt = (imgsize/2,imgsize/2)
        qimg=np.ones((imgsize,imgsize)).astype(np.float32) * 0.9
        qvis = (qvalues-qvalues.mean())/qvalues.std()
        qvis = np.exp(qvis)*2
        qvis = qvis/qvis.sum()
        for k,q in enumerate(qvis):
            endpt = self.action2coord(k,startpt,linelen=int(imgsize*q))
            cv2.arrowedLine(qimg,startpt,endpt,color=(0,0,1),thickness=2)
        print('Q values: {}'.format(qvalues))
        # cv2.imshow('qvalue',qimg)
        # cv2.waitKey(1)
        return qimg

    def loadDroneImg(self, imgname='droneimg.png'):
        droneimg = cv2.imread(imgname).astype(np.float32)
        droneimg = np.clip(droneimg/255.0, 0, 1)
        droneimg = cv2.resize(droneimg, (0,0), fx=0.3, fy=0.3)
        return droneimg

    def alignCenterImgs(self, imglist, shapelist, imggap = 50):
        imgw, imgh = 0, 0
        for imgshape in shapelist:
            if imgh<imgshape[0]:
                imgh = imgshape[0]
            imgw += (imggap + imgshape[1])
        imgh += imggap
        alignimg = np.ones((imgh,imgw,3),dtype=np.float32) * 0.9
        startx, starty = 0, imggap/2
        for imgshape,img in zip(shapelist,imglist):
            startx = imgh/2-imgshape[0]/2 # align middle 
            if len(imgshape)==2:
                img = np.tile(np.expand_dims(img,axis=2),3)
            alignimg[startx:startx+imgshape[0],starty:starty+imgshape[1],:] = img
            starty += (imggap + imgshape[1])
        return alignimg

    def vis_network(self, state, qvalue, disps):
        activationlist, shapelist = self.activation2imgs(disps)
        inputimg = self.state2img(state['img'])
        qimg = self.qvalue2img(qvalue)
        # draw images together
        actimg = self.alignCenterImgs(activationlist, shapelist, imggap = 30)
        dispimg = self.alignCenterImgs([inputimg,actimg],[inputimg.shape, actimg.shape],imggap=50)
        dispimg = self.alignCenterImgs([dispimg,self.droneimg,qimg],[dispimg.shape, self.droneimg.shape, qimg.shape],imggap=10)

        cv2.imshow('activation',dispimg)
        cv2.waitKey(1)



    def test_vis(self, ):
        
        self.countTest += 1

        qvalue, disps = self.agent.get_action_vis(self.current_state)

        action = np.argmax(qvalue)
        state, reward, finish, _ = self.env.step(action)

        # self.env.render(state)
        self.current_state = self.stateTransform(state)
        self.episode_reward.append(reward)
        # self.memory.append(self.current_state[0][0], action, reward, finish, self.current_state[1][0])
        self.visualizeAction(action)
        self.vis_network(state, qvalue, disps)

        print('')
        print('{}, action {}'.format(self.countTest, action))
        if finish:
            print('reward for this episode: {}'.format(sum(self.episode_reward)))
            self.episode_reward = []

            self.current_state = self.env.reset()
            self.current_state = self.stateTransform(self.current_state)

            return True

        return False

    def finalize(self):
        super(TrainDQNPeopleAvoidance, self).finalize()
        if not self.args.test and self.countTrain != self.args.train_step:
            self.write_accumulated_values()
            self.draw_accumulated_values()
            self.save_model(self.dqn, self.saveModelName+'_'+str(self.countTrain))
            self.memory.save_memory(prefix=self.prefix)
        self.env.close()
        self.logger.info("Finalized.")

        
if __name__ == '__main__':

    args = get_args()
    if args.use_int_plotter:
        plottertype = 'Int'
    else:
        plottertype = 'Visdom'
    try:
        # Instantiate an object for MyWF.
        traindqn = TrainDQNPeopleAvoidance(args.working_dir, args, prefix = args.exp_prefix, plotterType = plottertype)

        # Initialization.
        traindqn.initialize()
        while True:
            if not args.test:
                traindqn.train()
                if (traindqn.countTrain>=args.train_step):
                    break
            else:
                if traindqn.test_vis():
                    break

        traindqn.finalize()

    except WorkFlow.SigIntException as sie:
        print( sie.describe() )
        print( "Quit after finalize." )
        traindqn.finalize()
    except WorkFlow.WFException as e:
        print( e.describe() )
        traindqn.finalize()

    print("Done.")
