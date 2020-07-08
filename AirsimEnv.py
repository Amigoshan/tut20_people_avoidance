import airsim
from airsim.utils import to_eularian_angles
from airsim.types import YawMode

import cv2 # debug
import numpy as np
from math import cos, sin, tanh, pi
import pprint
import time

class AirsimEnvBase(object):
    def __init__(self, ip='', camlist=[0]):
        self.client = airsim.MultirotorClient(ip=ip)
        self.client.confirmConnection()

        self.IMGTYPELIST = [#'Scene', 
                            'DepthPlanner', 
                            'Segmentation',
                            ] 
        self.CAMLIST = camlist #[0,1,4,2] # front, left, back, right 

        self.imgRequest = []
        for k in self.CAMLIST:
            for imgtype in self.IMGTYPELIST:
                if imgtype == 'Scene':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.Scene, False, False))

                elif imgtype == 'DepthPlanner':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.DepthPlanner, True))

                elif imgtype == 'Segmentation':
                    self.imgRequest.append(airsim.ImageRequest(str(k), airsim.ImageType.Segmentation, False, False))

        # manage the loop period
        self.starttime = time.time()
        self.lasttimeind = 0

        self.observation_space = np.zeros(2) # dummpy
        self.imgwidth, self.imgheight = 0, 0

    def readimgs(self):
        responses = self.client.simGetImages(self.imgRequest)
        rgblist, depthlist, seglist = [], [], []
        idx = 0
        for k in range(len(self.CAMLIST)):
            for imgtype in self.IMGTYPELIST:
                response = responses[idx]
                # response_nsec = response.time_stamp
                # response_time = rospy.rostime.Time(int(response_nsec/1000000000),response_nsec%1000000000)
                if response.height == 0 or response.width == 0:
                    print 'Something wrong with image return..', idx
                    return None, None, None
                if imgtype == 'DepthPlanner': #response.pixels_as_float:  # for depth data
                    img1d = np.array(response.image_data_float, dtype=np.float32)
                    depthimg = img1d.reshape(response.height, response.width) 
                    depthlist.append(depthimg)

                elif imgtype == 'Scene':  # raw image data
                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
                    rgbimg = img1d.reshape(response.height, response.width, -1)
                    rgblist.append(rgbimg[:,:,:-1])

                elif imgtype == 'Segmentation':
                    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
                    img_rgba = img1d.reshape(response.height, response.width, -1)
                    img_seg = img_rgba[:,:,0]
                    seglist.append(img_seg)
                idx += 1

        # assume the images are the same size
        self.imgwidth = response.width
        self.imgheight = response.height

        return rgblist, depthlist, seglist

    def world2drone(self, vx, vy, yaw):
        vvx = vx * cos(yaw) - vy * sin(yaw)
        vvy = vx * sin(yaw) + vy * cos(yaw)

        return vvx, vvy

    def readKinematics(self, dronestate):
        kinematics_estimated = dronestate.kinematics_estimated
        drone_pos = kinematics_estimated.position
        drone_quat = kinematics_estimated.orientation
        world_drone_vel = kinematics_estimated.linear_velocity
        (pitch, roll, yaw) = to_eularian_angles(drone_quat)

        # convert velocity to the drone frame
        drone_vel_x, drone_vel_y = self.world2drone(world_drone_vel.x_val, world_drone_vel.y_val, yaw)

        return drone_pos.x_val, drone_pos.y_val, drone_pos.z_val, \
               pitch, roll, yaw, \
               drone_vel_x, drone_vel_y, world_drone_vel.z_val

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        self.client.simPause(False)
        self.client.reset() 
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

class PeopleAvoidEnvBase(AirsimEnvBase):
    def __init__(self, period=0.1, camlist=[0], strictsafe=False):
        super(PeopleAvoidEnvBase, self).__init__(camlist=camlist) 
        self.client.reset()
        time.sleep(1)
        dronestate = self.client.getMultirotorState()
        _, _, self.iniHeight, _, _, self.iniYaw, _, _, _ = self.readKinematics(dronestate)

        # self.iniHeight = self.getCurrentHeight()
        self.targetHeight = -1.0 + self.iniHeight 
        self.heightThresh = 0.1

        self.setSegmentation()
        self.peopleIdx = 42
        self.bgIdx = 130

        self.x, self.y, self.z = 0, 0, 0
        self.vx, self.vy, self.vz = 0, 0, 0
        self.yaw = 0 # used for coordinate transformation
        self.last_x = 0 # reward calculation
        self.rgbimg, self.depthimg, self.segimg = None, None, None

        self.stepCount = 0
        self.episodeLen = 1000
        self.period = period

        self.x_axis_positive = False
        # self.printControl = True

        self.strict_safe = strictsafe

    def setSegmentation(self):
        success1 = self.client.simSetSegmentationObjectID("[\w]*", 0, True);
        success2 = self.client.simSetSegmentationObjectID("BP_person[\w]*", 1, True);
        return success1 and success2

    # def getCurrentHeight(self):
    #     drone_height = self.client.getMultirotorState().kinematics_estimated.position.z_val
    #     return drone_height

    def computeReward(self, x, y, action):
        # because the drone is facing south in the neighborhood env
        reward_forward = -(x - self.last_x) * self.lambda_forward # reward for moving forward 
        self.last_x = x

        if y < -1: # punish for deviation
            reward_deviation = (y+1) 
        elif y > 1:
            reward_deviation = (1-y) 
        else:
            reward_deviation = 0.
        if reward_deviation < -5:
            reward_deviation = -5 
        reward_deviation *= self.lambda_deviation

        collision_info = self.client.simGetCollisionInfo()
        # if collision_info.has_collided:
        #     print collision_info
        if collision_info.has_collided: # BP_person_180
            collide_obj = collision_info.object_name
            if collide_obj[:9] == 'BP_person':
                reward_collision = -10.
            else:
                reward_collision = -5.
        else:
            reward_collision = 0.

        # action cost/punish speed changing
        reward_action = self.getActionCost(action)
        
        reward = reward_forward + reward_deviation + reward_collision + reward_action
        # print 'reward %.5f (forward %.5f, deviation %.5f, collision %.1f, action %.5f)' % (reward, reward_forward, reward_deviation, reward_collision, reward_action)
        # print 
        if reward<-1000:
            print('REWARD ERROR: {}, {}, {}, {}, {}'.format(reward, reward_forward, reward_deviation, reward_collision, reward_action))
            # import ipdb;ipdb.set_trace()
        return reward, collision_info.has_collided

    def getObservation(self):
        # read drone's kinematics
        dronestate = self.client.getMultirotorState()
        self.x, self.y, self.z, _, _, self.yaw, self.vx, self.vy, self.vz = self.readKinematics(dronestate)
        # print '    state: ', self.x, self.y, self.z, self.vx, self.vy, self.vz
        if self.x_axis_positive:
            self.x = - self.x
            self.y = - self.y

        rgblist, depthlist, seglist = self.readimgs()
        if rgblist is None: # How to deal with the environmental error?
            print('AirsimEnv: Observation error!!')
            return None
        self.depthimg, self.segimg = depthlist[0], seglist[0]
        self.segimg[self.segimg==self.peopleIdx] = 1
        self.segimg[self.segimg==self.bgIdx] = 0
        # assert(self.depthimg.shape==self.segimg.shape)
        newshape = (1,1,self.depthimg.shape[0], self.depthimg.shape[1])
        returny = np.clip(self.y, -5, 5)
        state = {'img': np.concatenate((self.depthimg.reshape(newshape),self.segimg.reshape(newshape)),axis=1),
                'vel': np.array([returny, self.vx, self.vy], dtype=np.float32)}
        return state

    def isDone(self, is_collide):
        if self.stepCount>=self.episodeLen:
            return True
        if self.strict_safe and is_collide:
            return True
        return False

    def waitPeriod(self):
        '''
        wait for a certain amount of time to maintain a certain frequency
        '''
        thistime = time.time()
        time_elapse = thistime-self.starttime
        timeind = int(time_elapse/self.period) 
        if timeind > self.lasttimeind:
            # if self.printControl:
            #     self.printControl = False
            #     print '    WARN: Simu frequency is lower than desired! ', time_elapse-self.lasttimeind*self.period, timeind
            self.lasttimeind = 0
            self.starttime = thistime
        else: # wait for some time 
            waittime = (timeind+1)*self.period - time_elapse 
            # print '    waittime',waittime
            time.sleep(waittime)
            self.lasttimeind = timeind + 1

            if timeind>100000000: # prevent overflow
                self.lasttimeind = 0
                self.starttime = thistime

    def step(self, action):
        self.stepCount += 1
        vx, vy, vz, vyaw = self.action2vel(action)
        yawrate_deg = vyaw * 180/pi 
        yawmode = YawMode(is_rate = True, yaw_or_rate = yawrate_deg)
        # moveByVelocity receive velocity in the world coordinates
        # this service provides velocity commands in the drone coordinates
        vvx, vvy = self.world2drone(vx, vy, self.yaw)
        self.client.moveByVelocityAsync(vvx, vvy, vz, duration=5, yaw_mode= yawmode)

        # print 'yaw:', self.yaw, vyaw

        self.client.simPause(False)
        time.sleep(self.period)
        self.client.simPause(True)

        # self.waitPeriod()
        # print time.time()-int(self.starttime/100)*100
        state = self.getObservation()

        reward, is_collide = self.computeReward(self.x, self.y, action)
        # print self.stepCount, ' - reward:', reward, ' action:', action

        finish = self.isDone(is_collide)

        # if self.stepCount%200==0:
        #     self.printControl = True
        if finish:
            episodeTime = time.time()-self.episodeStartTime
            print('Timeelapse:{}, average period: {}, speedup: {}'.format(episodeTime, episodeTime/self.stepCount, self.stepCount/episodeTime/10.) ) 
        return state, reward, finish, None

    def reset(self):
        self.client.simPause(False)
        self.client.reset() 
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        dronestate = self.client.getMultirotorState()
        self.x, self.y, self.z, _, _, self.yaw, self.vx, self.vy, self.vz = self.readKinematics(dronestate)
        while self.z > self.targetHeight:
            self.client.moveByVelocityAsync(0, 0, -0.5, duration=1)
            time.sleep(self.period)
            dronestate = self.client.getMultirotorState()
            self.x, self.y, self.z, _, _, self.yaw, self.vx, self.vy, self.vz = self.readKinematics(dronestate)
        # initialize the variables
        self.last_x = self.x
        self.rgbimg, self.depthimg, self.segimg = None, None, None
        self.stepCount = 0
        self.episodeStartTime = time.time()
        # print 'reset env done...'
        self.first_obs = self.getObservation()

        return self.first_obs

    def render(self):
        if self.depthimg is not None:
            depth_show = np.clip(self.depthimg*3,0,255).astype(np.uint8) # show 0-255/3 m
            cv2.imshow('img',depth_show)
            cv2.waitKey(2)
        if self.segimg is not None:
            cv2.imshow('seg',self.segimg.astype(np.float32))
            cv2.waitKey(2)


class PeopleAvoidDiscreteOneDirEnv(PeopleAvoidEnvBase):
    def __init__(self, period=0.1,camlist=[0], strictsafe=False):
        super(PeopleAvoidDiscreteOneDirEnv, self).__init__(period=period, camlist=camlist, strictsafe=strictsafe)
        # parameters for reward calculation
        self.lambda_forward = 1.0
        self.lambda_deviation = 0.1
        self.lambda_action = 0.01

        # kinematics of the drone
        self.maxAcc = 2.0 # m/s
        self.maxAccPerPeriod = self.maxAcc * self.period
        self.maxVel = 1.2 # m/s

        self.action_space = np.zeros(9)


    def action2vel(self, action):
        '''
        action: 0 - 8
        action[0]: delta vx
        action[1]: delta vy
        '''
        # if action == 0:
        #     vx, vy = self.vx,                        self.vy
        # elif action == 1:
        #     vx, vy = self.vx + self.maxAccPerPeriod, self.vy
        # elif action == 2:
        #     vx, vy = self.vx + self.maxAccPerPeriod, self.vy + self.maxAccPerPeriod
        # elif action == 3:
        #     vx, vy = self.vx,                        self.vy + self.maxAccPerPeriod  
        # elif action == 4:
        #     vx, vy = self.vx - self.maxAccPerPeriod, self.vy + self.maxAccPerPeriod  
        # elif action == 5:
        #     vx, vy = self.vx - self.maxAccPerPeriod, self.vy  
        # elif action == 6:
        #     vx, vy = self.vx - self.maxAccPerPeriod, self.vy - self.maxAccPerPeriod  
        # elif action == 7:
        #     vx, vy = self.vx,                        self.vy - self.maxAccPerPeriod  
        # elif action == 8:
        #     vx, vy = self.vx + self.maxAccPerPeriod, self.vy - self.maxAccPerPeriod  
        # else:
        #     print 'Invalid action:', action
        #     return 0., 0., 0.

        # vx = np.clip(vx, -self.maxVel, self.maxVel)
        # vy = np.clip(vy, -self.maxVel, self.maxVel)

        if action == 0:
            vx, vy = 0, 0
        elif action == 1:
            vx, vy = self.maxVel, 0
        elif action == 2:
            vx, vy = self.maxVel*0.8, self.maxVel*0.8
        elif action == 3:
            vx, vy = 0, self.maxVel  
        elif action == 4:
            vx, vy = -self.maxVel*0.8, self.maxVel*0.8  
        elif action == 5:
            vx, vy = -self.maxVel,  0
        elif action == 6:
            vx, vy = -self.maxVel*0.8, -self.maxVel*0.8  
        elif action == 7:
            vx, vy = 0, -self.maxVel  
        elif action == 8:
            vx, vy = self.maxVel*0.8, -self.maxVel*0.8  
        else:
            print('Invalid action:'.format(action))
            return 0., 0., 0.


        # height adjustment (bang bang)
        if self.z > self.targetHeight + self.heightThresh:
            vz = -0.2
        elif self.z < self.targetHeight - self.heightThresh:
            vz = 0.2
        else:
            vz = 0.

        # yaw adjustment
        yawdiff = self.yaw - self.iniYaw
        vyaw = 0
        if yawdiff < -pi:
            yawdiff += 2*pi
        if yawdiff > pi:
            yawdiff -= 2*pi
        if yawdiff > 0.02:
            vyaw = -0.1
        elif yawdiff < -0.02:
            vyaw = 0.1

        # print '    action vel:', vx, vy, vz
        return vx, vy, vz, vyaw

    def getActionCost(self, action):
        return 0
        # if action == 0: # no acceleration, no cost
        #     return 0
        # else: # constant cost for each acceleration
        #     return -self.maxAccPerPeriod * self.lambda_action

class PeopleAvoidDiscreteEnv(PeopleAvoidDiscreteOneDirEnv):
    def __init__(self, period=0.1, framenum=1, strictsafe=False):  # stack multiple frames of images
        super(PeopleAvoidDiscreteEnv, self).__init__(period=period, camlist=[0,1,4,2], strictsafe=strictsafe)
        # stack temporal inputs
        self.frame_interval = 2
        self.frame_num = framenum
        self.buf_len = self.frame_interval * self.frame_num
        self.cam_num = len(self.CAMLIST)

        self.readimgs() # update the image size 
        self.buf = np.zeros((self.buf_len, self.cam_num, self.imgheight, self.imgwidth),dtype=np.uint8) # frame x camnum x height x width
        self.buf_ind = -1

    def getObservation(self):
        # read drone's kinematics
        dronestate = self.client.getMultirotorState()
        self.x, self.y, self.z, _, _, self.yaw, self.vx, self.vy, self.vz = self.readKinematics(dronestate)
        # print '    state: ', self.x, self.y, self.z, self.vx, self.vy, self.vz
        if self.x_axis_positive:
            self.x = - self.x
            self.y = - self.y

        rgblist, depthlist, seglist = self.readimgs()
        if rgblist is None: # How to deal with the environmental error?
            print('AirsimEnv: Observation error!!')
            return None

        imglist = []
        self.buf_ind += 1
        if self.buf_ind>=self.buf_len:
            self.buf_ind = 0
        for k in range(self.cam_num):
            depthimg, segimg = depthlist[k], seglist[k]
            # import ipdb;ipdb.set_trace()
            segimg[segimg==self.peopleIdx] = 1
            segimg[segimg==self.bgIdx] = 0
            imgconcat = np.concatenate((np.expand_dims(depthimg, axis=0),np.expand_dims(segimg, axis=0)),axis=0)

            # add buf
            self.buf[self.buf_ind,k,:] = segimg.copy()

            # read buf and concate to the state
            # import ipdb; ipdb.set_trace()
            cat_ind = self.buf_ind
            for w in range(self.frame_num-1):
                cat_ind -= self.frame_interval
                if cat_ind<0:
                    cat_ind += self.buf_len
                imgconcat = np.concatenate((imgconcat, np.expand_dims(self.buf[cat_ind,k,:].copy(), axis=0)), axis=0)

            imglist.append(imgconcat)
        imglist_np = np.array(imglist).astype(np.float32)

        returny = np.clip(self.y, -5, 5)
        state = {'img': imglist_np, # 4 x (1+k) x height x width
                'vel': np.array([returny, self.vx, self.vy], dtype=np.float32)}
        return state

    def render(self, state):
        imgstate = state['img']
        imgshape = imgstate.shape
        print(imgshape)

        depth_show = np.zeros((imgshape[2], imgshape[3]*4),dtype=np.uint8)
        for k in range(imgshape[0]):
            depth_show[:,imgshape[3]*k:imgshape[3]*(k+1)] = np.clip(imgstate[k,0,:,:]*2,0,255).astype(np.uint8) # show 0-255/3 m
        cv2.imshow('img',depth_show)
        cv2.waitKey(2)

        # import ipdb;ipdb.set_trace()
        seg_show = np.zeros(((imgshape[1]-1)*imgshape[2],self.cam_num*imgshape[3]),dtype=np.float32)
        for k in range(self.cam_num):
            for w in range(imgshape[1]-1):
                seg_show[w*imgshape[2]:(w+1)*imgshape[2],k*imgshape[3]:(k+1)*imgshape[3]] = imgstate[k,w+1,:,:]
        cv2.imshow('frame',seg_show)
        cv2.waitKey(2)


class PeopleAvoidEnv(PeopleAvoidEnvBase):
    def __init__(self, period = 0.1):
        super(PeopleAvoidEnv, self).__init__()
        self.period = period
        # parameters for reward calculation
        self.lambda_forward = 1.0
        self.lambda_deviation = 0.1
        self.lambda_action = 0.01

        # kinematics of the drone
        self.maxAcc = 2.0 # m/s
        self.maxAccPerPeriod = self.maxAcc * self.period
        self.maxVel = 1.0 # m/s


    def action2vel(self, action):
        '''
        action: 1x2 array
        action[0]: delta vx
        action[1]: delta vy
        '''
        #delta_vx = tanh(action[0]) * self.maxAccPerPeriod
        #delta_vy = tanh(action[1]) * self.maxAccPerPeriod
        vx = np.clip(action[0], -self.maxVel, self.maxVel)
        vy = np.clip(action[1], -self.maxVel, self.maxVel)

        # height adjustment (bang bang)
        if self.z > self.targetHeight + self.heightThresh:
            vz = -0.2
        elif self.z < self.targetHeight - self.heightThresh:
            vz = 0.2
        else:
            vz = 0.

        # yaw adjustment
        yawdiff = self.yaw - self.iniYaw
        vyaw = 0
        if yawdiff < -pi:
            yawdiff += 2*pi
        if yawdiff > pi:
            yawdiff -= 2*pi
        if yawdiff > 0.02:
            vyaw = -0.1
        elif yawdiff < -0.02:
            vyaw = 0.1

        return vx, vy, vz, vyaw

    def getActionCost(self, action):
        action_cost = 0. # punish big action
        if action[0] > 2.0:
            action_cost -= (action[0]-2.0)
        elif action[0] < -2.0:
            action_cost -= (-2.0-action[0])
        if action[1] > 2.0:
            action_cost -= (action[1]-2.0)
        elif action[1] < -2.0:
            action_cost -= (-2.0-action[1])

        action_cost = action_cost * self.lambda_action
        if action_cost <-1:
            print('action cost error! {}, {}'.format(action, action_cost))
            action_cost = -1
        return action_cost


class DummyEnv(object):
    def __init__(self):
        self.count = 0
        self.action_space = np.zeros(9)
        self.observation_space = np.zeros(2)

    def reset(self):
        self.count = 0
        state={'img':np.random.rand(1,8,16,160).astype(np.float32),'vel':np.random.rand(3).astype(np.float32)}
        return state

    def step(self, action):
        self.count += 1
        if self.count>=100:
            finish = True
        else:
            finish = False

        state={'img':np.random.rand(4,2,16,160).astype(np.float32),'vel':np.random.rand(3).astype(np.float32)}
        return state, np.random.rand(), finish, None

    def render(self):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    '''
    env = PeopleAvoidEnv(period=0.05)
    env.reset()
    totalreward = 0
    for k in range(1000):
        state, reward, finish , _ = env.step([1.0,0.0])
        print reward
        env.render()
        totalreward += reward
    print '===total reward===',totalreward
    env.close()
    '''

    # # test omniDirEnv
    # env = PeopleAvoidDiscreteEnv()
    # env.reset()
    # for k in range(500):
    #     state, reward, finish, _ = env.step(1)#np.random.randint(9))
    #     print reward, state['vel']
    #     env.render()
    #     #time.sleep(0.1)
    # env.close()

    # # test oneDirEnv
    # env = PeopleAvoidDiscreteOneDirEnv()
    # env.reset()
    # for k in range(500):
    #     state, reward, finish, _ = env.step(1)#np.random.randint(9))
    #     print reward, state['vel']
    #     env.render()
    #     #time.sleep(0.1)
    # env.close()

    # test multiFrame
    env = PeopleAvoidDiscreteEnv(framenum = 10, strictsafe=True)
    env.reset()
    for k in range(200):
        state, reward, finish, _ = env.step(1)#np.random.randint(9))
        print('reward {}, vel {}'.format(reward, state['vel']))
        env.render(state)
        #time.sleep(0.1)
        if finish:
            break
    env.close()