import numpy as np
np.random.seed(0)    # so I can reproduce things
from physics_sim import PhysicsSim
class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.init_pose = init_pose
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 100.])
        #print(init_pose)
    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = np.tanh(1 - 0.03*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        #print("we are at position {} and target is {}".format(self.sim.pose[:3],self.target_pos))
        #print("sim.pose is {}".format(self.sim.pose))
        #print("velocities are {}".format(self.sim.v))
        # if we are below target height, we need a positive upward velocity, otherwise we need negative vertical velocity
        if self.sim.pose[2] < self.target_pos[2] and self.sim.v[2] > 0.0:
            up_down_reward = 2
            #print("correctly heading up")
        elif self.sim.pose[2] > self.target_pos[2] and self.sim.v[2] < 0.0:
            #print("correctly heading down")
            up_down_reward = 2
        else:
            up_down_reward = -5

        # give a penalty it it strays in x,y plane
        lat_err = np.tanh(1 - 0.075*(abs(self.sim.pose[:1] - self.target_pos[:1])).sum())
        #print("lat_err {}".format(lat_err))

        if abs(self.sim.pose[2] - self.target_pos[2]) < 5.0:     # had .2 here
            #print("we are near target height")
            #print("we are at position {} and target is {}".format(self.sim.pose[:3],self.target_pos))
            vertical_reward = 2
        else:
            #print("not near target height")
            vertical_reward = -1
        # give a reward for being off the ground
        if self.sim.pose[2] < 0.01:
            flying_reward = 25
        else:
            flying_reward = -1
        # give reward if rotor speeds are similar
        #print(self.rotor_speeds)
        if max(self.rotor_speeds) - min(self.rotor_speeds) < 200 and min(self.rotor_speeds) >400:
            similarspeed_reward = 15     # ACS  250 was reasonable
            #print("rotors have similar speeds :-)")
            #print(self.rotor_speeds)
        else:
            similarspeed_reward = -15     # had -50 here
        #reward = up_down_reward + vertical_reward + flying_reward + similarspeed_reward
        reward = up_down_reward + vertical_reward + lat_err + flying_reward + similarspeed_reward
        #print("which gives me a reward of {:7.3f}".format(reward))
        return reward


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        self.rotor_speeds = rotor_speeds
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            if done :
                reward += 1000
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.init_pose] * self.action_repeat) 
        #print("state reset to {}".format(state))
        return state