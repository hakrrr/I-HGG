import numpy as np

import os
from gym.utils import EzPickle
from gym.envs.robotics import rotations, robot_env, utils





def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class myUR5GripperFallEnv(robot_env.RobotEnv, EzPickle):
    """Superclass for all myUR5Gripper environments.
    """

    def __init__(self, reward_type='sparse'):
        """Initializes a new myUR5Gripper environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        # TODO: Requirements for .xml:
        # mocap body in worldbody
        # target body in worldbody
        # finder body welded to mocap body in equality
        # configured initial position of joints such that all joints are reachable

        # TODO: Init configs
        self.gripper_extra_height = 0.
        self.block_gripper = True
        self.has_object = False
        self.target_in_the_air = False
        self.target_offset = 0.
        self.obj_range = 0.
        self.target_range = 0.8 #default:0.2
        self.distance_threshold = 0.15 #default 0.05
        self.reward_type = reward_type
        self.mocap_name = 'robot:mocap' # type body
        self.target_name = 'target0' # type body
        self.finder_name = 'tool0'#'wrist_1_link' # type body
        self.ref_name = 'gripperfinger_middle_link_3'
        self.object_name = "object0"
        n_substeps = 20
        model_path = os.path.join('myUR5Gripper', 'myUR5gripper_fall.xml')
        initial_qpos = {
            'robot:wrist_2_joint': -1.57,
            'robot:shoulder_pan_joint': 0.0,
            'robot:shoulder_lift_joint': -0.6,
            'robot:elbow_joint': 1.1,
            #'wrist_1_joint': -0.7,
            #'wrist_2_joint': 1.57,
            #'gripperfinger_1_joint_1': 0.0,
            #'gripperfinger_middle_joint_1': 0.0,
            #'gripperpalm_finger_1_joint': 0.0,
        }
        # TODO: update myUR5<>env
        super(myUR5GripperFallEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos)
        EzPickle.__init__(self)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self): # set initial position of gripper!

        True



    def _set_action(self, action):
        # TODO: set action, n_action (= number of actuators) and action control
        assert action.shape == (4,)
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [0., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (3,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):

        # TODO: set position that has to reach target (object or on body), set to achieved goal
        # here: body_pos is the position that has to reach target
        # positions
        grip_pos = self.sim.data.get_body_xpos(self.finder_name)
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_body_xvelp(self.finder_name) * dt
        # for the following to work, all joints associated with the robot have to start with "robot"
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        # since object
        object_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
        gripper_state = robot_qpos[-1:]
        gripper_vel = robot_qvel[-1:] * dt  # changed to a scalar since the gripper is made symmetric [-3:] else
        achieved_goal = np.squeeze(object_pos.copy())
        #if not object_pos[2] == 0.: # TODO: had changes in her/rollout.py (see todo)
        #    achieved_goal = np.zeros_like(object_pos.copy())
        #    achieved_goal.fill(np.nan)

        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        # TODO: modify view (not necessary)
        body_id = self.sim.model.body_name2id(self.finder_name)
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # TODO: Target visualizetion, works better with bodys than sites!
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        body_id = self.sim.model.body_name2id(self.target_name)
        self.sim.data.body_xpos[body_id] = self.goal
        self.sim.model.body_pos[body_id] = self.goal #- sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        # TODO: start position of object (only if there is an object)
        self.sim.set_state(self.initial_state)
        # initial_state is a deep copy of the sim state defined in _env_setup (see robot_env)
        # Randomize start position of object.

        self.sim.forward()
        return True

    def _sample_goal(self):
        # TODO: sample goal (change range)
        x = self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal = x# goal = self.initial_object_xpos[:3] + x #initial_gripper_xpos is defined in _env_setup
        goal[2] = 0.
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        # TODO: initial state of joints
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = self.sim.data.get_body_xpos(self.finder_name).copy() #+ np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height])
        # gripper_rotation is 1 0 1 0 for wirst_1_link as finder_name
        gripper_rotation = np.array([0., 0., 1., 0.])
        self.sim.data.set_mocap_pos(self.mocap_name, gripper_target)
        self.sim.data.set_mocap_quat(self.mocap_name, gripper_rotation)

        # move gripper into position
        for i in range(3):
            self.sim.data.ctrl[i] = 0.8
        for _ in range(10):
            self.sim.step()

        # move object into position
        ref_pos = self.sim.data.get_body_xpos(self.finder_name)
        offset = np.array([0., 0., -0.12])
        pos = ref_pos + offset
        rot = [0, 0, 0, 0]
        ob_pos = list(pos) + rot
        self.sim.data.set_joint_qpos("object0:joint", ob_pos)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_object_xpos = self.sim.data.get_body_xpos(self.object_name).copy()
        self.initial_gripper_xpos = self.sim.data.get_body_xpos(self.finder_name).copy()



    def render(self, mode='human', width=500, height=500):
        # TODO: update myUR5Gripper<>Env
        return super(myUR5GripperFallEnv, self).render(mode, width, height)






