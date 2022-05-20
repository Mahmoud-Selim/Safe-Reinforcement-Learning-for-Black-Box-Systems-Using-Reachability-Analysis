
# this script is modified from tf-agents TD3 Examples
r"""Train and Eval TD3.
To run:
```bash
tensorboard --logdir $HOME/tmp/td3/gym/HalfCheetah-v2/ --port 2223 &
python tf_agents/agents/td3/examples/v2/train_eval.py \
  --root_dir=$HOME/tmp/td3/gym/HalfCheetah-v2/ \
  --num_iterations=2000000 \
  --alsologtostderr
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import rospy
from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf

from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.td3 import td3_agent
from tf_agents.drivers import dynamic_step_driver
#from tf_agents.environments import suite_mujoco
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from env_wrappers import TFEnv 
from tf_agents.policies.policy_saver import PolicySaver
#from safety_layer import safety_layer
import numpy as np 
from tensorboardX import SummaryWriter
####################################mbrl#####################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf

#import mbrl.env.cartpole_continuous as cartpole_env
#import mbrl.env.reward_fns as reward_fns
#import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util
import mbrl.types
from tf_agents.trajectories import TimeStep
from tf_agents.trajectories import PolicyStep
import joblib
from utils.MetricsCollector import MetricsCollector 
#############################################################################

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_integer('num_iterations', 1000000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_param', None, 'Gin binding parameters.')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(
    root_dir,
    env_name='Turtlebot-Vanilla',
    num_iterations=2000000,
    actor_fc_layers=(512, 256, 64),
    critic_obs_fc_layers=(512,),
    critic_action_fc_layers=None,
    critic_joint_fc_layers=(300,),
    # Params for collect
    initial_collect_steps=2000,
    collect_steps_per_iteration=1,
    replay_buffer_capacity=100000,
    exploration_noise_std=0.1,
    # Params for target update
    target_update_tau=0.05,
    target_update_period=5,
    # Params for train
    train_steps_per_iteration=1,
    batch_size=64,
    actor_update_period=2,
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-3,
    dqda_clipping=None,
    td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
    gamma=0.995,
    reward_scale_factor=1.0,
    gradient_clipping=None,
    use_tf_functions=True,
    # Params for eval
    num_eval_episodes=10,
    eval_interval=1000,
    # Params for checkpoints, summaries, and logging
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    eval_metrics_callback=None):


  
  
  rospy.init_node('Vanilla_TB')
  """A simple train and eval for TD3."""
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')
  safety_dir = os.path.join(root_dir, 'safety')
  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()
  #writer = SummaryWriter(safety_dir)
  metrics_collector = MetricsCollector(safety_dir)
  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000)
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
  ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    env_name = 'Vanilla_TB'
    #env = suite_gym.load(env_name)
    tf_env = tf_py_environment.TFPyEnvironment(TFEnv)#tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    eval_tf_env = tf_py_environment.TFPyEnvironment(TFEnv)#tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))

    actor_net = actor_network.ActorNetwork(
        tf_env.time_step_spec().observation,
        tf_env.action_spec(),
        fc_layer_params=actor_fc_layers,
    )

    critic_net_input_specs = (tf_env.time_step_spec().observation,
                              tf_env.action_spec())

    critic_net = critic_network.CriticNetwork(
        critic_net_input_specs,
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
    )
    noise = tf.Variable(0.2)
    print("*"*80)
    tf_agent = td3_agent.Td3Agent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        exploration_noise_std=noise,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        actor_update_period=actor_update_period,
        #dqda_clipping=dqda_clipping,
        td_errors_loss_fn=td_errors_loss_fn,
        target_policy_noise = noise,
        #target_policy_noise_clip = 0.2,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=global_step,
    )

    tf_agent.initialize()

    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric(),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity)

    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps)
    
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=collect_steps_per_iteration)

    if use_tf_functions:
      initial_collect_driver.run = common.function(initial_collect_driver.run)
      collect_driver.run = common.function(collect_driver.run)
      tf_agent.train = common.function(tf_agent.train)

    # Collect initial replay data.
    print(
        'Initializing replay buffer by collecting experience for {} steps with '
        'a random policy.'.format(initial_collect_steps))
    initial_collect_driver.run()


    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)
    timed_at_step = global_step.numpy()
    time_acc = 0

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    def train_step():
      experience, _ = next(iterator)
      return tf_agent.train(experience)

    if use_tf_functions:
      train_step = common.function(train_step)

    rate = rospy.Rate(5)

    dream = False
    collect_reachability_data = False
    load_models = False
    models_path = ""
    
    reachability_actions = []
    reachability_states = []

    restore_checkpoint = False
    restore_checkpoint_step = 410000
    if(restore_checkpoint):
      checkpoint_dir = '/home/mahmoud/td3/td3_policy_%d' % restore_checkpoint_step

      my_policy = tf_agent.collect_policy
      saver = PolicySaver(my_policy, batch_size=None)
      saver.save(checkpoint_dir + "/policy_saver")

      if(dream):
        torch.save(dynamics_model.state_dict(), checkpoint_dir + "/dynamics_model")
        joblib.dump(mbrl_replay_buffer, checkpoint_dir)
      
      train_checkpointer = common.Checkpointer(
          ckpt_dir=checkpoint_dir + "/checkpointer",
          max_to_keep=1000,
          agent=tf_agent,
          policy=tf_agent.policy,
          replay_buffer=replay_buffer,
          global_step=global_step
      )

      train_checkpointer.initialize_or_restore()
      global_step = tf.compat.v1.train.get_global_step()
      print("Checkpoint restoration complete")

    print("TD3 Agent...")
    print("Start Training...")
    tf_env.envs[0].enable_logging()
    for _ in range(num_iterations):
      #rate.sleep()
      start_time = time.time()
      times = []

      if time_step is None:
        time_step = tf_env.current_time_step()
      if policy_state is None:
        policy_state = collect_policy.get_initial_state(tf_env.batch_size)

      action_step = collect_policy.action(time_step, policy_state)
      

      if(dream):
        mbrl_time_step = TimeStep(discount = tf.identity(time_step.discount), reward = tf.identity(time_step.reward), 
                                  observation = tf.identity(time_step.observation), step_type = tf.identity(time_step.step_type))
        mbrl_action_step = PolicyStep(action = tf.identity(action_step.action))
        plan = [action_step.action.numpy()]
        for i in range(4):
          
          _current_obs = torch.from_numpy(mbrl_time_step.observation.numpy().astype(np.float64)).to(device)
          actions = torch.from_numpy(mbrl_action_step.action.numpy().astype(np.float64)).to(device)

          model_in = mbrl.types.TransitionBatch(
                      _current_obs, actions, None, None, None
                  )

          next_observs, pred_rewards = dynamics_model.sample(
                      model_in,
                      deterministic=True,
                      rng=_rng,
                  )

          mbrl_time_step = TimeStep(discount = mbrl_time_step.discount, reward = mbrl_time_step.reward, 
                                  observation = tf.identity(next_observs.clone().detach().cpu().numpy().astype(np.float32)), step_type = mbrl_time_step.step_type)
          mbrl_action_step = collect_policy.action(mbrl_time_step, policy_state)
          plan.append(mbrl_action_step.action.numpy())
        
      ##########################################################################################################################################
      if(collect_reachability_data):
        random_a = np.array([[0, 0]], dtype = np.float32)
        random_a[0][0] = np.random.uniform(0, .25)
        random_a[0][1] = np.random.uniform(-.5, .5)
        random_a[0][2] = np.random.uniform(-5, 5)
        action_step = action_step._replace(action = random_a)
        reachability_states.append(np.array(list(tf_env.envs[0].get_odom().odom) + [list(time_step.is_last())]))
        reachability_actions.append(action_step.action)
      #############################################################################################################################

      with tf.control_dependencies(tf.nest.flatten([time_step])):
        next_time_step = tf_env.step(action_step.action)


      policy_state = action_step.state

      traj = trajectory.from_transition(time_step, action_step, next_time_step)
      observer_ops = [observer(traj) for observer in collect_driver._observers]
      transition_observer_ops = [
          observer((time_step, action_step, next_time_step))
          for observer in collect_driver._transition_observers
      ]
      with tf.control_dependencies(
          [tf.group(observer_ops + transition_observer_ops)]):
        time_step, next_time_step, policy_state = tf.nest.map_structure(
            tf.identity, (time_step, next_time_step, policy_state))

      #counter += tf.cast(traj.is_boundary(), dtype=tf.int32)
      time_step = next_time_step


      metrics_collector.add_robot_speed(global_step, np.linalg.norm(tf_env.envs[0].get_odom().odom[-4:-1]))
      if(time_step.is_last() == True):

        step = tf_env.envs[0].episodes_number
        metrics_collector.add_goal_rate(step, tf_env.envs[0].arrivals_number/tf_env.envs[0].episodes_number)
        metrics_collector.add_collision_rate(step, tf_env.envs[0].collisions_number/tf_env.envs[0].episodes_number)
        metrics_collector.add_time_to_goal(step, tf_env.envs[0].one_round_step)
        
      for _ in range(train_steps_per_iteration):
        train_loss = train_step()

      time_acc += time.time() - start_time
      noise = noise * 0.999997

      if global_step.numpy() % log_interval == 0:
        print('step = {}, loss = {}, noise = {}'.format(global_step.numpy(),
                     train_loss.loss, noise))
        steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
        print('{} steps/sec'.format(steps_per_sec))
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = global_step.numpy()
        time_acc = 0
        if(collect_reachability_data):
          np.save("/home/mahmoud/exp/exp2/reachability_states", reachability_states)
          np.save("/home/mahmoud/exp/exp2/reachability_actions", reachability_actions)
    
      if (global_step.numpy() % 10000 == 0):
        checkpoint_dir = '/home/mahmoud/td3/td3_policy_%d' % global_step
        my_policy = tf_agent.collect_policy
        saver = PolicySaver(my_policy, batch_size=None)
        saver.save(checkpoint_dir + "/policy_saver")

        if(dream):
          torch.save(dynamics_model.state_dict(), checkpoint_dir + "/dynamics_model")
          joblib.dump(mbrl_replay_buffer, checkpoint_dir)
        
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir + "/checkpointer",
            max_to_keep=1000,
            agent=tf_agent,
            policy=tf_agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step
        )
        train_checkpointer.save(global_step)

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])
    return train_loss


def main(_):
  tf.compat.v1.enable_v2_behavior()
  logging.set_verbosity(logging.INFO)
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  train_eval(FLAGS.root_dir, num_iterations=FLAGS.num_iterations)


if __name__ == '__main__':
  FLAGS.root_dir = "/home/mahmoud/td3"
  flags.mark_flag_as_required('root_dir')
  app.run(main)
  #main()