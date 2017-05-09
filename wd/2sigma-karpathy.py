
# coding: utf-8

# In[ ]:

""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
#import cPickle as pickle
import _pickle as pickle
#import gym
import kagglegym as gym

# hyperparameters
#H = 200 # number of hidden layer neurons
H = 30
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 0.001 #1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False

# model initialization
#D = 80 * 80 # input dimensionality: 80x80 grid
D = 110
if resume:
  print('loading model')
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  #for t in reversed(xrange(0, r.size)):
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0.0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0 # ReLU nonlinearity
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  #print("eph", eph)
  #print("epdlogp", epdlogp)
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

#env = gym.make("Pong-v0")
env = gym.make()
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
steps = 0
report_every = batch_size / 2
row = 0
reward = 0.0
print(model["W1"][0:10])

while True:
  if render: env.render()
  if steps % report_every == 0:
    print("Episode %d, Step %d: " % (episode_number, steps), end="")
  print("%.2f " % reward, end="")
  if steps % report_every == report_every - 1:
    print()
  rows = observation.features.shape[0]
  action = observation.target
  #print("Forward prop for %d rows: " % rows)
  for row in range(0, rows):
      #if row % 50 == 0:
      #  print(".", end="")
      # preprocess the observation, set input to network to be difference image
      #cur_x = prepro(observation)
      cur_x = observation.features.head(row+1).tail(1).fillna(0).values.ravel()
      #cur_x = observation.features.fillna(0).T
      x = cur_x - prev_x if prev_x is not None else np.zeros(D) #np.zeros(cur_x.shape)
      prev_x = cur_x

      # forward the policy network and sample an action from the returned probability
      aprob, h = policy_forward(x)
      #action = 2 if np.random.uniform() < aprob else 3 # roll the dice!
      #action["y"][row] = aprob # SettingWithCopyWarning here
      action.loc[row, "y"] = -0.1 + 0.2 * aprob

      # record various intermediates (needed later for backprop)
      xs.append(x) # observation
      hs.append(h) # hidden state
      #y = 1 if action == 2 else 0 # a "fake label"
      y = 0.0 if aprob < 0.5 else 1.0 #aprob leads to 0 grad
      dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
      
  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  steps += 1

  for row in range(0, rows):
    reward_sum += reward * 1.0
    drs.append(reward * 1.0) # record reward (has to be done after we call step() to get reward for previous action)

  if done or steps % batch_size == 0: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory

    # compute the discounted reward backwards through time
    discounted_epr = discount_rewards(epr)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)
    discounted_epr = epr

    epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
    grad = policy_backward(eph, epdlogp)
    for k in model: 
        #print("grad", k, grad[k][0:2])
        grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    if episode_number % batch_size == 0:
      for k,v in model.items(): #iteritems():
        g = grad_buffer[k] # gradient
        #print("g", k, g[1])
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        #print("model", k, model[k][1])
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print('episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    if episode_number % 10 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    if done:
      print('resetting env.')
      print(action.head(10))
      steps = 0
      env = gym.make()
      observation = env.reset() # reset env
    prev_x = None

  #if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
  #  print (('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))


# In[ ]:



