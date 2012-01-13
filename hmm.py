import sys
import datetime
import operator
import math
import simplejson
import numpy as np


##### user defined hmm specific functions ###########

class Kleinberg:
  is_log = True
  num_states = 2
  _w = 1
  _p1p2_ratio = 1.1
  _state_rates = []
  _alpha_table = []

  def __init__(self,num_states,w,p1p2_ratio):
    self.num_states = num_states
    self._w = w
    self._p1p2_ratio = p1p2_ratio

  def init(self,observations):
    [counts, total_counts] = zip(*observations)
    counts = np.array(counts)
    total_counts = np.array(total_counts)
    rates = counts/total_counts # elementwise divide

    mean = np.mean(rates)
    if mean == 0:
      mean = 1/np.mean(total_counts)

    self.state_rates = []
    self.state_rates.append(mean)            # P_0
    self.state_rates.append(self.state_rates[0]*self._p1p2_ratio) # p_1
    
    self.__set_alpha__()
    #sys.stderr.write('[Kleinberg:init]\tstate_rates\t=%s\n' % self.state_rates)

  def set_w(self,w):
    self._w = w
    self.__set_alpha__()

  def __set_alpha__(self):
    self._alpha_table = [[0 for j in range(self.num_states)] for i in range(self.num_states)]
    for from_state in range(self.num_states):
      for to_state in range(self.num_states):
        self._alpha_table[from_state][to_state] = np.exp(-1*self._w*max(0,(to_state - from_state)))
    for from_state in range(self.num_states):
      z = sum(self._alpha_table[from_state])
      for to_state in range(self.num_states):
        if (self.is_log):
          self._alpha_table[from_state][to_state] = np.log(self._alpha_table[from_state][to_state]/z)
        else:
          self._alpha_table[from_state][to_state] = self._alpha_table[from_state][to_state]/z
    #sys.stderr.write('[__set_alpha__]\t_alpha_table\t=%s\n' % self._alpha_table)

  def log_pois_approx(self,k,lam):
    fact_approx = sum([np.log(i) for i in range(1,int(k)+1)])
    rval = k*np.log(lam) - lam - fact_approx
    if np.isnan(rval):
      sys.stderr.write('[log_pois_approx]\tk=%f, lam=%1.10f, fact_approx=%s, rval=%s\n' % (k,lam,fact_approx,rval))
    return rval

  def pois(self,k,lam):
    #sys.stderr.write('[pois]\tk=%d, lam=%1.10f\n' % (k, lam))
    return (math.pow(lam,k)*math.exp(-1*lam)) / math.factorial(k)

  # this is really beta
  # observations are a tuple(count,total-count)
  def beta(self,state, obs):
    #sys.stderr.write('[Kleinberg:beta]\tstate=%d,\tstate_rates[%d]=%f\n' % (state,state,self.state_rates[state]))
    state_rate = self.state_rates[state]
    [count, total_count] = obs
    if (self.is_log):
      return self.log_pois_approx(count,total_count*state_rate)
    else:
      return np.exp(self.log_pois_approx(count,total_count*state_rate))

  def alpha(self,from_state,to_state):
    return self._alpha_table[from_state][to_state]


###### HMM class #####################################

class embrHMM:
  model = 0

  # model provides user specific implementations of the cost functions
  # encapsulated in an inner class so that those functions can use their
  # own helper methods and state.  Just need to satisfy interface:
  #
  # model.num_states
  #          -- number of hmm states per slice
  # model.alpha(from_state_index, to_state_index)
  #          -- returns the probability of transitioning between states
  # model.beta(state_index,observation)
  #          -- computes prob. of emiting obs at time t.  obs can be anything
  # model.init(observations)
  #          -- will be called at the beginning of each call to viterbi
  #             useful for simple parameter learning algorithms
  # model.is_log
  #          -- field which specifies whether the model is giving log probabilities


  def __init__(self,_model):
    self.model = _model

  def get_max_path(self,trellis,backpointers):
    [final_max_ind, final_max_prob] = max(enumerate(trellis[-1]), key=operator.itemgetter(1))
    path = [0 for i in range(len(backpointers))]
    iters = range(len(backpointers))
    iters.reverse()
    path[iters[0]] = final_max_ind
    for i in iters[1:]:
      path[i] = backpointers[i+1][path[i+1]]
    return (path, final_max_prob)

  def viterbi(self,observations):
    self.model.init(observations)

    # build trellis
    trellis = [[-1 for state in range(self.model.num_states)] for t in range(len(observations))]

    # intialize trellis slice 0
    for state in range(self.model.num_states):
      trellis[0][state] = self.model.beta(state, observations[0])

    # initialize backpointer object
    backpointers = [[-1 for state in range(self.model.num_states)] for t in range(len(observations))]

    for i in range(1,len(observations)):
      #sys.stderr.write('[viterbi]\tITERATION %d ------------------------------\n' % i)
      #sys.stderr.write('[viterbi]\tobservations[%d]\t=%s\n' % (i,observations[i]))
      prev_probs = np.array(trellis[i-1])
      #sys.stderr.write('[viterbi]\tprev_probs\t=%s\n' % prev_probs)
      for to_state in range(self.model.num_states):
        trans_probs = []
        for from_state in range(self.model.num_states):
          trans_probs.append(self.model.alpha(from_state,to_state))
        #sys.stderr.write('[viterbi]\ttrans_probs\t=%s\n' % trans_probs)
        trans_probs = np.array(trans_probs)
        if (self.model.is_log):
          prev_combined = prev_probs+trans_probs
        else:
          prev_combined = prev_probs*trans_probs

        #sys.stderr.write('[viterbi]\tprev_combined\t=%s\n' % prev_combined)
        [max_prev_ind,max_prev_prob] = max(enumerate(prev_combined), key=operator.itemgetter(1))
        #sys.stderr.write('[viterbi]\tmax_ind\t=%d\t\tmax_prob=%f\n' % (max_prev_ind,max_prev_prob))
        curr_prob = self.model.beta(to_state,observations[i])
        #sys.stderr.write('[viterbi]\tcurr_prob\t=%f\n' % (curr_prob))
        if (self.model.is_log):
          trellis[i][to_state] = max_prev_prob+curr_prob
        else:
          trellis[i][to_state] = max_prev_prob*curr_prob
        backpointers[i][to_state] = max_prev_ind
    return self.get_max_path(trellis,backpointers)

  def forward(self,observations,end_state):
    self.model.init(observations)

    # build trellis
    trellis = [[-1 for state in range(self.model.num_states)] for t in range(len(observations))]

    # intialize trellis slice 0
    for state in range(self.model.num_states):
      trellis[0][state] = self.model.beta(state, observations[0])

    for i in range(1,len(observations)):
      #sys.stderr.write('[forward]\tt=%d ------------------------------\n' % i)
      #sys.stderr.write('[forward]\tobservations[%d]\t=%s\n' % (i,observations[i]))
      prev_probs = np.array(trellis[i-1])
      #sys.stderr.write('[forward]\tprev_probs\t=%s\n' % prev_probs)
      for to_state in range(self.model.num_states):
        trans_probs = []
        for from_state in range(self.model.num_states):
          trans_probs.append(self.model.alpha(from_state,to_state))
        #sys.stderr.write('[forward]\ttrans_probs\t=%s\n' % trans_probs)
        trans_probs = np.array(trans_probs)
        if (self.model.is_log):
          prev_combined = prev_probs+trans_probs
        else:
          prev_combined = prev_probs*trans_probs

        #sys.stderr.write('[forward]\tprev_combined\t=%s\n' % prev_combined)
        if (self.model.is_log):
          total_prev_prob = np.log(sum(np.exp(prev_combined)))
        else:
          total_prev_prob = sum(prev_combined)
        #sys.stderr.write('[forward]\ttotal_prev_prob\t=%f\n' % total_prev_prob)
        curr_state_prob = self.model.beta(to_state,observations[i])
        #sys.stderr.write('[forward]\tcurr_state_prob\t=%f\n' % (curr_state_prob))
        if (self.model.is_log):
          trellis[i][to_state] = total_prev_prob+curr_state_prob
        else:
          trellis[i][to_state] = total_prev_prob*curr_state_prob
        #sys.stderr.write('[forward]\ttrellis[%d][%d]\t=%f\n' % (i,to_state,trellis[i][to_state]))
    return trellis[-1][end_state]

  def total_forward(self,observations):
    total = 0
    for state in range(self.model.num_states):
      fwd = self.forward(observations,state)
      if self.model.is_log:
        total += np.exp(fwd)
      else:
        total += fwd
    return total

  def backward(self,observations,start_state):
    self.model.init(observations)

    # build trellis
    trellis = [[-1 for state in range(self.model.num_states)] for t in range(len(observations))]

    # intialize trellis slice 0
    for state in range(self.model.num_states):
      trellis[-1][state] = self.model.beta(state, observations[-1])
      # initializes LAST column of trellis to be prob of LAST observation
    #sys.stderr.write('[backward]\tinitial trellis\t= %s\n' % trellis[-1])

    iters = range(len(observations)-1)
    iters.reverse()
    for i in iters:
      #sys.stderr.write('[backward]\tITERATION %d ------------------------------\n' % i)
      #sys.stderr.write('[backward]\tobservations[%d]\t=%s\n' % (i,observations[i]))
      next_probs = np.array(trellis[i+1])
      #sys.stderr.write('[backward]\tnext_probs\t=%s\n' % next_probs)
      for from_state in range(self.model.num_states):
        trans_probs = []
        for to_state in range(self.model.num_states):
          trans_probs.append(self.model.alpha(from_state, to_state))
          # note this call to alpha is reverse of forward
        #sys.stderr.write('[backward]\ttrans_probs\t=%s\n' % trans_probs)
        trans_probs = np.array(trans_probs)
        if (self.model.is_log):
          next_combined = next_probs+trans_probs
        else:
          next_combined = next_probs*trans_probs

        #sys.stderr.write('[backward]\tnext_combined\t=%s\n' % next_combined)
        if (self.model.is_log):
          total_next_prob = np.log(sum(np.exp(next_combined)))
        else:
          total_next_prob = sum(next_combined)
        #sys.stderr.write('[backward]\ttotal_next_prob\t=%f\n' % total_next_prob)
        curr_prob = self.model.beta(to_state,observations[i])
        #sys.stderr.write('[backward]\tcurr_prob\t=%f\n' % curr_prob)
        if (self.model.is_log):
          trellis[i][from_state] = total_next_prob+curr_prob
        else:
          trellis[i][from_state] = total_next_prob*curr_prob
        #sys.stderr.write('[backward]\ttrellis[%d][%d]\t=%f\n' % (i,to_state,trellis[i][from_state]))
    return trellis[0][start_state]

  def baum_welch(self,observations,max_itrs=50):
    w_all = []
    #sys.stderr.write('[baum_welch]\tentering with max_itrs=%d\n' % max_itrs)
    #sys.stderr.write('[baum_welch] w[%d]\t=%1.15f\n' % (0,self.model._w))
    for i in range(max_itrs):
      #sys.stderr.write('\n\n\n[baum_welch]\tepoch %d #######################################################\n' % i)
      #sys.stderr.write('[baum_welch]\t################################################################\n')
      xi_table = self.e_step(observations)
      alpha_hat = self.m_step(xi_table)
      self.model._w = -1*alpha_hat[0][1]
      #sys.stderr.write('[baum_welch] w[%d]\t=%1.15f\n' % (i,self.model._w))
      w_all.append(self.model._w)
    return [self.model._w,w_all]
    
  def e_step(self,observations):
    #sys.stderr.write('[e_step]\tentering\n');
    xi_table = [[[0 for j in range(self.model.num_states)] for i in range(self.model.num_states)] for t in range(len(observations)-1)]
    for t in range(len(observations)-1):
      for from_state in range(self.model.num_states):
        for to_state in range(self.model.num_states):
          xi_table[t][from_state][to_state] = self.xi(observations,from_state,to_state,t)
    #sys.stderr.write('[e_step]\txi_table\t=%s' % xi_table)
    return xi_table

  def m_step(self,xi_table):
    #sys.stderr.write('[m_step]\tentering\n');
    alpha_hat = [[0 for j in range(self.model.num_states)] for i in range(self.model.num_states)]
    for from_state in range(self.model.num_states):
      for to_state in range(self.model.num_states):
        num = 0
        denom = 0
        #sys.stderr.write('[m_step]\txi_table=%s\n' % xi_table)
        for t in range(len(xi_table)-1):
          if (self.model.is_log):
            num += np.exp(xi_table[t][from_state][to_state])
          else:
            num += xi_table[t][from_state][to_state]
          for to_state_prime in range(self.model.num_states):
            if (self.model.is_log):
              denom += np.exp(xi_table[t][from_state][to_state_prime])
            else:
              denom += xi_table[t][from_state][to_state_prime]
        if (self.model.is_log):
          alpha_hat[from_state][to_state] = np.log(num) - np.log(denom)
        else:
          alpha_hat[from_state][to_state] = num/denom
        #sys.stderr.write('[m_step]\tnum=%s,\tdenom=%s\n' % (num,denom));
        #sys.stderr.write('[m_step]\talpha_hat\t=%s\n' % alpha_hat);
      return alpha_hat

  def xi(self,observations,from_state,to_state,t):
    #sys.stderr.write('\n\n[xi]\tentering with from_state=%d,\tto_state=%d,\tt=$=%d\n' % (from_state,to_state,t))
    fwd_total = self.total_forward(observations)
    #sys.stderr.write('[xi]\t\tfwd_total=%f\n' % fwd_total)
    fwd = self.forward(observations[:t+1],from_state)
    #sys.stderr.write('[xi]\t\tfwd=%f\n' % fwd)
    bwd = self.backward(observations[t+1:],to_state)
    #sys.stderr.write('[xi]\t\tbwd=%f\n' % bwd)
    p_trans = self.model.alpha(from_state,to_state)
    #sys.stderr.write('[xi]\t\tp_trans=%f\n' % p_trans)
    p_state = self.model.beta(to_state,observations[t+1])
    #sys.stderr.write('[xi]\t\tp_state=%f\n' % p_state)
    if (self.model.is_log):
      xi = fwd+p_trans+p_state+bwd-fwd_total
    else:
      xi = (fwd*p_trans*p_state*bwd)/fwd_total
    #sys.stderr.write('[xi]\t\txi=%f\n' % xi)
    return xi
