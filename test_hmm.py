import sys
import datetime
import simplejson

import hmm

def parse_date(date_str):
  splut=date_str.split();
  d = splut[0].split('-')
  t=splut[1].split(':')
  #uncomment if data is not from january 2011
  #months = ['Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12]
  #date = datetime.datetime(l[-1],months[l[1]],int(d[2]),int(t[0]),int(t[1]),int(t[2]))
  return datetime.datetime(2011,1,int(d[2]),int(t[0]),int(t[1]),int(t[2]))


def find_spikes(date_strs,counts,total_counts):
  model = hmm.Kleinberg(2,1,2)
  h = hmm.embrHMM(model)
  
  # parse dates
  dates = [parse_date(date_str) for date_str in date_strs]

  # sort counts and total_counts by dates
  all = zip(dates,counts,total_counts)
  all.sort()
  [dates, counts, total_counts] = zip(*all)
  
  observations = zip(counts,total_counts)
  #h.baum_welch(observations)
  [path, prob] = h.viterbi(observations)

  #print '[find_spikes]\tcounts\t=' + str(counts)
  #print '[find_spikes]\ttotal_counts\t=' + str(total_counts)
  #print '[find_spikes]\tpath\t=' + str(path)

  sparse = [date_strs[i] for i in range(len(path)) if path[i] == 1]
  return simplejson.dumps(sparse)



################### REDUCE SCRIPT BEGINS HERE ###################

cur_key = ''
times = []
counts = []
total_counts = []

for line in sys.stdin:
  [word,time,count,total_count] = line.split('\t')

  if cur_key == '':
    cur_key = word
    times.append(time)
    counts.append(float(count))
    total_counts.append(float(total_count))

  elif cur_key == word:
    times.append(time)
    counts.append(float(count))
    total_counts.append(float(total_count))

  elif cur_key != word:
    print str(cur_key) + "\t" + find_spikes(times,counts,total_counts)
    # reset accumulators
    times = []
    counts = []
    cur_key = word
    times.append(time)
    counts.append(float(count))
    total_counts.append(float(total_count))

print str(cur_key) + "\t" + find_spikes(times,counts,total_counts)

