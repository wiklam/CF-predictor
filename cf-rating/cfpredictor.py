#!/usr/bin/env python
# coding: utf-8

# In[3]:

from database import Database, LoadDatabase
import numpy as np


# In[4]:


DB = LoadDatabase()


# In[246]:


class CFRatingPredictor:
    def __init__(self, db):
        self.standings = db.standings
        self.contests = list(db.contests.index)
        self.contests.reverse()
    
    def intDiv(self, x,y):
        return - ((-x) // y) if x < 0 else x // y
    
    def Pij(self, ri, rj):
        return 1.0 / (1.0 + np.power(10.0, (rj-ri)/400.0))
    
    def getSeed(self, rating, contestant, contestStandings):
        prob = lambda r : self.Pij(r, rating)
        vecProb = np.vectorize(prob)
        oldRatings = contestStandings['oldRating'].to_numpy()
        seed = np.sum(vecProb(oldRatings)) - self.Pij(contestant['oldRating'],rating) + 1
        return seed
    
    def getInitialRatingChange(self, contestant, contestStandings):
        rank = contestant['rank']
        oldRating = contestant['oldRating']
        seed = self.getSeed(oldRating, contestant, contestStandings)
        print(contestant.name)
        print(seed)
        midRank = np.sqrt(rank*seed)
        R = self.binarySearch(midRank, lambda r: self.getSeed(r, contestant, contestStandings))
        print(R)
        print(oldRating)
        delta = self.intDiv(R-oldRating,2)
        print(delta)
        return delta
    
    def binarySearch(self, value, f, left=1, right=8000):
        while right - left > 1:
            mid = self.intDiv(left+right,2)
            print(left, right)
            print(mid, f(mid), value)
            if f(mid) < value:
                right = mid
            else:
                left = mid
        return left
     
    def fightAgainstInflation(self, contestStandings, deltas):
        contestantsSorted = contestStandings.sort_values(by=['oldRating'], ascending=False)
        pplCount = len(contestStandings)
        topPplCount = (min(pplCount, 4*(np.rint(np.sqrt(pplCount)))))
        topContestans = contestantsSorted.head(topPplCount)
        sum = np.sum([deltas[i] for i in topContestans.index])
        inc = min(0, max(-10, -(self.intDiv(sum, topPplCount))))
        print(inc)
        return deltas + inc
    
    def fixToSumZero(self, contestStandings, deltas):
        sum = np.sum(deltas)
        inc = -self.intDiv(sum, len(contestStandings)) - 1
        print(inc)
        return deltas + inc
    
    def processContest(self, contestStandings):
        deltas = contestStandings.apply(lambda c: self.getInitialRatingChange(c, contestStandings), axis=1)   
        deltas = self.fixToSumZero(contestStandings, deltas)
        deltas = self.fightAgainstInflation(contestStandings, deltas)
        return deltas
    


# In[247]:


rp = CFRatingPredictor(DB)


# In[248]:


standings = DB.standings[1299]
deltas = rp.processContest(standings)


# In[249]:


count = 0
for i in deltas.index:
    if deltas[i] != standings.loc[i]['delta']:
        print("nickname:     ", i)
        print("counted delta:", deltas[i])
        print("actual  delta:", standings.loc[i]['delta'])
        count += 1
print(count)


# In[233]:


standings.head(10)


# In[234]:


deltas.head(10)


# In[235]:


standings.loc['izban']


# In[245]:


standings = DB.standings[1299]
rp.getSeed(2821, standings.loc['izban'], standings)


# In[ ]:




