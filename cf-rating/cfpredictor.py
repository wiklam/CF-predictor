from database import Database, LoadDatabase
import numpy as np

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
    
    def getSeedCached(self, npCS):
        cache = {}
        def seedCacher(rating):
            if rating not in cache:
                prob = lambda r: self.Pij(r, rating)
                vecProb = np.vectorize(prob)
                cache[rating] = np.sum(vecProb(npCS)) + 0.5
            return cache[rating]
        
        return seedCacher
    
    def getInitialRatingChange(self, contestant, contestStandings):
        rank = contestant['rank']
        oldRating = contestant['oldRating']
        seed = self.getSeed(oldRating, contestant, contestStandings)
        midRank = np.sqrt(rank*seed)
        R = self.binarySearch(midRank, lambda r: self.getSeed(r, contestant, contestStandings))
        delta = self.intDiv(R-oldRating,2)
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
        topPplCount = (min(pplCount, 4*(np.rint(np.sqrt(pplCount))))).astype(int)
        topContestans = contestantsSorted.head(topPplCount)
        sum = np.sum([deltas[i] for i in topContestans.index])
        inc = min(0, max(-10, -(self.intDiv(sum, topPplCount))))
        print(inc)
        return deltas + inc
    
    def fixToSumZero(self, contestStandings, deltas):
        sum = np.sum(deltas)
        inc = -self.intDiv(sum, len(contestStandings)) - 1
        return deltas + inc
    
    def processContest(self, contestStandings):
        deltas = contestStandings.apply(lambda c: self.getInitialRatingChange(c, contestStandings), axis=1)   
        deltas = self.fixToSumZero(contestStandings, deltas)
        deltas = self.fightAgainstInflation(contestStandings, deltas)
        return deltas
    
    def calcErrorContest(self, contestStandings, errCalc):
        npCS = contestStandings['oldRating'].to_numpy()
        seedCalculator = self.getSeedCached(npCS)
        expectedRanks = contestStandings.apply(lambda c: seedCalculator(c['oldRating']), axis=1).to_numpy()
        actualRanks = contestStandings['oldRating'].to_numpy()
        vecErrCalc = np.vectorize(errCalc)
        return np.sum(vecErrCalc(expectedRanks, actualRanks)) / len(contestStandings)
    
    def genErrRateDic(self, errCalc):
        errRateDic = {}
        for key in self.contests:
            contestStandings = self.standings[key]
            print("Contest ", key, " started")
            print("There are ", len(contestStandings), " participants in this contest")
            errRateDic[key] = self.calcErrorContest(contestStandings, errCalc)
            print("Contest ", key, " is done!")
        return errRateDic

def GenCFRatingErrors(DB, errCalc):
    rp = CFRatingPredictor(DB)
    return rp.genErrRateDic(errCalc)

def AnadiErrorRate(a, b):
    return abs(a-b)


if __name__ == "__main__":
    DB = LoadDatabase()
    GenCFRatingErrors(DB, AnadiErrorRate)
