from database import Database, LoadDatabase
import numpy as np

# TODO:
#   - add parameters - for now make sure it is correct
#   - maybe add standarized method for calculating error rate - needs to be discussed
#     with others
#   - add standarized value to return - proposal dict contestId: error

# return value: ans, userRatings
#   ans - dict of contestId: average error on expected ranking
#   userRatings: dict of handle: inner rating class with AtCoder specific measures
# parameters:
#   - consider: eg. consider=50, will consider only 50 first contests (-1 means consider all)
#   - verbose: print verbosely
def AtCoderRatingSystem(data, consider=50, verbose=False, **kwargs):
    CENTER = 1200
    RATEDBOUND = np.inf

    def powersum(q, n):
        return q * (1 - q**n) / (1 - q)

    def F(n):
        return np.sqrt(powersum(0.81, n)) / powersum(0.9, n)

    def f(n):
        Finf = np.sqrt(0.81 / (1.0 - 0.81)) / (0.9 / (1.0 - 0.9))
        return (F(n) - Finf) / (F(1) - Finf) * CENTER

    def g(x):
        return 2.0 ** (x / 800.0)

    def ginv(y):
        return 800 * np.log2(y)

    class Rating:
        def __init__(self):
            self.perfs = []
            # sum(0.9**i * perf_i)
            self.num = 0
            # sum(0.9**i)
            self.den = 0
            self.aperf = 0
            self.ratings = []

        def getAPerf(self):
            return self.aperf if len(self.perfs) > 0 else CENTER

        def addPerf(self, perf):
            self.ratings.append(self.getRating())
            if len(self.perfs) == 0:
                perf = (perf - CENTER) * 1.5 + CENTER
            self.perfs.append(perf)
            self.num = 0.9 * (perf + self.num)
            self.den = 0.9 * (1 + self.den)
            self.aperf = self.num / self.den
        
        def getRating(self):
            if len(self.perfs) == 0:
                return 0

            res = 0
            mult = 1
            for perf in self.perfs:
                rperf = self.getRPerf(perf)
                mult *= 0.9
                res += g(rperf) * mult
            return ginv(res / self.den) - f(len(self.perfs))

        def getRPerf(self, perf):
            return min(perf, RATEDBOUND + 400)

    userRatings = {}

    def addNewUsers(standings):
        for handle in standings.index:
            if not handle in userRatings:
                userRatings[handle] = Rating()

    def getRatings(standings):
        return [(handle, userRatings[handle].getRating()) for handle in standings.index]

    def calcExpectedRanksFromRatings(ratings):
        sortedRatings = sorted(ratings, key=lambda x: -x[1])
        ranks = np.empty(len(sortedRatings))
        i = 0
        while i < len(ranks):
            j = i
            rating = sortedRatings[i][1]
            while j + 1 < len(ranks) and sortedRatings[j + 1][1] == rating:
                j += 1
            ranks[i:j+1] = (i + 1 + j + 1) / 2
            i = j + 1
        return [(handle, ranks[i]) for i, (handle, _) in enumerate(sortedRatings)]

    def calcErrorRate(expectedRanks, standings):
        res = 0.0
        for handle, rank in expectedRanks:
            res += np.sqrt(abs(rank - standings.loc[handle]["rank"]))
        return res / len(expectedRanks)

    def fixRanks(ranks):
        newRanks = np.empty(len(ranks))
        i = 0
        while i < slen:
            j = i
            curRank = ranks[i]
            while j + 1 < slen and ranks[j + 1] == curRank:
                j += 1
            n = j - i + 1
            first = curRank
            last = curRank + n - 1
            newRanks[i:j+1] = (first + last) / 2
            i = j + 1
        return newRanks

    def getAPerfs(standings):
        return np.array([userRatings[handle].getAPerf() for handle in standings.index])
    
    def calc(x, aperfs):
        return np.sum(1.0 / (1.0 + 6.0 ** ((x - aperfs) / 400.0)))

    def computePerf(handle, standings, aperfs):
        rank = standings.loc[handle]["rank"]
        l, r = 0, 5000
        maxIters = 80
        while maxIters > 0 and (r-l) > 1e-1:
            maxIters -= 1
            m = (l+r) / 2
            if calc(m, aperfs) > rank - 0.5:
                l = m
            else:
                r = m
        return l
    
    sortedStandings = [(k,v) for k,v in sorted(data.standings.items(),
                        key=lambda x: data.contests.loc[x[0]].startTime)]
    if consider != -1:
        sortedStandings = sortedStandings[:consider]
    standingsLeft = len(sortedStandings)
    ans = {}

    for contestId, standings in sortedStandings:
        standings = standings.copy()
        if verbose:
            print("Standings left:", standingsLeft, "contestId:", contestId)
        standingsLeft -= 1

        addNewUsers(standings)
        ratings = getRatings(standings)
        expectedRanks = calcExpectedRanksFromRatings(ratings)
        ans[contestId] = calcErrorRate(expectedRanks, standings)
        
        aperfs = getAPerfs(standings)
        for handle in standings.index:
            perf = computePerf(handle, standings, aperfs)
            userRatings[handle].addPerf(perf)

    return ans, userRatings


if __name__ == "__main__":
    db = LoadDatabase()
    ans, userRatings = AtCoderRatingSystem(db)
