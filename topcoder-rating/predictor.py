from database import Database
from database import LoadDatabase
from scipy.stats import norm
import math

DB = LoadDatabase()

class contestInfo:
    def __init__(self, contestId, contestTime, newTopcoderRating, newCodeforcesRating):
        self.contestId = contestId
        self.contestTime = contestTime
        self.newTopcoderRating = newTopcoderRating
        self.newCodeforcesRating = newCodeforcesRating    
    
    def __str__(self):
        return '[contest = %d, topcoder rating = %lf, codeforces rating = %d]' % (self.contestId, 
                                                                                self.newTopcoderRating,
                                                                                self.newCodeforcesRating)
    
def TopcoderRatingSystem(data, errFun, startRating = 1000, startVolatility = 500, capConstant1 = 150, capConstant2 = 1800, weightConstant1 = 0.6, weightConstant2 = 0.3,
    weightDecrease = [[2000, 0.9], [2500, 0.8]], verbose = True, **kwargs):

    contestsIds = list(data.contests.index)
    contestsIds = sorted(contestsIds, key = lambda x: data.contests.loc[x].startTime)
    
    ans, ratings, volatility, matches = {}, {}, {}, {}
    userHistory = {}
    
    def sqr(a):
        return a * a
    
    def getMatches(user):
        if user not in matches:
            matches[user] = 0
        return matches[user]
    
    def getRating(user):
        if user not in ratings:
            ratings[user] = startRating;
        return ratings[user]
    
    def getVolatility(user):
        if user not in volatility:
            volatility[user] = startVolatility
        return volatility[user]

    def pushHistory(user, contestId, rating):
        if user not in userHistory:
            userHistory[user] = []
        userHistory[user].append(contestInfo(contestId, data.contests.loc[contestId].startTime, 
                                             rating, data.standings[contestId].loc[user].newRating))
    
    def getCompetitionFactor(rating, volatility):
        n = len(rating)
        avgRating = sum(rating) / n
        volatilitySum, ratingSum = 0, 0
        
        for i in range(n):
            volatilitySum += sqr(volatility[i])
            ratingSum += sqr(rating[i] - avgRating)
        return math.sqrt(volatilitySum / n + ratingSum / (n - 1))

    def getExpectedRank(rating, volatility):
        expectedRank = [0.5] * n
        for i in range(n):
            for j in range(n):
                tmp = (rating[j] - rating[i]) / math.sqrt(2 * (sqr(volatility[i]) + sqr(volatility[j])))
                tmp = math.erf(tmp) + 1
                expectedRank[i] += tmp / 2.
        return expectedRank
    
    def getWeight(user):
        weight = weightConstant1 / (getMatches(user) + 1)
        weight += weightConstant2
        weight = 1 / (1 - weight) - 1
        
        coef = 1.
        rating = getRating(user)
        for s in weightDecrease:
            if s[0] <= rating and s[1] < coef:
                coef = s[1]
        return coef * weight
    
    def getCap(user):
        return capConstant1 + capConstant2 / (getMatches(user) + 2)
    
    def getError(expRank, actRank):
        ret = 0
        n = len(expRank)

        for s in range(n):
            ret += errFun(expRank[s], actRank[s])
        return ret / n
    
    for contest in contestsIds:
        if contest > 200:
            break

        df = data.standings[contest]
        n = df.shape[0]
        user = list(df.index)
        rank, oldRating, oldVolatility = [], [], []

        for i in range(n):
            rank.append(df.iloc[i]["rank"])
            oldRating.append(getRating(user[i]))
            oldVolatility.append(getVolatility(user[i]))

        competitionFactor = getCompetitionFactor(oldRating, oldVolatility)
        expectedRank = getExpectedRank(oldRating, oldVolatility)
        ans[contest] = getError(expectedRank, rank)
        
        for i in range(n):
            ePerf = -norm.ppf((expectedRank[i] - 0.5) / n)
            aPerf = -norm.ppf((rank[i] - 0.5) / n)
            
            perf = oldRating[i] + competitionFactor * (aPerf - ePerf)
            weight = getWeight(user[i])
            cap = getCap(user[i])
            
            
            newRating = (oldRating[i] + weight * perf) / (1 + weight)
            if newRating > oldRating[i] + cap:
                newRating = oldRating[i] + cap
            
            if oldRating[i] > newRating + cap:
                newRating = oldRating[i] - cap
            
            matches[user[i]] += 1
            ratings[user[i]] = newRating
            volatility[user[i]] = math.sqrt(sqr(oldVolatility[i]) / (weight + 1) + sqr(newRating - oldRating[i]) / weight)
            pushHistory(user[i], contest, newRating)

        if verbose and contest % 5 == 0:
            print('done contest %d' % contest)
    return ans, userHistory

def errFun(a, b):
    return math.log(max(a, b) / min(a, b))

contestErrors, userHistory = TopcoderRatingSystem(DB, errFun, capConstant2 = 1500, 
                                weightConstant1 = 0.5, weightConstant2 = 0.2)

