from database import Database
from database import LoadDatabase
from scipy.stats import norm
import math

DB = LoadDatabase()

def TopcoderRatingSystem(data, errFun, startRating = 0, startVolatility = 1, capConstant1 = 150, capConstant2 = 1500, weightConstant1 = 0.42, weightConstant2 = 0.18,
    weightDecrease = [[2000, 0.9], [2500, 0.8]], verbose = True, **kwargs):
    
    contestsIds = list(data.contests.index)
    contestsIds.reverse()
    
    ratings, volatility, matches = {}, {}, {}
    
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
    
    ans = []
    for contest in contestsIds:
        if contest > 100:
            break
        df = data.standings[contest]
        n = df.shape[0]

        user = list(df.index)
        rank, oldRating, oldVolatility = [[None] * n] * 3
        
        for i in range(n):
            rank[i] = df.iloc[i]["rank"]
            oldRating[i] = getRating(user[i])
            oldVolatility[i] = getVolatility(user[i])
        
        avgRating = 0
        for i in range(n):
            avgRating += oldRating[i]
        avgRating /= n
        
        volatilitySum = 0
        ratingSum = 0
        
        for i in range(n):
            volatilitySum += sqr(oldVolatility[i])
            ratingSum += sqr(oldRating[i] - avgRating)        
        competitionFactor = math.sqrt(volatilitySum / n + ratingSum / (n - 1))

        expectedRanks = [0.5] * n
        for i in range(n):
            for j in range(n):
                tmp = (oldRating[j] - oldRating[i]) / math.sqrt(2 * (sqr(oldVolatility[i]) + sqr(oldVolatility[j])))
                tmp = math.erf(tmp) + 1
                expectedRanks[i] += tmp / 2.
        ans.append(errFun(expectedRanks, rank))
        
        for i in range(n):
            ePerf = -norm.cdf((expectedRanks[i] - 0.5) / n)
            aPerf = -norm.cdf((rank[i] - 0.5) / n)
            perf = oldRating[i] + competitionFactor * (aPerf - ePerf)
            weight = getWeight(user[i])
            cap = getCap(user[i])
            
            matches[user[i]] += 1
            
            newRating = (oldRating[i] + weight * perf) / (1 + weight)
            if newRating > oldRating[i] + cap:
                newRating = oldRating[i] + cap
            
            if oldRating[i] > newRating + cap:
                newRating = oldRating[i] - cap
            
            ratings[user[i]] = newRating
            volatility[user[i]] = math.sqrt(sqr(oldVolatility[i]) / (weight + 1) + sqr(newRating - oldRating[i]) / weight)
        
        if verbose:
            print('done contest %d' % contest)
    return ans

def errFun(fa, fb):
    n = len(fa)
    ans = 0

    for i in range(n):
        ans += math.sqrt(abs(fa[i] - fb[i]))
    return ans / n

print("DB read")
print(TopcoderRatingSystem(DB, errFun))
