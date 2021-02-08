from database import Database, LoadDatabase
import numpy as np

# return value: ans, userRatings
#   ans - dict of contestId: average error of expected ranking
#   userRatings: dict of handle: inner rating class with AtCoder specific measures
# parameters:
#   - consider: eg. consider=50, will consider only 50 first contests (-1 means consider all)
#   - verbose: print verbosely
#   - basicRating: Ranking with which the user is starting
#   - k: constant scaling how many points will be gained in a duel
#   - eloConst1: constant by which the rating difference is divided in sigmoid function
#   - eloConst2: consant which is raised to the power in sigmoid function

def EloRatingSystem(data, consider=50, verbose=False, basicRating = 1200, k = 40, eloConst1 = 400, eloConst2 = 10, **kwargs):

    class Rating:
        def __init__(self):
            self.ratings = []
        
        def getRating(self):
            if len(self.ratings) == 0:
                return basicRating
            return self.ratings[len(self.ratings)-1]

        def addRating(self, rating):
            self.ratings.append(rating)

    userRatings = {}

    def addNewUsers(standings):
        for handle in standings.index:
            if not handle in userRatings:
                userRatings[handle] = Rating()

    def getRatings(standings):
        return [(handle, userRatings[handle].getRating()) for handle in standings.index]

    def calcEloDiff(myRating, enemyRating, myRank1, enemyRank):
        exp = (enemyRating - myRating) / eloConst1
        expectedScore = 1/(1 + eloConst2**exp)

        score = 0
        if myRank1 > enemyRank:
            score = 1
        elif myRank1 > enemyRank:
            score = 0.5

        return k * (score - expectedScore)


    def calcNewRating(handle, standings):
        oldRating = userRatings[handle].getRating()
        rank = standings.loc[handle]["rank"]
        diff = 0

        for handle2 in standings.index:
            if handle == handle2:
                continue
            enemyRating = userRatings[handle2].getRating()
            enemyRank = standings.loc[handle2]["rank"]
            diff += calcEloDiff(oldRating, enemyRating, rank, enemyRank)
        diff /= (len(standings.index) - 1)
        return oldRating + diff

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

        ratingUpdate = {}
        for handle in standings.index:
            newrating = calcNewRating(handle, standings)
            ratingUpdate[handle] = newrating

        for handle in standings.index:
            userRatings[handle].addRating(ratingUpdate[handle])

    return ans, userRatings


if __name__ == "__main__":
    db = LoadDatabase()
    ans, userRatings = AtCoderRatingSystem(db)
