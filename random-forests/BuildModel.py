# For proper running, ensure that there is
# a database.py and database.pickle (or their symlinks)
# in a given folder
# If run the first time, start with SaveUserDatabase

import pickle
import pandas as pd
import numpy as np

from database import UserContestRatingClass
from database import ContestInfoClass
from database import UserInfoClass
from database import UsersContestsDBClass
from database import LoadDataBase

from RandomForest import Forest
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import matplotlib.pyplot as plt

DB = LoadDataBase()

# Compute the correlation between user and author 
# based on contests (-inf, maxContestId)

def GetCorrelation(user, author, maxContestId):
    userContests = {}
    for cntst in DB.getUserContestsInfo(user):
        userContests[cntst.contestId] = cntst
    authorContests = DB.getUserContestsInfo(author)

    commonContests = 0
    scalarSum = 0

    for contest in authorContests:
        if contest.contestId not in userContests:
            continue
        
        if contest.contestId >= maxContestId:
            continue

        commonContests += 1
        scalarSum += contest.delta() * userContests[contest.contestId].delta()
    
    if commonContests == 0:
        return 0
    return scalarSum * (commonContests ** (1. / 2))

# Prepare data in form to create a data frame from it

def PrepareUserData(minContestId, maxContestId):
    activeUsers = DB.getActiveUsers()
    data = {}

    def push(key, val):
        if key in data:
            data[key].append(val)
        else:
            data[key] = [val]

    for user in activeUsers:
        userContests = DB.getUserContestsInfo(user)
        for contest in userContests:
            if contest.contestId < minContestId or maxContestId < contest.contestId:
                continue
            
            push("user", user)
            push("contest id", contest.contestId)

            country = None
            if user in DB.users:
                country = DB.users[user].country
            
            if country == None:
                country = "No country"
            push("country", country)

            push("target", contest.delta())
            push("rating", contest.oldRating)

            contestInfo = DB.contests[contest.contestId]
            push("start time", contestInfo.startTime % (24 * 60 * 60))
            push("start day", contestInfo.startTime)
            push("duration", contestInfo.duration)
            
            correlationCoef, authors = 0, 0
            for author in contestInfo.authors:
                authors += 1
                correlationCoef += GetCorrelation(user, author, contest.contestId)
            
            if authors > 0:
                correlationCoef /= authors
            push("correlation", correlationCoef)
    
    return data

def SaveUserDatabase():
    data = PrepareUserData(0, 1500)
    with open('user-data.pickle', 'wb') as outfile:
        pickle.dump(data, outfile)

def ReadUserDatabase():
    dataframeDB = None
    with open('user-data.pickle', 'rb') as outfile:
        dataframeDB = pickle.load(outfile)
    return dataframeDB

# Based on predictions (xs) and actual rating change (ys),
# treat these as points, and draw on plane
# Adds linear regression and line f(x) = x
# to ease results verification
# If save = True, then saves the result in
# drawings/name.png

def SaveErrors(xs, ys, name, save = False):
    xs = np.array(xs)
    ys = np.array(ys)

    zs = abs(xs - ys)
    avgErr = sum(zs) / len(zs)
    print('Average error is %.10lf' % avgErr)

    b, m = np.polynomial.polynomial.polyfit(xs, ys, 1)
    plt.plot(xs, ys, '.')
    plt.plot(xs, b + m * xs, 'r-')
    plt.plot(xs, xs, 'b-')
    plt.title(name)
    plt.xlabel('Expected change by Predictor')
    plt.ylabel('True change')

    if save == True:
        plt.savefig('drawings/%s.png' % name)
    else:
        plt.show()

# For user 'user' compute the predictions
# for any contest with id > 1200, such that
# user had written at least 20 contests
# before the chosen one

def TestUser(user, verbose = True):
    DF = pd.DataFrame(ReadUserDatabase())
    DF = DF[DF["user"] == user]

    contestList = DB.getUserContests(user)
    if len(contestList) <= 20:
        return
    
    xs, ys = [], []
    n = len(contestList)

    for i in range(n - 20):
        s = i + 20
        if DF.iloc[s]["contest id"] < 1200:
            continue

        trainDF = DF[i:s]
        forestModel = Forest(trainDF, 20, 3, "numerical")
        
        if verbose:
            print('done %d, left %d' % (i, n - 20 - i - 1))

        xs.append(forestModel.Query(DF.iloc[s]))
        ys.append(DF.iloc[s]["target"])

    SaveErrors(xs, ys, 'test-user-%s' % user, True)

# For user 'user' compute the predictions
# for any contest with id > 1200, such that
# user had written at least 20 contests
# before the chosen one. Use sklearn, probably
# TestUser or TestUserSklearn will be removed in
# the future.

def TestUserSklearn(user, ModelType, verbose = True, save = True, **kwargs):
    DF = pd.DataFrame(ReadUserDatabase())
    DF = DF[DF["user"] == user].drop(["user", "country"], axis=1)

    contestList = DB.getUserContests(user)
    assert len(contestList) > 20

    xs, ys = [], []
    n = len(contestList)

    for i in range(n - 20):
        s = i + 20
        if DF.iloc[s]['contest id'] < 1200:
            if verbose:
                print('Skipping contest of id %d' % DF.iloc[s]['contest id'])
            continue

        train_df = DF[i:s]
        X, y = train_df.drop('target', axis=1), train_df['target']
        model = ModelType(**kwargs).fit(X, y)
        pred = model.predict([DF.iloc[s].drop('target')])[0]
        
        if verbose:
            print('done %d, left %d' % (i, n - 20 - i - 1))
        
        xs.append(pred)
        ys.append(DF.iloc[s]['target'])
    
    xs = np.array(xs)
    ys = np.array(ys)

    SaveErrors(xs, ys, 'test-user-%s' % user, save)


if __name__ == '__main__':
    #  TestUser(input())
    TestUserSklearn(input(),
        RandomForestRegressor,
        n_estimators=20,
        max_features=3)
    #  TestUserSklearn(input(),
        #  AdaBoostRegressor,
        #  n_estimators=20)
