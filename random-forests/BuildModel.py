import sys
import pickle
import pandas as pd
sys.path.insert(1, '/home/anadi/General/Studia/ML/projekt/CF-predictor/data-fetchers')

from database import UserContestRatingClass
from database import ContestInfoClass
from database import UserInfoClass
from database import UsersContestsDBClass
from database import LoadDataBase

from RandomForest import Forest
import matplotlib.pyplot as plt

DB = LoadDataBase()

def GetCorrelation(user, author, maxContest):
    userContests = {}
    for cntst in DB.getUserContestsInfo(user):
        userContests[cntst.contestId] = cntst
    authorContests = DB.getUserContestsInfo(author)

    commonContests = 0
    scalarSum = 0

    for contest in authorContests:
        if contest.contestId not in userContests:
            continue
        
        if contest.contestId >= maxContest:
            continue

        commonContests += 1
        scalarSum += contest.delta() * userContests[contest.contestId].delta()

    if commonContests == 0:
        return 0
    return scalarSum * (commonContests ** (1. / 2))

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
            push("rank", contest.rank)

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

def SaveUserData():
    data = PrepareUserData(0, 1500)
    with open('user-data.pickle', 'wb') as outfile:
        pickle.dump(data, outfile)

def ReadUserData():
    DB = None
    with open('user-data.pickle', 'rb') as outfile:
        DB = pickle.load(outfile)
    return DB

def saveErrors(xs, ys, name, save = False):
    zs = [abs(xs[s] - ys[s]) for s in range(len(xs))]
    avgErr = sum(zs) / len(zs)
    print('Average error is %.10lf' % avgErr)

    plt.scatter(xs, ys)
    plt.title(name)
    plt.xlabel('Expected change by Predictor')
    plt.ylabel('True change')

    if save == True:
        plt.savefig('drawings/%s.png' % name)
    else:
        plt.show()

def TestGlobalModel():
    DF = pd.DataFrame(ReadUserData())
    train_DF = DF[DF["contest id"] <= 1450]
    train_DF = train_DF[:400]
    test_DF = DF[DF["contest id"] > 1450]
    forestModel = Forest(train_DF, 20, 3, "numerical", verbose = 1)

    xs, ys = [], []
    for s in range(test_DF.shape[0]):
        if test_DF.iat[s, 4] < 2800:
            continue

        val = int(forestModel.Query(test_DF.iloc[s]))
        exp_val = test_DF.iat[s, 3]

        xs.append(val)
        ys.append(exp_val)

    saveErrors(xs, ys, "TestGlobalModel")

def TestSingleUser(user):
    DF = pd.DataFrame(ReadUserData())
    DF = DF[DF["user"] == user]

    trainDF = DF[DF["contest id"] <= 1450]
    testDF = DF[DF["contest id"] > 1450]

    xs, ys = [], []
    curTrainDF = trainDF[trainDF["user"] == user]
    forestModel = Forest(curTrainDF, 20, 3, "numerical", verbose = 1)
    
    for s in range(testDF.shape[0]):
        curRow = testDF.iloc[s]
        xs.append(forestModel.Query(curRow))
        ys.append(curRow["target"])
    return xs, ys

def MultipleUsersTest():
    username = None
    try:
        while True:
            username = input()
            xs, ys = TestSingleUser(username)
            saveErrors(xs, ys, 'user-%s' % username, False)

    except EOFError:
        exit(0)

def TestUser(user):
    DF = pd.DataFrame(ReadUserData())
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

        print('done %d, left %d' % (i, n - 20 - i - 1))
        xs.append(forestModel.Query(DF.iloc[s]))
        ys.append(DF.iloc[s]["target"])
    saveErrors(xs, ys, 'test-user-%s' % user, True)

TestUser(input())
#TestGlobalModel()
#MultipleUsersTest()