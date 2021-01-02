import requests
import json
import pickle
import time

## This script allows to create a dictionary where the username is the key
## and the value is a contest list with structures holding information about:
## contest ID, contest place, previous rating, new rating and a second one (dictionary)
## which keys are contest IDs holding information about contest type, duration and if 
## suitable start date (in seconds) and contest author.
## We consider only users who participated in at least MIN_CONTESTS.


MAX_API_REQUESTS = 5
API_REQUESTS_INTERVAL = 1
MIN_CONTESTS = 5


class UserContestRatingClass:
    def __init__(self, contest):
        self.contestId = contest["contestId"]
        self.rank = contest["rank"]
        self.oldRating = contest["oldRating"]
        self.newRating = contest["newRating"]


class ContestInfoClass:
    def __init__(self, contest):
        self.type = contest["type"]
        self.duration = contest["durationSeconds"]
        if "startTimeSeconds" in contest:
            self.startTime = contest["startTimeSeconds"]
        else:
            self.startTime = None
        if "preparedBy" in contest:
            self.author = contest["preparedBy"]
        else:
            self.author = None


class UsersContestsDBClass:
    def __init__(self, users, contests):
        self.contests = contests
        self.users = users

    def getUsers(self):
        return self.users.keys()
    
    def getContestsId(self):
        return self.contests.keys()

    def getUserContests(self, nick):
        return [cntst.contestId for cntst in self.users[nick]]


def GetRequestStatusOk(res):
    if res["status"] != "OK":
        return False
    return True


def GetRequestBody(res):
    return res["result"]


def GetRequest(method):
    BlockAPICalls()
    res = requests.get("https://codeforces.com/api/" + method)
    if not(res):
        print("Unexpected status code: " + str(res.status_code))
        quit()
    return res.json()


def GetUserContestHistory(user):
    res = GetRequest("user.rating?handle=" + user)
    if(GetRequestStatusOk(res) == False):
        return None
    return GetRequestBody(res)


def GetActiveUsers():
    res = GetRequest("user.ratedList?activeOnly=true")
    if(GetRequestStatusOk(res) == False):
        print("Couldn't download active users")
        quit()
    return GetRequestBody(res)


def GetContestsList():
    res = GetRequest("contest.list?gym=false")
    if(GetRequestStatusOk(res) == False):
        print("Couldn't download contest list")
        quit()
    res = GetRequestBody(res)
    for cntst in res:
        if cntst["phase"] != "FINISHED":
            res.remove(cntst)
    return res


def ContestRatingInfo(usercntst):
    [cntst.pop('handle') for cntst in usercntst]
    [cntst.pop('contestName') for cntst in usercntst]
    [cntst.pop('ratingUpdateTimeSeconds') for cntst in usercntst]
    return [UserContestRatingClass(cntst) for cntst in usercntst]

def BlockAPICalls():
    BlockAPICalls.cnt += 1
    if BlockAPICalls.cnt >= MAX_API_REQUESTS:
        BlockAPICalls.now = time.time()
        diff =  BlockAPICalls.now - BlockAPICalls.lasttime
        if diff < API_REQUESTS_INTERVAL:
            time.sleep(diff)
        BlockAPICalls.cnt = 0
        BlockAPICalls.lasttime = BlockAPICalls.now

BlockAPICalls.cnt = 5
BlockAPICalls.lasttime = time.time()
BlockAPICalls.now = BlockAPICalls.lasttime


def UserFetch():
    res = {}
    users = GetActiveUsers()
    leftusers = len(users)

    for user in users:
        leftusers -= 1
        userName = user['handle']
        usercntst = GetUserContestHistory(userName)
        print("Users left " + str(leftusers) + " " + userName)

        if usercntst == None:
            print("PROBLEM WITH " + userName)
            with open('error-users.json', 'a') as outfile:
                json.dump(userName, outfile)
            continue

        elif len(usercntst) < MIN_CONTESTS:
            continue

        usercntst = ContestRatingInfo(usercntst)
        res[userName] = usercntst
        
    with open('user-info.pickle', 'wb') as outfile:
        pickle.dump(res, outfile)


def ContestFetch():
    res = {}
    contests = GetContestsList()
    
    for cntst in contests:
        contestId = cntst['id']
        res[contestId] = ContestInfoClass(cntst)

    with open('contest-info.pickle', 'wb') as outfile:
        pickle.dump(res, outfile)


def CreateDataBase():
    users = contests = None
    with open('user-info.pickle', 'rb') as outfile:
        users = pickle.load(outfile)
    with open('contest-info.pickle', 'rb') as outfile:
        contests = pickle.load(outfile)
    DB = UsersContestsDBClass(users, contests)
    with open('users-DB.pickle', 'wb') as outfile:
        pickle.dump(DB, outfile)


UserFetch()
ContestFetch()
CreateDataBase()
