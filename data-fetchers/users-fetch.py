import requests
import json
import pickle
import time

## This script allows to create a dictionary where the username is the key
## and the value is a contest list with structures holding information about:
## contest id, contest place, time of rating update, previous rating, new rating.
## We consider only users who participated in at least MIN_CONTESTS. 


MAX_API_REQUESTS = 5
API_REQUESTS_INTERVAL = 1
MIN_CONTESTS = 5


class ContestClass:
    def __init__(self, contest):
        self.type = contest["type"]
        self.duration = contest["durationSeconds"]
        if hasattr(contest, "startTimeSeconds"):
            self.startTime = contest["startTimeSeconds"]
        else:
            self.author = None
        if hasattr(contest, "preparedBy"):
            self.author = contest["preparedBy"]
        else:
            self.author = None


def RequestStatusOk(res):
    if res["status"] != "OK":
        return False
    return True


def GetRequestBody(res):
    return res["result"]


def GetRequest(method):
    BlockAPICalls()
    res = requests.get("https://codeforces.com/api/" + method)
    return res.json()


def GetUserContestHistory(user):
    res = GetRequest("user.rating?handle=" + user)
    if(RequestStatusOk(res) == False):
        return None
    return GetRequestBody(res)


def GetActiveUsers():
    res = GetRequest("user.ratedList?activeOnly=true")
    if(RequestStatusOk(res) == False):
        print("Couldn't download active users")
        quit()
    return GetRequestBody(res)


def GetContestsList():
    res = GetRequest("contest.list?gym=false")
    if(RequestStatusOk(res) == False):
        print("Couldn't download active users")
        quit()
    res = GetRequestBody(res)
    for cntst in res:
        if cntst["phase"] != "FINISHED":
            res.remove(cntst)
    return res


def ContestInfo(usercntst):
    [info.pop('handle') for info in usercntst]
    [info.pop('contestName') for info in usercntst]
    [info.pop('ratingUpdateTimeSeconds') for info in usercntst]
    return usercntst

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

        usercntst = ContestInfo(usercntst)
        res[userName] = usercntst

    with open('user-info.pickle', 'wb') as outfile:
        pickle.dump(res, outfile)


def ContestFetch():
    res = {}
    contests = GetContestsList()
    
    for cntst in contests:
        contestId = cntst['id']
        res[contestId] = ContestClass(cntst)

    with open('contest-info.pickle', 'wb') as outfile:
        pickle.dump(res, outfile)


UserFetch()
ContestFetch()
