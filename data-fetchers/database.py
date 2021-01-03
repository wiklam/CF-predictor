import requests
import json
import pickle
import time
import re
from bs4 import BeautifulSoup as bs

## This script allows you to create a database of all users,
## contests and active users' contest history. 

## All users are stored as a dictionary with username as a key
## and UserInfoClass as a value.
## UserInfoClass fields:
##      - nick
##      - country (possibly None)
##      - city (possibly None)
##      - organization (possibly None)
##      - contribution
##      - rating
##      - maxRating

## Contests are stored as a dictionary with contestId as a key
## and ContestInfoClass as a value.
## ContestInfoClass fields:
##      - duration (in seconds)
##      - startTime (in seconds)
##      - authors (list of usernames, possibly None)
## We consider only CodeForces (type CF) contests (not IOI or ICPC).

## Contest histories are stored as a dictionary with username as a key
## and UserContestRatingClass as a value:
## UserContestRatingClass fields:
##      - contestId
##      - rank (user ranking place in that contest)
##      - oldRating
##      - newRating
## We store contest history only for active users (participated in the rated
## contest during the last month) who participated in at least MIN_CONTESTS.

## Use LoadDataBase function to get database object with users, contests and contestHistory
## dictionaries described above.

MAX_API_REQUESTS = 5
API_REQUESTS_INTERVAL = 1
MIN_CONTESTS = 5


class UserContestRatingClass:
    def __init__(self, contest):
        self.contestId = contest["contestId"]
        self.rank = contest["rank"]
        self.oldRating = contest["oldRating"]
        self.newRating = contest["newRating"]

    def __str__(self):
        return 'contestId: %d, rank: %d, oldRating: %d, newRating: %d' % (
                self.contestId, self.rank, self.oldRating, self.newRating)


class ContestInfoClass:
    def __init__(self, contest):
        self.duration = contest["durationSeconds"]
        if "startTimeSeconds" in contest:
            self.startTime = contest["startTimeSeconds"]
        else:
            self.startTime = None
        if "preparedBy" in contest:
            self.authors = [contest["preparedBy"]]
        else:
            self.authors = FindAuthors(contest["id"])

    def __str__(self):
        res = 'duration: %dh %02dm' % (self.duration // 3600, self.duration % 3600 // 60)
        if self.startTime:
            res += ', startTime: %ds' % (self.startTime,)
        if self.authors:
            res += ', authors: %s' % (str(self.authors))
        return res


class UserInfoClass:
    def __init__(self, user):
        self.nick = user["handle"]
        if "country" in user:
            self.country = user["country"]
        else:
            self.country = None
        if "city" in user:
            self.city = user["city"]
        else:
            self.city = None
        if "organization" in user:
            self.organization = user["organization"]
        else:
            self.organization = None
        self.contribution = user["contribution"]
        self.rating = user["rating"]
        self.maxRating = user["maxRating"]

    def __str__(self):
        return 'nick: %s, country: %s, city: %s, organization: %s, contribution: %d, rating: %d, maxRating: %d' % (
                self.nick, self.country, self.city, self.organization, self.contribution, self.rating, self.maxRating)


class UsersContestsDBClass:
    def __init__(self, users, contests, contestHistory):
        self.users = users
        self.contests = contests
        self.contestHistory = contestHistory

    def getUsers(self):
        return self.users.keys()

    def getActiveUsers(self):
        return self.contestHistory.keys()
    
    def getContestsId(self):
        return self.contests.keys()

    def getUserContests(self, nick):
        return [cntst.contestId for cntst in self.contestHistory[nick]]


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


def GetAllUsers():
    res = GetRequest("user.ratedList?activeOnly=false")
    if(GetRequestStatusOk(res) == False):
        print("Couldn't download all users")
        quit()
    return GetRequestBody(res)


def FindAuthors(contestId):
    url = "http://codeforces.com/contests/" + str(contestId)
    res = requests.get(url)
    content = res.text
    soup = bs(content, "html.parser")
    return [tag.text for tag in soup.findAll("a", {"class", re.compile("rated-user*")})]


def GetContestsList():
    res = GetRequest("contest.list?gym=false")
    if(GetRequestStatusOk(res) == False):
        print("Couldn't download contest list")
        quit()
    res = GetRequestBody(res)
    return list(filter(lambda con: con["phase"] == "FINISHED" and con["type"] == "CF", res))


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


def ActiveUserFetch():
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
        
    with open('user-contest-history-info.pickle', 'wb') as outfile:
        pickle.dump(res, outfile)


def AllUserFetch():
    res = {}
    users = GetAllUsers()
    
    for user in users:
        userName = user['handle']
        userInfo = UserInfoClass(user)
        res[userName] = userInfo
    
    with open('user-info.pickle', 'wb') as outfile:
        pickle.dump(res, outfile)


def ContestFetch():
    res = {}
    contests = GetContestsList()
    cntstleft = len(contests)
    
    for cntst in contests:
        cntstleft -= 1
        contestId = cntst['id']
        res[contestId] = ContestInfoClass(cntst)
        print("Contests left", str(cntstleft), contestId)

    with open('contest-info.pickle', 'wb') as outfile:
        pickle.dump(res, outfile)


def CreateDataBase():
    users = contests = contestHistory = None
    with open('user-info.pickle', 'rb') as outfile:
        users = pickle.load(outfile)
    with open('contest-info.pickle', 'rb') as outfile:
        contests = pickle.load(outfile)
    with open('user-contest-history-info.pickle', 'rb') as outfile:
        contestHistory = pickle.load(outfile)
    DB = UsersContestsDBClass(users, contests, contestHistory)
    with open('database.pickle', 'wb') as outfile:
        pickle.dump(DB, outfile)


def LoadDataBase():
    DB = None
    with open('database.pickle', 'rb') as outfile:
        DB = pickle.load(outfile)
    return DB


if __name__ == '__main__':
    ActiveUserFetch()
    AllUserFetch()
    ContestFetch()
    CreateDataBase()
