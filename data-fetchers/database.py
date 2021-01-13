import requests
import json
import pickle
import time
import re
import pandas as pd
import numpy as np
import sys
import datetime
from bs4 import BeautifulSoup as bs

## This script allows you to create a database of all users,
## contests and active users' contest history.

## Data is stored in DataFrames, if value is missing it is
## NaN or None, depending on its type (NaN for numerical, None otherwise)

## All users are stored as a DataFrame with username as an index.
## User DataFrame fields:
##      - country (possibly None)
##      - city (possibly None)
##      - organization (possibly None)
##      - contribution
##      - rating
##      - maxRating

## Contests are stored as a DataFrame with contestId as an index.
## Contests DataFrame fields:
##      - duration (in seconds)
##      - startTime (in seconds)
##      - dayTime (in seconds from midnight)
##      - authors (list of usernames, possibly None)
## We consider only CodeForces (type CF) contests (not IOI or ICPC).

## Contest histories are stored as a dictionary with username as a key
## and user contest history DataFrame as a value:
## Contest history DataFrame fields:
##      - contestId
##      - rank (user ranking place in that contest)
##      - oldRating
##      - newRating
##      - delta (newRating - oldRating)
## We store contest history only for active users (participated in the rated
## contest during the last month) who participated in at least MIN_CONTESTS.
## Contest history for every user is sorted in descending order by contestId column.

## Use LoadDataBase function to get database object with
## users, contests and contestHistory described above.
## LoadDataBase has clean=True parameter, which cleans database from
## strange data, inconsistencies.

MAX_API_REQUESTS = 5
API_REQUESTS_INTERVAL = 1
MIN_CONTESTS = 5
MAX_SEGMENT = 18


class UserContestRatingClass:
    def __init__(self, contest):
        self.contestId = contest["contestId"]
        self.rank = contest["rank"]
        self.oldRating = contest["oldRating"]
        self.newRating = contest["newRating"]

    def delta(self):
        return self.newRating - self.oldRating

    def __str__(self):
        return 'contestId: %d, rank: %d, oldRating: %d, newRating: %d' % (
                self.contestId, self.rank, self.oldRating, self.newRating)

    def __repr__(self):
        return str(self)


class ContestInfoClass:
    def __init__(self, contest):
        self.duration = contest["durationSeconds"]
        if "startTimeSeconds" in contest:
            self.startTime = contest["startTimeSeconds"]
        else:
            self.startTime = np.nan
        if "preparedBy" in contest:
            self.authors = [contest["preparedBy"]]
        else:
            self.authors = FindAuthors(contest["id"])

    def __str__(self):
        res = 'duration: %dh %02dm' % (
                self.duration // 3600, self.duration % 3600 // 60)
        if self.startTime:
            res += ', startTime: %ds' % (self.startTime,)
        if self.authors:
            res += ', authors: %s' % (str(self.authors))
        return res

    def __repr__(self):
        return str(self)


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

    def __repr__(self):
        return str(self)


class UsersContestsDBClass:
    def __init__(self, users, contests, contestHistory):
        self.users = pd.DataFrame([
            self.userToDict(user) for user in users.values()],
            index=users.keys()).replace("", None)

        self.contests = pd.DataFrame([
            self.contestToDict(cntst) for cntst in contests.values()],
            index=contests.keys()).replace("", None)

        self.contestHistory = {}
        for user, history in contestHistory.items():
            self.contestHistory[user] = pd.DataFrame(
                self.contestHistoryToDict(cntstHist) for cntstHist in history) \
                .sort_values("contestId") \
                .replace("", None)

    def userToDict(self, user):
        return {"country": user.country,
                "city": user.city,
                "organization": user.organization,
                "contribution": user.contribution,
                "rating": user.rating,
                "maxRating": user.maxRating}

    def contestToDict(self, cntst):
        return {"duration": cntst.duration,
                "startTime": cntst.startTime,
                "dayTime": cntst.startTime % (60 * 60 * 24),
                "authors": cntst.authors}

    def contestHistoryToDict(self, cntstHist):
        return {"contestId": cntstHist.contestId,
                "rank": cntstHist.rank,
                "oldRating": cntstHist.oldRating,
                "newRating": cntstHist.newRating,
                "delta": cntstHist.delta()}

    def getAllUsers(self):
        return self.users.keys()

    def getActiveUsers(self):
        return self.contestHistory.keys()
    
    def getContestsId(self):
        return self.contests.keys()

    def getUserContests(self, nick):
        if nick in self.contestHistory:
            return [cntst.contestId for cntst in self.contestHistory[nick]]
        return []
    
    def getUserContestsInfo(self, nick):
        if nick in self.contestHistory:
            return self.contestHistory[nick]
        return []

    def getAllAuthors(self):
        authors = set()
        for contest in self.contest.values():
            if contest.authors:
                authors.update(set(contest.authors))
        return authors

    def getUserAuthors(self, nick):
        authors = set()
        history = self.contestHistory[nick]
        for entry in history:
            contest = self.contests[entry.contestId]
            if contest.authors:
                authors.update(set(contest.authors))
        return authors


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
    return [tag.text for tag in soup.findAll("a", {"class": re.compile("rated-user*")})]


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


def UserDataFetch(seg=0):
    if seg == 0:
        AllUserFetch()
        return
    
    res = {}
    with open('user-info.pickle', 'rb') as outfile:
        users = pickle.load(outfile)
    usersNumber = len(users)
    users = list(users.items())
    start = int(usersNumber * ((seg - 1) / MAX_SEGMENT))
    end = int(usersNumber * ((seg) / MAX_SEGMENT))
    for it in range(start, end):
        userName = users[it][0]
        userCntst = GetUserContestHistory(userName)
        timeleft = str(datetime.timedelta(hours=((end-it)/(5 * 60 * 60)))).rsplit(':', 1)[0]
        print("Users left " + str(end-it) + " " + userName + "Estimated time " + timeleft)

        if userCntst == None:
            print("PROBLEM WITH " + userName)
            with open('error-users.json', 'a') as outfile:
                json.dump(userName, outfile)
            continue
        elif len(userCntst) < MIN_CONTESTS:
            continue

        userCntst = ContestRatingInfo(userCntst)
        res[userName] = userCntst
        
    with open('user-contest-history-info' + str(seg) + '.pickle', 'wb') as outfile:
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
    users = contests = None
    contestHistory = {}
    with open('user-info.pickle', 'rb') as outfile:
        users = pickle.load(outfile)
    with open('contest-info.pickle', 'rb') as outfile:
        contests = pickle.load(outfile)
    for it in range(MAX_SEGMENT):
        with open('user-contest-history-info' + str(it+1) + '.pickle', 'rb') as outfile:
            contestHistory.update(pickle.load(outfile))
    DB = UsersContestsDBClass(users, contests, contestHistory)
    with open('database.pickle', 'wb') as outfile:
        pickle.dump(DB, outfile)


def CleanDataBase(DB):
    # add MikeMirzayanov
    admin = "MikeMirzayanov"
    DB.users.loc[admin] = {
        "country": "Russia",
        "city": "Saratov",
        "organization": None,
        "contribution": 256,
        "rating": 0,
        "maxRating": 0
    }

    # remove authors not in our database
    for authors in DB.contests["authors"]:
        to_pop = [author for author in authors if not author in DB.users.index]
        for tp in to_pop:
            authors.remove(tp)

    # remove contests with no authors
    DB.contests = DB.contests[DB.contests["authors"].map(lambda x: len(x) > 0)]

    # authors as set
    DB.contests["authors"] = DB.contests["authors"].map(lambda x: set(x))

    # remove non existing contest from contestHistory
    for user, history_df in DB.contestHistory.items():
        DB.contestHistory[user] = history_df[history_df.contestId.map(lambda x: x in DB.contests.index)]

    return DB
    

def LoadDataBase(clean=True):
    DB = None
    with open('database.pickle', 'rb') as outfile:
        DB = pickle.load(outfile)
    if clean:
        DB = CleanDataBase(DB)
    return DB


if __name__ == '__main__':
    if(len(sys.argv) == 1):
        ContestFetch()
        CreateDataBase()
    elif(len(sys.argv) == 2):
        if(int(sys.argv[1]) >= 0 and int(sys.argv[1]) <= MAX_SEGMENT):
            UserDataFetch(seg=int(sys.argv[1]))
        else:
            print("Argument should be an int from range [0," + str(MAX_SEGMENT) + ']')
    else:
        print("Unknown arguments")
