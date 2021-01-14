import re
import json
import time
import pickle
import requests
import pandas as pd
from bs4 import BeautifulSoup as bs

MAX_API_REQUESTS = 5
API_REQUESTS_INTERVAL = 1
MIN_CONTESTS = 5


def GetRequestStatusOk(res):
    if res["status"] != "OK":
        return False
    return True


def GetRequestBody(res):
    return res["result"]


def GetRequest(method):
    BlockAPICalls()
    res = requests.get("https://codeforces.com/api/" + method)
    if not res:
        print("Unexpected status code:", str(res.status_code))
    return res.json()


def GetAllUsers():
    res = GetRequest("user.ratedList?activeOnly=false")
    if GetRequestStatusOk(res) == False:
        print("Couldn't download all users")
        quit()
    return GetRequestBody(res)


def GetActiveUsers():
    res = GetRequest("user.ratedList?activeOnly=true")
    if(GetRequestStatusOk(res) == False):
        print("Couldn't download active users")
        quit()
    return GetRequestBody(res)


def GetContests():
    res = GetRequest("contest.list?gym=false")
    if GetRequestStatusOk(res) == False:
        print("Couldn't download contest list")
        quit()
    res = GetRequestBody(res)
    return list(filter(lambda con: con["phase"] == "FINISHED" and con["type"] == "CF", res))


def GetAuthors(contestId):
    url = "http://codeforces.com/contests/" + str(contestId)
    res = requests.get(url)
    content = res.text
    soup = bs(content, "html.parser")
    return set(tag.text for tag in soup.findAll("a", {"class": re.compile("rated-user*")}))


def GetHistory(user):
    res = GetRequest("user.rating?handle=" + user)
    if GetRequestStatusOk(res) == False:
        return None
    return GetRequestBody(res)


def GetStandings(contestId):
    res = GetRequest("contest.ratingChanges?contestId=" + str(contestId))
    if GetRequestStatusOk(res) == False:
        return None
    res = GetRequestBody(res)
    if not res:
        return None
    return res
    
    
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


def FetchUsers():
    print("Fetching users ...")
    users = GetAllUsers()
    columns = ["handle", "country", "city", "organization",
               "contribution", "rating", "maxRating"]
    users_df = pd.DataFrame(users)[columns].set_index("handle")
    print("Fetched users")
    
    with open("users.pickle", "wb") as outfile:
        pickle.dump(users_df, outfile)
        
    return users_df.index

        
def FetchContests():
    print("Fetching contests ...")
    contests = GetContests()
    columns = ["id", "durationSeconds", "startTimeSeconds"]
    contests_df = pd.DataFrame(contests)[columns].set_index("id")
    contests_df.columns = ["duration", "startTime"]
    contests_df["dayTime"] = contests_df["startTime"] % (24 * 60 * 60)
    authors = []
    left = len(contests_df)
    for contestId in contests_df.index:
        print(f"Contests left: {left}, doing contestId: {contestId}")
        authors.append(GetAuthors(contestId))
        left -= 1
    contests_df["authors"] = authors
    print("Fetched contests")
        
    with open("contests.pickle", "wb") as outfile:
        pickle.dump(contests_df, outfile)
        
    return contests_df.index
    
    
def FetchHistory(handle):
    history = GetHistory(handle)
    if history == None:
        print("ERROR: contest history of user", handle)
        with open("error.json", "a") as outfile:
            json.dump(handle, outfile)
        return None
    if len(history) < MIN_CONTESTS:
        return None
    columns = ["rank", "oldRating", "newRating"]
    history_df = pd.DataFrame(history)[columns]
    history_df["delta"] = history_df.newRating - history_df.oldRating
    return history_df
    
    
def FetchAllHistory(handles=None):
    if handles is None:
        with open("users.pickle", "rb") as infile:
            handles = pickle.load(infile).index

    print("Fetching all history ...")
    all_history = {}
    left = len(handles)
    for handle in handles:
        print(f"Contest history left: {left}, doing handle: {handle}")
        history = FetchHistory(handle)
        left -= 1
        if history is not None:
            all_history[handle] = history
    print("Fetched all history")
    
    with open("history.pickle", "wb") as outfile:
        pickle.dump(all_history, outfile)
    
    
def FetchStandings(contestId):
    standings = GetStandings(contestId)
    if standings is None:
        print("ERROR: standings of contest", contestId)
        return None
    columns = ["handle", "rank", "oldRating", "newRating"]
    standings_df = pd.DataFrame(standings)[columns].set_index("handle")
    standings_df["delta"] = standings_df.newRating - standings_df.oldRating
    return standings_df


def FetchAllStandings(contestIds=None):
    if contestIds is None:
        with open("contests.pickle", "rb") as infile:
            contestIds = pickle.load(infile).index

    print("Fetching all standings ...")
    all_standings = {}
    left = len(contestIds)
    for contestId in contestIds:
        print(f"Standings left: {left}, doing contestId: {contestId}")
        standings = FetchStandings(contestId)
        left -= 1
        if standings is not None:
            all_standings[contestId] = standings
    print("Fetched all standings")
    
    with open("standings.pickle", "wb") as outfile:
        pickle.dump(all_standings, outfile)
        
    
def FetchAll():
    handles = FetchUsers()
    contestIds = FetchContests()
    FetchAllContestHistory(handles)
    FetchAllStandings()
    
    
class Database:
    def __init__(self, users, contests, history, standings, clean=True):
        self.users = users
        self.contests = contests
        self.history = history
        self.standings = standings
        if clean:
            self.clean()
    
    def clean(self):
        pass
    
    
def LoadDataBase(clean=True):
    users = contests = None
    history = standings = None
    with open("users.pickle", "rb") as infile:
        users = pickle.load(infile)
    with open("contests.pickle", "rb") as infile:
        contests = pickle.load(infile)
    with open("history.pickle", "rb") as infile:
        history = pickle.load(infile)
    with open("standings.pickle", "rb") as infile:
        standings = pickle.load(infile)
    return Database(users, contests, history, standings, clean=clean)


if __name__ == "__main__":
    FetchAll()