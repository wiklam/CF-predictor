import requests
import json
import pickle
import time

## This script allows to create a dictionary where the username is the key
## and the value is a contest list with structures holding information about:
## contest id, contest place, time of rating update, previous rating, new rating

cnt = lasttime = now = 0


def GetChecker(res):
    if res["status"] != "OK":
        return False
    return True


def GetResult(res):
    return res["result"]


def GetRequest(method):
    global cnt 
    cnt += 1
    res = requests.get("https://codeforces.com/api/" + method)
    return res.json()


def GetUserRating(user):
    res = GetRequest("user.rating?handle=" + user)
    if(GetChecker(res) == False):
        return None
    return GetResult(res)


def GetActiveUsers():
    res = GetRequest("user.ratedList?activeOnly=true")
    if(GetChecker(res) == False):
        print("Couldn't download active users")
        quit()
    return GetResult(res)


def UserInfo(userinfo):
    [info.pop('handle') for info in userinfo]
    [info.pop('contestName') for info in userinfo]
    return userinfo


def AbleToCont():
    global cnt
    if cnt >= 4:
        global lasttime, now
        now = time.time()
        if now - lasttime < 1:
            time.sleep(now-lasttime)
        cnt = 0
        lasttime = now


def UserFetch():
    lasttime = time.time()
    res = {}
    users = GetActiveUsers()
    for user in users:
        user = user['handle']
        userinfo = GetUserRating(user)
        if userinfo == None:
            #print("PROBLEM WITH " + user)
            with open('error-users.json', 'a') as outfile:
                json.dump(user, outfile)
            continue
        userinfo = UserInfo(userinfo)
        res[user] = userinfo
        #print(user + " done")
        AbleToCont()
    with open('user-info.pickle', 'wb') as outfile:
        pickle.dump(res, outfile)
  

UserFetch()

