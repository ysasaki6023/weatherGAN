# -*- coding: utf-8 -*-
import lxml.html
import urllib,csv,os,datetime,time
import requests

def download(urlPath,filePath):
    print filePath,"..."
    timeout = 60
    if not os.path.exists(os.path.dirname(filePath)):
        os.makedirs(os.path.dirname(filePath))

    if os.path.exists(filePath):
        print "already downloaded. skip :",filePath
        return
    response = requests.get(urlPath, allow_redirects=False, timeout=timeout)
    if response.status_code != 200:
        e = Exception("HTTP status: " + response.status_code)
        raise e

    content_type = response.headers["content-type"]
    if 'image' not in content_type:
        e = Exception("Content-Type: " + content_type)
        raise e

    image = response.content
    with open(filePath, "wb") as fout:
        fout.write(image)

def getEveryday(urlBase,fileBase,date1="2014-10-01",date2="2017-10-01"):
    d1 = datetime.datetime.strptime(date1,"%Y-%m-%d")
    d2 = datetime.datetime.strptime(date2,"%Y-%m-%d")

    dd = d1
    while dd<d2:
        #time.sleep(1)
        dd += datetime.timedelta(days=1)
        url   = dd.strftime( urlBase)
        fname = dd.strftime(fileBase)
        try:
            download(url,fname)
        except:
            print "failed to download :", url
            continue

date1="2012-01-01"
date2="2017-08-01"

#for hour in ("01","02","04","05","07","08","10","11","13","14","16","17","19","20","22","23"):
"""
for hour in ["%02d"%h for h in range(0,24)]:
    getEveryday(urlBase ='http://az416740.vo.msecnd.net/static-images/satellite/%%Y/%%m/%%d/%s/00/00/japan_near/large.jpg'%(hour),
                fileBase='../data/satellite/%%Y_%%m_%%d_%s_00.jpg'%(hour),
                date1=date1, date2=date2)

"""
for hour in ["%02d"%h for h in range(3,24,3)]:
    getEveryday(urlBase ='http://az416740.vo.msecnd.net/static-images/archive/chart/entries/%%Y/%%m/%%d/%%Y-%%m-%%d-%s-00-00-large.jpg'%(hour),
                fileBase='../data/chart/%%Y_%%m_%%d_%s_00.jpg'%(hour),
                date1=date1, date2=date2)
"""
getEveryday(urlBase ='http://az416740.vo.msecnd.net/static-images/satellite/%Y/%m/%d/09/00/00/japan_near/large.jpg',
            fileBase='../data/satellite/%Y_%m_%d_09_00.jpg',
            date1=date1, date2=date2)

getEveryday(urlBase ='http://az416740.vo.msecnd.net/static-images/satellite/%Y/%m/%d/21/00/00/japan_near/large.jpg',
            fileBase='../data/satellite/%Y_%m_%d_21_00.jpg',
            date1=date1, date2=date2)

getEveryday(urlBase ='http://az416740.vo.msecnd.net/static-images/satellite/%Y/%m/%d/15/00/00/japan_near/large.jpg',
            fileBase='../data/satellite/%Y_%m_%d_15_00.jpg',
            date1=date1, date2=date2)

getEveryday(urlBase ='http://az416740.vo.msecnd.net/static-images/archive/chart/entries/%Y/%m/%d/%Y-%m-%d-15-00-00-large.jpg',
            fileBase='../data/chart/%Y_%m_%d_15_00.jpg',
            date1=date1, date2=date2)

getEveryday(urlBase ='http://az416740.vo.msecnd.net/static-images/rader/%Y/%m/%d/15/00/00/japan_detail/large.jpg',
            fileBase='../data/rader/%Y_%m_%d_15_00.jpg',
            date1=date1, date2=date2)

getEveryday(urlBase ='http://az416740.vo.msecnd.net/static-images/rader/%Y/%m/%d/15/00/00/japan_detail/large.jpg',
            fileBase='../data/rader/%Y_%m_%d_15_00.jpg',
            date1=date1, date2=date2)

getEveryday(urlBase ='http://az416740.vo.msecnd.net/static-images/live/daily/map/%Y/%m/%d/top/large.jpg',
            fileBase='../data/live/%Y_%m_%d_15_00.jpg',
            date1=date1, date2=date2)
"""
