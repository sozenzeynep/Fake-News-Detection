from bs4 import BeautifulSoup
from selenium import webdriver
import time
import csv


def veri_cek(path):
    sayfa = 1#Sayfayı aşağı kaydırmak,girilen değer kadar sayfaya bakmak gibi

    
    driver_path = "C:/webdriver/chromedriver"
    browser = webdriver.Chrome(driver_path)

    browser.get(path)


    #
    file = open("tweetler.csv","w",encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(["tweetler","begeni_sayisi","yorum_sayisi","retweet_sayisi"])
    
    
    #
    a = 0
    while a < sayfa:
    #
        lastHeight = browser.execute_script("return document.body.scrollHeight")
        i=0
        while i<1:
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
            newHeight = browser.execute_script("return document.body.scrollHeight")

            if newHeight == lastHeight:
                break
            else:
                lastHeight = newHeight

            i = i+1


        sayfa_kaynağı = browser.page_source
        soup = BeautifulSoup(sayfa_kaynağı, "html.parser")
        tweetler = soup.find_all("div",attrs={"data-testid":"tweet"})#data id kısmı hepsinde var
        dondur = []
        for i in tweetler:
            try: 
                yazı = i.find("div", attrs={"class":"css-1dbjc4n r-1iusvr4 r-16y2uox r-1777fci r-1mi0q7o"}).text#data id altında metni içine alan class
                #print(yazı)#Gelen tweetler bir sınır koyabiliriz
                writer.writerow([yazı])
                dondur.append(yazı)
                #return dondur
            
            except:
                print("**")
        else:
            break
        a = a+1
    return dondur
