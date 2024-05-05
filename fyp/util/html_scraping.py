def find_bpm(song_url):
    # This is so that if the user does not need to do web scraping, they can choose to not install bs4 and selenium
    from bs4 import BeautifulSoup as bs 
    from selenium import webdriver
    import time
    options = webdriver.ChromeOptions()
    options.add_argument("headless")

    driver = webdriver.Chrome(options=options)
    url = "https://bpmfinder.app/result/"

    driver.get(url+song_url[32:])

    print(f"fetching bpm result from {url}...")
    html = bs(driver.page_source, 'html5lib')
    bpm_tag = html.find("h2", attrs={"class":"text-sm xs:text-sm md:text-lg font-bold"})

    while bpm_tag is None:
        time.sleep(10)
        html = bs(driver.page_source, 'html5lib')
        bpm_tag = html.find("h2", attrs={"class":"text-sm xs:text-sm md:text-lg font-bold"})

    bpm = bpm_tag.text[4:]
    print("The BPM result is "+bpm)

    driver.quit()
    return int(bpm)
