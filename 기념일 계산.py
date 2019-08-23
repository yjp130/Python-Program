from datetime import datetime

def getMemorialDay(year, month, day, mem_day='내 생일', is_msg=True):
    m_day = datetime(year, month, day)
    today = datetime.now()
    elapsed = today - m_day


    elapsed_day   = elapsed.days
    elapsed_month = int(elapsed_day / 30)
    elapsed_year  = int(elapsed_day / 365)

    print('{}로부터 {:,}일이 지났고, 월수는 {}개월, 년수는 {}년이 지남!!'.format(mem_day,elapsed_day, elapsed_month, elapsed_year))


getMemorialDay(1991, 7, 7, mem_day='내 생일')

