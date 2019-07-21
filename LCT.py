import featdataprocess.FeatdataProcess as fp
import rawdataprocess.MySQL_helper as mh # 数据库帮助类库
import numpy as np
from numpy import linalg as la
from datetime import datetime
import matplotlib.pyplot as plt
import os
from datetime import datetime,timedelta

#排列组合#-----------------------------------------------------------  
def stationID_select(list_stationID):
    n = len(list_stationID)
    l = []
    for i in range(n):
        for j in range(i+1,n):
                l.append(list_stationID[i])
                l.append(list_stationID[j])
    return l

def choose_2_stationID(all_choices):
    n = len(all_choices)-1
    l = []
    for i in range(0,n,2):
        l.append(all_choices[i:i+2])
    return l    
#--------------------------------------------------------------------

#数据获取#-----------------------------------------------------------  
def AETA_data(StationID_1): 
   
    data1 = eq.get_final_data(stationID = StationID_1 , signalType='finaldata_lowfreq_magn',
    featureType_list=['average'], time_range=time_range)
    
    #台站名----------------------------------------------------------
    a = eq.get_station()
    title1 = a[a['StationID']==StationID_1]['Title'].values.tolist()
    title1.append(StationID_1)
    
#    title_list=[str(title1[1])+title1[0]+' & '+str(title2[1])+title2[0],str(title2[1])+title2[0]+' & '+str(title3[1])+title3[0],str(title3[1])+title3[0]+' & '+str(title1[1])+title1[0]]
    #-------------------------------------------------------------------- 
    
    #初始化FeatdataProcessorNew类：
    data_1 = fp.FeatdataProcessorNew(data=data1, time_range=time_range, interval=600)
    #注意：如果不需要对数据作图，仅仅是处理数据，可以不提供stationID
    
    #数据绘图
    #test.plot(is_eq=True, auto_scale=True)
    
    ##查看数据描述
#        data_1.describe()
    
#        data_1.plot()
    
    # 因为这个时间段的数据包括2种采样频率，所以我们先对数据进行统一降采样
    data_1 = data_1.down_sampling(interval=600)
    
    #数据补全
    compensate_data_1 = data_1.compensate(interval=None, bias=300)
    
    #降采样
    down_sample_data_1 = compensate_data_1.down_sampling(interval=3600, method='mean') # 以均值方式每1小时采样1个点
        
#        down_sample_data_1.plot()
#        down_sample_data_2.plot()
   
#        ###dataframe改成list ####必做-------------------(‘data_station_’必有)
#        data_station1 = list(down_sample_data_1.data['average'].values)

#    #操作一：选取数据的时间段
##    selected_data1 = compensate_data.select_range_data(hour_range=['00:00','04:00']) # 选择每天0点到4点的数据
#    selected_data1 = down_sample_data_1.select_range_data(hour_range=['00:00','04:00'])

#    #dataframe改成list
#    data_station1 = list(selected_data1.data['average'].values)

    
#        操作二：归一化 (method='z-score')
    norm_data_1 = down_sample_data_1.scaling(method='z-score')
#        #dataframe改成list
    data_station1 = list(norm_data_1.data['average'].values)   
#        norm_data_1.plot()
    
    time_start = list(down_sample_data_1.data['Timestamp'].values)
    
    return data_station1,title1,time_start
   
#--------------------------------------------------------------------

#LOCO#-----------------------------------------------------------  
def LocoScore_XY(data_station1,data_station2,months):
    
    w = 24     #每天24个数，那这个窗口就是7x24=168
    N = int(720*months-1*(w-1))  #去掉首（或尾），总共得到的Locoscore数量
    
#    Beta = 0.9

    X = data_station1
    Y = data_station2
    #先做x，矩阵的协方差是矩阵的各个向量间的协方差。cij = cov(xi,xj)得到一个数
    LocoScore_XY=[]
    Cwx = np.arange(576,dtype='float64').reshape(24,24)
    Cwy = np.arange(576,dtype='float64').reshape(24,24)
    for n in range(N):
        
        for i in range(w):      
            
            Cwx[:,i] = X[i+n:i+w+n]
        
        ACX = np.cov(Cwx)
        
        Ux,sigma,VT = la.svd(ACX)
        Ux_new = Ux[:,[0,1,2,3]]
        
        vector_Ux = Ux_new[:,0]
        
        for i in range(w):
            Cwy[:,i] = Y[i+n:i+w+n]
        ACY = np.cov(Cwy)
        
        Uy,sigma,VT = la.svd(ACY)
        Uy_new = Uy[:,[0,1,2,3]]
        
        vector_Uy = Uy_new[:,0]
        
        vector_proj1 = np.dot(Ux_new.T,vector_Uy) #最大特征值的特征向量uy 在UXnew特征空间上投影产生 投影向量1
        cos1 = np.linalg.norm(vector_proj1)/np.linalg.norm(vector_Uy) # 余弦值，norm函数可以轻松求出一个向量的模。
        
        vector_proj2 = np.dot(Uy_new.T,vector_Ux) #最大特征值的特征向量uy 在UXnew特征空间上投影产生 投影向量1
        cos2 = np.linalg.norm(vector_proj2)/np.linalg.norm(vector_Ux) # 余弦值，norm函数可以轻松求出一个向量的模。
        
        LocoScore_XY.append((cos1+cos2)/2)
        
    return LocoScore_XY 

# 取主特征向量时候，一般取前4个 即可。
#
#  [U S V]=svd(c) 即可做奇异值分解,好简单
#  最后一步 得到向量在矩阵空间上的投影  求其大小除以原来向量，即可得到余弦值。
# 例如  U=[1,2,3,4; 4,5,6,7] v=[1,2,3,4]'  U_new=U*v 得到一个向量；
# 计算这个向量的大小除以原来向量v的大小5.47  即可得到余弦值。



def pic_plot(LocoScore,StationID_1,StationID_2,months):
    eq_list_Time = eq_list['Timestamp'].values.tolist()
    eq_list_Magnitude = eq_list['Magnitude'].values.tolist()
    xlist=eq_list_Time
    ylist_min=[]
    for i in range(len(eq_list_Magnitude)):
        ylist_min.append(1-0.5/8*eq_list_Magnitude[i])
    ylist_max=[1]*len(ylist_min)

    #画图#----------------------------------------------------------
    time_start = AETA_data(StationID_1)[2]
#    time_start = list(selected_data1.data['Timestamp'].values)
    
    plot_start=int(time_start[25])
    plot_end=int(time_start[int(720*months)])
    
    xlist=list(range(plot_start,plot_end,3600))
    xlist.insert(0,xlist[0]-3600)
    xlist.append(xlist[-1]+3600)
#    print(xlist)
    plt.ylim(0,1.1)
    ytl=np.arange(0,1.1,0.5)
    plt.yticks(ytl,ytl)
    xtick_list=list(range(plot_start,plot_end,86400))
    xtick_list.append(xtick_list[-1]+86400)
    
    xname_list=[datetime.strftime(datetime.fromtimestamp(i),'%d') for i in xtick_list]


    plt.plot(xlist,LocoScore)
    plt.xticks(xtick_list,xname_list,rotation=0)
    plt.rcParams['font.sans-serif']=['SimHei']                #windows汉字显示
    plt.rcParams['axes.unicode_minus']=False
    plt.title('LCT（'+str(StationID_1)+AETA_data(StationID_1)[1][0]+' & '+str(StationID_2)+AETA_data(StationID_2)[1][0]+'）')
    plt.vlines(eq_list_Time,ylist_min,ylist_max,color='red')

     #--------------------------------------------------------------------


if __name__=="__main__":
    eq = mh.EqMySql()
#    time_range = ['2017-07-14', '2018-10-13']
    months = 1
    
#    time_end='2017-8-10'
#    time_end='2017-8-13'
    time_end='2019-7-21'
#    time_end='2019-4-20'
#    time_end='2018-6-26'  #第一组
#    time_end='2018-10-22'  #第二组
    t2=datetime.strptime(time_end,'%Y-%m-%d')
    t1=t2-timedelta(days=30*months)
    time_range=[datetime.strftime(item,format='%Y-%m-%d') for item in [t1,t2]]
 
#    list_stationID = [90,121,129,43,75,105,38,93,125]
#    list_stationID = [43,90,121,129,116]
#    list_stationID = [75,246,240,38,129,121,122,131]
#    list_stationID = [122,43,48,38,116,240,121,246] #4.18台湾6.7级
#    list_stationID = [121,116,150,129] #第一组
#    list_stationID = [38,240,75,48] #第二组
#    list_stationID = [75,246,240,38,129,121,122,131]
    list_stationID = [129,90,43,116,121,240]
#    list_stationID = [131,75,122,121,246,129,38,48,240] ###近期四川2019.03.14
#    stations_title_list = [AETA_data(i)[1][0] for i in list_stationID]
    eq_list = eq.get_earthquake(time_range= time_range, distance_range=(75, 129, 22, 40), min_mag=4.5, update=True)   #(73, 135, 13, 54) (97, 109, 26, 35)
#    eq_list = eq.get_earthquake(time_range= time_range, distance_range=(73, 145, -53, 54), min_mag=5, update=True)   #(73, 135, 13, 54) (97, 109, 26, 35)
    stations_amount = len(list_stationID)
    stations_title = [str(list_stationID[i])+str(AETA_data(list_stationID[i])[1][0]) for i in range(stations_amount)]
    
    
    all_choices = stationID_select(list_stationID)
    every_two_stations_choices = choose_2_stationID(all_choices)
    plt.figure()
    plt.suptitle(time_range[0]+'至'+time_range[1])
    for i in range(0,len(every_two_stations_choices)):
        StationID_1 = every_two_stations_choices[i][0]
        StationID_2 = every_two_stations_choices[i][1] 
#        print(StationID_1)
#        print(StationID_2) 

        LocoScore = LocoScore_XY(AETA_data(StationID_1)[0],AETA_data(StationID_2)[0],months)
        plt.subplots_adjust(left=None,bottom=0.2,right=None,top=0.86,wspace=None,hspace=2)   
        fig=plt.gcf()
        fig.set_size_inches(9*months,len(every_two_stations_choices)+2)
        plt.subplot(len(every_two_stations_choices),1,i+1) 
        pic_plot(LocoScore,StationID_1,StationID_2,months)
    plt.show() 
    
    path = 'Only LCT'+time_range[0]+'至'+time_range[1]
    if not os.path.isdir(path):
        os.mkdir(path)
    fig.savefig(path+"/"+time_range[0]+'至'+time_range[1]+str(stations_title),dpi=200)

#    fig.savefig() 
    print(" ")
    print(eq_list) 
    print(" ")
    print(len(every_two_stations_choices))
    print(" ")
    print('done!')
#    print(every_two_stations_choices)
    