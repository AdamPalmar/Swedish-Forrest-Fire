# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:24:50 2016

@author: thor
"""

#import file
def getIndeces(datafile, year):
    lines = datafile.readlines();
    counter = 0    
    indexStart = None
    indexEnd = None
    startFound = False 
    prevYear = None
    for line in lines:
        strLine = str(line)
        strLine = strLine.split(',')
        
        counter = counter+1
        if indexStart is None and strLine[1] == year:
            indexStart = counter
            startFound = True
        
        elif startFound and indexEnd is None and strLine[1] != prevYear:
            indexEnd = counter
        
        prevYear = strLine[1]
    return (indexStart, indexEnd)
    #print str(indexStart) + ', ' + str(indexEnd)    
        
def extractPeriods(datafile):
    lines = datafile.readlines();
    yearList = [] 
    for line in lines:
        strLine = str(line)
        strLine = strLine.split(',')
        if strLine[0] == 'kpi':
            continue
        if not strLine[1] in yearList: 
            yearList.append(strLine[1])
    return yearList    

def reorganizeData(year, inputFile):
    kpiFile = file
    interval = getIndeces(year)
    line = kpiFile.readline();
    strLine = str(line)
    strLine = strLine.split(',')
    #print ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa " + strLine[2])
    #==============================================================================
    # newString = str(strLine[2] +', ' + strLine[4])
    # print newString
    # newKpiFile.write(newString + ' \n')
    # print strLine
    # print(line);
    #==============================================================================
    attCounter = 0
    curYear = None
    curId = None
    newKpiFile.write('municipality_id') 
    for line in kpiFile:
        strLine = str(line)
        strLine = strLine.split(',')
        if curYear is None:
            curYear = strLine[1] 
            
        if strLine[1] != curYear or str(strLine[0]) == 'kpi':
            newKpiFile.write('\n')
            print 'done ' + str(attCounter)
            break
        
        if curId is None or (strLine[0] != curId):
            curId = strLine[0]
            newKpiFile.write(', '+str(strLine[0]))    
            attCounter = attCounter +1          
            print str(strLine[0] + ' ' + str(attCounter))
            
        
    
    kpiFile.close
    kpiFile = open("/home/thor/Desktop/5.semester/swedish-school-fires/kpis_2009_2011.csv", 'r');
    kpiFile.readline
    linelist = kpiFile.readlines()
    idBreakPoint = None
    
    for iline in linelist:
        #print 'new i line'
        strLine = str(iline)
        strLine = strLine.split(',')
        if strLine[2] == 'municipality_id':
            print 'skipping first line'        
            continue
       
        
        
        if idBreakPoint is not None and (strLine[2] == idBreakPoint or strLine[2] == 'municipality_id'):
            print 'break found exiting'
            break
       
        if idBreakPoint is None:
            idBreakPoint = strLine[2] 
        curId = str(strLine[2])
        print curId
        newKpiFile.write(str(curId))
        
        for jline in linelist:
           
            jstrLine = str(jline)
            jstrLine = jstrLine.split(',')  
            #print strLine[1]
           # print jstrLine[4]
           # print curId + 'aa------------------------'
            if jstrLine[1] != year:
                #print 'new year breaking ------'
                break
            if jstrLine[2] == curId and jstrLine[1] == year:
                #print '----------------------------------------------'
                newKpiFile.write(str(', ' + jstrLine[4]))
            
            
        newKpiFile.write('\n')          
    print 'done'
    counter = 0;
    for line in kpiFile:
        counter = counter + 1
    print counter
    
    newKpiFile.close
    
kpiFile = open("/home/thor/Desktop/5.semester/swedish-school-fires/kpis_2009_2011.csv", 'r');
#newKpiFile = open('/home/thor/Desktop/5.semester/swedish-school-fires/plebtest2.csv', 'w')
getIndeces(kpiFile, "2010")
#extractPeriods(kpiFile)
kpiFile.close