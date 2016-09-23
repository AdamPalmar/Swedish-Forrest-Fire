# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 11:24:50 2016

@author: thor
"""

#import file

def converMunicipalNamesToCodes():
    dictionaryFile = open("simplified_municipality_indicators.csv", 'r');    
    dictlines = dictionaryFile.readlines()
    dictionaryFile.close()    
    dictionary = {}
    for dline in dictlines:
        strLine = str(dline)
        strLine = strLine.split(',')
        if strLine[0] == "code":
            continue ##skipping first line
        #print strLine[1]
        string = repr(strLine[1])
        dictionary.update({string:strLine[0]})
    #print dictionary #test
    #print dictionary.get(repr('V\xc3\xa4nersborg')) #test
    
    kpiFile = open("school_fire_cases_1998_2014.csv", 'r')
    newkpifile = open("school_fire_cases_1998_2014_wcodes1.csv", 'w')
    lines = kpiFile.readlines()
    kpiFile.close
    
    for line in lines:
        line = line.replace('\n','')
        strLine = str(line)
        strLine = strLine.split(',')
        cityname = repr(strLine[0]).replace('"','')
        #print cityname
#==============================================================================
#         if strLine[1] == 'Cases':
#             print "skipping"
#             newkpifile.write(str(line) + ', code\n')
#==============================================================================
        #print cityname
        #print dictionary.get(cityname)
        dashit = dictionary.get(cityname)
        if dashit is None:
            newkpifile.write(line + ', code\n')
        else:
            newkpifile.write(line + ','+ dashit +'\n')
        
        newkpifile.close


#bruges til at finde indexet for de forskellige år såden ikke skal lede milioner af linier igennem
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

#utrækker en liste der beskriver hvilke årstal datasættet indeholder, kan bruges til at finde index, oprette filer, 
#og til at begrændse søgningnen i filen         
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

##skal være skriptet der opretter filen der skrives til og som transformerer datasættet
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
    
#kpiFile = open("/home/thor/Desktop/5.semester/swedish-school-fires/kpis_2009_2011.csv", 'r');
#newKpiFile = open('/home/thor/Desktop/5.semester/swedish-school-fires/plebtest2.csv', 'w')
#getIndeces(kpiFile, "2010")
#extractPeriods(kpiFile)
#kpiFile.close
converMunicipalNamesToCodes()