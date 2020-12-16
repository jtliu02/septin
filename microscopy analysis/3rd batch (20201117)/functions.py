#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from PIL import Image

import gsd.fl
import gsd.hoomd
import gsd.pygsd


# In[2]:


#microscopy images


# In[3]:


#edit imageJ raw data

def microscopy(filename,im): 
    #filename = .csv files as string. example: filename='L1.csv'
    #im = .tif fluorescence microscopy images
    
    df = pd.read_csv(filename)
    
    #drop extra index column at beginning if imagej generated one
    if len(df.columns) == 12:
        df.drop(df.columns[0],axis=1,inplace=True)
        
    if len(df.columns)==11:
        df.rename(columns={'Contour ID': 'Contour_ID'},inplace=True)
        df.rename(columns={'Line width': 'Width'},inplace=True)
        df.rename(columns={'Angle of normal': 'Aon'},inplace=True) #actually tangent angle
        
        #drop irrelevant columns (frame, pos., contrast, asymmetry, class)
        df.drop(df.columns[[0,2,6,7,10]],axis=1,inplace=True)

        #clean up edges (frequent wrong detecetion) and duplications
        xmax = df.X.max()
        ymax = df.Y.max()
        df.query('X!=0 and X!=@xmax',inplace=True)
        df.query('Y!=0 and Y!=@ymax',inplace=True)
        df.drop_duplicates(subset=['X','Y'],inplace=True)
        df.reset_index(drop=True,inplace=True)

        #intensity (13.6 pixel = 1 micron)
        imarray = np.array(im)
        df['Intensity'] = imarray[round(13.6*df['Y']).astype(int).values,round(13.6*df['X']).astype(int).values]

        #cross sectional fluorescence intensity
        def cord(x,y,ang,d):
            inrad = np.array(range(int(-d/2),int((d+1)/2)))
            xcord = np.around(inrad*np.cos(ang-np.pi/2)).astype(int)
            ycord = np.around(inrad*np.sin(ang-np.pi/2)).astype(int)
            try:
                imsum = np.sum(imarray[ycord+y,xcord+x])
                return imsum
            except IndexError:
                return 0
    
        cord = np.vectorize(cord)
        df['TotalCross'] = cord(round(13.6*df['X']).astype(int),round(13.6*df['Y']).astype(int),df.Aon,round(13.6*df['Width']).astype(int))
        
        #clean up messy detections
        df.query('TotalCross!=0',inplace=True)
        df.query('Intensity>40',inplace=True)
        df.query('Length>=1',inplace=True)
        
        df.reset_index(inplace=True,drop=True)
            
        return df


# In[4]:


#local orientation alignment

def alignmentM(dataframe,num):
    #dataframe = results from microscopy function
    #num = number of sample pairs for analysis
    
    result = pd.DataFrame()
    
    #change 2pi periodic angles to [0,pi)
    df = dataframe.copy(deep=True)
    df['Aon'].mask(df['Aon']>=np.pi, df['Aon']-np.pi,inplace=True)
    
    #convert microns to pixel
    df['X'] = round(13.6*df['X']).astype(int)
    df['Y'] = round(13.6*df['Y']).astype(int)
    df['Width'] = round(13.6*df['Width'])
    df['Length'] = round(13.6*df['Length'])
    
    #assign regions (50p x 50p) as (Xreg, Yreg). alignment only computed with points pairs in neighboring or same regions.
    df['Xreg'] = round(df['X']/50).astype(int)
    df['Yreg'] = round(df['Y']/50).astype(int)
    
    #pick random pairs distributed across all regions based on total points in each region
    for (xr,yr) in list(product(df.Xreg.unique(),df.Yreg.unique())):
        
        #cen is points in a specific 50p x 50p; out is 150p x 150p surrounding or at cen
        cen = df.query('Xreg==@xr and Yreg==@yr').reset_index(drop=True)
        out = df.query('Xreg==@xr-1 | Xreg==@xr | Xreg==@xr+1')
        out = out.query('Yreg==@yr-1 | Yreg==@yr | Yreg==@yr+1').reset_index(drop=True)
        
        #randomly picks n indices within cen and out each
        n = int(np.round(num*len(cen)/len(df)))
        a = np.random.randint(len(cen),size=n)
        b = np.random.randint(len(out),size=n)
        
        #pair up cen and out points
        aa = cen.iloc[a].drop(columns=['Contour_ID','Length','Width','Intensity','TotalCross','Xreg','Yreg']).reset_index(drop=True)
        bb = out.iloc[b].drop(columns=['Contour_ID','Length','Width','Intensity','TotalCross','Xreg','Yreg']).reset_index(drop=True)
        #note: could opt to keep contour ids of each point to assess alignment between points that are not on the same filament. however, we find it to be irrelevant
        dd = aa.join(bb,lsuffix='1',rsuffix='2')
        result = pd.concat([result,dd])
    
    #find distance (micron) and alignment between each pair
    result['r'] = np.round(np.sqrt((result.X1-result.X2)**2+(result.Y1-result.Y2)**2)/13.6,1)
    result['align'] = np.cos(2*(result.Aon1-result.Aon2))
    
    #keep r and align column
    result.drop(columns=['X1','X2','Y1','Y2','Aon1','Aon2'],inplace=True)
    
    return result


# In[5]:


#calculate samples of chosen points not in same filament IF chooses to include contour id above

def nsamp(dataframe,align):
    #dataframe = results from microscopy function
    #align = alignment result returned above
   
    possible = 0
    
    for (xr,yr) in list(product(dataframe.Xreg.unique(),dataframe.Yreg.unique())):
        cen = dataframe.query('Xreg==@xr and Yreg==@yr').reset_index(drop=True)
        out = dataframe.query('Xreg==@xr-1 | Xreg==@xr | Xreg==@xr+1')
        out = out.query('Yreg==@yr-1 | Yreg==@yr | Yreg==@yr+1').reset_index(drop=True)
        possible += len(cen)*len(out)
    
    return len(align)/possible


# In[6]:


#flexibility

def flexibility(dataframe,num):
    #dataframe = results from microscopy function
    #num = number of sample pairs for analysis
    
    df = dataframe.copy(deep=True)
    df.sort_values(by='Contour_ID',inplace=True)
    
    #filter out filaments <= 3 pixels long
    dfa = df.drop_duplicates(subset='Contour_ID',keep='first',ignore_index=True)
    dfb = df.drop_duplicates(subset='Contour_ID',keep='last',ignore_index=True)
    dfc = dfa.join(dfb,lsuffix='1',rsuffix='2')
    dfc = dfc.loc[np.absolute(dfc['X1']-dfc['X2'])>3/13.6]
    dfc = dfc.loc[np.absolute(dfc['Y1']-dfc['Y2'])>3/13.6]
    
    #pick out random filament id
    result = pd.DataFrame()
    a = dfc['Contour_ID1'].unique()
    b = np.random.randint(len(a),size=num)
    c = np.array(a[b])
    
    #flexibility of each filament
    for i in c:
        
        #calculate angle differences between each possible pairs of points within a filament
        d = df.query('Contour_ID == @i')
        ang = np.array(d['Aon'])
        inv = ang.reshape([1,-1]).T
        cos = np.cos(ang-inv).flatten()
        
        #determine distance corresponding to eachc position of array
        l = len(ang)
        x = np.arange(l)
        y = x.reshape([1,-1]).T
        r = np.absolute((x-y).flatten())
        
        #compile into dataframe
        e = pd.DataFrame({'r':r/13.6,'cos':cos})
        result = pd.concat([result,e]) 

    return result  


# In[7]:


#calculate proportion of filaments that satisfied the minimum length requirement
def psamp(dataframe):
    df = dataframe.copy(deep=True)
    df.sort_values(by='Contour_ID',inplace=True)
    
    dfa = df.drop_duplicates(subset='Contour_ID',keep='first',ignore_index=True)
    dfb = df.drop_duplicates(subset='Contour_ID',keep='last',ignore_index=True)
    df = dfa.join(dfb,lsuffix='1',rsuffix='2')
    l = len(df)
    df = df.loc[np.absolute(df['X1']-df['X2'])<3/13.6]
    df = df.loc[np.absolute(df['Y1']-df['Y2'])<3/13.6]
    m = len(df)
    
    return m/l


# In[8]:


#simulation


# In[9]:


#process .gsd simulation files into dataframe

def sim(filename):
    #filename = .gsd file in string format. example: 's0.gsd'
    
    #final frame trajectory data
    f = gsd.pygsd.GSDFile(open(filename,'rb'))
    t = gsd.hoomd.HOOMDTrajectory(f)[0]
    
    #xy coordinate position
    pos = t.particles.position
    x = np.array([i[0] for i in pos]).flatten()
    y = np.array([i[1] for i in pos]).flatten()
    
    #0: A, 1: ghost
    tid = t.particles.typeid
    
    df = pd.DataFrame({'n':np.NaN,'Contour_ID':np.NaN,'X':x,'Y':y,'Length':np.NaN,'Type':tid})
    
    #find septin at filament ends
    bond = t.bonds.group
    a = np.array([i[0] for i in bond])
    b = np.array([i[1] for i in bond])
    dfa = pd.DataFrame({'a':a, 'b':b})
    dfb = pd.DataFrame({'a':b, 'b':a})
    dup = dfa.append(dfb) #compile all the septin id that appeared in bond data
    edf = dup.drop_duplicates(subset='a',keep=False).reset_index(drop=True) #kept edge septin (appeared once or one bond only). middle appeared twice and lone septin none
    ea = edf.a #array of septin id on either end of a filament
    
    #assign contour id and keep position of septin in its filament
    for i in ea:
        n = [i] #ongoing list of septin id within a filament
        if np.isnan(df.iloc[i,1]) == True: #avoid duplicate operation as both the beginning and end septin id are in ea array
            while True:
                try:
                    m = np.array(dfa[(dfa.a==i)|(dfa.b==i)]).flatten() #find septins paired with last documented septin i
                    n.append(m[np.isin(m,n)==False][0]) #in filament abc, b has 4 pairs (ab,cb,ba,bc). append only c
                    i = n[-1]
                except IndexError: #indexError in n.append step has there's no pair (i.e. reached the last septin)
                    df.loc[n,'Contour_ID'] = i #assign contour id based on last septin id
                    df.loc[n,'Length'] = len(n) #number of septin molecules in the filament
                    for l in n: #position of each septin in the filament
                        df.loc[l,'n'] = n.index(l)
                    break
    
    #drop ghost molecules
    df = df.query('Type == 0').drop(columns='Type')
    
    #drop lone septin
    ind = df[df['Contour_ID'].isna()==True].index #ghost molecule doesn't have assigned contour id
    df.drop(ind)
    
    df = df.sort_values(by=['Contour_ID','n'])
    df.reset_index(inplace=True,drop=True)
    
    #angle. determined by angle of line connecting neighboring septin(s)
    a = t.configuration.box[0] #box side dimension
    df['xn'] = df['X']
    df['yn'] = df['Y']
    
    for i in range(1,len(df)): #unwrap xy coordinate due connected left/right and top/bottom border
        if df['Contour_ID'][i]==df['Contour_ID'][i-1]: #if not the first septin in a filament 
            if np.absolute(df['xn'][i]-df['xn'][i-1])>15: #if crosses over right/left border (15 is just a random distance too big for the spacing between two consecutive filament)
                df.loc[i,'xn'] = df.loc[i,'xn']+a #try adding a distance to x coordinate (move the point one box to the right)
                if np.absolute(df['xn'][i]-df['xn'][i-1])>15: #still too far apart (should move left instead)
                    df.loc[i,'xn'] = df.loc[i,'xn']-2*a #move two boxes left          
            if np.absolute(df['yn'][i]-df['yn'][i-1])>15: #if crosses over top/bottom border (same idea as x)
                df.loc[i,'yn'] = df.loc[i,'yn']+a
                if np.absolute(df['yn'][i]-df['yn'][i-1])>15:
                    df.loc[i,'yn'] = df.loc[i,'yn']-2*a
                
    row = pd.DataFrame({'xn':[np.NaN],'yn':[np.NaN]})
    refn = df.drop(df.index[0]).drop(columns=['X','Y','n','Length','Contour_ID']).append(row,ignore_index=True) #all rows moved up (take angle from next septin)
    refn.rename(columns={'xn':'X1','yn':'Y1'},inplace=True)
    refp = row.append(df.drop(df.index[-1]).drop(columns=['X','Y','n','Length','Contour_ID']),ignore_index=True) #all rows moved down (take angle from previous septin)
    refp.rename(columns={'xn':'X2','yn':'Y2'},inplace=True)
    df = pd.concat([df,refn,refp],axis=1)
    
    df['Aon'] = np.arctan((df.Y2-df.Y1)/(df.X2-df.X1)) #find angles for middle septin (by two neighboring septins) 
    df.loc[df['n']==0,'Aon'] = np.arctan((df.yn-df.Y1)/(df.xn-df.X1)) #find angles for first septin of each filament (by position of itself and the next septin)
    df.loc[df['n']==(df['Length']-1), 'Aon'] = np.arctan((df.yn-df.Y2)/(df.xn-df.X2)) #find angles for the last septin of each filament (by position of itself and the previous septin)
    
    df['Aon'].mask(df['Aon']<0, df['Aon']+np.pi,inplace=True) #change [-pi/2,pi/2] to [0,pi)
    
    df = df.drop(columns=['n','xn','yn','X1','Y1','X2','Y2'])
    
    return df


# In[10]:


def alignmentS(dataframe,num):
    #dataframe = results from sim function
    #num = number of sample pairs for analysis
    
    df = dataframe.copy(deep=True)
    
    #pick random index for pairing
    a = np.random.randint(len(df),size=num)
    b = np.random.randint(len(df),size=num)
    aa = df.iloc[a].drop(columns=['Contour_ID','Length']).reset_index(drop=True)
    bb = df.iloc[b].drop(columns=['Contour_ID','Length']).reset_index(drop=True)
    d = aa.join(bb,lsuffix='1',rsuffix='2')
    
    #find minimum distance between each pair. note: 125.33141 determined by t.configuration.box[0] in sim()
    cx = pd.DataFrame()
    cy = pd.DataFrame()
    cx['Ax1'] = np.absolute(d.X2-125.33141-d.X1) #if X2 shift left by 1 box
    cy['Ay1'] = np.absolute(d.Y2-125.33141-d.Y1)
    cx['Ax2'] = np.absolute(d.X2-d.X1) #if X2 doesn't shift
    cy['Ay2'] = np.absolute(d.Y2-d.Y1)
    cx['Ax3'] = np.absolute(d.X2+125.33141-d.X1) #if X2 shift right by 1 box
    cy['Ay3'] = np.absolute(d.Y2+125.33141-d.Y1)
    
    d['Ax'] = cx.min(axis=1) #find the minimum distance between X1 and X2 from the 3 possible distances
    d['Ay'] = cy.min(axis=1)

    d['r'] = np.round(np.sqrt((d.Ax)**2+(d.Ay)**2),0)*0.032
    d['align'] = np.cos(2*(d.Aon1-d.Aon2))
    d.drop(columns=['X1','X2','Y1','Y2','Ax','Ay','Aon1','Aon2'],inplace=True)
    
    return d.sort_values(by='r')

