##### Run Steamlit

import streamlit as st
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import joblib
from statsbombpy import sb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from scipy.spatial import ConvexHull
from matplotlib.font_manager import FontProperties
import joblib
from matplotlib.lines import Line2D
from adjustText import adjust_text
import matplotlib.patches as patches
import zipfile

def plot_pca(position_group,metrics,no_clusters,random_state,cluster_names,legend_loc = 'upper right',player_ids=0,season_ids = 0,am_change=0):
    
    work_sans_font = FontProperties(fname='input/www/WorkSans-Regular.ttf')
    work_sans_font_legend = FontProperties(fname='input/www/WorkSans-Regular.ttf',size=16)
    work_sans_font_sb = FontProperties(fname='input/www/WorkSans-SemiBold.ttf')

    figs = []
    text_output = []
        
    
    if am_change == 1:
        
        zip_path = 'data/GB1_GB2_TR1_22_23_23_24_24_25.csv.zip'
        
        with zipfile.ZipFile(zip_path) as z:
            # Find the actual CSV (not starting with __MACOSX/)
            csv_filename = [f for f in z.namelist() if f.endswith('.csv') and not f.startswith('__MACOSX')][0]
            with z.open(csv_filename) as f:
                player_data = pd.read_csv(f)
        
                
        
        
        player_data = player_data.drop_duplicates()
        player_data['PlayerNickName'] = player_data['PlayerNickName'].fillna(player_data['PlayerName'])
        player_data = player_data.fillna(0)
        
        conditions = [
            player_data['PositionId'].isin([1]),
            player_data['PositionId'].isin([2, 7, 6, 8]),
            player_data['PositionId'].isin([3, 4, 5]),
            player_data['PositionId'].isin([9, 10, 11,13,14,15]),
            player_data['PositionId'].isin([12, 17, 16, 21,18,19,20]),
            player_data['PositionId'].isin([25,22,23,24])
        ]
        
        # 1 = GK, 2 = FB, 3 = CB, 4 = DM/CM, 5 = AM/W, 6 = AT
        values = [1, 2, 3, 4, 5,6]
        player_data['GroupID'] = np.select(conditions, values, default=np.nan)
        
        
        #Get Mins. Played
        player_data['ShiftStartTime'] = pd.to_datetime(player_data['ShiftStartTime'], format='%H:%M:%S').dt.time
        player_data['ShiftEndTime'] = pd.to_datetime(player_data['ShiftEndTime'], format='%H:%M:%S').dt.time
        
        # Compute difference in minutes
        player_data['minutes'] = (
            pd.to_datetime(player_data['ShiftEndTime'], format='%H:%M:%S') - 
            pd.to_datetime(player_data['ShiftStartTime'], format='%H:%M:%S')
        ).dt.total_seconds().div(60).round().astype(int)
        
        
        
        team_data_22_23 = sb.team_season_stats(2, 235,creds={"user": "craig.e@sportrepublic.com", "passwd": "3pmxmPEs"})
        team_data_23_24 = sb.team_season_stats(2, 281,creds={"user": "craig.e@sportrepublic.com", "passwd": "3pmxmPEs"})
        team_data_24_25 = sb.team_season_stats(2, 317,creds={"user": "craig.e@sportrepublic.com", "passwd": "3pmxmPEs"})
        
        champ_team_data_22_23 = sb.team_season_stats(3, 235,creds={"user": "craig.e@sportrepublic.com", "passwd": "3pmxmPEs"})
        champ_team_data_23_24 = sb.team_season_stats(3, 281,creds={"user": "craig.e@sportrepublic.com", "passwd": "3pmxmPEs"})
        champ_team_data_24_25 = sb.team_season_stats(3, 317,creds={"user": "craig.e@sportrepublic.com", "passwd": "3pmxmPEs"})
        
        tr1_team_data_22_23 = sb.team_season_stats(85, 235,creds={"user": "craig.e@sportrepublic.com", "passwd": "3pmxmPEs"})
        tr1_team_data_23_24 = sb.team_season_stats(85, 281,creds={"user": "craig.e@sportrepublic.com", "passwd": "3pmxmPEs"})
        tr1_team_data_24_25 = sb.team_season_stats(85, 317,creds={"user": "craig.e@sportrepublic.com", "passwd": "3pmxmPEs"})
        
        all_team_data = pd.concat([team_data_22_23,team_data_23_24,team_data_24_25,
                                   champ_team_data_22_23,champ_team_data_23_24,champ_team_data_24_25,
                                   tr1_team_data_22_23,tr1_team_data_23_24,tr1_team_data_24_25],axis=0)
            
        player_data = pd.merge(player_data,all_team_data[['team_id','team_name','season_id','team_season_possession']]
                               ,left_on=['TeamId','SeasonId'],right_on=['team_id','season_id'])
        

        
        #Sort out right-left columns
        player_data['Low Received Passes In Atk Half Half Space Outside Box OP'] = player_data['Low Received Passes In Atk Half Left Half Space Outside Box OP'] + player_data['Low Received Passes In Atk Half Right Half Space Outside Box OP'] 
        player_data['Low Received Passes In Atk Half Channel OP'] = player_data['Low Received Passes In Atk Half Left Channel OP'] + player_data['Low Received Passes In Atk Half Right Channel OP'] 
        player_data['Low Channel Forward Received Passes Inside OP'] = player_data['Low Left Channel Forward Received Passes Inside OP'] + player_data['Low Right Channel Forward Received Passes Inside OP'] 
        player_data['Low Channel Forward Received Passes Outside OP'] = player_data['Low Left Channel Forward Received Passes Outside OP'] + player_data['Low Right Channel Forward Received Passes Outside OP'] 
        player_data['Carries Wide'] = player_data['Left Half Carries Left'] + player_data['Right Half Carries Right'] 
        player_data['Carries In'] = player_data['Left Half Carries Right'] + player_data['Right Half Carries Left'] 
        player_data['Carries Into Final Third From Half Space'] = player_data['Carries Into Final Third From Left Half Space'] + player_data['Carries Into Final Third From Right Half Space'] 
        player_data['Carries Into Final Third From Channel'] = player_data['Carries Into Final Third From Left Channel'] + player_data['Carries Into Final Third From Right Channel'] 
        player_data['Pressures Wide Third Def Third'] = player_data['Pressures Left Third Def Third'] + player_data['Pressures Right Third Def Third'] 
        
        player_data = player_data.drop(columns=[col for col in player_data.columns if 'Right' in col or 'Left' in col])
        
        
        #Get ratios
        
       
        
        player_data['Total Dribbles'] = player_data['Successful Dribbles'] + player_data['Unsuccessful Dribbles']
        
        player_data['Progressive Passes Attempted'] = (player_data['Unsuccessful Progressive Passes 10Perc'] + player_data['Successful Progressive Passes 10Perc'])
        player_data['Def Third Passes Attempted'] = (player_data['Successful Passes To Def Third'] + player_data['Unsuccessful Passes To Def Third'])

        player_data['Total Cutbacks'] = player_data['Successful Cutbacks'] + player_data['Unsuccessful Cutbacks']
        player_data['Total Crosses'] = player_data['Successful Crosses OP'] + player_data['Unsuccessful Crosses']
        player_data['Total Crosses From Deep'] = player_data['Successful Crosses From Deep'] + player_data['Unsuccessful Crosses From Deep']
        
        player_data['Passes Attempted'] = player_data['Successful Passes OP'] + player_data['Unsuccessful Passes OP']
        player_data['Long Balls Attempted'] = player_data['Successful Long Balls'] + player_data['Unsuccessful Long Balls']
        
        player_data['Aerial Duels'] = player_data['Aerial Duels Won'] + player_data['Aerial Duels Lost']
        
        player_data['Headed Shots'] = player_data['Headed Shots Off T'] +player_data['Headed Shots On T']

        player_data['Defensive Aerial Duels'] = player_data['Def Aerial Duels Won'] + player_data['Def Aerial Duels Lost']
        
        player_data['Ball Recovery and Interceptions'] = player_data['Recoveries Total'] + player_data['Interceptions Total']

        player_data['Attempted Passes to Def Third'] = player_data['Successful Passes To Def Third'] + player_data['Unsuccessful Passes To Def Third']
        
        player_data['GK Total Collections Attempted'] = player_data['GK Collections'] + player_data['GK Collections Failed']
        player_data['GK Total Punches Attempted'] = player_data['GK Punches Failed'] + player_data['GK Punches']
        
        player_data['GK High Passes OP'] = player_data['GK Successful High Pass OP'] + player_data['GK Unsuccessful High Pass OP']
        player_data['GK Low Passes OP'] = player_data['GK Successful Low Pass OP'] + player_data['GK Unsuccessful Low Pass OP']
        player_data['GK Ground Passes OP'] = player_data['GK Successful Ground Pass OP'] + player_data['GK Unsuccessful Ground Pass OP']
        
        player_data['GK Passes OP'] = player_data['GK High Passes OP'] + player_data['GK Low Passes OP'] +  player_data['GK Ground Passes OP']
        player_data['GK Short Dist OP'] = player_data['GK Successful Short Dist OP'] + player_data['GK Unsuccessful Short Dist OP']
        player_data['GK Long Dist OP'] = player_data['GK Successful Long Dist OP'] + player_data['GK Unsuccessful Long Dist OP']

        player_data['GK Throws'] = player_data['GK Successful Throws'] + player_data['GK Unsuccessful Throws']
        
        player_data['Low Pass Received in Def Third Outside Box'] = player_data['Low Received Passes In Def Third OP']


        player_data['pAdj_Received Passes OP'] = player_data['Received Passes OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Successful Passes OP'] = player_data['Successful Passes OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Low Received Passes In Atk Half OP'] = player_data['Low Received Passes In Atk Half OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Low Received Passes In Final Third OP'] = player_data['Low Received Passes In Final Third OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Low Received Passes In Def Third OP'] = player_data['Low Received Passes In Def Third OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Low Received Passes In Box OP'] = player_data['Low Received Passes In Box OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Low Received Passes In 10 Space OP'] = player_data['Low Received Passes In 10 Space OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Low Received Progressive Passes 10 Perc OP'] = player_data['Low Received Progressive Passes 10 Perc OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Long Ball Received OPP'] = player_data['Long Ball Received OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Layoff Received OP'] = player_data['Layoff Received OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Layoff Received In Atk Half OP'] = player_data['Layoff Received In Atk Half OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Through Ball Received In Final Third OP'] = player_data['Through Ball Received In Final Third OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Expected Progressive Passes 10Perc  '] = player_data['Expected Progressive Passes 10Perc'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Expected Switch Passes OP'] = player_data['Expected Switch Passes OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Expected Diagonal Passes OP '] = player_data['Expected Diagonal Passes OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Expected Quick Passes OP'] = player_data['Expected Quick Passes OP'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Expected Long Balls'] = player_data['Expected Long Balls'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Expected Passes To Def Third'] = player_data['Expected Passes To Def Third'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Carries Over 3 Seconds'] = player_data['Carries Over 3 Seconds'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Carries Over 1 Second'] = player_data['Carries Over 1 Second'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Carries Into Opp Half'] = player_data['Carries Into Opp Half'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Def Actions'] = player_data['Def Actions'] * (0.5/(1-player_data['team_season_possession']))
        player_data['pAdj_Ball Receipts In Box'] = player_data['Ball Receipts In Box'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Expected Crosses From Deep'] = player_data['Expected Crosses From Deep'] * (0.5/player_data['team_season_possession'])
        player_data['pAdj_Low Received Passes In Atk Half Half Space Outside Box OP'] = player_data['Low Received Passes In Atk Half Half Space Outside Box OP']* (0.5/player_data['team_season_possession'])
        player_data['pAdj_Total Crosses'] = player_data['Total Crosses']* (0.5/player_data['team_season_possession'])
        player_data['pAdj_Carries Into Opp Half'] = player_data['Carries Into Opp Half']* (0.5/player_data['team_season_possession'])
        player_data['pAdj_Foot Clearances'] = player_data['Foot Clearances']*  (0.5/(1-player_data['team_season_possession']))
        player_data['pAdj_Progressive Passes Attempted'] = player_data['Progressive Passes Attempted']*   (0.5/player_data['team_season_possession'])
        player_data['pAdj_Def Third Passes Attempted'] =  player_data['Def Third Passes Attempted']*   (0.5/player_data['team_season_possession'])

        #Possession is possessiona against
        player_data['pAdj_Pressures Wide Third Def Third'] = player_data['Pressures Wide Third Def Third'] * (0.5/(1-player_data['team_season_possession']))
        
      
        
        player_data = player_data[player_data['minutes']>10]
        
        
        
        first_columns = ['PlayerNickName', 'TeamName', 'SeasonId','GroupID','PlayerId','CompetitionId','team_name','PositionId']
        drop_columns = ['TeamId', 'MatchId', 'ShiftStartTime', 'ShiftEndTime','PlayerName']
        avg_cols = ['Pressures AvX','Def Actions AvX']

        
        agg_dict = {}
        for col in player_data.columns:
            if col in drop_columns or col in first_columns:
                pass
            elif col in avg_cols:
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'sum'
        
        # Apply groupby and aggregation
        grouped_player_data = (
            player_data
            .drop(columns=drop_columns)
            .groupby(['PlayerId', 'PlayerNickName','GroupID', 'SeasonId', 'CompetitionId','team_name'], as_index=False)
            .agg(agg_dict)
        )
        exclude_cols = ['PlayerId', 'CompetitionId', 'SeasonId', 'PlayerNickName', 'GroupID','team_name', 'minutes',
                        'Pressures AvX','Def Actions AvX']
        
        
        grouped_player_data.loc[:, ~grouped_player_data.columns.isin(exclude_cols)] = (
            grouped_player_data.loc[:, ~grouped_player_data.columns.isin(exclude_cols)]
            .apply(lambda x: (x * 90) / grouped_player_data['minutes'])
        )
        
        grouped_player_data = grouped_player_data[(grouped_player_data['minutes']>500)&
                                                  (grouped_player_data['minutes']<4000)
                                                  ]

        
        #get average pressure X per team
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name'])['Pressures AvX'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name'])['Pressures AvX'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Pressures AvX'] = grouped_player_data['Pressures AvX'] / other_avg

        #get average pressure x per team per pos
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Pressures AvX'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Pressures AvX'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Pos Pressures AvX'] = grouped_player_data['Pressures AvX'] / other_avg

        #get average progressive passess per team per pos
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name'])['Progressive Passes Attempted'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name'])['Progressive Passes Attempted'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Progressive Passes Attempted'] = grouped_player_data['Progressive Passes Attempted'] / other_avg
       
        #get average progressive passess per team per pos
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Progressive Passes Attempted'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Progressive Passes Attempted'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Pos Progressive Passes Attempted'] = grouped_player_data['Progressive Passes Attempted'] / other_avg
        
        #get average aerila clearances in box per team
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Aerial Clearances In Box'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Aerial Clearances In Box'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Pos Aerial Clearances In Box'] = grouped_player_data['Aerial Clearances In Box'] / other_avg
        
        
        #get average aerila key pass 
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Key Passes'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Key Passes'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Key Pass'] = grouped_player_data['Key Passes'] / other_avg

        #get average aerila pass received 
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name'])['Received Passes OP'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name'])['Received Passes OP'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Pass Received OP'] = grouped_player_data['Received Passes OP'] / other_avg
        
        #get average aerila pass received f3 
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name'])['Low Received Passes In Final Third OP'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name'])['Low Received Passes In Final Third OP'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Low F3 Received'] = grouped_player_data['Low Received Passes In Final Third OP'] / other_avg
        
        
        #get average aerila passes to def third
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name'])['Attempted Passes to Def Third'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name'])['Attempted Passes to Def Third'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Attempted Passes to Def Third'] = grouped_player_data['Attempted Passes to Def Third'] / other_avg
        
        
        #get average aerila ball recov. interceptiosn
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Ball Recovery and Interceptions'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Ball Recovery and Interceptions'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Pos Ball Recovery and Interceptions'] = grouped_player_data['Ball Recovery and Interceptions'] / other_avg
        
        
        #get average aerila def actions
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Def Actions'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Def Actions'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Pos Def Actions'] = grouped_player_data['Def Actions'] / other_avg
        
        
        #get relative pressure opp half 
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Pressures Opp Half'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Pressures Opp Half'].transform('count')
        other_avg = group_sums/group_counts
        grouped_player_data['Relative Pos Pressures Opp Half'] = grouped_player_data['Pressures Opp Half'] / other_avg
        
        #get average aerila ball receipts in box
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name'])['Ball Receipts In Box'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name'])['Ball Receipts In Box'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Ball Receipts In Box'] = grouped_player_data['Ball Receipts In Box'] / other_avg
        
        #get average Def third received %
        grouped_player_data['Low Def Third Received Pass %'] = grouped_player_data['Low Received Passes In Def Third OP']/grouped_player_data['Received Passes OP']
        group_sums = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Low Def Third Received Pass %'].transform('sum')
        group_counts = grouped_player_data.groupby(['SeasonId', 'team_name','GroupID'])['Low Def Third Received Pass %'].transform('count')
        other_avg = group_sums/group_counts
        
        grouped_player_data['Relative Pos Low Def Third Received Pass %'] = grouped_player_data['Low Def Third Received Pass %'] / other_avg
        
        grouped_player_data['Carries'] = grouped_player_data['Carries In'] + grouped_player_data['Carries Wide']


        grouped_player_data['Headed Shot %'] = grouped_player_data['Headed Shots']/grouped_player_data['Non-Penalty Shots OP']
        
        grouped_player_data['Long Ball %'] = grouped_player_data['Long Balls Attempted']/grouped_player_data['Passes Attempted']
        
        grouped_player_data['OBV per Carry'] = (grouped_player_data['Carry OBV'])/grouped_player_data['Carries']
        grouped_player_data['OBV per Pass OP'] = (grouped_player_data['Pass OP OBV'])/grouped_player_data['Successful Passes OP']
        
        grouped_player_data['xG per Shot'] = grouped_player_data['Non-Penalty xG']/(grouped_player_data['Non-Penalty Shots OP'])
        grouped_player_data['Shots From Crosses %'] = grouped_player_data['Shots From Crosses']/grouped_player_data['Non-Penalty Shots OP']
        grouped_player_data['Shots per Pass'] = grouped_player_data['Non-Penalty Shots OP']/player_data['Passes Attempted']
        
        grouped_player_data['% of Atk Half Passes Received in 10 Space'] = grouped_player_data['Low Received Passes In 10 Space OP']/grouped_player_data['pAdj_Low Received Passes In Atk Half OP']
        grouped_player_data['% of Atk Half Passes Received in Half Space'] = grouped_player_data['Low Received Passes In Atk Half Half Space Outside Box OP']/grouped_player_data['pAdj_Low Received Passes In Atk Half OP']


        grouped_player_data['Aerial Win Rate'] = grouped_player_data['Aerial Duels Won']/(grouped_player_data['Aerial Duels Won'] + grouped_player_data['Aerial Duels Lost'])
        grouped_player_data['Aerial Pass Rate'] = grouped_player_data['Headed Passes']/(grouped_player_data['Headed Passes'] + grouped_player_data['Aerial Clearances'])

        grouped_player_data['Total Quick Passes'] = grouped_player_data['Unsuccessful Quick Passes OP'] + grouped_player_data['Successful Quick Passes OP']
        grouped_player_data['Total Passes'] = grouped_player_data['Unsuccessful Passes OP'] + grouped_player_data['Successful Passes OP']

        grouped_player_data['Quick Pass %'] = grouped_player_data['Total Quick Passes']/grouped_player_data['Total Passes']
        
        grouped_player_data['FWD Pass Rate'] = grouped_player_data['Successful FWD Passes OP']/(grouped_player_data['Successful Passes OP'])
        grouped_player_data['Distance per FWD Pass'] = grouped_player_data['Successful Pass Dist FWD OP']/grouped_player_data['Successful FWD Passes OP']

        grouped_player_data['OBV per Pass Received'] = grouped_player_data['Pass Received OP OBV']/grouped_player_data['Received Passes OP']

        grouped_player_data['Ball Received in Box %'] = grouped_player_data['Ball Receipts In Box']/grouped_player_data['Received Passes OP']

        grouped_player_data['Take Ons per Pass Received'] = grouped_player_data['Total Dribbles']/(grouped_player_data['Carries'])
        
        grouped_player_data['Pressures in Opp Third Def Third Ratio'] = abs(1-(grouped_player_data['Pressures Opp Third']/grouped_player_data['Pressures Own Third']))
        grouped_player_data['Low Pass Received Box Def Third Ratio'] = abs(1-(grouped_player_data['Low Received Passes In Final Third OP']/grouped_player_data['Pressures Own Third']))
        
        
        grouped_player_data['Low Received Passes In Atk Half Space vs Channel'] = grouped_player_data['Low Received Passes In Atk Half Half Space Outside Box OP']/(grouped_player_data['Low Received Passes In Atk Half Channel OP'] + grouped_player_data['Low Received Passes In Atk Half Half Space Outside Box OP'])
        grouped_player_data['Low Channel Forward Received Passes Inside Ratio'] = grouped_player_data['Low Channel Forward Received Passes Inside OP']/(grouped_player_data['Low Channel Forward Received Passes Outside OP'] + grouped_player_data['Low Channel Forward Received Passes Inside OP'])
        grouped_player_data['Carries In Ratio'] = grouped_player_data['Carries In']/(grouped_player_data['Carries Wide']+grouped_player_data['Carries In'])
        
        
        grouped_player_data['Carries Into Final Third From Half Space vs Channel'] = grouped_player_data['Carries Into Final Third From Half Space']/(grouped_player_data['Carries Into Final Third From Channel']+grouped_player_data['Carries Into Final Third From Half Space'])
        grouped_player_data['Cross Cutback %'] = grouped_player_data['Total Cutbacks']/(grouped_player_data['Total Cutbacks']+grouped_player_data['Total Crosses'])
        grouped_player_data['Cross Deep %'] = grouped_player_data['Total Crosses From Deep']/grouped_player_data['Total Crosses']
        
        grouped_player_data['Half Space Received Pass %'] = grouped_player_data['Low Received Passes In Atk Half Half Space Outside Box OP']/grouped_player_data['Received Passes OP']
        grouped_player_data['F3 Low Received Pass %'] = grouped_player_data['Low Received Passes In Final Third OP']/grouped_player_data['Received Passes OP']
        grouped_player_data['Atk Half Channel Low Received Pass %'] = grouped_player_data['Low Received Passes In Atk Half Channel OP']/grouped_player_data['Received Passes OP']

        grouped_player_data['Atk. Half Pressure %'] = grouped_player_data['Pressures Opp Half']/grouped_player_data['Pressures']
        grouped_player_data['Def Third Pressure %'] = grouped_player_data['Pressures Own Third']/grouped_player_data['Pressures']
        grouped_player_data['Pressure Wide Def Third %'] = grouped_player_data['Pressures Wide Third Def Third']/grouped_player_data['Pressures']
        grouped_player_data['Def Third Pressure Wide %'] = grouped_player_data['Pressures Wide Third Def Third']/grouped_player_data['Pressures Own Third']

        
        grouped_player_data['Ball Rceovery Own Third % Def Action'] = grouped_player_data['Recoveries Total Own Third']/grouped_player_data['Def Actions']
        grouped_player_data['Interceptions Def Action %'] = grouped_player_data['Interceptions Total']/grouped_player_data['Def Actions']
        grouped_player_data['Ball Recovery Def Action %'] = grouped_player_data['Recoveries Total']/grouped_player_data['Def Actions']
        grouped_player_data['Interceptions Atk Half %'] = grouped_player_data['Interceptions Total Opp Half']/grouped_player_data['Interceptions Total']
        grouped_player_data['Ball Recovery Own Third %'] = grouped_player_data['Recoveries Total Own Third']/grouped_player_data['Recoveries Total']

        grouped_player_data['Tackle Def Action %'] = grouped_player_data['Tackles Total']/grouped_player_data['Def Actions']
        grouped_player_data['Block Def Action %'] = grouped_player_data['Blocks Total']/grouped_player_data['Def Actions']

        grouped_player_data['Fouls per Def Action'] = grouped_player_data['Fouls Committed']/grouped_player_data['Def Actions']


        grouped_player_data['Ball Recovery Interception Def Action %'] = (grouped_player_data['Interceptions Total'] + grouped_player_data['Recoveries Total'])/grouped_player_data['Def Actions']

        grouped_player_data['Pass Received Low %'] = grouped_player_data['Low Received Passes OP']/grouped_player_data['Received Passes OP']

        grouped_player_data['Progressive Carry %'] = grouped_player_data['Progressive Carries 10Perc']/grouped_player_data['Carries']
        grouped_player_data['Progressive Low Pass %'] = (grouped_player_data['Unsuccessful Progressive Passes 10Perc'] + grouped_player_data['Successful Progressive Passes 10Perc'])/grouped_player_data['Passes Attempted']
        
        grouped_player_data["Clearance % from Def Third"] = grouped_player_data['Foot Clearances']/(grouped_player_data['Unsuccessful Passes From Own Third OP'] + grouped_player_data['Successful Passes From Own Third OP'])

        grouped_player_data['Key Pass Cross %'] = grouped_player_data['Key Passes Crosses']/grouped_player_data['Key Passes']
        grouped_player_data['Key Pass Progressive Carry %'] = grouped_player_data['Progressive Carry 10Perc Key Pass']/grouped_player_data['Key Passes']
        
        grouped_player_data['All Box Entries'] = grouped_player_data['Carries Into Box'] + grouped_player_data['Successful Passes Into Box'] + grouped_player_data['Ball Receipts In Box'] + grouped_player_data['Successful Crosses OP']
        grouped_player_data['Carry Box Entry %'] = grouped_player_data['Carries Into Box']/grouped_player_data['All Box Entries']
        grouped_player_data['Pass Box Entry %'] = grouped_player_data['Successful Passes Into Box']/grouped_player_data['All Box Entries']
        grouped_player_data['Ball Receipt Box Entry %'] = grouped_player_data['Ball Receipts In Box']/grouped_player_data['All Box Entries']
        

        grouped_player_data['Sweeper Rate %'] = grouped_player_data['GK Sweeper Keeper']/grouped_player_data['OppSweepablePasses']
        grouped_player_data['Claim Rate %'] = grouped_player_data['GK Total Collections Attempted']/grouped_player_data['OppClaimableHighBall']
        grouped_player_data['Punch Rate %'] = grouped_player_data['GK Total Punches Attempted']/grouped_player_data['GK Total Collections Attempted']

        grouped_player_data['Distance per Sweep'] = grouped_player_data['GK Total Sweeper Keeper Distance From Goal']/grouped_player_data['GK Sweeper Keeper']
        grouped_player_data['Distance per Collection Punch'] = (grouped_player_data['GK Total Punch Distance From Goal'] + grouped_player_data['GK Total Collection Distance From Goal'])/(grouped_player_data['GK Total Collections Attempted'] + grouped_player_data['GK Total Punches Attempted'])

        grouped_player_data['GK High Ball %'] = grouped_player_data['GK High Passes OP']/grouped_player_data['GK Passes OP']
        grouped_player_data['GK Low Ball %'] = grouped_player_data['GK Low Passes OP']/grouped_player_data['GK Passes OP']
        grouped_player_data['GK Ground Ball %'] = grouped_player_data['GK Ground Passes OP']/grouped_player_data['GK Passes OP']

        grouped_player_data['GK Throw %'] = player_data['GK Throws']/(grouped_player_data['GK Passes OP']+ player_data['GK Throws'])
        
        
        grouped_player_data['GK Short Dist %'] = grouped_player_data['GK Short Dist OP']/(grouped_player_data['GK Long Dist OP']+grouped_player_data['GK Short Dist OP'])

        grouped_player_data = grouped_player_data.fillna(0)

        grouped_player_data[grouped_player_data == np.inf] = 1
        grouped_player_data[grouped_player_data == -np.inf] = 1
 
        

    else:
        grouped_player_data = pd.read_csv('data/Grouped_Player_Final.csv',index_col=0)

    
    
    pos_5 = grouped_player_data[grouped_player_data['GroupID'] == position_group]
    
    pos_5 = pos_5[['PlayerId', 'CompetitionId', 'SeasonId', 'PlayerNickName','GroupID', 'team_name','minutes'] + 
                 metrics]
    
    drop_cols = ['PlayerId', 'CompetitionId', 'SeasonId', 'PlayerNickName', 'GroupID', 'team_name','minutes']
    feature_cols = [col for col in pos_5.columns if col not in drop_cols]
    
    # Extract features for clustering
    
    
    

    
    pos_5= pos_5.reset_index()
  
    
    data = pos_5[feature_cols].copy()
    
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled_df = pd.DataFrame(data_scaled, columns=feature_cols)
    
    #manual_indices = manual_player_rows.index.tolist()
    #selected_players_scaled = data_scaled[manual_indices]
    
        # Number of clusters
    n_clusters = no_clusters
    
    # Remaining centroids
    
 
    
    # Final KMeans
    clusterer = KMeans(
        n_clusters=n_clusters,
        #init=selected_players_scaled,
        n_init='auto',
        random_state=random_state
    )
    clusterer.fit(data_scaled)


    # Compute probabilities
    distances = clusterer.transform(data_scaled)
    inverse_distances = 1 / (distances + 1e-6)
    probabilities = inverse_distances / inverse_distances.sum(axis=1, keepdims=True)

    # Create a DataFrame for probabilities
    prob_df = pd.DataFrame(probabilities, columns=[f'Cluster_{i}_prob' for i in range(n_clusters)])

    # Assign clusters to data
    pos_5['cluster'] = clusterer.labels_

    # Merge with original identifier columns
    final_df = pos_5[drop_cols + ['cluster']].reset_index(drop=True).join(prob_df)

    # Compute range of means for each feature
    means_by_cluster_raw = pd.concat([pos_5[['cluster']], data], axis=1).groupby('cluster').mean()
    means_by_cluster = pd.concat([pos_5[['cluster']].reset_index(drop=True), data_scaled_df], axis=1).groupby('cluster').mean()
    ranges = means_by_cluster.max() - means_by_cluster.min()
    
    
    
    pca_data = data_scaled_df

    # Run PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(pca_data)

    # Create PCA DataFrame
    pca_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
    pca_df['cluster'] = pos_5['cluster'].values
    pca_df['PlayerId'] = pos_5['PlayerId'].values
    pca_df['PlayerNickName'] = pos_5['PlayerNickName'].values
    pca_df['SeasonId'] = pos_5['SeasonId'].values
    pca_df['CompetitionId'] = pos_5['CompetitionId'].values
    pca_df['team_name'] = pos_5['team_name'].values



    pca_df['ShortName'] = pca_df['PlayerNickName'].apply(
        lambda x: f"{x.split()[0][0]}. {' '.join(x.split()[1:])}" if len(x.split()) > 1 else x
    )



    filtered_pca = pca_df[pca_df['SeasonId'].isin([317]) & 
                          pca_df['CompetitionId'].isin([2])]



        
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 20),  facecolor='#fbf6ef',
                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
    

    import itertools

    # Your custom list of colors
    custom_colors = ['#d81159', '#93cf7d', '#fbb13c', '#73d2de','#fc79d6']
    
    num_clusters = len(filtered_pca['cluster'].unique())
    
    palette = list(itertools.islice(itertools.cycle(custom_colors), no_clusters))

    
    #ax1.set_facecolor('#fbf6ef')

    
    sns.scatterplot(data=filtered_pca, x='PC1', y='PC2', hue='cluster', palette=palette, s=80, alpha=0.8, ax=ax1)
    
    texts = []
    for _, row in filtered_pca.iterrows():
        text = ax1.text(row['PC1'] + 0.01, row['PC2'] + 0.01, row['ShortName'],
                        fontproperties=work_sans_font, fontsize=14, alpha=0.75)
        texts.append(text)
    
    
    # Adjust text labels to avoid overlap

    test = adjust_text(texts, ax=ax1,ensure_inside_axes=False,
                max_move=0.2,
                min_arrow_len=10)
    
    
    def plot_cluster_ellipse(ax, data, n_std=2.0, facecolor='none', edgecolor='black', alpha=0.2):
        """
        Plot an ellipse based on the covariance of the data points.
        """
        from matplotlib.patches import Ellipse

        if data.shape[0] < 2:
            return
        
        cov = np.cov(data.T)
        mean = np.mean(data, axis=0)
    
        # Eigen decomposition
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
    
        # Compute width, height, angle
        width, height = 2 * n_std * np.sqrt(vals)
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        ax.add_patch(ellipse)
    
   
    # Add Convex Hull polygons for each cluster
    for cluster in filtered_pca['cluster'].unique():
        cluster_data_all = pca_df[pca_df['cluster'] == cluster][['PC1', 'PC2']].values
        cluster_data = filtered_pca[filtered_pca['cluster'] == cluster][['PC1', 'PC2']].values
        cluster_color = palette[cluster]
        
        # Plot points (optional if already done)
        ax1.scatter(cluster_data[:, 0], cluster_data[:, 1], color=cluster_color, s=80, alpha=0.8)
    
        # Plot ellipse
        plot_cluster_ellipse(ax1, cluster_data_all, n_std=2.2, facecolor=cluster_color, alpha=0.2)  # Plot the cluster points
                
            
    if player_ids != 0:
        
        
        for i in range(len(player_ids)):
            player_id = player_ids[i]
            season_id = season_ids[i]
            
            print(i,player_id,season_id)
        
            player_pca = pca_df[(pca_df['PlayerId'] == player_id) &
                                   (pca_df['SeasonId'] == season_id)].iloc[0]
            
            player_cluster = player_pca['cluster']
            
            if season_id == 317:
                season = '24/25'
            
            if season_id == 281:
                season = '23/24'
            
            if season_id == 235:
                season = '22/23'
            
            ax1.scatter(player_pca['PC1'], player_pca['PC2'], color=palette[player_cluster], s=160, alpha=1, edgecolors='black')
            ax1.text(player_pca['PC1'] + 0.01, player_pca['PC2'] + 0.01, player_pca['ShortName'] + ' - ' +season, 
                    fontproperties = work_sans_font_sb, fontsize=22, alpha=1)
        
    
    # Create the legend with the number of players in each cluster
    handles, labels = ax1.get_legend_handles_labels()
    
    # Update the labels to include the number of players in each cluster
    new_labels = cluster_names
    
    # Add the updated legend with title, larger font size, and markers
    ax1.legend(handles=handles, labels=new_labels,  
               prop=work_sans_font_legend,fontsize=12, markerscale=1.5,
               loc=legend_loc)
    
    ax1.set_title('PCA Scatter Plot', 
                 fontproperties = work_sans_font_sb,
                 fontsize=22,
                 loc='left',
                 ha='left')
    
    fig.suptitle('Premier League 2024-2025\nPosition Group ' + str(position_group),
             fontproperties=work_sans_font_sb, fontsize=28, y=0.94)
    
    
    ax1.grid(True)
    
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    
    # --- Heatmap table ---
    wrapped_labels = ['\n'.join(textwrap.wrap(col, 15)) for col in means_by_cluster.columns]
    
    # Create a custom color normalization for each column
    
    cmap = sns.color_palette("blend:#ffffff,#2E145D", as_cmap=True)
    
    player_metrics = pos_5[(pos_5['PlayerId']==player_ids[0]) &
                            (pos_5['SeasonId']==season_ids[0])]
    
    player_name = player_metrics.iloc[0]['PlayerNickName']
    
    player_metrics = player_metrics[metrics]
    
    player_metrics_scaled = scaler.transform(player_metrics)
    player_metrics_scaled = pd.DataFrame(player_metrics_scaled, columns=feature_cols)
    
    
    means_by_cluster = pd.concat([means_by_cluster,player_metrics_scaled])
    
  
    
    means_by_cluster_raw = pd.concat([means_by_cluster_raw,player_metrics])
    
    
    sns.heatmap(
        means_by_cluster,
        annot=means_by_cluster_raw,
        cmap=cmap,
        cbar=False,
        fmt=".2g",  # Format for 3 significant figures
        ax=ax2,
        annot_kws={"size": 14, "fontproperties": work_sans_font},
        linewidths=0.5,  # Thin white lines between cells
        linecolor='white',  # Line color for grid
        )# Make each cell a square
    
    ax2.axhline(no_clusters, color='#fbf6ef', linestyle='-', linewidth=4)

    
    ax2.set_xticklabels(
        wrapped_labels,
        rotation=0,
        ha='center',
        fontproperties=work_sans_font,
        fontsize=13
    )
    ax2.set_yticklabels(
        cluster_names + [player_name],
        fontproperties=work_sans_font_sb,
        fontsize=15,
        rotation=0
    )
    
    # Set axis labels
    ax2.set_ylabel(
        "",
        fontproperties=work_sans_font_sb,
        fontsize=20
    )
    ax2.set_title(
        "Cluster Averages",
        fontproperties=work_sans_font_sb,
        fontsize=22,
        loc='left',
        ha='left',
        pad=8
    )
    
    
    # Remove axis spines for a cleaner look
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    # Adjust layout and save
    plt.tight_layout()
    figs.append(fig)
    
    
    pl_2425 = final_df[(final_df['CompetitionId']==2) &
                       (final_df['SeasonId']==317)]
        
    text = ""

    for i in range(no_clusters):
        col = f'Cluster_{i}_prob'
        color = palette[i]  # Get color for this cluster
        cluster_marquee = pl_2425.nlargest(3, col)[['PlayerNickName', col]]
        
        # Use cluster_names[i] instead of "Cluster i"
        cluster_name = cluster_names[i]  # Get the name of the cluster
        text += f"<h6 style='color:{color};'>{cluster_name} Example Players:</h6>"  # Update header with cluster name
        
        for _, row in cluster_marquee.iterrows():
            text += f"<p style='color:{color}; margin-bottom: 15px;'>• {row['PlayerNickName']}: {row[col]*100:.1f}%</p>"
            
            
    text_output.append(text)
        
    
    if player_ids !=0:
        
        for i in range(len(player_ids)):
            
            player_id = player_ids[i]
            season_id = season_ids[i]
                
                    # Get the colors (assuming you're already using this)
            
            player_data = final_df[(final_df['PlayerId'] == player_id) &
                                   (final_df['SeasonId'] == season_id)].iloc[0]
            
            player_name = player_data['PlayerNickName']
            team_name = player_data['team_name']
            
            if season_id == 317:
                season = '24/25'
            
            if season_id == 281:
                season = '23/24'
            
            if season_id == 235:
                season = '22/23'
            
            player_data = player_data[8:]
            
            # Prepare labels and values
            labels = cluster_names
            sizes = player_data.values*100
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 6.5),facecolor='#fbf6ef')
            bars = ax.barh(labels, sizes, color=custom_colors,alpha=0.75)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{width:.1f}%', fontproperties=work_sans_font,va='center',fontsize=15)
            
            # Style adjustments
            ax.set_xlim(0, sizes.max()+7)
            ax.set_yticklabels(labels, fontproperties=work_sans_font,fontsize=14)
            for label in ax.get_xticklabels():
                label.set_fontproperties(work_sans_font)       
            
            ax.set_xlabel('Cluster Probability %',fontproperties=work_sans_font,fontsize=14)
    
            ax.invert_yaxis()  # Highest value on top
            ax.grid(alpha=0.4)
            plt.title(f"{player_name} - {season} - {team_name}", fontproperties=work_sans_font_sb, fontsize=16)
            plt.tight_layout()
            
            figs.append(fig)
            
            
    def find_closest_rows(df, player_id, season_id, metrics):
        # Get the row to compare
        target_row = df[(df['PlayerId'] == player_id) & (df['SeasonId'] == season_id)]
    
        if target_row.empty:
            raise ValueError("No matching row found for given PlayerId and SeasonId.")
    
        target_row = target_row.iloc[0]  # Convert to Series
        target_vec = target_row[metrics].values.astype(float)
    
        # Scale the data
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[metrics] = scaler.fit_transform(df[metrics])
    
        # Scale the target row's metrics
        target_vec_scaled = scaler.transform([target_vec])[0]
    
        def get_closest(df_subset):
            df_subset = df_subset.copy()
            df_subset = df_subset[~((df_subset['PlayerId'] == player_id) & (df_subset['SeasonId'] == season_id))]
    
            # Scale the subset data
            df_subset[metrics] = scaler.transform(df_subset[metrics])
    
            cluster_matrix = df_subset[metrics].values.astype(float)
            mse = np.mean((cluster_matrix - target_vec_scaled) ** 2, axis=1)
            df_subset['mse'] = mse
            closest_row = df_subset.loc[df_subset['mse'].idxmin()]
            
            # Calculate similarity percentage
            similarity_pct = 100 * (1 - closest_row['mse'] / np.mean(target_vec_scaled ** 2))
            return closest_row, similarity_pct
    
        # 1. Closest within SeasonId=317 and CompetitionId=2
        filtered_df = df[(df['SeasonId'] == 317) & (df['CompetitionId'] == 2)]
        closest_row_filtered, similarity_filtered = get_closest(filtered_df)
    
        # 2. Closest overall
        closest_row_all, similarity_all = get_closest(df)
    
        return {
            'closest_filtered': closest_row_filtered,
            'similarity_filtered_pct': similarity_filtered,
            'closest_overall': closest_row_all,
            'similarity_overall_pct': similarity_all
            }

    
    closest_dict = find_closest_rows(pos_5, player_id=player_ids[0], season_id=season_ids[0],metrics=metrics)
    
    closest_pl = closest_dict['closest_filtered']
    closest = closest_dict['closest_overall']
    
    closest_pl_name = closest_pl['PlayerNickName']
    closest_pl_team = closest_pl['team_name']
    
    closest_name = closest['PlayerNickName']
    closest_team = closest['team_name']
    closest_season_id = closest['SeasonId']
    
    if closest_season_id == 317:
        closest_season = '24/25'
    
    if closest_season_id == 281:
        closest_season = '23/24'
    
    if closest_season_id == 235:
        closest_season = '22/23'
    
    text = ""
    
    # Add header for most similar players
    text += "<h6 style='color:black;'>Most Similar Players:</h6>"
    
    # Closest PL 24/25 Player
    text += f"<p style='color:black;'>• Closest PL 24/25 Player – {closest_pl_name} ({closest_pl_team}) - {round(closest_dict['similarity_filtered_pct'],1)}%</p>"
    
    # Closest Overall
    text += f"<p style='color:black;'>• Closest Overall – {closest_name} - {closest_season} ({closest_team}) - {round(closest_dict['similarity_overall_pct'],1)}%</p>"
    
    text += "<br>"
    
    text_output.append(text)
    
    
                
    return figs,text_output


st.title("Player Clustering")

# Create a row with four columns: Position Group, Season, Player, Go button
select_col1, select_col2, select_col3, select_col4 = st.columns([2, 3, 4, 1])

with select_col1:
    labels = ['GK',"FB",'CB','DM/CM','AM','W','AM/W','CF']
    output_values = [1,2, 3, 4, 5, 6,7,8]
    
    # Create a dictionary to map labels to output values
    label_to_value = dict(zip(labels, output_values))
    
    # Display the selectbox with labels
    selected_label = st.selectbox("Position Group", options=labels, index=0)
    
    # Get the corresponding numeric value
    position_group = label_to_value[selected_label]
    
    if position_group == 1:
    
        gk_breakdown_labels = ['IP','OOP']
        output_values_gk = [1,2]
        
        # Create a dictionary to map labels to output values
        gk_label_to_value = dict(zip(gk_breakdown_labels, output_values_gk))
        
        # Display the selectbox with labels
        selected_label_gk = st.selectbox("Position Group", options=gk_breakdown_labels, index=0)
        
        # Get the corresponding numeric value
        gk_group = gk_label_to_value[selected_label_gk]
        
    

if position_group == 1 and gk_group == 1:
    
    
#GOOD
    metrics =  ['GK High Ball %',
                   'GK Ground Ball %',
                #  'pAdj_Progressive Passes Attempted',
                #  'Progressive Low Pass %',
                  'Quick Pass %',
                  'GK Throw %',
                  'OBV per Pass OP',
                  'GK Short Dist %'
                
               ]
    
    no_clusters = 2
    random_state = 0
    am_change = 0
    
    cluster_names = ['Direct','Ball-Playing']


if position_group == 1 and gk_group == 2:
    
    metrics = [#'GK Total Collection Distance From Goal',
                  #'GK Total Punch Distance From Goal',
                  #'GK Total Smothers Distance From Goal',
                  #'GK Total Sweeper Keeper Distance From Goal',
                  'GK Total Collections Attempted',
                  'GK Total Punches Attempted',
                  'GK Sweeper Keeper',
                  'GK Sweeper Keeper Claim',
                  'GK Sweeper Keeper Clear',
                 # 'GK Smothers',
                  'Sweeper Rate %',
                  'Claim Rate %',
                  'Punch Rate %',
                  'Distance per Collection Punch',
                  'Relative Pos Pressures AvX',
                 # 'Distance per Sweep'
            
                 ]
    
    no_clusters = 3
    random_state = 0
    am_change = 0
    
    cluster_names = ['Sweeping','Box Dominant','On-Line']


if position_group == 2:
    
    metrics = ['Progressive Low Pass %',
                  #'FWD Pass Rate',
                  'pAdj_Total Crosses',
                  'Relative Key Pass',
                 # 'Long Ball %',
                 # 'Cross Deep %',
                  'Progressive Carry %',
                  'Take Ons per Pass Received',
                  'Relative Pressures AvX',
                 # 'Half Space Received Pass %',
                  'Key Pass Cross %',
                  'pAdj_Received Passes OP',
                  'pAdj_Successful Passes OP'
                 ]

    
    no_clusters = 3
    random_state = 7
    am_change = 0
    
    cluster_names = ['Deep','Overlapping','Supporting']

    
if position_group == 3:
    
    metrics =  [#'Def Third Pressure Wide %',
                 
                  'Relative Pos Pressures AvX',
             #     'Aerial Pass Rate',
                  'Relative Pos Aerial Clearances In Box',
                  'Progressive Carry %',
                #  'Relative Pos Progressive Passes Attempted',
                  'pAdj_Progressive Passes Attempted',
                  'Progressive Low Pass %',
                #  'Ball Recovery Def Action %',
                  'Interceptions Def Action %',
                  'Tackle Def Action %',
                 # 'Block Def Action %',
                 # 'Fouls per Def Action'
                 # 'Long Ball %'

                 # 'Progressive Low Pass %',
               #   'Relative Pos Pressures AvX',
                  #'Interceptions Def Action %',
                  #'Aerial Duels',
               ]

    
    no_clusters = 4
    random_state = 0
    am_change = 0
    
    cluster_names = ['Covering','Front-Footed','No-Nonsense','Ball-Playing']


if position_group == 4:
    
    metrics = ['pAdj_Total Crosses',
                  'Take Ons per Pass Received',
                  'pAdj_Received Passes OP',
              #    'pAdj_Low Received Passes In Atk Half OP',
               #   'pAdj_Def Third Passes Attempted',
                #  '% of Atk Half Passes Received in 10 Space',
                  'Relative Pressures AvX',
               #  'Relative Attempted Passes to Def Third',
               #'pAdj_Carries Over 3 Seconds',
                 'Shots per Pass',
                 'pAdj_Ball Receipts In Box',
                  'Relative Pos Progressive Passes Attempted',
                 # 'Long Ball %',
                 'Progressive Carry %',
                #  'Key Pass Progressive Carry %',
                  'Relative Pos Def Actions',
                #  'Def Third Pressure %'
              #   'Relative Key Pass',
                 #'Relative Pos Low Def Third Received Pass %'
               #  'Relative Ball Recovery and Interceptions',
              #  'Low Pass Received Box Def Third Ratio'
                 ]
    
    no_clusters  = 4
    random_state = 14
    am_change = 0
    
    cluster_names = ['Deep Playmaker','Anchor','Advanced Creator','Box-to-Box']



if position_group == 5:
    
        
    metrics = [ #'Carry Box Entry %',
                 # '% of Atk Half Passes Received in Half Space',
                  'pAdj_Ball Receipts In Box',
                  'Ball Receipt Box Entry %',
                  'Relative Key Pass',
                  'xG per Shot',
                  #'Quick Pass %',
                  'Shots From Crosses %']

    
    no_clusters = 2
    random_state = 3
    am_change = 0
    
    cluster_names = ['Box Crasher','Central Creator']



if position_group == 6:
    metrics = ['pAdj_Total Crosses',
                 'Cross Deep %',
                 'Take Ons per Pass Received',
                 '% of Atk Half Passes Received in Half Space',
                 'pAdj_Ball Receipts In Box',
                 'pAdj_Through Ball Received In Final Third OP',
                 'Carries Into Final Third From Half Space vs Channel',
                 'Low Received Passes In Atk Half Space vs Channel',
                 '% of Atk Half Passes Received in 10 Space',
                 ]
    
    no_clusters = 3
    random_state = 3
    am_change = 0
    
    cluster_names = ['Direct Winger','Inverted Winger','Inside Forward']

        

if position_group == 7:
    
        
    metrics = ['pAdj_Total Crosses',
                  'Cross Deep %',
                  'Take Ons per Pass Received',
                  '% of Atk Half Passes Received in Half Space',
                  'pAdj_Ball Receipts In Box',
                  'pAdj_Through Ball Received In Final Third OP',
                  'Carries Into Final Third From Half Space vs Channel',
                  'Low Received Passes In Atk Half Space vs Channel',
                  '% of Atk Half Passes Received in 10 Space',]
        
    
    cluster_names = ['Inside Forward','Inverted Winger','Direct Winger','Box Crasher','Central Creator']
    
    no_clusters = 5
    random_state = 19
    
    position_group = 5
    am_change = 1
    
    
if position_group == 8:
   
    
      
    
    metrics =[ 'pAdj_Ball Receipts In Box',
                  'pAdj_Through Ball Received In Final Third OP',
                  'Shots From Crosses %',
                  'Aerial Duels',
                  'pAdj_Total Crosses',
                  'pAdj_Successful Passes OP'   ,
                 # 'Progressive Carry %',
                  'Atk Half Channel Low Received Pass %',
                 # 'Ball Received in Box %',
                  #'Headed Shot %',
                  'Carry Box Entry %',
                  'Pass Box Entry %',
                  'Quick Pass %',
                  'Ball Receipt Box Entry %'
                   ]
    
    cluster_names = ['Wide Forward','Target Man','False 9','Poacher']

    
    
    no_clusters = 4
    random_state = 2
    am_change = 0
    
    position_group = 7
    
    
    
players_available = pd.read_csv('data/Grouped_Player_Final.csv', index_col=0)





with select_col2:
    
    season_labels = ['24/25', '23/24', '22/23']
    season_values = [317, 281, 235]  # Corresponding SeasonId values
    
    # Create a dictionary to map labels to season IDs
    label_to_season_id = dict(zip(season_labels, season_values))
    
    # Display the selectbox with the season labels
    selected_label = st.selectbox("Season", season_labels)
    
    # Get the corresponding season ID
    selected_season = label_to_season_id[selected_label]

with select_col3:
    
    if position_group == 5 and am_change == 1:
        players_in_season = players_available[
            (players_available['SeasonId'] == selected_season) &
            (players_available['GroupID'].isin([5,6]))
        ]
        
        
    else:
        players_in_season = players_available[
            (players_available['SeasonId'] == selected_season) &
            (players_available['GroupID']==position_group)
        ]
        

        
    players_in_season = players_in_season.sort_values('PlayerNickName')
    player_options = players_in_season[['PlayerNickName', 'PlayerId']].drop_duplicates()
    player_choices = [(row['PlayerNickName'], row['PlayerId']) for _, row in player_options.iterrows()]
    
    selected_player = st.selectbox("Player", options=player_choices, format_func=lambda x: x[0])
    selected_player_id = selected_player[1]

with select_col4:
    go_clicked = st.button("Go")

# Only run if "Go" is clicked
if go_clicked:
    st.title("PCA Plot")

    player_ids = [selected_player_id] if selected_player_id is not None else []
    season_ids = [selected_season] if selected_season is not None else []

    col1, col2 = st.columns([3, 1])  # Display columns for plot/text

    figs, text = plot_pca(position_group, metrics, no_clusters, random_state, cluster_names,
                          legend_loc='upper right', player_ids=player_ids, season_ids=season_ids,
                          am_change = am_change)

    with col1:
        st.pyplot(figs[0])

    with col2:
        st.markdown(f'<div style="font-size: 15px;">{text[0]}</div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)  # Creates an empty line gap between columns

    st.title("Player Breakdown")

    col1, col2 = st.columns([3, 1]) 
    
    with col1:
        st.pyplot(figs[1])

    with col2:
        st.markdown(f'<div style="font-size: 15px;">{text[1]}</div>', unsafe_allow_html=True)




    