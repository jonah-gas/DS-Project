import torch 
import itertools
import numpy as np
import pandas as pd
import itertools
import os
import sys
root_path = os.path.abspath(os.path.join('../..'))
if root_path not in sys.path:
    sys.path.append(root_path)
import database_server.db_utilities as dbu 
import database_server.db_utilities as dbu 
import pickle as pkl
from Help_functions import preprocess, game_dict, inputs, club_dict, points_and_co, points_and_co_oppon, data_to_lstm, Sport_pred_2LSTM_1, predict, two_team_inputs


query_str = """
    SELECT ms.*, 
        m.schedule_date, m.schedule_time, m.schedule_round, m.schedule_day,
        w.annual_wages_eur AS annual_wage_team, 
        w.weekly_wages_eur AS weekly_wages_eur,
        w.annual_wages_eur/w.n_players AS annual_wage_player_avg
    FROM matchstats ms 
    LEFT JOIN matches m ON ms.match_id = m.id
    LEFT JOIN teamwages w ON ms.team_id = w.team_id
    AND ms.season_str = w.season_str
    ORDER BY m.schedule_date DESC, m.schedule_time DESC; 
    """
df_allinfo = dbu.select_query(query_str)
new_data_test = preprocess(df_allinfo)
scale_df = new_data_test.data_frame
clubs = club_dict(scale_df)
result_dict = new_data_test.return_dicts("result")
clubs = points_and_co(clubs, result_dict)
clubs = points_and_co_oppon(clubs, result_dict)

abcdefg = list(scale_df.columns)
abc = abcdefg[:abcdefg.index("annual_wage_player_avg")+1]
defg = abcdefg[abcdefg.index("annual_wage_player_avg")+1:]

rearange_list = ['result', 'gf', 'ga', 'goal_diff', 'xg', 'xga', 'shooting_standard_gls',
       'shooting_standard_sh', 'shooting_standard_sot',
       'shooting_standard_sot_perc', 'shooting_standard_g_per_sh',
       'shooting_standard_g_per_sot', 'shooting_standard_dist',
       'shooting_standard_fk', 'shooting_standard_pk',
       'shooting_standard_pkatt', 'shooting_expected_npxg',
       'shooting_expected_npxg_per_sh', 'shooting_expected_g_minus_xg',
       'shooting_expected_npg_minus_xg', 'keeper_performance_sota',
       'keeper_performance_saves', 'keeper_performance_save_perc',
       'keeper_performance_cs', 'keeper_performance_psxg',
       'keeper_performance_psxg_plus_minus', 'keeper_penaltykicks_pkatt',
       'keeper_penaltykicks_pka', 'keeper_penaltykicks_pksv',
       'keeper_penaltykicks_pkm', 'keeper_launched_cmp', 'keeper_launched_att',
       'keeper_launched_cmp_perc', 'keeper_passes_att', 'keeper_passes_thr',
       'keeper_passes_launch_perc', 'keeper_passes_avglen',
       'keeper_goalkicks_att', 'keeper_goalkicks_launch_perc',
       'keeper_goalkicks_avglen', 'keeper_crosses_opp', 'keeper_crosses_stp',
       'keeper_crosses_stp_perc', 'keeper_sweeper_number_opa',
       'keeper_sweeper_avgdist', 'passing_total_cmp', 'passing_total_att',
       'passing_total_cmp_perc', 'passing_total_totdist',
       'passing_total_prgdist', 'passing_short_cmp', 'passing_short_att',
       'passing_short_cmp_perc', 'passing_medium_cmp', 'passing_medium_att',
       'passing_medium_cmp_perc', 'passing_long_cmp', 'passing_long_att',
       'passing_long_cmp_perc', 'passing_attacking_ast',
       'passing_attacking_xag', 'passing_attacking_xa', 'passing_attacking_kp',
       'passing_attacking_1_per_3', 'passing_attacking_ppa',
       'passing_attacking_crspa', 'passing_attacking_prgp',
       'passing_types_passtypes_live', 'passing_types_passtypes_dead',
       'passing_types_passtypes_fk', 'passing_types_passtypes_tb',
       'misc_aerialduels_won_perc','attendance', 'points', 'mean_points',
       'weekly_wages_eur', 'season_str',  'league_id', 'venue', 'team_id',
       'opponent_id', 'last_results', 'oppon_points', 'oppon_mean_points', 'schedule_round',
        'captain', 'formation', 'referee',  'match_id', 'schedule_date', 'schedule_time',
        'schedule_day', 'annual_wage_team', 'annual_wage_player_avg',]

rearange_list = list(itertools.chain.from_iterable(defg if item == "team_id" else [item] for item in rearange_list))
del rearange_list[rearange_list.index("opponent_id")]
    

def sequence_models(team1, team2, clubs, rearrange_list, scale_df, result_dict, num_seas):
        
    input_to_lstm = two_team_inputs(team1, team2, rearrange_list, scale_df, clubs)
    
    width_input = input_to_lstm[0][0].shape[1]
    model = Sport_pred_2LSTM_1(width_input, width_input, 3, 2)
    model.state_dict(torch.load("./models/sequence_model_2seas/LSTM/2e-05/accur_49.45"))
    model.eval()
    
    prediction1 = model(input_to_lstm[0][0])[-1,:] 
    prediction2 = model(input_to_lstm[1][0])[-1,:]
    prediction1 = torch.index_select(prediction1, 0, torch.LongTensor([result_dict["W"], result_dict["L"], result_dict["D"]]))
    prediction2 = torch.index_select(prediction2, 0, torch.LongTensor([result_dict["L"], result_dict["W"], result_dict["D"]]))
    
    #expect_result = predict(prediction1, prediction2, result_dict)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax((prediction1 + prediction2), dim = 0)
        
    pd_to_return = pd.DataFrame({"home_win_prob": float(prediction[0]),
                                "draw_prob": float(prediction[2]),
                                "away_win_prob": float(prediction[1])}, index = [0])
    return pd_to_return































