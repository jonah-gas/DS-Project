-- The following commands should be executed as postgres superuser.


-- Create the database
CREATE DATABASE soccerdb;

-- connect to new database
\c soccerdb



-- Tables -----------------------------------------------------------------------------------------------------------------

-- Table overview: 
    -- matches -> contains match_ids and base data (but not statistics)
    -- match_stats -> contains statistics for each match, one row per (match_id, team_id), i.e. two rows per match (one for each team)
    -- teams -> contains team_ids and team names
    -- (players -> contains player_ids and player names)
    -- (player_stats -> contains statistics for each player, one row per (player_id, match_id))
    -- wages -> contains wages per team and season
    -- leagues -> contains league_ids and league names
    -- countries -> country codes and names

-- Table COUNTRIES
CREATE TABLE countries (
    code char(3) PRIMARY KEY,
    name text UNIQUE NOT NULL
);


-- Table LEAGUES
CREATE TABLE leagues (
    id SERIAL PRIMARY KEY,
    fbref_id text UNIQUE NOT NULL,
    name text UNIQUE NOT NULL,
    country char(3) REFERENCES countries (code)
);

-- Table TEAMS
CREATE TABLE teams (
    id SERIAL PRIMARY KEY,
    fbref_id text UNIQUE,
    name text UNIQUE NOT NULL,
    country char(3) REFERENCES countries (code)
);

-- Table MATCHES

-- primary key: match_id -> auto gen 
-- constraints: 
    -- other ids (fbref, etc.) must also be unique
    -- no null values
-- foreign keys: league_id, home_team_id, away_team_id

-- define weekday type for 'schedule_day' column
CREATE TYPE weekday AS ENUM ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun');

CREATE TABLE matches (
    id SERIAL PRIMARY KEY,
    fbref_id text UNIQUE NOT NULL,
    league_id integer REFERENCES leagues (id) NOT NULL,
    home_team_id integer REFERENCES teams (id) NOT NULL,
    away_team_id integer REFERENCES teams (id) NOT NULL,
    -- also include a few of the base data columns which don't really belong in matchstats
    schedule_date date,
    schedule_time time,
    schedule_round text,
    schedule_day weekday
);


-- Table TEAMWAGES

-- squad_name;n_players;pct_estimated

CREATE TABLE teamwages (
    team_id integer REFERENCES teams (id) NOT NULL,
    season_str text NOT NULL,
    n_players integer,
    pct_estimated float(4),
    weekly_wages_eur bigint,
    weekly_wages_gbp bigint,
    weekly_wages_usd bigint,
    annual_wages_eur bigint,
    annual_wages_gbp bigint,
    annual_wages_usd bigint
);


-- Table MATCHSTATS

-- there will be two rows for every match, one for each team's statistics
-- primary key: (match_id, team_id) 
-- foreign keys: match_id, team_id, opponent_id, league_id

-- special types
CREATE TYPE matchresult AS ENUM ('L', 'D', 'W'); -- ordered type might be useful

CREATE DOMAIN venuetype AS TEXT 
    CHECK (VALUE IN ('Home', 'Away', 'Neutral'));

CREATE TABLE matchstats (
    venue text,
    result matchresult,
    gf int,
    ga int,
    xg float,
    xga float,
    attendance int,
    captain text,
    formation text,
    referee text,
    season_str text,
    league_id int REFERENCES leagues (id) ,
    team_id int REFERENCES teams (id) NOT NULL,
    opponent_id int REFERENCES teams (id) NOT NULL,
    match_id int REFERENCES matches (id) NOT NULL,
    shooting_standard_gls int,
    shooting_standard_sh int,
    shooting_standard_sot int,
    shooting_standard_sot_perc float,
    shooting_standard_g_per_sh float,
    shooting_standard_g_per_sot float,
    shooting_standard_dist float,
    shooting_standard_fk int,
    shooting_standard_pk int,
    shooting_standard_pkatt int,
    shooting_expected_npxg float,
    shooting_expected_npxg_per_sh float,
    shooting_expected_g_minus_xg float,
    shooting_expected_npg_minus_xg float,
    keeper_performance_sota int,
    keeper_performance_saves int,
    keeper_performance_save_perc float,
    keeper_performance_cs int,
    keeper_performance_psxg float,
    keeper_performance_psxg_plus_minus float,
    keeper_penaltykicks_pkatt int,
    keeper_penaltykicks_pka int,
    keeper_penaltykicks_pksv int,
    keeper_penaltykicks_pkm int,
    keeper_launched_cmp int,
    keeper_launched_att int,
    keeper_launched_cmp_perc float,
    keeper_passes_att int,
    keeper_passes_thr int,
    keeper_passes_launch_perc float,
    keeper_passes_avglen float,
    keeper_goalkicks_att int,
    keeper_goalkicks_launch_perc float,
    keeper_goalkicks_avglen float,
    keeper_crosses_opp int,
    keeper_crosses_stp int,
    keeper_crosses_stp_perc float,
    keeper_sweeper_number_opa int,
    keeper_sweeper_avgdist float,
    passing_total_cmp int,
    passing_total_att int,
    passing_total_cmp_perc float,
    passing_total_totdist float,
    passing_total_prgdist float,
    passing_short_cmp int,
    passing_short_att int,
    passing_short_cmp_perc float,
    passing_medium_cmp int,
    passing_medium_att int,
    passing_medium_cmp_perc float,
    passing_long_cmp int,
    passing_long_att int,
    passing_long_cmp_perc float,
    passing_attacking_ast int,
    passing_attacking_xag float,
    passing_attacking_xa float,
    passing_attacking_kp int,
    passing_attacking_1_per_3 int,
    passing_attacking_ppa int,
    passing_attacking_crspa int,
    passing_attacking_prgp int,
    passing_types_passtypes_live int,
    passing_types_passtypes_dead int,
    passing_types_passtypes_fk int,
    passing_types_passtypes_tb int,
    passing_types_passtypes_sw int,
    passing_types_passtypes_crs int,
    passing_types_passtypes_ti int,
    passing_types_passtypes_ck int,
    passing_types_cornerkicks_in int,
    passing_types_cornerkicks_out int,
    passing_types_cornerkicks_str int,
    passing_types_outcomes_off int,
    passing_types_outcomes_blocks int,
    gca_scatypes_sca int,
    gca_scatypes_passlive int,
    gca_scatypes_passdead int,
    gca_scatypes_to int,
    gca_scatypes_sh int,
    gca_scatypes_fld int,
    gca_scatypes_def int,
    gca_gcatypes_gca int,
    gca_gcatypes_passlive int,
    gca_gcatypes_passdead int,
    gca_gcatypes_to int,
    gca_gcatypes_sh int,
    gca_gcatypes_fld int,
    gca_gcatypes_def int,
    defense_tackles_tkl int,
    defense_tackles_tklw int,
    defense_tackles_def3rd int,
    defense_tackles_mid3rd int,
    defense_tackles_att3rd int,
    defense_challenges_tkl int,
    defense_challenges_att int,
    defense_challenges_tkl_perc float,
    defense_challenges_lost int,
    defense_blocks_blocks int,
    defense_blocks_sh int,
    defense_blocks_pass int,
    defense_general_int int,
    defense_general_tkl_plus_int int,
    defense_general_clr int,
    defense_general_err int,
    possession_general_poss float,
    possession_touches_touches int,
    possession_touches_defpen int,
    possession_touches_def3rd int,
    possession_touches_mid3rd int,
    possession_touches_att3rd int,
    possession_touches_attpen int,
    possession_touches_live int,
    possession_takeons_att int,
    possession_takeons_succ int,
    possession_takeons_succ_perc float,
    possession_takeons_tkld int,
    possession_takeons_tkld_perc float,
    possession_carries_carries int,
    possession_carries_totdist float,
    possession_carries_prgdist float,
    possession_carries_prgc int,
    possession_carries_1_per_3 int,
    possession_carries_cpa int,
    possession_carries_mis int,
    possession_carries_dis int,
    possession_receiving_rec int,
    possession_receiving_prgr int,
    misc_performance_crdy int,
    misc_performance_crdr int,
    misc_performance_2crdy int,
    misc_performance_fls int,
    misc_performance_fld int,
    misc_performance_off int,
    misc_performance_og int,
    misc_performance_recov int,
    misc_aerialduels_won int,
    misc_aerialduels_lost int,
    misc_aerialduels_won_perc float,
    PRIMARY KEY (match_id, team_id) -- composite primary key
);





-- Users & Access -----------------------------------------------------------------------------------------------------------------
-- Create user for the database
CREATE USER project_client WITH ENCRYPTED PASSWORD 'XXXXXX'; -- replace with actual password

-- grant privileges to new user
GRANT ALL ON ALL TABLES IN SCHEMA public TO project_client; -- note: access is only granted for existing tables
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO project_client;





