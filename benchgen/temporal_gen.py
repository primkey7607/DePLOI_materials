from utils.temporal_utils import MyInterval, StartTime, unit_lst
from utils.df_utils import modify_dataframe, generic_modify
from rolehier_gen import ceo_pair
import random
import pandas as pd
import os
import random
from ast import literal_eval
import re
from datetime import datetime

# unit_lst = ['hour', 'day', 'week', 'month', 'year']

random.seed(42)

weekday_dct = {0 : 'Sunday',
               1 : 'Monday',
               2 : 'Tuesday',
               3 : 'Wednesday',
               4 : 'Thursday',
               5 : 'Friday',
               6 : 'Saturday'}

def longest_consecutive_subsequence(nums):
    """
    Finds the longest consecutive subsequence in a list of integers.

    Args:
        nums (list): List of integers.

    Returns:
        list: The longest consecutive subsequence.
    """
    if not nums:
        return []
    
    # Create a set of numbers for O(1) lookups
    num_set = set(nums)
    longest_streak = 0
    best_start = None

    for num in num_set:
        # Only start if it's the beginning of a sequence
        if num - 1 not in num_set:
            current_num = num
            current_streak = 1

            while current_num + 1 in num_set:
                current_num += 1
                current_streak += 1

            if current_streak > longest_streak:
                longest_streak = current_streak
                best_start = num

    # Build the longest consecutive subsequence
    return [best_start + i for i in range(longest_streak)] if best_start is not None else []

def test_lcs():
    nums = [100, 4, 200, 1, 3, 2]
    print(longest_consecutive_subsequence(nums))

def simple_nl(raw_st_time, raw_end_time, days_of_week):
    #adjust times as necessary
    if raw_st_time == 0:
        st_time = 12
    else:
        st_time = raw_st_time
    
    if raw_end_time == 0:
        end_time = 12
    else:
        end_time = raw_end_time
    
    #find the longest consecutive subsequence
    lcs = longest_consecutive_subsequence(days_of_week)
    if len(lcs) > 2:
        day_lst = [weekday_dct[lcs[0]] + ' through ' + weekday_dct[lcs[-1]]] + [weekday_dct[d] for d in days_of_week if d not in lcs]
    else:
        day_lst = [weekday_dct[d] for d in days_of_week]
    
    day_st = ', '.join(day_lst[:-1]) + ', and ' +  day_lst[-1]
    
    #Example Sentence: "This role can access this view from 9am to 5pm on Monday through Wednesday and Friday"
    
    full_nl = 'This role can access this view from ' + str(st_time) + 'am to ' + str(end_time) + 'pm on the days of the week: ' + day_st
    return full_nl

def is_ceo(row, pertfile='amazon_roleperts.json'):
    role_st = row['Role']
    base_desc = ceo_pair[1]
    with open(pertfile, 'r') as fh:
        dct = literal_eval(fh.read())
    
    pert_dct = dct[base_desc]
    all_descs = [base_desc] + [pert_dct[p] for p in pert_dct]
    
    if role_st in all_descs:
        return True
    
    return False

def unit_range(unit_name):
    if unit_name == 'hour':
        return range(1, 24)
    elif unit_name == 'day':
        return range(1, 8)
    elif unit_name == 'week':
        return range(1, 5)
    elif unit_name == 'month':
        return range(1, 13)
    elif unit_name == 'year':
        return range(2021, 2024)
    
    raise Exception("Unit Type Not Valid: {}, {}".format(unit_name, unit_lst))

def simple_nl2sql(nl_string: str) -> str:
    """
    Converts a natural language temporal constraint into a PostgreSQL SQL function.
    
    Args:
        nl_string (str): Natural language constraint string.
        
    Returns:
        str: PostgreSQL SQL function as a string.
    """
    # Extract hours and AM/PM
    time_match = re.search(r'from (\d{1,2})(am|pm) to (\d{1,2})(am|pm)', nl_string)
    if not time_match:
        raise ValueError("Invalid time format in input string.")
    
    hour1, ampm1, hour2, ampm2 = time_match.groups()
    
    def to_military(hour, period):
        hour = int(hour)
        if period == 'am':
            return hour if hour != 12 else 0
        else:
            return hour + 12 if hour != 12 else 12
    
    hour1_military = to_military(hour1, ampm1)
    hour2_military = to_military(hour2, ampm2)
    
    # Extract days
    days_match = re.search(r'on (.+)', nl_string)
    if not days_match:
        raise ValueError("Invalid days format in input string.")
    
    days_part = days_match.group(1)
    days = []
    
    # Handle day ranges
    day_mapping = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    day_ranges = re.findall(r'(\w+) through (\w+)', days_part)
    for start, end in day_ranges:
        if start in day_mapping and end in day_mapping:
            start_idx = day_mapping.index(start)
            end_idx = day_mapping.index(end)
            days.extend(day_mapping[start_idx:end_idx + 1])
    
    # Handle individual days
    individual_days = re.findall(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', days_part)
    days.extend(individual_days)
    
    # Remove duplicates and format
    days = sorted(set(days), key=lambda d: day_mapping.index(d))
    days_sql = ', '.join(f"'{day}'" for day in days)
    
    # Generate SQL function
    sql_function = f"""
    CREATE OR REPLACE FUNCTION is_access_allowed() RETURNS BOOLEAN AS $$
    DECLARE
        current_time TIME;
        current_day TEXT;
    BEGIN
        -- Get current time and day
        current_time := NOW();
        current_day := to_char(CURRENT_DATE, 'Day');

        -- Check if current time is between {hour1_military} and {hour2_military} and it's on specified days
        IF EXTRACT('Hour' FROM current_time) >= {hour1_military} AND EXTRACT('Hour' FROM current_time) < {hour2_military} AND current_day IN ({days_sql}) THEN
            RETURN true;
        ELSE
            RETURN false;
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    """
    return sql_function.strip()

def test_simple_nl2sql():
    # Example Usage
    nl_input = "This role can access this view from 9am to 5pm on Monday through Wednesday and Friday"
    print(simple_nl2sql(nl_input))

def simple_nlacm2temp(nlacm_path, sqlacm_path, outdir, outname, sqldir, sqlname):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(sqldir):
        os.mkdir(sqldir)
    
    nlacmdf = pd.read_csv(nlacm_path)
    sqlacmdf = pd.read_csv(sqlacm_path)
    
    
    new_nlacm = modify_dataframe(nlacmdf, 12, 12, 7, 3, simple_nl, 'Role')
    new_sqlacm = generic_modify(new_nlacm, simple_nl2sql, 'Role')
    
    new_sqlacm['Role'] = sqlacmdf['Role']
    change_cols = {new_nlacm.columns[i] : sqlacmdf.columns[i] for i in range(new_nlacm.shape[1])}
    new_sqlacm = new_sqlacm.rename(columns=change_cols)
    
    outpath = os.path.join(outdir, outname)
    sqlpath = os.path.join(sqldir, sqlname)
    
    new_nlacm.to_csv(outpath, index=False)
    new_sqlacm.to_csv(sqlpath, index=False)

def add_extra_hour(sql_string: str) -> str:
    """
    Adds an extra hour to the SQL time range condition.
    Preference is given to adjusting the lower bound (>= x to >= x - 1).
    If not possible, adjusts the upper bound (< y to < y + 1).
    """
    # Match EXTRACT('Hour' FROM current_time) >= x
    lower_bound_match = re.search(r"EXTRACT\('Hour' FROM current_time\) >= (\d+)", sql_string)
    upper_bound_match = re.search(r"EXTRACT\('Hour' FROM current_time\) < (\d+)", sql_string)
    
    if lower_bound_match:
        lower_bound = int(lower_bound_match.group(1))
        if lower_bound > 0:  # Ensure we don't go below 0
            return re.sub(
                r"EXTRACT\('Hour' FROM current_time\) >= \d+",
                f"EXTRACT('Hour' FROM current_time) >= {lower_bound - 1}",
                sql_string,
                count=1
            )
    
    if upper_bound_match:
        upper_bound = int(upper_bound_match.group(1))
        if upper_bound < 24:  # Ensure we don't exceed 24 hours
            return re.sub(
                r"EXTRACT\('Hour' FROM current_time\) < \d+",
                f"EXTRACT('Hour' FROM current_time) < {upper_bound + 1}",
                sql_string,
                count=1
            )
    
    # If no valid modification was possible, return the original string
    return sql_string

def add_extra_day(sql_string: str, seed: int = 42) -> str:
    """
    Adds an extra day to the SQL day condition if possible.
    Randomly selects a day not currently in the condition.
    """
    all_days = {'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'}
    
    # Find the current day condition
    day_match = re.search(r"current_day IN \(\s*'([\w\s]+)'(?:,\s*'([\w\s]+)')*\s*\)", sql_string)
    if day_match:
        current_days = set(re.findall(r"'(\w+)'", day_match.group(0)))
        available_days = all_days - current_days
        
        if available_days:
            # random.seed(seed)
            new_day = random.choice(list(available_days))
            updated_days = "', '".join(sorted(current_days | {new_day}))
            return re.sub(
                r"current_day IN \(\s*'([\w\s]+)'(?:,\s*'([\w\s]+)')*\s*\)",
                f"current_day IN ('{updated_days}')",
                sql_string
            )
    
    # If no modification was possible (e.g., all days are already included)
    return sql_string

def test_pert_funcs():
    sql_string = """
    CREATE OR REPLACE FUNCTION is_access_allowed() RETURNS BOOLEAN AS $$
        DECLARE
            current_time TIME;
            current_day TEXT;
        BEGIN
            current_time := NOW();
            current_day := to_char(CURRENT_DATE, 'Day');
            IF EXTRACT('Hour' FROM current_time) >= 6 AND EXTRACT('Hour' FROM current_time) < 15 
               AND current_day IN ('Sunday', 'Wednesday', 'Thursday') THEN
                RETURN true;
            ELSE
                RETURN false;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
    """
    
    # Perturbation 1
    print(add_extra_hour(sql_string))
    
    # Perturbation 2
    print(add_extra_day(sql_string))

def perturb_simple_sql(sql_st):
    
    coin_flip = random.choice([0,1])
    if coin_flip == 0:
        return sql_st
    
    new_sql_st = add_extra_hour(sql_st)
    if new_sql_st == sql_st:
        new_sql_st = add_extra_day(sql_st)
    
    #at this point, new_sql_st could be the same,
    #but we will not handle this case.
    if new_sql_st == sql_st:
        print("WARNING: could not add extra hour or day to SQL: " + sql_st)
    return new_sql_st

def simple_perturb(sqlpath, outdir, outname):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    sqldf = pd.read_csv(sqlpath)
    
    pert_df = generic_modify(sqldf, perturb_simple_sql, 'Role')
    
    outpath = os.path.join(outdir, outname)
    pert_df.to_csv(outpath, index=False)
    
    


def nlacm2temp(nlacm_path, outdir, outname, sqldir, sqlname, sqldf):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.exists(sqldir):
        os.mkdir(sqldir)
        
    nlacmdf = pd.read_csv(nlacm_path)
    new_dct = {}
    new_dct['Role'] = nlacmdf['Role'].tolist()
    for c in nlacmdf.columns.tolist():
        if c == 'Role':
            continue
        new_dct[c] = []
    
    
    sql_dct = {}
    sql_dct['Role'] = sqldf['Role'].tolist()
    for c in sqldf.columns.tolist():
        if c == 'Role':
            continue
        sql_dct[c] = []
    
    random.seed(10)
    
    for i,row in enumerate(nlacmdf.to_dict(orient='records')):
        if is_ceo(row):
            for j,v in enumerate(row):
                if v == 'Role':
                    continue
                new_dct[v].append('This role can access this view at all times.')
                sql_dct[sqldf.columns.tolist()[j]].append('--None.')
        else:
            for j,v in enumerate(row):
                if v == 'Role':
                    continue
                st_unit = random.choice(unit_lst[1:]) #we will not support the duration and start times having the same unit
                dur_unit = 'hour' if st_unit == 'day' else unit_lst[unit_lst.index(st_unit) - 1]
                val_range = unit_range(st_unit)
                if st_unit == 'year':
                    pts = [random.choice(val_range)]
                else:
                    pts = random.sample(val_range, k=3)
                dur_length = random.choice(range(1, 12))
                st_times = StartTime(pts, st_unit)
                dur_tup = (dur_length, dur_unit)
                intval = MyInterval(st_times, dur_tup)
                intval_st = intval.to_string()
                # intval_sql = intval.to_sql(sqldf.loc[i]['Role'], sqldf.columns[j], i, j)
                intval_sql = intval.to_sql(i, j)
                intval_st = 'This role can access this view ' + intval_st
                new_dct[v].append(intval_st)
                sql_dct[sqldf.columns.tolist()[j]].append(intval_sql)
    #for debugging
    with open('test_temp_dct.json', 'w+') as fh:
        print(sql_dct, file=fh)
    
    new_df = pd.DataFrame(new_dct)
    out_sql_df = pd.DataFrame(sql_dct)
    outpath = os.path.join(outdir, outname)
    sqlpath = os.path.join(sqldir, sqlname)
    new_df.to_csv(outpath, index=False)
    out_sql_df.to_csv(sqlpath, index=False)

def generate_full_base():
    testdir = 'amazon_bird_nlacm_deep_nl'
    testfile = 'amazon_bird_nlacm_deep_nl_base.csv'
    test_sql = 'amazon_bird_nlacm_deep_sql_base.csv'
    test_sql_df = pd.read_csv(test_sql)
    
    testpath = os.path.join(testdir, testfile)
    outdir = testdir.replace('nlacm', 'time')
    outfile = testfile.replace('nlacm', 'time')
    
    sqldir = outdir.replace('_nl', '_sql')
    sqlfile = outfile.replace('nl', 'sql')
    
    nlacm2temp(testpath, outdir, outfile, sqldir, sqlfile, test_sql_df)

def generate_simple():
    #generate a simpler version over the same roles and views, for smaller-scale easier evaluation
    orig_nlpath = os.path.join('bird_nlacm_european_football_2_nl', 'bird_nlacm_nl_chunk_11.csv')
    orig_sqlpath = os.path.join('bird_nlacm_european_football_2_sql', 'bird_nlacm_sql_chunk_11.csv')
    simple_nldir = 'bird_nlacm_european_football_2_nl'.replace('nlacm', 'simpletime')
    simple_nlname = 'bird_nlacm_nl_chunk_11.csv'.replace('nlacm', 'simpletime')
    simple_sqldir = 'bird_nlacm_european_football_2_sql'.replace('nlacm', 'simpletime')
    simple_sqlname = 'bird_nlacm_sql_chunk_11.csv'.replace('nlacm', 'simpletime')
    
    
    simple_nlacm2temp(orig_nlpath, orig_sqlpath, simple_nldir, simple_nlname, simple_sqldir, simple_sqlname)

def pert_simple():
    orig_sqlpath = os.path.join('bird_simpletime_european_football_2_sql', 'bird_simpletime_sql_chunk_11.csv')
    pert_sqldir = 'bird_simpletime_european_football_2_badsql'
    pert_sqlname = 'bird_simpletime_badsql_chunk_11.csv'
    
    simple_perturb(orig_sqlpath, pert_sqldir, pert_sqlname)

if __name__=='__main__':
    # generate_full_base()
    # generate_simple()
    # test_pert_funcs()
    pert_simple()


