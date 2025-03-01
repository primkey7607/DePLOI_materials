import psycopg2
import sqlite3
import os
import re
from sqlalchemy import create_engine, text
import pandas as pd
from abc import ABC, abstractmethod
import traceback
import sys
from gen_sociomeperts import sociome_schema, read_sociome_file

class DBAPI(ABC):
    @abstractmethod
    def __init__(self, con_details : dict):
        #we store the con_details because if an error happens, we may have to
        #reconnect, and storing them makes it more convenient.
        self.con_details = con_details
        self.con = self.connect()
    
    @abstractmethod
    def connect(self):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def query(self, query_st : str):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def execute(self, stmt_st : str):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def tbl_to_df(self, tbl_name : str):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def df_to_tbl(self, df_path : str, tbl_name : str):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def memdf_to_tbl(self, df, tbl_name : str):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def get_schema(self):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def get_schema_wvals(self):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def get_privs(self):
        raise Exception("Must be implemented")

class PostgresAPI(DBAPI):
    
    def __init__(self, con_details : dict):
        self.con_details = con_details
        self.con = self.connect()
    
    def connect(self):
        con = psycopg2.connect(user=self.con_details['user'], 
                                    password=self.con_details['password'], 
                                    host=self.con_details['host'], 
                                    port=self.con_details['port'], 
                                    database=self.con_details['database'])
        return con
    
    def query(self, query_st):
        cur = self.con.cursor()
        try:
            cur.execute(query_st)
            out_tups = cur.fetchall()
            return out_tups
        except:
            tb = traceback.format_exc()
            print(tb)
            #don't assume autocommit--reinitialize the connection
            cur.close()
            self.con.close()
            self.con = self.connect()
            return tb
    
    #in some cases, we will want to return query results as a table
    def query_as_df(self, query_st):
        cur = self.con.cursor()
        try:
            cur.execute(query_st)
            out_tups = cur.fetchall()
            out_df = pd.DataFrame(out_tups, columns=[desc[0] for desc in cur.description])
            return out_df
        except:
            tb = traceback.format_exc()
            print(tb)
            #don't assume autocommit--reinitialize the connection
            cur.close()
            self.con.close()
            #in this case, we absolutely want to return a dataframe,
            #so if an error occurred, we need to stop everything and figure it out
            raise Exception("Error Occurred: {}".format(tb))
            # self.con = self.connect()
            
    
    def execute(self, stmt_st):
        cur = self.con.cursor()
        try:
            cur.execute(stmt_st)
            self.con.commit()
            cur.close()
            self.con.close()
            self.con = self.connect()
            return ''
        except:
            tb = traceback.format_exc()
            print(tb)
            #don't assume autocommit--reinitialize the connection
            cur.close()
            self.con.close()
            self.con = self.connect()
            return tb
    
    def teardown(self):
        # cur = self.con.cursor()
        # get_myroles = "select rolname from pg_authid where rolname != 'postgres' and not rolname ilike 'pg' || '%';"
        # cur.execute(get_myroles)
        # cur_rolelst = [tup[0] for tup in cur.fetchall()]
        # drop_st = ''
        # for role in cur_rolelst:
        #     drop_st +=  'drop owned by ' + role + '; '
        #     drop_st += 'drop role ' + role + '; '
        
        # if drop_st != '':
        #     cur.execute(drop_st)
        #     self.con.commit()
        
        #also drop views
        view_drop = '''SELECT 'DROP VIEW ' || table_name || ';' FROM information_schema.views WHERE table_schema NOT IN ('pg_catalog', 'information_schema') AND table_name !~ '^pg_';'''
        drop_stmts = self.query(view_drop)
        for ds in drop_stmts:
            self.execute(ds[0])
        # cur.close()
        # self.con.close()
    
    def tbl_to_df(self, tbl_name, nrows=None):
        eng_st = 'postgresql://postgres:' + self.con_details['password'] + '@localhost:' + self.con_details['port'] + '/' + self.con_details['database']
        eng = create_engine(eng_st)
        tbl_query = 'select * from "' + tbl_name + '";'
        if nrows != None:
            tbl_query = 'select * from "' + tbl_name + '" limit ' + str(nrows) + ';'
        
        with eng.connect() as conn:
            df = pd.read_sql_query(text(tbl_query), con=conn)
        
        return df
    
    def df_to_tbl(self, df_path, tbl_name):
        eng_st = 'postgresql://postgres:' + self.con_details['password'] + '@localhost:' + self.con_details['port'] + '/' + self.con_details['database']
        eng = create_engine(eng_st)
        df = pd.read_csv(df_path)
        df.to_sql(tbl_name, eng, if_exists='replace')
    
    def memdf_to_tbl(self, df, tbl_name):
        eng_st = 'postgresql://postgres:' + self.con_details['password'] + '@localhost:' + self.con_details['port'] + '/' + self.con_details['database']
        eng = create_engine(eng_st)
        clean_name = tbl_name
        # if tbl_name == 'order':
        #    clean_name = '"order"'
        df.to_sql(clean_name, eng, if_exists='replace', index=False)
    
    def get_schema(self):
        #the below also gets views, which we don't want.
        # schema_details = self.query('SELECT table_name, column_name, data_type FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = \'public\';')
        #instead, we only get base tables using the following query
        schema_details = self.query('SELECT INFORMATION_SCHEMA.COLUMNS.table_name, INFORMATION_SCHEMA.COLUMNS.column_name, INFORMATION_SCHEMA.COLUMNS.data_type FROM INFORMATION_SCHEMA.COLUMNS INNER JOIN INFORMATION_SCHEMA.TABLES ON INFORMATION_SCHEMA.COLUMNS.table_name = INFORMATION_SCHEMA.TABLES.table_name WHERE INFORMATION_SCHEMA.COLUMNS.table_schema = \'public\' and table_type = \'BASE TABLE\';')
        tabcols = {}
        for tup in schema_details:
            tab = tup[0]
            col = tup[1]
            if tab not in tabcols:
                tabcols[tab] = [col]
            else:
                tabcols[tab] += [col]
        
        return tabcols
    
    def get_schema_wvals(self):
        schema_details = self.query('SELECT INFORMATION_SCHEMA.COLUMNS.table_name, INFORMATION_SCHEMA.COLUMNS.column_name, INFORMATION_SCHEMA.COLUMNS.data_type FROM INFORMATION_SCHEMA.COLUMNS INNER JOIN INFORMATION_SCHEMA.TABLES ON INFORMATION_SCHEMA.COLUMNS.table_name = INFORMATION_SCHEMA.TABLES.table_name WHERE INFORMATION_SCHEMA.COLUMNS.table_schema = \'public\' and table_type = \'BASE TABLE\';')
        all_tbls = set([tup[0] for tup in schema_details])
        
        tabcolvals = {}
        
        for tbl in all_tbls:
            tabcolvals[tbl] = {}
            tbl_df = self.tbl_to_df(tbl, nrows=3)
            for c in tbl_df.columns:
                tabcolvals[tbl][c] = tbl_df[c].tolist()
        
        return tabcolvals
    
    def get_privs(self):
        priv_query = 'SELECT grantee, privilege_type, table_name FROM information_schema.role_table_grants where table_schema = \'public\''
        privs = self.query(priv_query)
        return privs

class SociomeAPI(DBAPI):
    
    def __init__(self, con_details : dict):
        self.con_details = con_details
        self.connect()
    
    def connect(self):
        if not os.path.exists(self.con_details['sociome_path']):
            raise Exception("Sociome data not at specified directory: {}".format(self.con_details['sociome_path']))
            
    
    def query(self, query_st):
        raise Exception("Queries Not supported on sociome csv repo")
    
    def execute(self, stmt_st):
        raise Exception("Statements Not supported on sociome csv repo")
    
    def teardown(self):
        print("Nothing to tear down")
        pass
    
    def tbl_to_df(self, tbl_name):
        print("Keep in Mind, this is not an actual database table. It's a table describing a sociome table.")
        tbl_path = os.path.join(self.con_details['sociome_path'], tbl_name)
        df = pd.read_csv(tbl_path)
        return df
        
    
    def df_to_tbl(self, df_path, tbl_name):
        raise Exception("Not allowed to write to the sociome database. It's proprietary")
    
    def memdf_to_tbl(self, df, tbl_name):
        raise Exception("Not allowed to write to the sociome database. It's proprietary")
    
    def get_schema(self):
        #the below also gets views, which we don't want.
        # schema_details = self.query('SELECT table_name, column_name, data_type FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = \'public\';')
        #instead, we only get base tables using the following query
        out_schema = sociome_schema(self.con_details['sociome_path'])
        return out_schema
    
    def get_schema_wvals(self):
        print("Sociome Database Values Not Available--simply return tables and columns")
        out_schema = sociome_schema(self.con_details['sociome_path'])
        return out_schema
    
    def get_privs(self):
        print("NOTE: The privileges returned constitute the ground truth, and not those that would be returned by a database.")
        sociome_dir = self.con_details['sociome_path']
        clean_dir = os.path.join(sociome_dir, 'clean')
        tbl_dir = os.path.join(sociome_dir, 'tbls')
        if not os.path.exists(tbl_dir):
            os.mkdir(os.path.join(tbl_dir))
        
        out_privs = {}
        
        for f in os.listdir(clean_dir):
            if f.endswith('_clean.xlsx'):
                fullf = os.path.join(clean_dir, f)
                df = read_sociome_file(fullf)
                out_privs[f] = df
        
        return out_privs
        

def views_are_equal(view_def1, view_def2, db_details):
    #let's not go through the DB for the obvious...
    if 'CREATE VIEW' in view_def1 and 'CREATE VIEW' not in view_def2:
        return False
    elif 'CREATE VIEW' in view_def2 and 'CREATE VIEW' not in view_def1:
        return False
    elif 'CREATE VIEW' not in view_def1 and 'CREATE VIEW' not in view_def2:
        #mostly, this is a base table, so we can just check for string equality
        return (view_def1 == view_def2)
    elif view_def1 == view_def2:
        return True
    else:
        vname1 = view_def1.split(' ')[2]
        vname2 = view_def2.split(' ')[2]
        pgapi = PostgresAPI(db_details)
        
        pgapi.teardown()
        #first, create the views
        err_st1 = pgapi.execute(view_def1)
        if err_st1 != '':
            raise Exception("View_def1 failed: {}, {}".format(view_def1, err_st1))
        view_q1 = 'select * from ' + vname1 + ';'
        v1_recs = pgapi.query(view_q1)
        pgapi.teardown()
        
        
        err_st2 = pgapi.execute(view_def2)
        if err_st2 != '':
            raise Exception("View_def2 failed: {}, {}".format(view_def2, err_st2))
        view_q2 = 'select * from ' + vname2 + ';'
        v2_recs = pgapi.query(view_q2)
        pgapi.teardown()
        #now, get the records and check if they match        
        if type(v1_recs) == str:
            raise Exception("Query did not execute correctly: {}, {}".format(view_q1, v1_recs))
        if type(v2_recs) == str:
            raise Exception("Query did not execute correctly: {}, {}".format(view_q2, v2_recs))
        
        #if the sets of sets of attribute values are the same, we can assume
        #these are the same view.
        v1_recset = [frozenset(tup) for tup in v1_recs]
        v2_recset = [frozenset(tup) for tup in v2_recs]
        v1_set = set(v1_recset)
        v2_set = set(v2_recset)
        if v1_set == v2_set:
            return True
        else:
            return False

def check_sub(sub_rows, full_rows):
    for r1 in sub_rows:
        r1_found = False
        for r2 in full_rows:
            vals_exist = True
            for k in r1:
                if r1[k] != r2[k]:
                    vals_exist = False
                    break
            
            if vals_exist:
                print(f"Row {r1}\nFound in {r2}")
                r1_found = True
                break
        
        if not r1_found:
            print(f"Row {r1}\nNot Found in any rows of full_rows")
            return False
    
    for r2 in full_rows:
        r2_found = False
        for r1 in sub_rows:
            vals_exist = True
            for k in r1:
                if r1[k] != r2[k]:
                    vals_exist = False
                    break
            
            if vals_exist:
                print(f"Row {r2}\nFound in {r1}")
                r2_found = True
                break
        
        if not r2_found:
            print(f"Row {r2}\nNot Found in any rows of sub_rows")
            return False
    
    return True

def check_rows_only(rows1, rows2):
    r1sets = [set(r1.values()) for r1 in rows1]
    r2sets = [set(r2.values()) for r2 in rows2]
    
    for r1 in r1sets:
        if r1 not in r2sets:
            print(f"Tuple {r1} not in rows2:\n\n{r2sets}")
            return False
    
    return True

def match_st_col(st_col, df):
    max_col = None
    max_len = 0
    for c in df.columns:
        if str(df.dtypes[c]) != 'object':
            continue
        c_list = df[c].tolist()
        c_int = set(st_col).intersection(set(c_list))
        if c_int != set():
            if len(c_int) > max_len:
                max_col = c
                max_len = len(c_int)
    
    return max_col, max_len

def match_int_col(int_col, df):
    max_col = None
    max_len = 0
    for c in df.columns:
        if 'int' not in str(df.dtypes[c]):
            continue
        c_list = df[c].tolist()
        c_int = set(int_col).intersection(set(c_list))
        if c_int != set():
            if len(c_int) > max_len:
                max_col = c
                max_len = len(c_int)
    
    return max_col, max_len

def match_float_col(f_col, df):
    max_col = None
    max_len = 0
    for c in df.columns:
        if 'float' not in str(df.dtypes[c]) and 'double' not in str(df.dtypes[c]):
            continue
        c_list = df[c].tolist()
        c_max = max(c_list)
        c_min = min(c_list)
        c_int = [f_val for f_val in f_col if f_val <= c_max and f_val >= c_min]
        if c_int != []:
            if len(c_int) > max_len:
                max_col = c
                max_len = len(c_int)
    
    return max_col, max_len

def cols_equal(df1, df2):
    are_equal = False
    out_map = {}
    
    cols1 = df1.columns.tolist()
    cols2 = df2.columns.tolist()
    
    if sorted(cols1) == sorted(cols2):
        are_equal = True
        out_map = {c2 : c2 for c2 in cols2}
        return are_equal, out_map
    elif set(cols1).union(set(cols2)) == set(cols1):
        are_equal = False
        out_map = {c2 : c2 for c2 in cols2}
        return are_equal, out_map
    else:
        #we need some strict domain checks to map columns
        for c in cols2:
            cur_col_lst = df2[c].tolist()
            if str(df2.dtypes[c]) == 'object':
                st_match, st_len = match_st_col(cur_col_lst, df1)
                if st_match != None:
                    out_map[c] = st_match
            elif 'int' in str(df2.dtypes[c]):
                int_match, int_len = match_int_col(cur_col_lst, df1)
                if int_match != None:
                    out_map[c] = int_match
            elif 'float' in str(df2.dtypes[c]) or 'double' in str(df2.dtypes[c]):
                fl_match, fl_len = match_float_col(cur_col_lst, df1)
                if fl_match != None:
                    out_map[c] = fl_match
    
    return are_equal, out_map
                

#checks whether views are equal, or one view is a column-wise subset of the other
def views_sub_equal(view_def1, view_def2, db_details, get_gt=None):
    #let's not go through the DB for the obvious...
    if view_def1 == view_def2:
        return True
    else:
        view_q1 = view_def1
        view_q2 = view_def2
        if 'CREATE VIEW' in view_def1:
            sel_ind = view_def1.index('SELECT')
            view_q1 = view_def1[sel_ind:]
        
        if 'CREATE VIEW' in view_def2:
            sel_ind = view_def2.index('SELECT')
            view_q2 = view_def2[sel_ind:]
        
        #now, query the database
        pg_api = PostgresAPI(db_details)
        clean_q1 = fix_pgcaps(view_q1, db_details)
        #in our context, q1 is a gpt-generated query that we are evaluating
        #so, instead of just failing if it cannot run, we should count it wrong
        #downstream, hence the try-except
        try:
            vdf1 = pg_api.query_as_df(clean_q1)
        except Exception as err:
            traceback.print_exc(file=sys.stdout)
            print(f"Could not run query:\n{clean_q1}\n\ndue to error:\n{err}")
            return False
            
            
        if get_gt != None:
            vdf2_raw = get_gt(view_q2, db_details['database'])
        else:
            vdf2_raw = pg_api.query_as_df(view_q2)
        
        are_equal, col_map = cols_equal(vdf1, vdf2_raw)
        if col_map != {}:
            vdf2 = vdf2_raw.rename(columns=col_map)
        else:
            vdf2 = vdf2_raw
        
        if are_equal:
            #then, do straight tuple inclusion checking
            rows1 = vdf1.to_dict(orient='records')
            rows2 = vdf2.to_dict(orient='records')
            for r1 in rows1:
                if r1 not in rows2:
                    print("Row not in view 2: {}".format(r1))
                    return False
            for r2 in rows2:
                if r2 not in rows1:
                    print("Row not in view 1: {}".format(r2))
                    return False
        else:
            v1_cols = sorted(vdf1.columns.tolist())
            v2_cols = sorted(vdf2.columns.tolist())
            
            if set(v1_cols).union(set(v2_cols)) == set(v2_cols):
                rows1 = vdf1.to_dict(orient='records')
                rows2 = vdf2.to_dict(orient='records')
                return check_sub(rows1, rows2)
            elif set(v1_cols).union(set(v2_cols)) == set(v1_cols):
                rows1 = vdf1.to_dict(orient='records')
                rows2 = vdf2.to_dict(orient='records')
                return check_sub(rows2, rows1)
            elif len(v1_cols) == len(v2_cols):
                rows1 = vdf1.to_dict(orient='records')
                rows2 = vdf2.to_dict(orient='records')
                left_ans = check_rows_only(rows1, rows2)
                right_ans = check_rows_only(rows2, rows1)
                return left_ans and right_ans
            else:
                print(f"Unsure how to compare result schemas:\n{v1_cols}\n{v2_cols}, assuming unequal")
                return False

def bird_subs_equal(view_def1, view_def2, db_details):
    return views_sub_equal(view_def1, view_def2, db_details, get_gt=bird2df)

def spider_subs_equal(view_def1, view_def2, db_details):
    return views_sub_equal(view_def1, view_def2, db_details, get_gt=spider2df)

def unfuse_word(word, query_st):
    if word not in query_st:
        raise Exception("Very malformed query missing {}: {}".format(word, query_st))
    cur_pts = query_st.split(' ')
    new_pts = []
    if word not in cur_pts:
        for pt in cur_pts:
            if word in pt:
                sel_ind = pt.index(word)
                new_pt = pt[:sel_ind] + ' ' + word + ' ' + pt[sel_ind + len(word):]
                new_pts.append(new_pt)
            else:
                new_pts.append(pt)
    
    if new_pts == []:
        new_pts = cur_pts
    
    new_st = ' '.join(new_pts)
    return new_st

def fixed_sql(raw_st):
    if ' ' not in raw_st:
        #most likely this is a table/column/role name, so just return it.
        return raw_st
    new_st = raw_st
    #one mistake is that ChatGPT can repeat words.
    if raw_st.startswith('CREATE VIEW VIEW'):
        rest = raw_st[len('CREATE VIEW VIEW'):]
        new_st = 'CREATE VIEW' + rest
    
    #another is that words get fused
    new_st = unfuse_word('SELECT', new_st)
    new_st = unfuse_word('FROM', new_st)
    return new_st

#gpt-4o generated--this regex seems to work...
def split_respecting_quotes(input_string):
    # Regular expression to match quoted substrings or unquoted words
    pattern = r"'[^']*'|\"[^\"]*\"|\S+"
    # Use re.findall to extract all matches
    result = re.findall(pattern, input_string)
    return result

# Example usage
# input_string = "SELECT * FROM customer WHERE customer.name = 'Brian Davis'"
# print(split_respecting_quotes(input_string))


#in postgres, capitalized column names have to be double-quoted
#and so do capitalized table names
def fix_pgcaps(query_st : str, db_details : dict):
    pgapi = PostgresAPI(db_details)
    schema_query = 'SELECT table_schema, table_name, column_name, data_type FROM INFORMATION_SCHEMA.COLUMNS where table_schema = \'public\';'
    schema_tups = pgapi.query(schema_query)
    col_names = [tup[2] for tup in schema_tups]
    tbl_names = [tup[1] for tup in schema_tups]
    print("Col_names: {}".format(col_names))
    print("Table names: {}".format(tbl_names))
    
    q_pts = split_respecting_quotes(query_st)
    new_pts = []
    #we need to examine equality among words likely to be column names
    #we do this to avoid column names that are substrings
    for pt in q_pts:
        if pt == '':
            continue
        clean_pt = pt.replace(' ', '')
        rel_pt = pt.replace(',', '')
        #condition for capitalization
        title_cond = False
        for tbl_name in tbl_names:
            if rel_pt.startswith(tbl_name): #this condition will not handle TPC-H because of the "order" table, and that is okay
                title_cond = True
        
        cap_cond = rel_pt in col_names or title_cond
        if cap_cond and rel_pt[0].isupper():
            new_pt = clean_pt[:clean_pt.index(rel_pt)] + '"' + rel_pt + '"' + clean_pt[clean_pt.index(rel_pt) + len(rel_pt):]
            new_pts.append(new_pt)
        else:
            new_pts.append(pt)
    
    new_st = ' '.join(new_pts)
    
    return new_st

#convert sqlite tables to postgres tables.
#this is needed if we want to run spider queries
def sqlite2postgres(sqlite_path : str, pg_details : dict):
    pgapi = PostgresAPI(pg_details)
    cnx = sqlite3.connect(sqlite_path)
    
    #first, get the list of tables
    cur = cnx.cursor()
    res = cur.execute("select name from sqlite_master where type='table';")
    tbls = [tup[0] for tup in res.fetchall()]
    for tbl in tbls:
        clean_tbl = tbl
        if tbl == 'order':
            clean_tbl = '"order"'
        cur_df = pd.read_sql_query("SELECT * from " + clean_tbl, cnx)
        pgapi.memdf_to_tbl(cur_df, tbl)
    
#execute the BIRD benchmark query against the original SQLite databases
def bird2df(query, db_name):
    db_path = os.path.expanduser('~/birdbench/dev/dev_databases/' + db_name + '/' + db_name + '.sqlite')
    conn = sqlite3.connect(db_path)
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

#execute the spider/Dr. Spider benchmark query against the original SQLite databases
def spider2df(query, db_name):
    db_path = os.path.expanduser('~/diagnostic-robustness-text-to-sql/data/Spider-dev/databases/' + db_name + '/' + db_name + '.sqlite')
    conn = sqlite3.connect(db_path)
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

