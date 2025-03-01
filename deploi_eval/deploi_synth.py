import pandas as pd
import os
from ast import literal_eval
from utils.chat_utils import OpenAIHandler, parse_elv2, parse_c3, parse_lst, read_parsed_chat, parse_el_sem
from utils.metadata_utils import BIRD_Context
from utils.context_utils import Tabcolval_Context, Sociome_Context
from acmdiff.bird_diff import BIRD_META, BIRD_OTHER
from utils.db_utils import PostgresAPI, bird_subs_equal, spider_subs_equal
from utils.z3_utils import parse_sql_temporal_conditions, check_equivalence
import pickle

handler = OpenAIHandler('gpt-4o')

def schema_by_dbtype(db_type, db_api):
    if db_type == 'BIRD':
        tc_context = Tabcolval_Context(db_api)
        schema_st = tc_context.get_context()
        meta_context = BIRD_Context(BIRD_META, {'dict_path' : BIRD_OTHER})
    elif db_type == 'sociome':
        tc_context = Sociome_Context(db_api)
        schema_st = tc_context.get_context()
        meta_context = None
    else:
        tc_context = Tabcolval_Context(db_api)
        schema_st = tc_context.get_context()
        meta_context = None
        
    return schema_st, meta_context

def role_prompt(nl_role, sql_role_lst, nl_ind, outdir, outname):
    chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
    prompt = 'Consider the following description of a Postgres database role: ' + nl_role
    prompt += '\n\nConsider the following list of SQL roles on the database: ' + str(sql_role_lst)
    prompt += '\n\nWhich SQL role from the list best matches the given description?'
    chat += [{'role' : 'user', 'content' : prompt}]
    
    # resp = handler.get_response(chat, 1.0, write_dir=outdir, write_file=outname + '_rd2term' + str(i) + '_chat.json')
    # term = parse_elv2(resp, all_role_sql, 'None')
    outfile = os.path.join(outdir, outname + '_rd2term' + str(nl_ind) + '_parsedchat.pkl')
    print("Rhiesys Synth: Storing Results in File: {}".format(outfile))
    term, resp = handler.get_parsed_response({'element' : nl_role, 'list' : sql_role_lst}, chat, 1.0, parse_el_sem, write_dir=outdir, write_file=outname + '_rd2term_' + str(nl_ind) + '_parsedchat.pkl')
    return term, resp

def role_map_only(nl_role_lst, sql_role_lst, outdir, outname):
    out_schema = ['Role', 'SQL Role', 'Response']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    for i,r in enumerate(nl_role_lst):
        cur_term, cur_resp = role_prompt(r, sql_role_lst, i, outdir, outname)
        out_dct['Role'].append(r)
        out_dct['SQL Role'].append(cur_term)
        out_dct['Response'].append(cur_resp)
    
    outpath = os.path.join(outdir, outname + '_rolemap.csv')
    out_df = pd.Dataframe(out_dct)
    out_df.to_csv(outpath, index=False)

#TODO: fix the ordering bug below--results are not properly aligned with inputs
def rolehier_synth(rhdf, rh_sql, outdir, outname):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    out_schema = ['Role', 'SQL Role', 'Child', 'SQL Child', 'Full SQL']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    #NOTE: doing list(set(...)) causes different ordering on different runs.
    #so we need to sort the output results to make sure the inputs are still aligned
    #with the outputs on every run.
    all_role_descs = sorted(list(set(rhdf['Role'].tolist() + rhdf['Child'].tolist())))
    all_role_sql = sorted(list(set(rh_sql['Role'].tolist() + rh_sql['Child'].tolist())))
    
    rdterm_schema = ['Role No.', 'Role', 'SQL Role', 'Explanation']
    
    rdterm = {}
    rdterm_dct = {}
    for rt in rdterm_schema:
        rdterm[rt] = []
    
    for i,rd in enumerate(all_role_descs):
        print("Current i: {}".format(i))
        chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
        prompt = 'Consider the following description of a Postgres database role: ' + rd
        prompt += '\n\nConsider the following list of SQL roles on the database: ' + str(all_role_sql)
        prompt += '\n\nWhich SQL role from the list best matches the given description?'
        chat += [{'role' : 'user', 'content' : prompt}]
        
        # resp = handler.get_response(chat, 1.0, write_dir=outdir, write_file=outname + '_rd2term' + str(i) + '_chat.json')
        # term = parse_elv2(resp, all_role_sql, 'None')
        outfile = os.path.join(outdir, outname + '_rd2term' + str(i) + '_parsedchat.pkl')
        print("Rhiesys Synth: Storing Results in File: {}".format(outfile))
        term, resp = handler.select_from_lst(chat, rd, all_role_sql, 1.0, write_dir=outdir, write_file=outname + '_rd2term' + str(i) + '_parsedchat.pkl')
        rdterm['Role No.'].append(i)
        rdterm['Role'].append(rd)
        rdterm['SQL Role'].append(term)
        rdterm['Explanation'].append(resp)
        rdterm_dct[rd] = term
    
    rdterm_df = pd.DataFrame(rdterm)
    rdterm_df.to_csv(os.path.join(outdir, outname + '_rh_rd2term.csv'), index=False)
    
    #now, construct the GRANT statements
    for i,row in enumerate(rhdf.to_dict(orient='records')):
        cur_role = row['Role']
        cur_child = row['Child']
        sql_role = rdterm_dct[cur_role]
        sql_child = rdterm_dct[cur_child]
        out_dct['Role'].append(cur_role)
        out_dct['Child'].append(cur_child)
        out_dct['SQL Role'].append(sql_role)
        out_dct['SQL Child'].append(sql_child)
        cur_full_sql = 'GRANT \'' + sql_child + '\' TO \'' + sql_role + '\';'
        if 'None' in sql_child:
            cur_full_sql = f'--None. No GRANT statement needed for {sql_role} and {sql_child}'
        out_dct['Full SQL'].append(cur_full_sql)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(os.path.join(outdir, outname + '_rh_fullsql.csv'), index=False)

#function that properly constructs results from chats
def rh_fromchats(rhdf, rh_sql, outdir, outname):
    out_schema = ['Role', 'SQL Role', 'Child', 'SQL Child', 'Full SQL']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    all_role_descs = list(set(rhdf['Role'].tolist() + rhdf['Child'].tolist()))
    all_role_sql = list(set(rh_sql['Role'].tolist() + rh_sql['Child'].tolist()))
    
    if os.path.exists(os.path.join(outdir, outname + '_rh_fullsql.csv')):
        os.rename(os.path.join(outdir, outname + '_rh_fullsql.csv'), os.path.join(outdir, outname + '_rh_fullsql.bak'))
    
    rh_chats = [os.path.join(outdir, f) for f in os.listdir(outdir) if '_rd2term' in f and f.endswith('_chat.json')]
    
    term_tups = []
    for rhf in rh_chats:
        with open(rhf, 'r') as fh:
            ents = literal_eval(fh.read())
        inp_text = ents[1]['content']
        out_text = ents[-1]['content']
        
        inp_rd = [rd for rd in all_role_descs if rd in inp_text][0]
        out_term = parse_elv2(out_text, all_role_sql, 'None')
        term_tups.append((inp_rd, out_term, rhf))
    
    print(term_tups)

def map_roles(roles_lst, outdir, outname, in_suffix, out_suffix):
    
    out_schema = ['Role', 'Match', 'Term', 'Explanation']
    
    out_dct = {}
    
    for o in out_schema:
        out_dct[o] = []
    
    rd2term_df = pd.read_csv(os.path.join(outdir, outname + in_suffix))
    rd2term = {}
    for row in rd2term_df.to_dict(orient='records'):
        rd2term[row['Role']] = row['SQL Role']
    
    all_descs = list(rd2term.keys())
    all_terms = list(rd2term.values())
    
    for i,r in enumerate(roles_lst):
        chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
        prompt = f'Consider the given description of a Postgres database role: {r}\n\n'
        prompt += f'Consider the following list of descriptions of Postgres database roles: \n\n{all_descs}\n\n'
        prompt += 'Which description from the list best matches the given description?'
        chat += [{'role' : 'user', 'content' : prompt}]
        # init_resp = handler.get_response(chat, 1.0, write_dir=outdir, write_file=outname + out_suffix + '_role' + str(i) + '_chat.json')
        # parsed = parse_elv2(init_resp, all_descs, 'None')
        parsed, init_resp = handler.select_from_lst(chat, r, all_descs, 1.0, write_dir=outdir, write_file=outname + out_suffix + '_role' + str(i) + '_parsedchat.pkl')
        out_dct['Role'].append(r)
        out_dct['Match'].append(parsed)
        out_dct['Explanation'].append(init_resp)
        
        term = rd2term[parsed]
        out_dct['Term'].append(term)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(os.path.join(outdir, outname + out_suffix + '_rolemap.csv'), index=False)
    
    return out_df



def view_synth(view_lst, nl2sql_type, db_api, outdir, outname):
    out_schema = ['View Name', 'View', 'SQL View', 'Explanation']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    schema_st, meta_context = schema_by_dbtype(nl2sql_type, db_api)
    # tc_context = Tabcolval_Context(pg_api)
    # schema_st = tc_context.get_context()
    
    # meta_context = BIRD_Context(BIRD_META, {'dict_path' : BIRD_OTHER})
    
    for i,v in enumerate(view_lst):
        if meta_context != None:
            meta_st = meta_context.get_context_nl(v)
        else:
            meta_st = ''
        
        prompt = 'Consider the following question to answer using a database: ' + str(v)
        prompt += schema_st + meta_st
        if meta_st != '':
            prompt += 'Based on the schema and relevant metadata,'
        else:
            prompt += 'Based on the schema,'
        prompt += ' write SQL code for Postgres that would answer this question.'
        chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
        chat += [{'role' : 'user', 'content' : prompt}]
        raw_resp = handler.get_response(chat, 1.0, write_dir=outdir, write_file=outname + '_view' + str(i) + '_chat.json')
        parsed = parse_c3(raw_resp)
        out_dct['View'].append(v)
        out_dct['View Name'].append('view' + str(i))
        view_sql = 'CREATE VIEW view' + str(i) + ' AS ' + parsed
        out_dct['SQL View'].append(view_sql)
        out_dct['Explanation'].append(raw_resp)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(os.path.join(outdir, outname + '_viewmap.csv'), index=False)
    
    return out_df

def priv_map_only(nl_priv_lst, outdir, outname):
    ops_lst = ['SELECT', 'UPDATE', 'INSERT', 'CREATE', 'DELETE', 'GRANT']
    test_lst = ['UPDATE', 'INSERT', 'SELECT']
    out_schema = ['Priv', 'SQL Priv', 'Explanation']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    for i,priv_nl in enumerate(nl_priv_lst):
        prompt = 'Consider the following sentence/phrase describing Postgres database permissions for a role on a table: Privilege 1-' + priv_nl + '.'
        prompt += 'Consider the following list of allowed PostgreSQL operators for the same role on the same table: Privilege 2-' + str(test_lst) + '.'
        prompt += 'In the same format as Privilege 2, give me the smallest list of SQL operators that need to be allowed to satisfy Privilege 1.'
        prompt += ' Choose SQL operators from the following: SELECT, UPDATE, INSERT, CREATE, DELETE, and GRANT.'
        prompt += ' Also note that in this case, the SQL operator GRANT is shorthand for WITH GRANT OPTION--it indicates that a role can pass down any of its permitted operations.'
        
        chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
        chat += [{'role' : 'user', 'content' : prompt}]
        raw_resp = handler.get_response(chat, 1.0, write_dir=outdir, write_file=outname + '_priv' + str(i) + '_chat.json')
        parsed = parse_lst(raw_resp, quote_lst=ops_lst)
        
        
        out_dct['Priv'].append(priv_nl)
        out_dct['SQL Priv'].append(parsed)
        # out_dct['Full SQL'].append('GRANT ' + ', '.join(parsed) + ' ON ' + view_name + ' TO \'' + role_sql + '\';')
        out_dct['Explanation'].append(raw_resp)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(os.path.join(outdir, outname + '_privonly_map.csv'), index=False)

def priv_synth(role_ind, role_nl, role_sql, view_name, view_nl, view_sql, priv_nl, outdir, outname, cache=None):
    #role_ind here should be the index of the role in the NLACM, and NOT the index of the role hierarchy!
    ops_lst = ['SELECT', 'UPDATE', 'INSERT', 'CREATE', 'DELETE', 'GRANT']
    out_schema = ['Role', 'SQL Role', 'View', 'SQL View', 'View Name', 'Priv', 'SQL Priv', 'Full SQL', 'Explanation']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    if cache != None:
        if priv_nl in cache:
            parsed = cache[priv_nl]['SQL Priv']
            full_sql = 'GRANT ' + ', '.join(parsed) + ' ON ' + view_name + ' TO \'' + role_sql + '\';'
            raw_resp = cache[priv_nl]['Explanation']
            
            out_dct['Role'].append(role_nl)
            out_dct['SQL Role'].append(role_sql)
            out_dct['View'].append(view_nl)
            out_dct['SQL View'].append(view_sql)
            out_dct['View Name'].append(view_name)
            out_dct['Priv'].append(priv_nl)
            out_dct['SQL Priv'].append(parsed)
            out_dct['Full SQL'].append(full_sql)
            out_dct['Explanation'].append(raw_resp)
            
            return out_dct
    
    test_lst = ['UPDATE', 'INSERT', 'SELECT']
    prompt = 'Consider the following sentence/phrase describing Postgres database permissions for a role on a table: Privilege 1-' + priv_nl + '.'
    prompt += 'Consider the following list of allowed PostgreSQL operators for the same role on the same table: Privilege 2-' + str(test_lst) + '.'
    prompt += 'In the same format as Privilege 2, give me the smallest list of SQL operators that need to be allowed to satisfy Privilege 1.'
    prompt += ' Choose SQL operators from the following: SELECT, UPDATE, INSERT, CREATE, DELETE, and GRANT.'
    prompt += ' Also note that in this case, the SQL operator GRANT is shorthand for WITH GRANT OPTION--it indicates that a role can pass down any of its permitted operations.'
    
    chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
    chat += [{'role' : 'user', 'content' : prompt}]
    raw_resp = handler.get_response(chat, 1.0, write_dir=outdir, write_file=outname + '_priv_role' + str(role_ind) + '_' + view_name + '_chat.json')
    parsed = parse_lst(raw_resp, quote_lst=ops_lst)
    
    out_dct['Role'].append(role_nl)
    out_dct['SQL Role'].append(role_sql)
    out_dct['View'].append(view_nl)
    out_dct['SQL View'].append(view_sql)
    out_dct['View Name'].append(view_name)
    out_dct['Priv'].append(priv_nl)
    out_dct['SQL Priv'].append(parsed)
    out_dct['Full SQL'].append('GRANT ' + ', '.join(parsed) + ' ON ' + view_name + ' TO \'' + role_sql + '\';')
    out_dct['Explanation'].append(raw_resp)
    
    return out_dct

def nlacm_synth(nlacm_df, pg_api, outdir, outname):
    out_schema = ['Role', 'SQL Role', 'View', 'SQL View', 'View Name', 'Priv', 'SQL Priv', 'Full SQL', 'Explanation']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    nl_roles = nlacm_df['Role'].tolist()
    nl_views = [c for c in nlacm_df.columns.tolist() if c != 'Role']
    
    #step 1: translate roles using map from role hierarchy roles
    role_df = map_roles(nl_roles, outdir, outname, '_rh_rd2term.csv', '_nlacm')
    
    #step 2: translate views
    view_df = view_synth(nl_views, pg_api, outdir, outname)
    
    #step 3: translate privileges using roles and views
    for role_ind, row in enumerate(nlacm_df.to_dict(orient='records')):
        cur_role = row['Role']
        cur_rmap = role_df[role_df['Role'] == cur_role]
        if cur_rmap.empty:
            raise Exception("Dataframe should not be empty: {}, {}".format(cur_role, role_df['Role'].tolist()))
        
        #get the first row
        rmap_row = cur_rmap.to_dict(orient='records')[0]
        cur_term = rmap_row['Term']
        for c in nlacm_df.columns:
            if c == 'Role':
                continue
            view_sel = view_df[view_df['View'] == c]
            if view_sel.empty:
                raise Exception("Dataframe should not be empty: {}, {}".format(c, view_df['View'].tolist()))
            
            view_row = view_sel.to_dict(orient='records')[0]
            cur_view = view_row['View']
            cur_name = view_row['View Name']
            cur_view_sql = view_row['SQL View']
            
            cur_priv = row[c]
            
            new_ent = priv_synth(role_ind, cur_role, cur_term, cur_name, cur_view, cur_view_sql, cur_priv, outdir, outname)
            for k in new_ent:
                out_dct[k] += new_ent[k]
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(os.path.join(outdir, outname + '_privmap.csv'), index=False)
    
    # return out_df

def temp_map_only(nl_temp_lst, outdir, outname):
    out_schema = ['Temp', 'SQL Temp', 'Response']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    for i,nl_temp in enumerate(nl_temp_lst):
        fs_inp = """Consider the following temporal access control constraint, which needs to be implemented on a Postgres database: "This role can access this view from 10am to 11pm on the days of the week: Saturday, Tuesday, and Sunday"
        Write a SQL function for the above temporal constraint."""
        fs_out = """ This function will check if the current time falls within the allowed access window (10am to 11pm on Saturdays, Tuesdays, and Sundays).
        ```
        CREATE OR REPLACE FUNCTION is_access_allowed() RETURNS BOOLEAN AS $$
        DECLARE
            current_time TIME;
            current_day TEXT;
        BEGIN
            -- Get current time and day
            current_time := NOW();
            current_day := to_char(CURRENT_DATE, 'Day');

            -- Check if current time is between 10 and 23 and it's on specified days
            IF EXTRACT('Hour' FROM current_time) >= 10 AND EXTRACT('Hour' FROM current_time) < 23 AND current_day IN ('Sunday', 'Tuesday', 'Saturday') THEN
                RETURN true;
            ELSE
                RETURN false;
            END IF;
        END;
        $$ LANGUAGE plpgsql;
        ```
        """
        
        chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
        chat += [{'role' : 'user', 'content' : fs_inp}]
        chat += [{'role' : 'assistant', 'content' : fs_out}]
        
        prompt = f"""Consider the following temporal access control constraint, which needs to be implemented on a Postgres database: "{nl_temp}"
        Write a SQL function for the above temporal constraint."""
        
        chat += [{'role' : 'user', 'content' : prompt}]
        raw_resp = handler.get_response(chat, 1.0, write_dir=outdir, write_file=outname + '_temp' + str(i) + '_chat.json')
        parsed = parse_c3(raw_resp)
        out_dct['Temp'].append(nl_temp)
        out_dct['SQL Temp'].append(parsed)
        out_dct['Response'].append(raw_resp)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(os.path.join(outdir, outname + '_tempmap.csv'), index=False)

def fill_in_policy(policy_name, priv_lst, role_sql, view_name, func_name):
    full_sql = 'CREATE POLICY ' + policy_name + ' ON ' + view_name + ' FOR '
    full_sql += ', '.join(priv_lst) + ' USING ' + ' ( ' + func_name + ' AND current_user = \'' + role_sql + '\' );'
    return full_sql
    

def temp_synth(role_ind, role_nl, role_sql, view_name, view_nl, view_sql, cond_nl, outdir, outname, cache=None):
    out_schema = ['Role', 'SQL Role', 'View', 'SQL View', 'View Name', 'Cond', 'SQL Cond', 'Policy Name', 'Explanation']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    policy_name = 'role' + str(role_ind) + '_' + view_name + '_temppolicy'
    
    if cache != None:
        if cond_nl in cache:
            parsed = cache[cond_nl]['SQL Cond']
            raw_resp = cache[cond_nl]['Explanation']
            
            out_dct['Role'].append(role_nl)
            out_dct['SQL Role'].append(role_sql)
            out_dct['View'].append(view_nl)
            out_dct['SQL View'].append(view_sql)
            out_dct['View Name'].append(view_name)
            out_dct['Cond'].append(cond_nl)
            out_dct['SQL Cond'].append(parsed)
            out_dct['Policy Name'].append(policy_name)
            out_dct['Explanation'].append(raw_resp)
            
            return out_dct
        
    fs_inp = """Consider the following temporal access control constraint, which needs to be implemented on a Postgres database: "This role can access this view from 10am to 11pm on the days of the week: Saturday, Tuesday, and Sunday"
    Write a SQL function for the above temporal constraint."""
    fs_out = """ This function will check if the current time falls within the allowed access window (10am to 11pm on Saturdays, Tuesdays, and Sundays).
    ```
    CREATE OR REPLACE FUNCTION is_access_allowed() RETURNS BOOLEAN AS $$
    DECLARE
        current_time TIME;
        current_day TEXT;
    BEGIN
        -- Get current time and day
        current_time := NOW();
        current_day := to_char(CURRENT_DATE, 'Day');

        -- Check if current time is between 10 and 23 and it's on specified days
        IF EXTRACT('Hour' FROM current_time) >= 10 AND EXTRACT('Hour' FROM current_time) < 23 AND current_day IN ('Sunday', 'Tuesday', 'Saturday') THEN
            RETURN true;
        ELSE
            RETURN false;
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    ```
    """
    
    chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
    chat += [{'role' : 'user', 'content' : fs_inp}]
    chat += [{'role' : 'assistant', 'content' : fs_out}]
    
    prompt = f"""Consider the following temporal access control constraint, which needs to be implemented on a Postgres database: "{cond_nl}"
    Write a SQL function for the above temporal constraint."""
    
    chat += [{'role' : 'user', 'content' : prompt}]
    raw_resp = handler.get_response(chat, 1.0, write_dir=outdir, write_file=outname + '_role' + str(role_ind) + '_' + view_name + '_temp_chat.json')
    parsed = parse_c3(raw_resp)
    out_dct['Role'].append(role_nl)
    out_dct['SQL Role'].append(role_sql)
    out_dct['View'].append(view_nl)
    out_dct['SQL View'].append(view_sql)
    out_dct['View Name'].append(view_name)
    out_dct['Cond'].append(cond_nl)
    out_dct['SQL Cond'].append(parsed)
    out_dct['Policy Name'].append(policy_name)
    out_dct['Explanation'].append(raw_resp)
    
    return out_dct

def timeacm_synth(temp_df, pg_api, outdir, outname):
    out_schema = ['Role', 'SQL Role', 'View', 'SQL View', 'View Name', 'Cond', 'SQL Cond', 'Policy Name', 'Explanation']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    nl_roles = temp_df['Role'].tolist()
    nl_views = temp_df.columns.tolist()
    
    #step 1: translate roles using map from role hierarchy roles
    role_df = map_roles(nl_roles, outdir, outname, '_rh_rd2term.csv', '_time')
    
    #step 2: we would already have generated the view SQL, so read that in
    view_df = pd.read_csv(os.path.join(outdir, outname + '_viewmap.csv'))
    
    priv_df = pd.read_csv(os.path.join(outdir, outname + '_privmap.csv'))
    
    #step 3: translate temporal conditions using roles and views
    for role_ind, row in enumerate(temp_df.to_dict(orient='records')):
        cur_role = row['Role']
        cur_rmap = role_df[role_df['Role'] == cur_role]
        if cur_rmap.empty:
            raise Exception("Dataframe should not be empty: {}, {}".format(cur_role, role_df['Role'].tolist()))
        
        #get the first row
        rmap_row = cur_rmap.to_dict(orient='records')[0]
        cur_term = rmap_row['Term']
        for c in temp_df.columns:
            if c == 'Role':
                continue
            view_sel = view_df[view_df['View'] == c]
            if view_sel.empty:
                raise Exception("View Dataframe should not be empty: {}, {}".format(c, view_df['View'].tolist()))
            
            #we also need the privileges generated in the previous step of the overall algorithm
            priv_sel = priv_df[(priv_df['SQL Role'] == cur_term) & (priv_df['View'] == c)]
            if priv_sel.empty:
                raise Exception("Priv Dataframe should not be empty: {}, {}\n\n{}".format(cur_term, c, priv_df))
            priv_lst = priv_sel['SQL Priv'].tolist()[0]
            
            view_row = view_sel.to_dict(orient='records')[0]
            cur_view = view_row['View']
            cur_name = view_row['View Name']
            cur_view_sql = view_row['SQL View']
            
            cur_cond = row[c]
            
            new_ent = temp_synth(role_ind, cur_role, cur_term, cur_name, cur_view, cur_view_sql, cur_cond, outdir, outname)
            cond_pts = new_ent['SQL Cond'][0].split(' ')
            func_cands = [pt for pt in cond_pts if '()' in pt]
            if len(func_cands) < 1:
                raise Exception("LLM did not name the function: {}".format(new_ent['SQL Cond']))
            func_name = func_cands[0]
            policy_name = 'role' + str(role_ind) + '_' + cur_name + '_temppolicy'
            policy_sql = fill_in_policy(policy_name, priv_lst, cur_term, cur_name, func_name)
            full_sql = new_ent['SQL Cond'][0] + '\n\n' + policy_sql
            new_ent['SQL Cond'][0] = full_sql
            for k in new_ent:
                out_dct[k] += new_ent[k]
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(os.path.join(outdir, outname + '_tempmap.csv'), index=False)
    # return out_df

def check_temps(temp_strs, role_sql, view_name):
    view_test = 'ON ' + view_name
    current_user = 'AND current_user = \'' + role_sql + '\''
    
    temp_lst = [t for t in temp_strs if view_test in t and current_user in t]
    
    if len(temp_lst) == 0:
        return False
    return True

def construct_program(outdir, outname):
    rh_file = os.path.join(outdir, outname + '_rh_fullsql.csv')
    nlacm_file = os.path.join(outdir, outname + '_privmap.csv')
    temp_file = os.path.join(outdir, outname + 'tempmap.csv')
    
    rhdf = pd.read_csv(rh_file)
    nlacmdf = pd.read_csv(nlacm_file)
    tempdf = pd.read_csv(temp_file)
    
    rh_sql = rhdf['Full SQL'].tolist()
    temp_sql = tempdf['SQL Cond'].tolist()
    
    nlacm_sql = []
    for row in enumerate(nlacmdf.to_dict(orient='records')):
        is_temp = check_temps(temp_sql, row['SQL Role'], row['View Name'])
        if not is_temp:
            nlacm_sql.append(row['Full SQL'])
        else:
            #we should end up skipping a lot...
            print("Skipping GRANT statement: {}".format(row['Full SQL']))
    
    #now, put together the sequence
    full_st = ''
    full_st += '\n\n'.join(rh_sql) + '\n\n'
    full_st += '\n\n'.join(temp_sql) + '\n\n'
    full_st += '\n\n'.join(nlacm_sql)
    
    with open(os.path.join(outdir, outname + '_fullprogram.sql'), 'w+') as fh:
        print(full_st, file=fh)

#store and return a dictionary of role descriptions to role SQL
def rh_gtmap(raw_rh, gt_rh, outdir, outname):
    outfile = os.path.join(outdir, outname + '_rh_gtmap.json')
    if os.path.exists(outfile):
        with open(outfile, 'r') as fh:
            out_dct = literal_eval(fh.read())
        return out_dct
    
    out_dct = {}
    
    raw_rows = raw_rh.shape[0]
    gt_rows = gt_rh.shape[0]
    if raw_rows != gt_rows:
        print(f"WARNING: raw_rh rows: {raw_rows} is not equal to gt_rh rows: {gt_rows}")
    
    num_rows = min(raw_rows, gt_rows)
    for i in range(num_rows):
        raw_role = raw_rh.loc[i]['Role']
        raw_child = raw_rh.loc[i]['Child']
        gt_role = gt_rh.loc[i]['Role']
        gt_child = gt_rh.loc[i]['Child']
        out_dct[raw_role] = gt_role
        out_dct[raw_child] = gt_child
    
    
    with open(outfile, 'w+') as fh:
        print(out_dct, file=fh)
    
    return out_dct

def nlacm_gtmap(raw, gt, outdir, outname):
    role_out = os.path.join(outdir, outname + '_nlacm_role_gtmap.json')
    view_out = os.path.join(outdir, outname + '_nlacm_view_gtmap.json')
    priv_out = os.path.join(outdir, outname + '_nlacm_priv_gtmap.json')
    
    if os.path.exists(role_out) and os.path.exists(view_out) and os.path.exists(priv_out):
        with open(role_out, 'r') as fh:
            role_dct = literal_eval(fh.read())
        
        with open(view_out, 'r') as fh:
            view_dct = literal_eval(fh.read())
        
        with open(priv_out, 'r') as fh:
            priv_dct = literal_eval(fh.read())
        
        return role_dct, view_dct, priv_dct
    
    role_dct = {}
    view_dct = {}
    priv_dct = {}
    
    raw_roles = raw['Role'].tolist()
    gt_roles = gt['Role'].tolist()
    raw_views = [c for c in raw.columns.tolist() if c != 'Role']
    gt_views = [c for c in gt.columns.tolist() if c != 'Role']
    
    
    if len(raw_roles) != len(gt_roles):
        print(f"WARNING: raw roles: {raw_roles} is not equal to gt roles: {gt_roles}")
    
    if len(raw_views) != len(gt_views):
        print(f"WARNING: raw views: {raw_views} is not equal to gt views: {gt_views}")
    
    role_dim = min(len(raw_roles), len(gt_roles))
    view_dim = min(len(raw_views), len(gt_views))
    
    for i in range(role_dim):
        role_dct[raw_roles[i]] = gt_roles[i]
    
    for i in range(view_dim):
        view_dct[raw_views[i]] = gt_views[i]
    
    for i,row in enumerate(raw.to_dict(orient='records')):
        gt_row = gt.loc[i]
        for j,c in enumerate(raw_views):
            priv_dct[row[c]] = gt_row[gt_views[j]]
    
    with open(role_out, 'w+') as fh:
        print(role_dct, file=fh)
    
    with open(view_out, 'w+') as fh:
        print(view_dct, file=fh)
    
    with open(priv_out, 'w+') as fh:
        print(priv_dct, file=fh)
    
    return role_dct, view_dct, priv_dct

def temp_gtmap(raw, gt, outdir, outname):
    role_out = os.path.join(outdir, outname + '_time_role_gtmap.json')
    view_out = os.path.join(outdir, outname + '_time_view_gtmap.json')
    priv_out = os.path.join(outdir, outname + '_time_cond_gtmap.json')
    
    if os.path.exists(role_out) and os.path.exists(view_out) and os.path.exists(priv_out):
        with open(role_out, 'r') as fh:
            role_dct = literal_eval(fh.read())
        
        with open(view_out, 'r') as fh:
            view_dct = literal_eval(fh.read())
        
        with open(priv_out, 'r') as fh:
            temp_dct = literal_eval(fh.read())
        
        return role_dct, view_dct, temp_dct
    
    role_dct = {}
    view_dct = {}
    temp_dct = {}
    
    raw_roles = raw['Role'].tolist()
    gt_roles = gt['Role'].tolist()
    raw_views = [c for c in raw.columns.tolist() if c != 'Role']
    gt_views = [c for c in gt.columns.tolist() if c != 'Role']
    
    if len(raw_roles) != len(gt_roles):
        print(f"WARNING: raw roles: {raw_roles} is not equal to gt roles: {gt_roles}")
    
    if len(raw_views) != len(gt_views):
        print(f"WARNING: raw views: {raw_views} is not equal to gt views: {gt_views}")
    
    role_dim = min(len(raw_roles), len(gt_roles))
    view_dim = min(len(raw_views), len(gt_views))
    
    for i in range(role_dim):
        role_dct[raw_roles[i]] = gt_roles[i]
    
    for i in range(view_dim):
        view_dct[raw_views[i]] = gt_views[i]
    
    for i,row in enumerate(raw.to_dict(orient='records')):
        gt_row = gt.loc[i]
        for j,c in enumerate(raw_views):
            temp_dct[row[c]] = gt_row[gt_views[j]]
    
    role_out = os.path.join(outdir, outname + '_time_role_gtmap.json')
    view_out = os.path.join(outdir, outname + '_time_view_gtmap.json')
    priv_out = os.path.join(outdir, outname + '_time_cond_gtmap.json')
    
    with open(role_out, 'w+') as fh:
        print(role_dct, file=fh)
    
    with open(view_out, 'w+') as fh:
        print(view_dct, file=fh)
    
    with open(priv_out, 'w+') as fh:
        print(temp_dct, file=fh)
    
    return role_dct, view_dct, temp_dct

def temps_are_equal(temp_st1, temp_st2):
    z3cond1 = parse_sql_temporal_conditions(temp_st1)
    z3cond2 = parse_sql_temporal_conditions(temp_st2)
    
    are_equal = check_equivalence(z3cond1, z3cond2)
    return are_equal

#only compute RH and NLACM results--do not worry about temporal for now
def simple_eval_synth(outdir, outname, nl2sql_type, db_details):
    rh_file = os.path.join(outdir, outname + '_rh_fullsql.csv')
    nlacm_file = os.path.join(outdir, outname + '_privmap.csv')
    temp_file = os.path.join(outdir, outname + 'tempmap.csv')
    
    rh_gtfile = os.path.join(outdir, outname + '_rh_gtmap.json')
    
    nlacm_rolegtfile = os.path.join(outdir, outname + '_nlacm_role_gtmap.json')
    nlacm_viewgtfile = os.path.join(outdir, outname + '_nlacm_view_gtmap.json')
    nlacm_privgtfile = os.path.join(outdir, outname + '_nlacm_priv_gtmap.json')
    
    time_rolegtfile = os.path.join(outdir, outname + '_time_role_gtmap.json')
    time_viewgtfile = os.path.join(outdir, outname + '_time_view_gtmap.json')
    time_privgtfile = os.path.join(outdir, outname + '_time_cond_gtmap.json')
    
    with open(rh_gtfile, 'r') as fh:
        rh_gt = literal_eval(fh.read())
    
    with open(nlacm_rolegtfile, 'r') as fh:
        nlacm_rolegt = literal_eval(fh.read())
    with open(nlacm_viewgtfile, 'r') as fh:
        nlacm_viewgt = literal_eval(fh.read())
    with open(nlacm_privgtfile, 'r') as fh:
        nlacm_privgt = literal_eval(fh.read())
    
    with open(time_rolegtfile, 'r') as fh:
        time_rolegt = literal_eval(fh.read())
    with open(time_viewgtfile, 'r') as fh:
        time_viewgt = literal_eval(fh.read())
    with open(time_privgtfile, 'r') as fh:
        time_privgt = literal_eval(fh.read())
    
    rhdf = pd.read_csv(rh_file)
    nlacmdf = pd.read_csv(nlacm_file)
    tempdf = pd.read_csv(temp_file)
    
    #outputs
    rh_res_dct = {}
    nlacm_res_dct = {}
    temp_res_dct = {}
    
    for c in rhdf.columns:
        rh_res_dct[c] = []
    
    rh_res_dct['GT Role'] = []
    rh_res_dct['GT Child'] = []
    rh_res_dct['Correct?'] = []
    
    for c in nlacmdf.columns:
        nlacm_res_dct[c] = []
    
    nlacm_res_dct['GT Role'] = []
    nlacm_res_dct['GT View'] = []
    nlacm_res_dct['GT Priv'] = []
    nlacm_res_dct['Correct?'] = []
    nlacm_res_dct['Error Type'] = []
    
    for c in tempdf.columns:
        temp_res_dct[c] = []
    
    temp_res_dct['GT'] = []
    temp_res_dct['Correct?'] = []
    temp_res_dct['Error Type'] = []
    
    rh_cnt = 0
    nlacm_cnt = 0
    temp_cnt = 0
    
    rh_wrong = 0
    nlacm_wrong = 0
    temp_wrong = 0
    
    for i,row in enumerate(rhdf.to_dict(orient='records')):
        
        for c in rhdf.columns:
            rh_res_dct[c] += [row[c]]
        
        rh_res_dct['GT Role'] += [rh_gt[row['Role']]]
        rh_res_dct['GT Child'] += [rh_gt[row['Child']]]
        
        rh_cond = (row['Role'] in rh_gt and rh_gt[row['Role']] == row['SQL Role']) and (row['Child'] in rh_gt and rh_gt[row['Child']] == row['SQL Child'])
        if rh_cond:
            rh_cnt += 1
            rh_res_dct['Correct?'] += [1]
            
        else:
            rh_wrong += 1
            rh_res_dct['Correct?'] += [0]
    
    #check lengths
    for c in rh_res_dct:
        c_len = len(rh_res_dct[c])
        print(f"RH Column {c} Length: {c_len}")
        
    rh_accuracy = rh_cnt / (rh_cnt + rh_wrong)
    
    #NLACM results
    for i,row in enumerate(nlacmdf.to_dict(orient='records')):
        
        for c in nlacmdf.columns:
            nlacm_res_dct[c] += [row[c]]
        
        nlacm_res_dct['GT Role'] += [nlacm_rolegt[row['Role']]]
        nlacm_res_dct['GT View'] += [nlacm_viewgt[row['View']]]
        nlacm_res_dct['GT Priv'] += [nlacm_privgt[row['Priv']]]
        
        
        error_type = []
        is_correct = True
        #checking for role correctness
        if row['Role'] not in nlacm_rolegt or nlacm_rolegt[row['Role']] != row['SQL Role']:
            is_correct = False
            nlacm_res_dct['Correct?'] += [0]
            error_type += ['Role']
        
        #checking for view correctness
        if row['View'] in nlacm_viewgt:
            if nl2sql_type == 'BIRD':
                are_sub_equal = bird_subs_equal(row['SQL View'], nlacm_viewgt[row['View']], db_details)
            else:
                are_sub_equal = spider_subs_equal(row['SQL View'], nlacm_viewgt[row['View']], db_details)
        else:
            print(f"View {row['View']} was not in ground truth:\n\n{nlacm_viewgt}")
            are_sub_equal = False
        
        if not are_sub_equal:
            if is_correct:
                is_correct = False
                nlacm_res_dct['Correct?'] += [0]
            error_type += ['View']
        
        if row['Priv'] not in nlacm_privgt or nlacm_privgt[row['Priv']] != row['SQL Priv']:
            if is_correct:
                is_correct = False
                nlacm_res_dct['Correct?'] += [0]
            error_type += ['Priv']
        
        if is_correct:
            nlacm_res_dct['Correct?'] += [1]
            nlacm_cnt += 1
        else:
            nlacm_wrong += 1
        
        nlacm_res_dct['Error Type'] += [error_type]
    
    nlacm_accuracy = nlacm_cnt / (nlacm_cnt + nlacm_wrong)
    
    #Temporal results
    for i,row in enumerate(tempdf.to_dict(orient='records')):
        
        for c in tempdf.columns:
            temp_res_dct[c] += [row[c]]
        
        temp_res_dct['GT Role'] += [time_rolegt[row['Role']]]
        temp_res_dct['GT View'] += [time_viewgt[row['View']]]
        temp_res_dct['GT Cond'] += [time_privgt[row['Cond']]]
        
        
        error_type = []
        is_correct = True
        if row['Role'] not in time_rolegt or time_rolegt[row['Role']] != row['SQL Role']:
            is_correct = False
            temp_res_dct['Correct?'] += [0]
            error_type += ['Role']
        
        #checking for view correctness
        if row['View'] in nlacm_viewgt:
            if nl2sql_type == 'BIRD':
                are_sub_equal = bird_subs_equal(row['SQL View'], nlacm_viewgt[row['View']], db_details)
            else:
                are_sub_equal = spider_subs_equal(row['SQL View'], nlacm_viewgt[row['View']], db_details)
        else:
            print(f"View {row['View']} was not in ground truth:\n\n{nlacm_viewgt}")
            are_sub_equal = False
        
        if not are_sub_equal:
            if is_correct:
                is_correct = False
                nlacm_res_dct['Correct?'] += [0]
            error_type += ['View']
        
        if row['Priv'] not in time_privgt or time_privgt[row['Priv']] != row['SQL Priv']:
            if is_correct:
                is_correct = False
                temp_res_dct['Correct?'] += [0]
            error_type += ['Priv']
        
        if is_correct:
            temp_res_dct['Correct?'] += [1]
            temp_cnt += 1
        else:
            temp_wrong += 1
        
        temp_res_dct['Error Type'] += [error_type]
                
    temp_accuracy = temp_cnt / (temp_cnt + temp_wrong)
    
    rh_res = pd.DataFrame(rh_res_dct)
    nlacm_res = pd.DataFrame(nlacm_res_dct)
    temp_res = pd.DataFrame(temp_res_dct)
    
    rh_res.to_csv(os.path.join(outdir, outname + '_rh_synth_results.csv'), index=False)
    nlacm_res.to_csv(os.path.join(outdir, outname + '_nlacm_synth_results.csv'), index=False)
    temp_res.to_csv(os.path.join(outdir, outname + '_time_synth_results.csv'), index=False)
    
    print("Role Hierarchy Accuracy: {}".format(rh_accuracy))
    print("NLACM Accuracy: {}".format(nlacm_accuracy))
    print("Time Accuracy: {}".format(temp_accuracy))
    
    return rh_accuracy, nlacm_accuracy, temp_accuracy


def partial_eval_synth(outdir, outname, nl2sql_type, db_details):
    rh_file = os.path.join(outdir, outname + '_rh_fullsql.csv')
    nlacm_file = os.path.join(outdir, outname + '_privmap.csv')
    # temp_file = os.path.join(outdir, outname + 'tempmap.csv')
    
    rh_gtfile = os.path.join(outdir, outname + '_rh_gtmap.json')
    
    nlacm_rolegtfile = os.path.join(outdir, outname + '_nlacm_role_gtmap.json')
    nlacm_viewgtfile = os.path.join(outdir, outname + '_nlacm_view_gtmap.json')
    nlacm_privgtfile = os.path.join(outdir, outname + '_nlacm_priv_gtmap.json')
    
    # time_rolegtfile = os.path.join(outdir, outname + '_time_role_gtmap.json')
    # time_viewgtfile = os.path.join(outdir, outname + '_time_view_gtmap.json')
    # time_privgtfile = os.path.join(outdir, outname + '_time_cond_gtmap.json')
    
    with open(rh_gtfile, 'r') as fh:
        rh_gt = literal_eval(fh.read())
    
    with open(nlacm_rolegtfile, 'r') as fh:
        nlacm_rolegt = literal_eval(fh.read())
    with open(nlacm_viewgtfile, 'r') as fh:
        nlacm_viewgt = literal_eval(fh.read())
    with open(nlacm_privgtfile, 'r') as fh:
        nlacm_privgt = literal_eval(fh.read())
    
    # with open(time_rolegtfile, 'r') as fh:
    #     time_rolegt = literal_eval(fh.read())
    # with open(time_viewgtfile, 'r') as fh:
    #     time_viewgt = literal_eval(fh.read())
    # with open(time_privgtfile, 'r') as fh:
    #     time_privgt = literal_eval(fh.read())
    
    rhdf = pd.read_csv(rh_file)
    nlacmdf = pd.read_csv(nlacm_file)
    # tempdf = pd.read_csv(temp_file)
    
    #outputs
    rh_res_dct = {}
    nlacm_res_dct = {}
    temp_res_dct = {}
    
    for c in rhdf.columns:
        rh_res_dct[c] = []
    
    rh_res_dct['GT Role'] = []
    rh_res_dct['GT Child'] = []
    rh_res_dct['Correct?'] = []
    
    for c in nlacmdf.columns:
        nlacm_res_dct[c] = []
    
    nlacm_res_dct['GT Role'] = []
    nlacm_res_dct['GT View'] = []
    nlacm_res_dct['GT Priv'] = []
    nlacm_res_dct['Correct?'] = []
    nlacm_res_dct['Error Type'] = []
    
    # for c in tempdf.columns:
    #     temp_res_dct[c] = []
    
    # temp_res_dct['GT'] = []
    # temp_res_dct['Correct?'] = []
    # temp_res_dct['Error Type'] = []
    
    rh_cnt = 0
    nlacm_cnt = 0
    # temp_cnt = 0
    
    rh_wrong = 0
    nlacm_wrong = 0
    # temp_wrong = 0
    
    for i,row in enumerate(rhdf.to_dict(orient='records')):
        
        for c in rhdf.columns:
            rh_res_dct[c] += [row[c]]
        
        rh_res_dct['GT Role'] += [rh_gt[row['Role']]]
        rh_res_dct['GT Child'] += [rh_gt[row['Child']]]
        
        rh_cond = (row['Role'] in rh_gt and rh_gt[row['Role']] == row['SQL Role']) and (row['Child'] in rh_gt and rh_gt[row['Child']] == row['SQL Child'])
        if rh_cond:
            rh_cnt += 1
            rh_res_dct['Correct?'] += [1]
            
        else:
            rh_wrong += 1
            rh_res_dct['Correct?'] += [0]
    
    #check lengths
    for c in rh_res_dct:
        c_len = len(rh_res_dct[c])
        print(f"RH Column {c} Length: {c_len}")
        
    rh_accuracy = rh_cnt / (rh_cnt + rh_wrong)
    
    #NLACM results
    for i,row in enumerate(nlacmdf.to_dict(orient='records')):
        
        for c in nlacmdf.columns:
            nlacm_res_dct[c] += [row[c]]
        
        nlacm_res_dct['GT Role'] += [nlacm_rolegt[row['Role']]]
        nlacm_res_dct['GT View'] += [nlacm_viewgt[row['View']]]
        nlacm_res_dct['GT Priv'] += [nlacm_privgt[row['Priv']]]
        
        
        error_type = []
        is_correct = True
        #checking for role correctness
        if row['Role'] not in nlacm_rolegt or nlacm_rolegt[row['Role']] != row['SQL Role']:
            is_correct = False
            nlacm_res_dct['Correct?'] += [0]
            error_type += ['Role']
        
        #checking for view correctness
        if row['View'] in nlacm_viewgt:
            if nl2sql_type == 'BIRD':
                are_sub_equal = bird_subs_equal(row['SQL View'], nlacm_viewgt[row['View']], db_details)
            else:
                are_sub_equal = spider_subs_equal(row['SQL View'], nlacm_viewgt[row['View']], db_details)
        else:
            print(f"View {row['View']} was not in ground truth:\n\n{nlacm_viewgt}")
            are_sub_equal = False
        
        if not are_sub_equal:
            if is_correct:
                is_correct = False
                nlacm_res_dct['Correct?'] += [0]
            error_type += ['View']
        
        if row['Priv'] not in nlacm_privgt or nlacm_privgt[row['Priv']] != row['SQL Priv']:
            if is_correct:
                is_correct = False
                nlacm_res_dct['Correct?'] += [0]
            error_type += ['Priv']
        
        if is_correct:
            nlacm_res_dct['Correct?'] += [1]
            nlacm_cnt += 1
        else:
            nlacm_wrong += 1
        
        nlacm_res_dct['Error Type'] += [error_type]
    
    nlacm_accuracy = nlacm_cnt / (nlacm_cnt + nlacm_wrong)
    
    #Temporal results
    # for i,row in enumerate(tempdf.to_dict(orient='records')):
        
    #     for c in tempdf.columns:
    #         temp_res_dct[c] += [row[c]]
        
    #     temp_res_dct['GT Role'] += [time_rolegt[row['Role']]]
    #     temp_res_dct['GT View'] += [time_viewgt[row['View']]]
    #     temp_res_dct['GT Cond'] += [time_privgt[row['Cond']]]
        
        
    #     error_type = []
    #     is_correct = True
    #     if row['Role'] not in time_rolegt or time_rolegt[row['Role']] != row['SQL Role']:
    #         is_correct = False
    #         temp_res_dct['Correct?'] += [0]
    #         error_type += ['Role']
        
    #     if row['View'] not in time_viewgt or time_viewgt[row['View']] != row['SQL View']:
    #         if is_correct:
    #             is_correct = False
    #             temp_res_dct['Correct?'] += [0]
    #         error_type += ['View']
        
    #     if row['Priv'] not in time_privgt or time_privgt[row['Priv']] != row['SQL Priv']:
    #         if is_correct:
    #             is_correct = False
    #             temp_res_dct['Correct?'] += [0]
    #         error_type += ['Priv']
        
    #     if is_correct:
    #         temp_res_dct['Correct?'] += [1]
    #         temp_cnt += 1
    #     else:
    #         temp_wrong += 1
        
    #     temp_res_dct['Error Type'] += [error_type]
                
    # temp_accuracy = temp_cnt / (temp_cnt + temp_wrong)
    
    rh_res = pd.DataFrame(rh_res_dct)
    nlacm_res = pd.DataFrame(nlacm_res_dct)
    # temp_res = pd.DataFrame(temp_res_dct)
    
    rh_res.to_csv(os.path.join(outdir, outname + '_rh_synth_results.csv'), index=False)
    nlacm_res.to_csv(os.path.join(outdir, outname + '_nlacm_synth_results.csv'), index=False)
    # temp_res.to_csv(os.path.join(outdir, outname + '_time_synth_results.csv'), index=False)
    
    print("Role Hierarchy Accuracy: {}".format(rh_accuracy))
    print("NLACM Accuracy: {}".format(nlacm_accuracy))
    # print("Time Accuracy: {}".format(temp_accuracy))
    
    return rh_accuracy, nlacm_accuracy # , temp_accuracy

if __name__=='__main__':
    testout = 'rhiesys_european_football_2_test'
    testpref = 'rhiesys_test'
    
    #test data
    test_nlacm = pd.read_csv('amazon_bird_nlacm_deep_nl/amazon_bird_nlacm_deep_nl_priv_inf.csv')
    test_rh = pd.read_csv('amazon_rhl_deep_nl/amazon_rhl_deep_nl_priv_inf.csv')
    test_temp = pd.read_csv('amazon_bird_time_deep_nl/amazon_bird_time_deep_nl_base.csv')
    
    #test gt
    nlacm_gt = pd.read_csv('amazon_bird_nlacm_deep_sql/amazon_bird_nlacm_deep_sql_priv_inf.csv')
    rh_gt = pd.read_csv('amazon_rhl_deep_sql/amazon_rhl_deep_sql_priv_inf.csv')
    # temp_gt = pd.read_csv('amazon_bird_time_deep_sql/amazon_bird_time_deep_sql_base.csv')
    
    db_name = 'european_football_2'
    pg_details = {'user' : 'YOUR_USER', 'password' : 'YOUR_PASS', 'host' : 'XXXXXXXXXX', 'port' : 'XXXX', 'database' : db_name}
    pg_api = PostgresAPI(pg_details)
    
    #synthesis code
    # rolehier_synth(test_rh, rh_gt, testout, testpref)
    # nlacm_synth(test_nlacm, pg_api, testout, testpref)
    # timeacm_synth(test_temp, pg_api, testout, testpref)
    
    #ground truth generation
    rh_gtmap(test_rh, rh_gt, testout, testpref)
    nlacm_gtmap(test_nlacm, nlacm_gt, testout, testpref)
    # temp_gtmap(test_temp, temp_gt, testout, testpref)
    
    #partial evaluation
    partial_eval_synth(testout, testpref, pg_details)
    
    #work around bugs, if needed
    # rh_fromchats(test_rh, rh_gt, testout, testpref)
    
    
    
            
    
    
    
    
        
