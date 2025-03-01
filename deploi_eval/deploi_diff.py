from utils.chat_utils import OpenAIHandler, parse_elv2, parse_yn, parse_yn_llm, parse_el_sem, ParsedOut, read_chat_overwrite_parse, add_and_to_last_ordinal, fix_ending
from utils.db_utils import DBAPI, PostgresAPI, SociomeAPI
from utils.metadata_utils import BIRD_Context
from utils.context_utils import Tabcol_Context, Sociome_Context
from utils.df_utils import filter_and_select, extract_single, select_rows_and_columns
from acmdiff.bird_diff import BIRD_META, BIRD_OTHER
from abc import ABC, abstractmethod
from caching import ChatCacheObject, ChatCache
import pandas as pd
import os
import shutil
import pickle
from ast import literal_eval
import json

TEMP_RULES = ["""If you see the SQL temporal syntax, EXTRACT(DAY FROM x), this extracts a number representing the day of the month from a datetime string, not the day of the week.""",
              """You will see temporal conditions expressed in natural language using constructions such as, \"starting on the fifth, twelfth, and ninth months of the year...\". 
              In such cases, you should interpret it as a duration starting on the first time unit of each specified time point. For example, \"starting on the fifth, twelfth, and ninth months\"
              will mean starting on the first day of May, first day of December, and the first day of September."""]

handler = OpenAIHandler('gpt-4o')

def schema_by_dbtype(db_type, db_api):
    if db_type == 'BIRD':
        tc_context = Tabcol_Context(db_api)
        schema_st = tc_context.get_context()
        meta_context = BIRD_Context(BIRD_META, {'dict_path' : BIRD_OTHER})
    elif db_type == 'sociome':
        tc_context = Sociome_Context(db_api)
        schema_st = tc_context.get_context()
        meta_context = None
    else:
        #assume this is a postgres database, and extract its schema only
        tc_context = Tabcol_Context(db_api)
        schema_st = tc_context.get_context()
        meta_context = None
        
    return schema_st, meta_context

def cot_temp_chat(nl_cell, sql_cell, parse_func, outdir, outname):
    #Example call
    #ans2, raw_resp2 = handler.get_parsed_response({'NL Temp' : nl_cell, 'SQL Temp' : sql_cell}, chat, 1.0, parse_func=parse_func, write_dir=self.outdir, write_file='nlvssql_temp_role' + str(role_ind) + '_view' + str(view_ind) + '_chat.json')
    
    ex_prompt = 'Consider a NL description of a temporal condition:\n\n '
    ex_prompt += 'This role can access this view from 3am to 9pm on the days of the week: Wednesday, Friday, and Saturday'
    ex_prompt += '\n\n Consider a SQL temporal condition:\n\n '
    ex_prompt += """CREATE OR REPLACE FUNCTION is_access_allowed() RETURNS BOOLEAN AS $$
    DECLARE
        current_time TIME;
        current_day TEXT;
    BEGIN
        -- Get current time and day
        current_time := NOW();
        current_day := to_char(CURRENT_DATE, 'Day');

        -- Check if current time is between 3 and 22 and it's on specified days
        IF EXTRACT('Hour' FROM current_time) >= 3 AND EXTRACT('Hour' FROM current_time) < 22 AND current_day IN ('Tuesday', 'Wednesday', 'Thursday') THEN
            RETURN true;
        ELSE
            RETURN false;
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    """
    ex_prompt += '\n\n Does this SQL temporal condition match or fit within the temporal condition described by the NL?'
    
    ex_ans = """No, the SQL temporal condition in fact allows accesses outside the times indicated by the NL. To determine this, we reason using the following steps:
        1. Compare Times: The NL description says, "from 3am to 9pm". On the other hand, the SQL temporal condition contains conditions: 
            EXTRACT('Hour' FROM current_time) >= 3 AND EXTRACT('Hour' FROM current_time) < 22. 22 is military time for 10pm, so the SQL temporal condition would
            allow access from 3am until 10pm, not until 9pm as indicated in the NL description.
        2. Compare Days: The NL descriptions says, "Wednesday, Friday, and Saturday". On the other hand, the SQL temporal condition contains condition:
            current_day IN ('Tuesday', 'Wednesday', 'Thursday'), which means it allows access on Tuesday, and Thursday, which are not permitted according to the NL condition.
        
        Therefore, the SQL temporal condition does not match or fit within the NL temporal condition.
    """
    
    ex_prompt2 = 'Consider a NL description of a temporal condition:\n\n '
    ex_prompt2 += 'This role can access this view from 9am to 5pm on weekdays.'
    ex_prompt2 += '\n\n Consider a SQL temporal condition:\n\n '
    ex_prompt2 += """CREATE OR REPLACE FUNCTION is_access_allowed() RETURNS BOOLEAN AS $$
    DECLARE
        current_time TIME;
        current_day TEXT;
    BEGIN
        -- Get current time and day
        current_time := NOW();
        current_day := to_char(CURRENT_DATE, 'Day');

        -- Check if current time is between 9 and 5 and it's on specified days
        IF EXTRACT('Hour' FROM current_time) >= 9 AND EXTRACT('Hour' FROM current_time) < 17 AND current_day IN ('Monday', 'Wednesday', 'Thursday') THEN
            RETURN true;
        ELSE
            RETURN false;
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    """
    ex_prompt2 += '\n\n Does this SQL temporal condition match or fit within the temporal condition described by the NL?'
    
    
    ex_ans2 = """Yes, the SQL temporal condition fits within the temporal condition described by the NL. To determine this, we reason using the following steps:
        1. Compare Times: The NL description says, "from 9am to 5pm". On the other hand, the SQL temporal condition contains conditions: 
            EXTRACT('Hour' FROM current_time) >= 9 AND EXTRACT('Hour' FROM current_time) < 17. 17 is military time for 5pm, so the SQL temporal condition would
            allow access from 9am until 5pm, which fits within the NL specification of 9am to 5pm.
        2. Compare Days: The NL descriptions says, "on weekdays". On the other hand, the SQL temporal condition contains condition:
            current_day IN ('Monday', 'Wednesday', 'Thursday'), which means it allows access on Mondays, Wednesdays and Thursdays, which are all weekdays. Therefore,
            this fits within the NL specification of weekdays.
        
        Therefore, the SQL temporal condition does match or fit within the NL temporal condition.
    """
    
    prompt = 'Consider a NL description of a temporal condition:\n\n ' + nl_cell
    prompt += '\n\n Consider a SQL temporal condition:\n\n ' + sql_cell
    prompt += '\n\n Does the temporal condition described by the NL match or fit within the SQL temporal condition?'
    
    chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
    chat += [{'role' : 'user', 'content' : ex_prompt}]
    chat += [{'role' : 'assistant', 'content' : ex_ans}]
    chat += [{'role' : 'user', 'content' : ex_prompt2}]
    chat += [{'role' : 'assistant', 'content' : ex_ans2}]
    chat += [{'role' : 'user', 'content' : prompt}]
    
    final_ans, final_resp = handler.get_parsed_response({'NL Temp' : nl_cell, 'SQL Temp' : sql_cell}, chat, 1.0, parse_func, write_dir=outdir, write_file=outname)
    return final_ans, final_resp

def fewshot_temp_chat(nl_cell, sql_cell, parse_func, outdir, outname):
    ex_prompt = 'Consider a NL description of a temporal condition:\n\n '
    ex_prompt += 'This role can access this view from 3am to 9pm on the days of the week: Wednesday, Friday, and Saturday'
    ex_prompt += '\n\n Consider a SQL temporal condition:\n\n '
    ex_prompt += """CREATE OR REPLACE FUNCTION is_access_allowed() RETURNS BOOLEAN AS $$
    DECLARE
        current_time TIME;
        current_day TEXT;
    BEGIN
        -- Get current time and day
        current_time := NOW();
        current_day := to_char(CURRENT_DATE, 'Day');

        -- Check if current time is between 3 and 22 and it's on specified days
        IF EXTRACT('Hour' FROM current_time) >= 3 AND EXTRACT('Hour' FROM current_time) < 22 AND current_day IN ('Tuesday', 'Wednesday', 'Thursday') THEN
            RETURN true;
        ELSE
            RETURN false;
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    """
    ex_prompt += '\n\n Does this SQL temporal condition match or fit within the temporal condition described by the NL?'
    
    ex_ans = """No, the SQL temporal condition in fact allows accesses outside the times indicated by the NL, because the condition, "from 3am to 9pm" does not match the SQL conditions:
        "EXTRACT('Hour' FROM current_time) >= 3 AND EXTRACT('Hour' FROM current_time) < 22". Further, the SQL condition allows accesses on Tuesdays and Thursdays, which are not allowed
        according to the NL condition.
    """
    
    ex_prompt2 = 'Consider a NL description of a temporal condition:\n\n '
    ex_prompt2 += 'This role can access this view from 9am to 5pm on weekdays.'
    ex_prompt2 += '\n\n Consider a SQL temporal condition:\n\n '
    ex_prompt2 += """CREATE OR REPLACE FUNCTION is_access_allowed() RETURNS BOOLEAN AS $$
    DECLARE
        current_time TIME;
        current_day TEXT;
    BEGIN
        -- Get current time and day
        current_time := NOW();
        current_day := to_char(CURRENT_DATE, 'Day');

        -- Check if current time is between 9 and 5 and it's on specified days
        IF EXTRACT('Hour' FROM current_time) >= 9 AND EXTRACT('Hour' FROM current_time) < 17 AND current_day IN ('Monday', 'Wednesday', 'Thursday') THEN
            RETURN true;
        ELSE
            RETURN false;
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    """
    ex_prompt2 += '\n\n Does this SQL temporal condition match or fit within the temporal condition described by the NL?'
    
    ex_ans2 = """Yes, the SQL temporal condition fits within the temporal condition described by the NL. The SQL temporal condition contains conditions: 
        EXTRACT('Hour' FROM current_time) >= 9 AND EXTRACT('Hour' FROM current_time) < 17, which fits within the 9am to 5pm interval specified by the NL.
        Further, the SQL temporal condition allows accesses on Mondays, Wednesdays, and Thursdays, which fits within the NL specification of weekdays.
    """
    
    prompt = 'Consider a NL description of a temporal condition:\n\n ' + nl_cell
    prompt += '\n\n Consider a SQL temporal condition:\n\n ' + sql_cell
    prompt += '\n\n Does the temporal condition described by the NL match or fit within the SQL temporal condition?'
    
    chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
    chat += [{'role' : 'user', 'content' : ex_prompt}]
    chat += [{'role' : 'assistant', 'content' : ex_ans}]
    chat += [{'role' : 'user', 'content' : ex_prompt2}]
    chat += [{'role' : 'assistant', 'content' : ex_ans2}]
    chat += [{'role' : 'user', 'content' : prompt}]
    
    final_ans, final_resp = handler.get_parsed_response({'NL Temp' : nl_cell, 'SQL Temp' : sql_cell}, chat, 1.0, parse_func, write_dir=outdir, write_file=outname)
    return final_ans, final_resp

def sc_temp_chat(nl_cell, sql_cell, parse_func, outdir, outname, temp_val=1.0, trials=5):
    if temp_val < 1.0:
        print("WARNING: Temperature less than 1, self-consistency may not receive varied enough answers.")
    
    all_outs = []
    all_fnames = []
    chat_template = None
    for i in range(trials):
        prompt = 'Consider a NL description of a temporal condition:\n\n ' + nl_cell
        prompt += '\n\n Consider a SQL temporal condition:\n\n ' + sql_cell
        prompt += '\n\n Does the temporal condition described by the NL match or fit within the SQL temporal condition?'
        prompt += ' To clarify, I am looking for an answer along the lines of either, "Yes, the SQL temporal condition matches or fits within..." or "No, the SQL temporal condition in fact allows accesses not permitted by the NL..."'
        
        chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
        chat += [{'role' : 'user', 'content' : prompt}]
        if chat_template == None:
            chat_template = chat
        trialname = outname[:outname.index('.')] + '_trial' + str(i) + '.json'
        cur_parsed, cur_exp = handler.get_parsed_response({'NL Temp' : nl_cell, 'SQL Temp' : sql_cell}, chat, temp_val, parse_func, write_dir=outdir, write_file=trialname)
        full_fpath = os.path.join(outdir, trialname)
        all_fnames.append(full_fpath)
        all_outs.append((cur_parsed, cur_exp))
    
    val_dct = {}
    for a in all_outs:
        if a[0] not in val_dct:
            val_dct[a[0]] = 1
        else:
            val_dct[a[0]] += 1
    final_answer = max(val_dct, key=val_dct.get)
    final_exp_lst = ['Explanation ' + str(i) + ':\n\n' + a[1] for i,a in enumerate(all_outs)]
    final_exp = '\n\n'.join(final_exp_lst)
    final_inputs = {'NL Temp' : nl_cell, 'SQL Temp' : sql_cell}
    final_dets = 'Parse details for each run can be found in files: ' + ', '.join(all_fnames)
    
    sc_parse_obj = ParsedOut(final_inputs, chat_template, final_dets, final_exp, final_answer)
    with open(os.path.join(outdir, outname), 'wb') as fh:
        pickle.dump(sc_parse_obj, file=fh)
    
    return final_answer, final_exp

def persona_temp_chat(nl_cell, sql_cell, parse_func, outdir, outname):
    prompt = "You are a database administrator attempting to ensure that your database's access control implementation matches the company's access control policy. "
    prompt += "Currently, you are making sure that database accesses are permitted only during the times specified by the policy."
    
    prompt += 'Consider a NL description of a temporal condition:\n\n ' + nl_cell
    prompt += '\n\n Consider a SQL temporal condition:\n\n ' + sql_cell
    prompt += '\n\n Does the temporal condition described by the NL match or fit within the SQL temporal condition?'
    prompt += 'To clarify, I am looking for an answer along the lines of either, "Yes, the SQL temporal condition matches or fits within..." or "No, the SQL temporal condition in fact allows accesses not permitted by the NL..."'
    
    chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
    chat += [{'role' : 'user', 'content' : prompt}]
    
    final_ans, final_resp = handler.get_parsed_response({'NL Temp' : nl_cell, 'SQL Temp' : sql_cell}, chat, 1.0, parse_func, write_dir=outdir, write_file=outname)
    return final_ans, final_resp

def least_to_most(nl_cell, sql_cell, parse_func, outdir, outname):
    
    ex_prompt1_s1 = 'Consider the following PostgreSQL function that checks whether the current time satisfies a temporal condition:\n\n'
    ex_prompt1_s1 += """CREATE OR REPLACE FUNCTION is_access_allowed() RETURNS BOOLEAN AS $$
    DECLARE
        current_time TIME;
        current_day TEXT;
    BEGIN
        -- Get current time and day
        current_time := NOW();
        current_day := to_char(CURRENT_DATE, 'Day');

        -- Check if current time is between 3 and 10 and it's on specified days
        IF EXTRACT('Hour' FROM current_time) >= 3 AND EXTRACT('Hour' FROM current_time) < 22 AND current_day IN ('Monday', 'Wednesday', 'Thursday') THEN
            RETURN true;
        ELSE
            RETURN false;
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    """
    ex_prompt1_s1 += '\n\nDescribe the temporal condition this function checks. That is, during what times would this function return true?'
    
    ex_ans1_s1 = 'This function returns true when the time is between 3am until 10pm, and when the day is Monday, Wednesday, or Thursday. Specifically:\n\n'
    ex_ans1_s1 += "1. EXTRACT('Hour' FROM current_time) >= 3 AND EXTRACT('Hour' FROM current_time) < 22 means that the time must be from 3 until but not including 10pm to be true."
    ex_ans1_s1 += "\n2. current_day IN ('Monday', 'Wednesday', 'Thursday') means that the day of the week must be either Monday, Wednesday or Thursday."
    
    ex_prompt1 = 'Now, consider the following sentence/phrase describing a temporal condition on a PostgreSQL database:\n\n'
    ex_prompt1 += 'This role can access this view from 3am to 9pm on the days of the week: Wednesday, Friday, and Saturday'
    ex_prompt1 += '\n\nDoes the previous function implement a condition that would be permitted according to this sentence? Begin your answer with Yes or No.'
    ex_ans1 = """No. The function's implemented time interval is not contained within the sentence's time interval of 3am to 9pm, because it permits access from 3am to 10pm, allowing one extra hour.
    Further, the function allows accesses on Mondays and Thursdays, which are not permitted according to the sentence."""
    
    ex_prompt2_s1 = 'Consider the following PostgreSQL function that checks whether the current time satisfies a temporal condition:\n\n'
    ex_prompt2_s1 += """CREATE OR REPLACE FUNCTION is_access_allowed() RETURNS BOOLEAN AS $$
    DECLARE
        current_time TIME;
        current_day TEXT;
    BEGIN
        -- Get current time and day
        current_time := NOW();
        current_day := to_char(CURRENT_DATE, 'Day');

        -- Check if current time is between 9 and 5 and it's on specified days
        IF EXTRACT('Hour' FROM current_time) >= 9 AND EXTRACT('Hour' FROM current_time) < 17 AND current_day IN ('Tuesday', 'Wednesday', 'Friday') THEN
            RETURN true;
        ELSE
            RETURN false;
        END IF;
    END;
    $$ LANGUAGE plpgsql;
    """
    ex_prompt2_s1 += '\n\nDescribe the temporal condition this function checks. That is, during what times would this function return true?'
    
    ex_ans2_s1 = 'This function returns true when the time is between 9am until 5pm, and when the day is Tuesday, Wednesday, or Friday. Specifically:\n\n'
    ex_ans2_s1 += "1. EXTRACT('Hour' FROM current_time) >= 9 AND EXTRACT('Hour' FROM current_time) < 17 means that the time must be from 9 until but not including 5pm to be true."
    ex_ans2_s1 += "\n2. current_day IN ('Tuesday', 'Wednesday', 'Friday') means that the day of the week must be either Tuesday, Wednesday or Friday."
    
    ex_prompt2 = 'Now, consider the following sentence/phrase describing a temporal condition on a PostgreSQL database:\n\n'
    ex_prompt2 += 'This role can access this view from 9am to 5pm on weekdays.'
    ex_prompt2 += '\n\nDoes the previous function implement a condition that would be permitted according to this sentence? Begin your answer with Yes or No.'
    ex_ans2 = """Yes. The function's implemented time interval of 9am until 5pm is contained within the sentence's time interval of 3am to 9pm.
    Further, the function allows accesses on Tuesdays, Wednesdays and Fridays, which are all weekdays."""
    
    chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
    chat += [{'role' : 'user', 'content' : ex_prompt1_s1}]
    chat += [{'role' : 'assistant', 'content' : ex_ans1_s1}]
    chat += [{'role' : 'user', 'content' : ex_prompt1}]
    chat += [{'role' : 'assistant', 'content' : ex_ans1}]
    chat += [{'role' : 'user', 'content' : ex_prompt2_s1}]
    chat += [{'role' : 'assistant', 'content' : ex_ans2_s1}]
    chat += [{'role' : 'user', 'content' : ex_prompt2}]
    chat += [{'role' : 'assistant', 'content' : ex_ans2}]
    
    prompt = 'Consider the following PostgreSQL function that checks whether the current time satisfies a temporal condition:\n\n' + sql_cell + '\n\n'
    prompt += 'Describe the temporal condition this function checks. That is, during what times would this function return true?'
    # prompt += ' Is Privilege 2 a sufficient set of permissions for allowing Privilege 1? Begin your answer with YES or NO. If you are unsure, make your best guess.'
    chat += [{'role' : 'user', 'content' : prompt}]
    raw_resp1 = handler.get_response(chat, 1.0)
    chat += [{'role' : 'assistant', 'content' : raw_resp1}]
    
    #now, compare and get the yes/no
    prompt2 = 'Now, consider the following sentence/phrase describing a temporal condition on a PostgreSQL database:\n\n' + str(nl_cell) + '\n\n'
    prompt2 += 'Does the previous function implement a condition that would be permitted according to this sentence? Begin your answer with Yes or No.'
    chat += [{'role' : 'user', 'content' : prompt2}]

    ans2, raw_resp2 = handler.get_parsed_response({'NL Temp' : nl_cell, 'SQL Temp' : sql_cell}, chat, 1.0, parse_func=parse_func, write_dir=outdir, write_file=outname)
    return ans2, raw_resp2
    
    
def is_none(st1):
    if 'None' in st1 or pd.isna(st1):
        return True
    return False

def fix_temp_sentence(sentence):
    new_sent = add_and_to_last_ordinal(sentence)
    new_sent = fix_ending(new_sent)
    
    return new_sent
    

class DiffPrompt(ABC):
    @abstractmethod
    def __init__(self, nlacm, sqlacm, gtsql_acm, nlpriv, sqlpriv, gtsql_priv, db_type, db_details, outdir, cache_dir=None):
        self.nlacm = nlacm
        self.sqlacm = sqlacm
        self.gtsql_acm = gtsql_acm
        self.nlpriv = nlpriv
        self.sqlpriv = sqlpriv
        self.gtsql_priv = gtsql_priv
        self.db_type = db_type
        self.db_details = db_details
        self.outdir = outdir
        self.cache_dir = cache_dir
        if cache_dir != None:
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            self.full_dct = {}
            for f in os.listdir(cache_dir):
                cache_file = os.path.join(cache_dir, f)
                with open(cache_file, 'rb') as fh:
                    cur_obj = pickle.load(fh)
                for k in cur_obj.cache:
                    self.full_dct[k] = cur_obj.cache[k]
            
            self.full_cache = ChatCache(audit_objs=[], from_cache=self.full_dct)
            
                
    
    @abstractmethod
    def role_map(self, nl_views, sql_views):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def view_map(self, nl_roles, sql_roles):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def rv_map(self):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def is_permitted_cot(self, nl_cell, sql_cell, role_ind, view_ind):
        raise Exception("Must be implemented")
    
    @abstractmethod
    def temp_sub(self):
        raise Exception("Must be implemented")

class TempDiff(DiffPrompt):
    def __init__(self, nlacm, sqlacm, gtsql_acm, nlpriv, sqlpriv, gtsql_priv, db_type, db_details, outdir, cache_dir=None):
        super(TempDiff, self).__init__(nlacm, sqlacm, gtsql_acm, nlpriv, sqlpriv, gtsql_priv, db_type, db_details, outdir, cache_dir)
        if db_type == 'postgres':
            self.pg_api = PostgresAPI(db_details)
        elif db_type == 'sociome':
            self.pg_api = SociomeAPI(db_details)
        else:
            raise Exception(f"Database type not supported: {db_type}")
        
    
    def role_map(self, nl_roles, sql_roles):
        out_dct = {}
        for i,acm_r in enumerate(nl_roles):
            outfile = os.path.join(self.outdir, 'nlvssql_role' + str(i) + '_chat.json')
            #first, check the cache
            add_cache = False
            if self.cache_dir != None:
                if (acm_r, tuple(sql_roles)) in self.full_cache.cache:
                    chat_resp = self.full_cache.cache[(acm_r, tuple(sql_roles))]
                    raw_resp = chat_resp[-1]['content']
                    parsed = parse_elv2(raw_resp, sql_roles, 'None')
                    if parsed == None:
                        print("Got nothing! {}, {}, {}".format(parsed, raw_resp, sql_roles))
                    # answer = (parsed, raw_resp)
                    out_dct[acm_r] = {}
                    out_dct[acm_r]['SQL'] = parsed
                    out_dct[acm_r]['GT'] = sql_roles[i]
                    out_dct[acm_r]['Explanation'] = raw_resp
                    continue
                else:
                    add_cache = True
            
            if os.path.exists(outfile):
                with open(outfile, 'r') as fh:
                    chat_resp = literal_eval(fh.read())
                    
                raw_resp = chat_resp[-1]['content']
                parsed = parse_elv2(raw_resp, sql_roles, 'None')
                if parsed == None:
                    print("Got nothing! {}, {}, {}".format(parsed, raw_resp, sql_roles))
                # answer = (parsed, raw_resp)
                out_dct[acm_r] = {}
                out_dct[acm_r]['SQL'] = parsed
                out_dct[acm_r]['GT'] = sql_roles[i]
                out_dct[acm_r]['Explanation'] = raw_resp
                
                if add_cache:
                    self.full_cache.cache[(acm_r, tuple(sql_roles))] = chat_resp
                    
                
            else:
                chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
                prompt = 'Consider the following list of roles in a database:\n\n' + str(sql_roles) + '\n\n'
                prompt += 'Consider the following phrase describing a role: ' + acm_r + '. '
                prompt += 'Which database role from the list does this phrase most likely describe? Begin your answer with this role. If none of them match, begin your answer with None.'
                prompt += ' If you are unsure, make your best guess.'
                
                chat += [{'role' : 'user', 'content' : prompt}]
                # raw_resp = handler.get_response(chat, 1.0, write_dir=self.outdir, write_file='nlvssql_role' + str(i) + '_chat.json')
                # chat += [{'role' : 'assistant', 'content' : raw_resp}]
                
                # parsed, raw_resp = handler.select_from_lst(chat, acm_r, sql_roles.tolist(), 1.0, none_option="None", write_dir=self.outdir, write_file='nlvssql_role' + str(i) + '_parsedchat.pkl')
                parsed, raw_resp = handler.get_parsed_response({'element' : acm_r, 'list' : sql_roles}, chat, 1.0, parse_el_sem, write_dir=self.outdir, write_file='nlvssql_role' + str(i) + '_parsedchat.pkl')
                chat += [{'role' : 'assistant', 'content' : raw_resp}]
                
                # parsed = parse_elv2(raw_resp, sql_roles, 'None')
                # if parsed == None:
                #     print("Got nothing! {}, {}, {}".format(parsed, raw_resp, sql_roles))
                # answer = (parsed, raw_resp)
                out_dct[acm_r] = {}
                out_dct[acm_r]['SQL'] = parsed
                out_dct[acm_r]['GT'] = sql_roles[i]
                out_dct[acm_r]['Explanation'] = raw_resp
                
                if add_cache:
                    self.full_cache.cache[(acm_r, tuple(sql_roles))] = chat
        
        if self.cache_dir:
            outcache = os.path.join(self.cache_dir, 'rolecache.pkl')
            with open(outcache, 'wb') as fh:
                pickle.dump(self.full_cache, file=fh)
        
        return out_dct
            
    
    def view_map(self, nl_views, sql_views, db_type='BIRD'):
        out_dct = {}
        schema_st, meta_context = schema_by_dbtype(db_type, self.pg_api)
        
        for i, db_v in enumerate(sql_views):
            # outfile = os.path.join(outdir, outname + '_sqlvsnl_view' + str(i) + '.txt')
            outchat = os.path.join(self.outdir, 'sqlvsnl_view' + str(i) + '_chat.json')
            
            #first, check the cache
            add_cache = False
            if self.cache_dir != None:
                if (db_v, tuple(nl_views)) in self.full_cache.cache:
                    chat_resp = self.full_cache.cache[(db_v, tuple(nl_views))]
                    raw_resp = chat_resp[-1]['content']
                    parsed = parse_elv2(raw_resp, nl_views, 'None')
                    if parsed == None:
                        print("Got nothing! {}, {}, {}".format(parsed, raw_resp, nl_views))
                    
                    out_dct[db_v] = {}
                    out_dct[db_v]['NL'] = parsed
                    out_dct[db_v]['GT'] = nl_views[i]
                    out_dct[db_v]['Explanation'] = raw_resp
                    
                    if not os.path.exists(outchat):
                        with open(outchat, 'w+') as fh:
                            print(chat_resp, file=fh)
                    
                    continue
                else:
                    add_cache = True
            
            if os.path.exists(outchat):
                with open(outchat, 'r') as fh:
                    ents = literal_eval(fh.read())
                raw_resp = ents[-1]['content']
                parsed = parse_elv2(raw_resp, nl_views, 'None')
                
                out_dct[db_v] = {}
                out_dct[db_v]['NL'] = parsed
                out_dct[db_v]['GT'] = nl_views[i]
                out_dct[db_v]['Explanation'] = raw_resp
                
                if add_cache:
                    self.full_cache.cache[(db_v, tuple(nl_views))] = ents
                
            else:
                
                if meta_context != None:
                    meta_st = meta_context.get_context(db_v)
                else:
                    meta_st = ''
                
                chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
                
                prompt1 = 'Consider the following SQL for a table or view: ' + db_v + '. '
                prompt1 += schema_st + meta_st
                prompt1 += 'Explain what this query does in plain english.'
                chat += [{'role' : 'user', 'content' : prompt1}]
                raw_resp1 = handler.get_response(chat, 1.0, write_dir=self.outdir, write_file='sqlvsnl_view' + str(i) + '_step1chat.json')
                chat += [{'role' : 'assistant', 'content' : raw_resp1}]
                
                prompt2 = 'Consider the following list of descriptions of views in a database:\n\n' + str(nl_views) + '\n\n'
                prompt2 += 'Which description from the list does the given query most likely describe? Begin your answer with this description. If none of them match, begin your answer with None.'
                prompt2 += ' If you are unsure, make your best guess.'
                
                chat += [{'role' : 'user', 'content' : prompt2}]
                # raw_resp = handler.get_response(chat, 1.0, write_dir=self.outdir, write_file='sqlvsnl_view' + str(i) + '_chat.json')
                parsed, raw_resp = handler.get_parsed_response({'element' : db_v, 'list' : nl_views}, chat, 1.0, parse_el_sem, write_dir=self.outdir, write_file='sqlvsnl_view' + str(i) + '_parsedchat.pkl')
                # parsed, raw_resp = handler.select_from_lst(chat, db_v, nl_views, 1.0, none_option="None", write_dir=self.outdir, write_file='sqlvsnl_view' + str(i) + '_parsedchat.pkl')
                
                chat += [{'role' : 'assistant', 'content' : raw_resp}]
                # parsed = parse_elv2(raw_resp, nl_views, 'None')
                # print("Parsed Answer: {}".format(parsed))
                
                out_dct[db_v] = {}
                out_dct[db_v]['NL'] = parsed
                out_dct[db_v]['GT'] = nl_views[i]
                out_dct[db_v]['Explanation'] = raw_resp
                
                if add_cache:
                    self.full_cache.cache[(db_v, tuple(nl_views))] = chat
        
        if self.cache_dir:
            outcache = os.path.join(self.cache_dir, 'viewcache.pkl')
            with open(outcache, 'wb') as fh:
                pickle.dump(self.full_cache, file=fh)
        
        return out_dct
    
    def rv_map(self, nl2sql_type):
        nldf = pd.read_csv(self.nlacm)
        sqldf = pd.read_csv(self.sqlacm)
        
        nl_roles = nldf['Role']
        sql_roles = sqldf['Role']
        
        nl_views = [c for c in nldf.columns if c != 'Role']
        sql_views = [c for c in sqldf.columns if c != 'Role']
        
        nl2sql_roles = self.role_map(nl_roles, sql_roles)
        sql2nl_views = self.view_map(nl_views, sql_views, db_type=nl2sql_type)
        
        rout_schema = ['NL Role', 'SQL Role', 'GT SQL Role', 'Explanation']
        rout_dct = {}
        for o in rout_schema:
            rout_dct[o] = []
        
        vout_schema = ['NL View', 'SQL View', 'GT NL View', 'Explanation']
        vout_dct = {}
        for o in vout_schema:
            vout_dct[o] = []
        
        for nlr in nl2sql_roles:
            rout_dct['NL Role'].append(nlr)
            rout_dct['SQL Role'].append(nl2sql_roles[nlr]['SQL'])
            rout_dct['GT SQL Role'].append(nl2sql_roles[nlr]['GT'])
            rout_dct['Explanation'].append(nl2sql_roles[nlr]['Explanation'])
        
        rout_df = pd.DataFrame(rout_dct)
        routpath = os.path.join(self.outdir, 'rolemap.csv')
        rout_df.to_csv(routpath, index=False)
        
        self.rout_df = rout_df
        print("Stored Role Out Dataframe")
        
        for sqlv in sql2nl_views:
            vout_dct['SQL View'].append(sqlv)
            vout_dct['NL View'].append(sql2nl_views[sqlv]['NL'])
            vout_dct['GT NL View'].append(sql2nl_views[sqlv]['GT'])
            vout_dct['Explanation'].append(sql2nl_views[sqlv]['Explanation'])
        
        vout_df = pd.DataFrame(vout_dct)
        voutpath = os.path.join(self.outdir, 'viewmap.csv')
        vout_df.to_csv(voutpath, index=False)
        
        self.vout_df = vout_df
        print("Stored View Out Dataframe")
    
    def is_permitted_cot(self, nl_cell, sql_cell, role_ind, view_ind, parse_func=parse_yn, prompt_type='betterl2m'):
        outchat = os.path.join(self.outdir, 'nlvssql_temp_role' + str(role_ind) + '_view' + str(view_ind) + '_chat.json')
        
        #first, check the cache
        add_cache = False
        if self.cache_dir != None:
            if (nl_cell, sql_cell) in self.full_cache.cache:
                chat_resp = self.full_cache.cache[(nl_cell, sql_cell)]
                raw_resp1 = chat_resp[-1]['content']
                ans2 = parse_func(raw_resp1)
                
                return ans2, raw_resp1
            else:
                add_cache = True
            
        
        if os.path.exists(outchat):
            ans2, raw_resp1 = read_chat_overwrite_parse(outchat, parse_func)
            
            #TODO: fix the caching logic below if needed
            # if add_cache:
            #     self.full_cache.cache[(nl_cell, sql_cell)] = ents
            
            # if self.cache_dir:
            #     with open(os.path.join(self.cache_dir, 'tempcache.pkl'), 'wb') as fh:
            #         pickle.dump(self.full_cache, file=fh)
            
            return ans2, raw_resp1
        else:
            cur_writefile = 'nlvssql_temp_role' + str(role_ind) + '_view' + str(view_ind) + '_chat.json'
            
            #NOTE: method block. Change prompting strategy here--exactly ONE of the below should be uncommented at any
            #point in time.
            if prompt_type == 'betterl2m':
                ans2, raw_resp2 = least_to_most(nl_cell, sql_cell, parse_func, self.outdir, cur_writefile)
            elif prompt_type == 'realcot':
                ans2, raw_resp2 = cot_temp_chat(nl_cell, sql_cell, parse_func, self.outdir, cur_writefile)
            elif prompt_type == 'simplefewshot':
                ans2, raw_resp2 = fewshot_temp_chat(nl_cell, sql_cell, parse_func, self.outdir, cur_writefile)
            elif prompt_type == 'simplepersona':
                ans2, raw_resp2 = persona_temp_chat(nl_cell, sql_cell, parse_func, self.outdir, cur_writefile)
            
            #TODO: skipping self-consistency for now, it will not even run. 
            # ans2, raw_resp2 = sc_temp_chat(nl_cell, sql_cell, parse_func, self.outdir, cur_writefile)
    
            # ans2, raw_resp2 = handler.get_parsed_response({'NL Temp' : nl_cell, 'SQL Temp' : sql_cell}, chat, 1.0, parse_func=parse_func, write_dir=self.outdir, write_file='nlvssql_temp_role' + str(role_ind) + '_view' + str(view_ind) + '_chat.json')
            # ans2 = parse_yn(raw_resp2)
            with open(os.path.join(self.outdir, cur_writefile), 'rb') as fh:
                chat_obj = pickle.load(fh)
            chat = chat_obj.chat
            if type(chat) == list:
                chat += [{'role' : 'assistant', 'content' : raw_resp2}]
            
            if add_cache:
                self.full_cache.cache[(nl_cell, sql_cell)] = chat
            
            if self.cache_dir:

                with open(os.path.join(self.cache_dir, 'tempcache.pkl'), 'wb') as fh:
                    pickle.dump(self.full_cache, file=fh)
            
            return ans2, raw_resp2
    
    def privs_permitted_cot(self, nl_cell, sql_cell, role_ind, view_ind, parse_func=parse_yn):
        outchat = os.path.join(self.outdir, 'nlvssql_priv_role' + str(role_ind) + '_view' + str(view_ind) + '_chat.json')
        
        #first, check the cache
        add_cache = False
        if self.cache_dir != None:
            if (nl_cell, sql_cell) in self.full_cache.cache:
                chat_resp = self.full_cache.cache[(nl_cell, sql_cell)]
                raw_resp1 = chat_resp[-1]['content']
                ans2 = parse_func(raw_resp1)
                
                return ans2, raw_resp1
            else:
                add_cache = True
            
        
        if os.path.exists(outchat):
            ans2, raw_resp1 = read_chat_overwrite_parse(outchat, parse_func)
            
            #TODO: fix the caching logic below if needed
            # if add_cache:
            #     self.full_cache.cache[(nl_cell, sql_cell)] = ents
            
            # if self.cache_dir:
            #     with open(os.path.join(self.cache_dir, 'tempcache.pkl'), 'wb') as fh:
            #         pickle.dump(self.full_cache, file=fh)
            
            return ans2, raw_resp1
        else:
            sql_example = "['INSERT', 'UPDATE']"
            chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
            prompt = 'Consider the following sentence/phrase describing Postgres database permissions for a role on a table: Privilege 1-' + nl_cell + '.'
            prompt += 'Consider the following list of allowed PostgreSQL operators for the same role on the same table: Privilege 2-' + sql_example + '.'
            prompt += 'In the same format as Privilege 2, give me the smallest list of SQL operators that need to be allowed to satisfy Privilege 1.'
            prompt += ' Choose SQL operators from the following: SELECT, UPDATE, INSERT, CREATE, DELETE, and GRANT.'
            prompt += ' Also note that in this case, the SQL operator GRANT is shorthand for WITH GRANT OPTION--it indicates that a role can pass down any of its permitted operations.'
            chat += [{'role' : 'user', 'content' : prompt}]
            raw_resp1 = handler.get_response(chat, 1.0)
            
            chat += [{'role' : 'assistant', 'content' : raw_resp1}]
            prompt2i = 'Based on your answer, consider the following list of SQL operators:\n\n' + sql_cell + '\n\nAre the operators in this list as or less permissive than the ones that would satisfy Privilege 1?'
            prompt2i += ' Just to clarify, I am asking for an answer of the form, "Yes, they are as or less permissive..." or "No, they are in fact more permissive...".'
            chat += [{'role' : 'user', 'content' : prompt2i}]
            raw_resp_exp = handler.get_response(chat, 1.0)
            prompt2 = 'Summarize your answer as Yes or No. If you are unsure, make your best guess.'
            chat += [{'role' : 'assistant', 'content' : raw_resp_exp}]
            chat += [{'role' : 'user', 'content' : prompt2}]
            
            ans2, raw_resp2 = handler.get_parsed_response({'NL Priv' : nl_cell, 'SQL Priv' : sql_cell}, chat, 1.0, parse_func=parse_func, write_dir=self.outdir, write_file='nlvssql_priv_role' + str(role_ind) + '_view' + str(view_ind) + '_chat.json')
            # ans2 = parse_yn(raw_resp2)
            chat += [{'role' : 'assistant', 'content' : raw_resp2}]
            
            if add_cache:
                self.full_cache.cache[(nl_cell, sql_cell)] = chat
            
            if self.cache_dir:

                with open(os.path.join(self.cache_dir, 'privcache.pkl'), 'wb') as fh:
                    pickle.dump(self.full_cache, file=fh)
            
            return ans2, raw_resp2
        
    def is_permitted_rules(self, nl_cell, sql_cell, role_ind, view_ind):
        outchat = os.path.join(self.outdir, 'nlvssql_temp_role' + str(role_ind) + '_view' + str(view_ind) + '_chat.json')
        
        #first, check the cache
        add_cache = False
        if self.cache_dir != None:
            if (nl_cell, sql_cell) in self.full_cache.cache:
                chat_resp = self.full_cache.cache[(nl_cell, sql_cell)]
                raw_resp1 = chat_resp[-1]['content']
                ans2 = parse_yn(raw_resp1)
                
                return ans2, raw_resp1
            else:
                add_cache = True
            
        
        if os.path.exists(outchat):
            ans2, raw_resp1 = read_chat_overwrite_parse(outchat, parse_yn_llm)
            
            #TODO: fix the caching logic below if needed
            # if add_cache:
            #     self.full_cache.cache[(nl_cell, sql_cell)] = ents
            
            # if self.cache_dir:
            #     with open(os.path.join(self.cache_dir, 'tempcache.pkl'), 'wb') as fh:
            #         pickle.dump(self.full_cache, file=fh)
            
            return ans2, raw_resp1
        else:
            chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
            prompt = 'Consider the following PostgreSQL function that checks whether the current time satisfies a temporal condition:\n\n' + sql_cell + '\n\n'
            prompt += 'Describe the temporal condition this function checks. That is, during what times would this function return true?'
            # prompt += ' Is Privilege 2 a sufficient set of permissions for allowing Privilege 1? Begin your answer with YES or NO. If you are unsure, make your best guess.'
            chat += [{'role' : 'user', 'content' : prompt}]
            raw_resp1 = handler.get_response(chat, 1.0)
            chat += [{'role' : 'assistant', 'content' : raw_resp1}]
            
            #now, compare and get the yes/no
            prompt2 = 'Now, consider the following sentence/phrase describing a temporal condition on a PostgreSQL database:\n\n' + str(nl_cell) + '\n\n'
            prompt2 += 'Does the previous function implement the condition described by this sentence? '
            temp_rules_lst = ['Tip ' + str(i) + ': ' + st for i,st in enumerate(TEMP_RULES)]
            temp_rules_st = '\n'.join(temp_rules_lst)
            prompt2 += f'Consider the following tips when answering this question: \n\n {temp_rules_st} \n\n Begin your answer with Yes or No.'
            chat += [{'role' : 'user', 'content' : prompt2}]
    
            ans2, raw_resp2 = handler.get_parsed_response({'NL Temp' : nl_cell, 'SQL Temp' : sql_cell}, chat, 1.0, parse_func=parse_yn_llm, write_dir=self.outdir, write_file='nlvssql_temp_role' + str(role_ind) + '_view' + str(view_ind) + '_chat.json')
            # ans2 = parse_yn(raw_resp2)
            chat += [{'role' : 'assistant', 'content' : raw_resp2}]
            
            if add_cache:
                self.full_cache.cache[(nl_cell, sql_cell)] = chat
            
            if self.cache_dir:

                with open(os.path.join(self.cache_dir, 'tempcache.pkl'), 'wb') as fh:
                    pickle.dump(self.full_cache, file=fh)
            
            return ans2, raw_resp2
    
    def extract_gt(self, gtdf, sql_cell, sql_role, sql_view):
        #in this case, the ground truth will only be different from the sql
        #if the strings don't match, and vice versa
        gt_sub = filter_and_select(gtdf, 'Role', [sql_role], [sql_view])
        gt_rind, gt_cell = extract_single(gt_sub)
        
        if sql_cell != gt_cell:
            return 'No'
        return 'Yes'
    
    def temp_sub(self):
        out_schema = ['NL Role', 'SQL Role', 'NL View', 'SQL View', 'NL Temp', 'SQL Temp', 'GT', 'Response', 'Parsed']
        out_dct = {}
        for o in out_schema:
            out_dct[o] = []
        
        nldf = pd.read_csv(self.nlacm)
        sqldf = pd.read_csv(self.sqlacm)
        
        for i,role_row in enumerate(self.rout_df.to_dict(orient='records')):
            r_1 = role_row['NL Role']
            r_2 = role_row['SQL Role']
            if is_none(r_1) or is_none(r_2):
                continue
                
            for j,view_row in enumerate(self.vout_df.to_dict(orient='records')):
                v_1 = view_row['NL View']
                v_2 = view_row['SQL View']
                if is_none(v_1) or is_none(v_2):
                    continue
                
                #the below are pandas Series, not dataframes
                #and they should only have one value
                nlrows = nldf[nldf['Role'] == r_1][v_1].to_list()
                sqlrows = sqldf[sqldf['Role'] == r_2][v_2].to_list()
                
                #now, compare every pair of cells
                for nlr in nlrows:
                    for sqlr in sqlrows:
                        nl_cell = nlr
                        sql_cell = sqlr
                        cur_gt = self.extract_gt(sql_cell, i, v_2)
                        parsed, exp = self.is_permitted_cot(nl_cell, sql_cell, i, j)
                        out_dct['NL Role'].append(r_1)
                        out_dct['SQL Role'].append(r_2)
                        out_dct['NL View'].append(v_1)
                        out_dct['SQL View'].append(v_2)
                        out_dct['NL Temp'].append(nl_cell)
                        out_dct['SQL Temp'].append(sql_cell)
                        out_dct['GT'].append(cur_gt)
                        out_dct['Response'].append(exp)
                        out_dct['Parsed'].append(parsed)
        
        out_df = pd.DataFrame(out_dct)
        outpath = os.path.join(self.outdir, 'tempmap.csv')
        out_df.to_csv(outpath, index=False)
        
        self.priv_df = out_df
        print("Stored Temporal Constraint Dataframe")
    
    def temp_sub_src(self, prompt_type):
        out_schema = ['NL Role', 'SQL Role', 'NL View', 'SQL View', 'NL Temp', 'SQL Temp', 'GT', 'Response', 'Parsed']
        out_dct = {}
        for o in out_schema:
            out_dct[o] = []
        
        nldf = pd.read_csv(self.nlacm)
        sqldf = pd.read_csv(self.sqlacm)
        gtdf = pd.read_csv(self.gtsql_acm)
        
        """
        role pkl--write_dir=self.outdir, write_file='nlvssql_role' + str(i) + '_parsedchat.pkl'
        """
        
        """
        view pkl--write_dir=self.outdir, write_file='sqlvsnl_view' + str(i) + '_parsedchat.pkl'
        """
        
        view_outs = [os.path.join(self.outdir, f) for f in os.listdir(self.outdir) if f.endswith('_parsedchat.pkl') and 'sqlvsnl_view' in f]
        role_outs = [os.path.join(self.outdir, f) for f in os.listdir(self.outdir) if f.endswith('_parsedchat.pkl') and 'nlvssql_role' in f]
        
        for r_file in role_outs:
            with open(r_file, 'rb') as fh:
                r_obj = pickle.load(fh)
            
            nl_role = r_obj.inputs['element']
            sql_role = r_obj.parsed
            
            for v_file in view_outs:
                with open(v_file, 'rb') as fh:
                    v_obj = pickle.load(fh)
                
                sql_view = v_obj.inputs['element']
                sql_view_ind = sqldf.columns.get_loc(sql_view)
                nl_view = v_obj.parsed
                
                nl_sub = filter_and_select(nldf, 'Role', [nl_role], [nl_view])
                nl_role_ind, nl_cell = extract_single(nl_sub)
                
                sql_sub = filter_and_select(sqldf, 'Role', [sql_role], [sql_view])
                sql_role_ind, sql_cell = extract_single(sql_sub)
                
                # outfile = os.path.join(self.outdir, 'nlvssql_temp_role' + str(nl_role_ind) + '_view' + str(sql_view_ind) + '_parsedchat.pkl')
                parsed, exp = self.is_permitted_cot(nl_cell, sql_cell, nl_role_ind, sql_view_ind, parse_func=parse_yn, prompt_type=prompt_type)
                # parsed, exp = self.is_permitted_rules(nl_cell, sql_cell, nl_role_ind, sql_view_ind)
                
                cur_gt = self.extract_gt(gtdf, sql_cell, sql_role, sql_view)
                
                #now, add dataframe attributes
                out_dct['NL Role'].append(nl_role)
                out_dct['SQL Role'].append(sql_role)
                out_dct['NL View'].append(nl_view)
                out_dct['SQL View'].append(sql_view)
                out_dct['NL Temp'].append(nl_cell)
                out_dct['SQL Temp'].append(sql_cell)
                out_dct['GT'].append(cur_gt)
                out_dct['Response'].append(exp)
                out_dct['Parsed'].append(parsed)
                
        
        out_df = pd.DataFrame(out_dct)
        outpath = os.path.join(self.outdir, 'tempmap.csv')
        out_df.to_csv(outpath, index=False)
        
        self.time_df = out_df
        print("Stored Temporal Constraint Dataframe")
    
    def priv_sub_src(self):
        out_schema = ['NL Role', 'SQL Role', 'NL View', 'SQL View', 'NL Priv', 'SQL Priv', 'GT', 'Response', 'Parsed']
        out_dct = {}
        for o in out_schema:
            out_dct[o] = []
        
        nldf = pd.read_csv(self.nlpriv)
        sqldf = pd.read_csv(self.sqlpriv)
        gtdf = pd.read_csv(self.gtsql_priv)
        
        """
        role pkl--write_dir=self.outdir, write_file='nlvssql_role' + str(i) + '_parsedchat.pkl'
        """
        
        """
        view pkl--write_dir=self.outdir, write_file='sqlvsnl_view' + str(i) + '_parsedchat.pkl'
        """
        
        view_outs = [os.path.join(self.outdir, f) for f in os.listdir(self.outdir) if f.endswith('_parsedchat.pkl') and 'sqlvsnl_view' in f]
        role_outs = [os.path.join(self.outdir, f) for f in os.listdir(self.outdir) if f.endswith('_parsedchat.pkl') and 'nlvssql_role' in f]
        
        for r_file in role_outs:
            with open(r_file, 'rb') as fh:
                r_obj = pickle.load(fh)
            
            nl_role = r_obj.inputs['element']
            sql_role = r_obj.parsed
            
            for v_file in view_outs:
                with open(v_file, 'rb') as fh:
                    v_obj = pickle.load(fh)
                
                sql_view = v_obj.inputs['element']
                sql_view_ind = sqldf.columns.get_loc(sql_view)
                nl_view = v_obj.parsed
                
                nl_sub = filter_and_select(nldf, 'Role', [nl_role], [nl_view])
                nl_role_ind, nl_cell = extract_single(nl_sub)
                
                sql_sub = filter_and_select(sqldf, 'Role', [sql_role], [sql_view])
                sql_role_ind, sql_cell = extract_single(sql_sub)
                
                # outfile = os.path.join(self.outdir, 'nlvssql_temp_role' + str(nl_role_ind) + '_view' + str(sql_view_ind) + '_parsedchat.pkl')
                parsed, exp = self.privs_permitted_cot(nl_cell, sql_cell, nl_role_ind, sql_view_ind, parse_func=parse_yn)
                # parsed, exp = self.is_permitted_rules(nl_cell, sql_cell, nl_role_ind, sql_view_ind)
                
                cur_gt = self.extract_gt(gtdf, sql_cell, sql_role, sql_view)
                
                #now, add dataframe attributes
                out_dct['NL Role'].append(nl_role)
                out_dct['SQL Role'].append(sql_role)
                out_dct['NL View'].append(nl_view)
                out_dct['SQL View'].append(sql_view)
                out_dct['NL Priv'].append(nl_cell)
                out_dct['SQL Priv'].append(sql_cell)
                out_dct['GT'].append(cur_gt)
                out_dct['Response'].append(exp)
                out_dct['Parsed'].append(parsed)
                
        
        out_df = pd.DataFrame(out_dct)
        outpath = os.path.join(self.outdir, 'privmap.csv')
        out_df.to_csv(outpath, index=False)
        
        self.priv_df = out_df
        print("Stored Permission Constraint Dataframe")

def mismatch(respath):
    df = pd.read_csv(respath)
    
    if 'Parsed' in df.columns:
        matches = df[df['Parsed'] == df['GT']]
        mismatches = df[df['Parsed'] != df['GT']]
    elif 'SQL Role' in df.columns:
        matches = df[df['SQL Role'] == df['GT SQL Role']]
        mismatches = df[df['SQL Role'] != df['GT SQL Role']]
    elif 'NL View' in df.columns:
        matches = df[df['NL View'] == df['GT NL View']]
        mismatches = df[df['NL View'] != df['GT NL View']]
    
    return matches, mismatches

def temp_full_eval(resdir):
    # basedf = pd.read_csv(base_path)
    # col_len = basedf.shape[0]
    
    out_schema = ['Role', 'View', 'Privilege', 'Error Type', 'Error Stage']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = {}
    
    rpath = os.path.join(resdir, 'rolemap.csv')
    vpath = os.path.join(resdir, 'viewmap.csv')
    privpath = os.path.join(resdir, 'tempmap.csv')
    
    privdf = pd.read_csv(privpath)
    
    r_good, r_miss = mismatch(rpath)
    v_good, v_miss = mismatch(vpath)
    # p_good, p_miss = mismatch(privpath)
    
    pv_good = pd.merge(privdf, v_good, left_on=['NL View', 'SQL View'], right_on=['NL View', 'SQL View'])
    prv_good = pd.merge(pv_good, r_good, left_on=['NL Role', 'SQL Role'], right_on=['NL Role', 'SQL Role'])
    # priv_good = prv_good[prv_good['Parsed'] == prv_good['GT']]
    priv_good = prv_good[(prv_good['Parsed'].str.lower() == prv_good['GT'].str.lower()) & (prv_good['GT'] == 'Yes')]
    priv_good.to_csv(os.path.join(resdir, 'results_temp_tps.csv'), index=False)
    
    priv_good_tn = prv_good[(prv_good['Parsed'].str.lower() == prv_good['GT'].str.lower()) & (prv_good['GT'] == 'No')]
    priv_good_tn.to_csv(os.path.join(resdir, 'results_temp_tns.csv'), index=False)
    
    # priv_bad_fn = prv_good[(prv_good['Parsed'] != prv_good['GT']) & (prv_good['GT'] == 'Yes')]
    priv_bad_fn = prv_good[(prv_good['Parsed'].str.lower() != prv_good['GT'].str.lower()) & (prv_good['GT'] == 'Yes')]
    priv_bad_fn.to_csv(os.path.join(resdir, 'results_temp_fns.csv'), index=False)
    
    priv_bad_fp = prv_good[(prv_good['Parsed'].str.lower() != prv_good['GT'].str.lower()) & (prv_good['GT'] != 'Yes')]
    priv_bad_fp.to_csv(os.path.join(resdir, 'results_temp_fps.csv'), index=False)
    
    pv_bad = pd.merge(privdf, v_miss, left_on=['NL View', 'SQL View'], right_on=['NL View', 'SQL View'])
    prv_bad = pd.merge(pv_bad, r_miss, left_on=['NL Role', 'SQL Role'], right_on=['NL Role', 'SQL Role'])
    prv_bad.to_csv(os.path.join(resdir, 'results_rv_fps.csv'), index=False)
    
    tp = priv_good.shape[0]
    tn = priv_good_tn.shape[0]
    fn = priv_bad_fn.shape[0] + prv_bad.shape[0] #RV correct but priv wrong + RV wrong
    fp = priv_bad_fp.shape[0]
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 Score: {}".format(f1))
    stat_dct = {'TP' : [tp], 'FN' : [fn], 'FP' : fp, 'Precision' : precision, 'Recall' : recall, 'F1' : f1}
    stat_df = pd.DataFrame(stat_dct)
    stat_df.to_csv(os.path.join(resdir, 'final_stats.csv'), index=False)

def priv_full_eval(resdir):
    # basedf = pd.read_csv(base_path)
    # col_len = basedf.shape[0]
    
    out_schema = ['Role', 'View', 'Privilege', 'Error Type', 'Error Stage']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = {}
    
    rpath = os.path.join(resdir, 'rolemap.csv')
    vpath = os.path.join(resdir, 'viewmap.csv')
    privpath = os.path.join(resdir, 'privmap.csv')
    
    privdf = pd.read_csv(privpath)
    
    r_good, r_miss = mismatch(rpath)
    v_good, v_miss = mismatch(vpath)
    # p_good, p_miss = mismatch(privpath)
    
    pv_good = pd.merge(privdf, v_good, left_on=['NL View', 'SQL View'], right_on=['NL View', 'SQL View'])
    prv_good = pd.merge(pv_good, r_good, left_on=['NL Role', 'SQL Role'], right_on=['NL Role', 'SQL Role'])
    # priv_good = prv_good[prv_good['Parsed'] == prv_good['GT']]
    priv_good = prv_good[(prv_good['Parsed'].str.lower() == prv_good['GT'].str.lower()) & (prv_good['GT'] == 'Yes')]
    priv_good.to_csv(os.path.join(resdir, 'results_priv_tps.csv'), index=False)
    
    priv_good_tn = prv_good[(prv_good['Parsed'].str.lower() == prv_good['GT'].str.lower()) & (prv_good['GT'] == 'No')]
    priv_good_tn.to_csv(os.path.join(resdir, 'results_priv_tns.csv'), index=False)
    
    # priv_bad_fn = prv_good[(prv_good['Parsed'] != prv_good['GT']) & (prv_good['GT'] == 'Yes')]
    priv_bad_fn = prv_good[(prv_good['Parsed'].str.lower() != prv_good['GT'].str.lower()) & (prv_good['GT'] == 'Yes')]
    priv_bad_fn.to_csv(os.path.join(resdir, 'results_priv_fns.csv'), index=False)
    
    priv_bad_fp = prv_good[(prv_good['Parsed'].str.lower() != prv_good['GT'].str.lower()) & (prv_good['GT'] != 'Yes')]
    priv_bad_fp.to_csv(os.path.join(resdir, 'results_priv_fps.csv'), index=False)
    
    pv_bad = pd.merge(privdf, v_miss, left_on=['NL View', 'SQL View'], right_on=['NL View', 'SQL View'])
    prv_bad = pd.merge(pv_bad, r_miss, left_on=['NL Role', 'SQL Role'], right_on=['NL Role', 'SQL Role'])
    prv_bad.to_csv(os.path.join(resdir, 'results_rv_fps.csv'), index=False)
    
    tp = priv_good.shape[0]
    tn = priv_good_tn.shape[0]
    fn = priv_bad_fn.shape[0] + prv_bad.shape[0] #RV correct but priv wrong + RV wrong
    fp = priv_bad_fp.shape[0]
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 Score: {}".format(f1))
    stat_dct = {'TP' : [tp], 'FN' : [fn], 'FP' : fp, 'Precision' : precision, 'Recall' : recall, 'F1' : f1}
    stat_df = pd.DataFrame(stat_dct)
    stat_df.to_csv(os.path.join(resdir, 'priv_final_stats.csv'), index=False)

def get_spider_questions(db_name, num_range):
    nl2sql_dct = {}
    sp_path = os.path.expanduser('~/diagnostic-robustness-text-to-sql/data/Spider-dev/questions.json')
    with open(sp_path, 'r') as fh:
        ents = json.load(fh)
    
    db_ents = [e for e in ents if e['db_id'] == db_name]
    #we have to do this to make sure the order is the same on every run
    db_ents = sorted(db_ents, key=lambda e: e['question'])
    if db_ents == []:
        raise Exception(f"NO spider entries for database: {db_name}")
    
    for n in num_range:
        if n >= len(db_ents):
            print(f"Index {n} did not exist, ignoring")
        else:
            cur_ent = db_ents[n]
            nl2sql_dct[cur_ent['question']] = cur_ent['query']
    
    if len(nl2sql_dct) == 0:
        raise Exception(f'The range {num_range} was not needed.')
    
    return nl2sql_dct
            
def get_drspider_questions(db_name, num_range):
    nl2sql_dct = {}
    top_dir = os.path.expanduser('~/diagnostic-robustness-text-to-sql/data')
    pert_dirs = [os.path.join(top_dir, d) for d in os.listdir(top_dir) if d.startswith('NLQ')]
    
    all_ents = []
    for pdir in pert_dirs:
        ppath = os.path.join(pdir, 'questions_post_perturbation.json')
        with open(ppath, 'r') as fh:
            p_ents = json.load(fh)
        
        db_ents = [e for e in p_ents if e['db_id'] == db_name]
        all_ents += db_ents
    
    #we have to do this to make sure the order is the same on every run
    all_ents = sorted(all_ents, key=lambda e: e['question'])
    if len(all_ents) == 0:
        raise Exception(f"NO Dr. Spider entries for database {db_name}")
    
    for n in num_range:
        if n >= len(all_ents):
            print(f"Index {n} did not exist, ignoring")
        else:
            cur_ent = all_ents[n]
            nl2sql_dct[cur_ent['question']] = cur_ent['query']
    
    if len(nl2sql_dct) == 0:
        raise Exception(f'The range {num_range} was not needed.')
    
    return nl2sql_dct

def replace_nlacm_views(oldnl, oldsql, oldgt, db_name, nl2sql_type, num_range):
    if nl2sql_type == 'spider':
        nl2sql_dct = get_spider_questions(db_name, num_range)
    elif nl2sql_type == 'drspider':
        nl2sql_dct = get_drspider_questions(db_name, num_range)
    else:
        raise Exception(f"Unsupported NL2SQL Type: {nl2sql_type}")
    
    nldf = pd.read_csv(oldnl)
    old_nlcols = [c for c in nldf.columns if c != 'Role']
    sqldf = pd.read_csv(oldsql)
    old_sqlcols = [c for c in sqldf.columns if c != 'Role']
    gtdf = pd.read_csv(oldgt)
    
    if len(nl2sql_dct) < len(old_nlcols):
        new_nl = select_rows_and_columns(nldf, nldf.shape[0], len(nl2sql_dct))
        new_sql = select_rows_and_columns(sqldf, sqldf.shape[0], len(nl2sql_dct))
        new_gt = select_rows_and_columns(gtdf, gtdf.shape[0], len(nl2sql_dct))
        
        nlcols = {}
        sqlcols = {}
        cnt = 0
        for k,v in nl2sql_dct.items():
            cur_nlcol = old_nlcols[cnt]
            cur_sqlcol = old_sqlcols[cnt]
            nlcols[cur_nlcol] = k
            sqlcols[cur_sqlcol] = v
            cnt += 1
        
        new_nl = new_nl.rename(columns=nlcols)
        new_sql = new_sql.rename(columns=sqlcols)
        new_gt = new_gt.rename(columns=sqlcols)
        
    else:
        nlcols = {}
        sqlcols = {}
        cnt = 0
        for k,v in nl2sql_dct.items():
            cur_nlcol = old_nlcols[cnt]
            cur_sqlcol = old_sqlcols[cnt]
            nlcols[cur_nlcol] = k
            sqlcols[cur_sqlcol] = v
            cnt += 1
        
        new_nl = nldf.rename(columns=nlcols)
        new_sql = sqldf.rename(columns=sqlcols)
        new_gt = gtdf.rename(columns=sqlcols)
    
    return new_nl, new_sql, new_gt

def run_with_views(old_temp_nl, old_temp_sql, old_temp_gt, old_priv_nl, old_priv_sql, old_priv_gt, outdir, db_name, nl2sql_type, num_range, persist_name=None):
    new_temp_nl, new_temp_sql, new_temp_gt = replace_nlacm_views(old_temp_nl, old_temp_sql, old_temp_gt, db_name, nl2sql_type, num_range)
    new_priv_nl, new_priv_sql, new_priv_gt = replace_nlacm_views(old_priv_nl, old_priv_sql, old_priv_gt, db_name, nl2sql_type, num_range)
    
    if persist_name != None:
        tempnlpath = persist_name + '_temp_nl.csv'
        tempsqlpath = persist_name + '_temp_sql.csv'
        tempgtpath = persist_name + '_temp_gt.csv'
        
        privnlpath = persist_name + '_priv_nl.csv'
        privsqlpath = persist_name + '_priv_sql.csv'
        privgtpath = persist_name + '_priv_gt.csv'
    else:
        pref = 'tmp'
        tempnlpath = pref + '_temp_nl.csv'
        tempsqlpath = pref + '_temp_sql.csv'
        tempgtpath = pref + '_temp_gt.csv'
        
        privnlpath = pref + '_priv_nl.csv'
        privsqlpath = pref + '_priv_sql.csv'
        privgtpath = pref + '_priv_gt.csv'
    
    new_temp_nl.to_csv(tempnlpath, index=False)
    new_temp_sql.to_csv(tempsqlpath, index=False)
    new_temp_gt.to_csv(tempgtpath, index=False)
    new_priv_nl.to_csv(privnlpath, index=False)
    new_priv_sql.to_csv(privsqlpath, index=False)
    new_priv_gt.to_csv(privgtpath, index=False)
    
    diff_obj = TempDiff(tempnlpath, tempsqlpath, tempgtpath, privnlpath, privsqlpath, privgtpath, 'postgres', pg_details, outdir)
    diff_obj.rv_map(nl2sql_type)
    diff_obj.temp_sub_src()
    temp_full_eval(outdir)

def run_sociome(prompt_type='betterl2m'):
    socpriv_nl = os.path.expanduser('~/sociome_data/sociome_revised_nl.csv')
    socpriv_gt = os.path.expanduser('~/sociome_data/sociome_revised_sql.csv')
    socpriv_sql = os.path.expanduser('~/sociome_data/sociome_revised_badsql.csv')
    
    soctemp_nl = os.path.expanduser('~/sociome_data/sociome_temp_nl.csv')
    soctemp_gt = os.path.expanduser('~/sociome_data/sociome_temp_sql.csv')
    soctemp_sql = os.path.expanduser('~/sociome_data/sociome_temp_badsql.csv')
    
    cur_dbtype = 'sociome'
    soc_resdir = 'sociome_' + prompt_type
    soc_details = {'sociome_path' : os.path.expanduser('~/sociome_data')}
    diff_obj = TempDiff(soctemp_nl, soctemp_sql, soctemp_gt, socpriv_nl, socpriv_sql, socpriv_gt, cur_dbtype, soc_details, soc_resdir)
    
    nl2sql_type = 'sociome'
    diff_obj.rv_map(nl2sql_type)
    diff_obj.temp_sub_src()
    temp_full_eval(soc_resdir)

def run_amazon(hier_type='wide', prompt_type='betterl2m', roles_from='wide_betterl2m', views_from='wide_betterl2m'):
    db_name = 'european_football_2'
    
    tmp_nl = hier_type + '_simpletime_nl.csv'
    tmp_sql = hier_type + '_simpletime_badsql.csv'
    tmp_gt = hier_type + '_simpletime_sql.csv'
    
    priv_nl = hier_type + '_nlacm_nl.csv'
    priv_sql = hier_type + '_nlacm_badsql.csv'
    priv_gt = hier_type + '_nlacm_sql.csv'
    
    amazon_dir = hier_type + '_' + prompt_type
    
    #then copy view files
    if os.path.exists(views_from) and views_from != amazon_dir:
        if not os.path.exists(amazon_dir):
            os.mkdir(amazon_dir)
        
        v_files = [f for f in os.listdir(views_from) if f.startswith('sqlvsnl_view')]
        v_files.append('viewmap.csv')
        
        for v_f in v_files:
            cur_f = os.path.join(views_from, v_f)
            new_f = os.path.join(amazon_dir, v_f)
            shutil.copyfile(src=cur_f, dst=new_f)
    
    #and copy role files
    if os.path.exists(roles_from) and roles_from != amazon_dir:
        if not os.path.exists(amazon_dir):
            os.mkdir(amazon_dir)
        
        r_files = [f for f in os.listdir(roles_from) if f.startswith('nlvssql_role')]
        r_files.append('rolemap.csv')
        
        for r_f in r_files:
            cur_f = os.path.join(roles_from, r_f)
            new_f = os.path.join(amazon_dir, r_f)
            shutil.copyfile(src=cur_f, dst=new_f)
    
    pg_details = {'user' : 'YOUR_USER', 'password' : 'YOUR_PASS', 'host' : 'XXXXXXXXXX', 'port' : 'XXXX', 'database' : db_name}
    diff_obj = TempDiff(tmp_nl, tmp_sql, tmp_gt, priv_nl, priv_sql, priv_gt, 'postgres', pg_details, amazon_dir)
    
    diff_obj.rv_map('BIRD')
    diff_obj.temp_sub_src(prompt_type=prompt_type)
    temp_full_eval(amazon_dir)

def run_all_amazon():
    all_prompts = ['realcot', 'simplefewshot', 'simplepersona']
    role_hiers = ['wide', 'balance']
    
    for pt in all_prompts:
        for rh in role_hiers:
            if rh == 'balance':
                cur_view_from = 'balance_betterl2m'
                cur_role_from = 'balance_betterl2m'
            else:
                cur_view_from = 'wide_betterl2m'
                cur_role_from = 'wide_betterl2m'
            run_amazon(hier_type=rh, prompt_type=pt, roles_from=cur_role_from, views_from=cur_view_from)

if __name__=='__main__':
    tnlacm_path = 'bird_simpletime_european_football_2_nl/bird_simpletime_nl_chunk_11.csv'
    sqlacm_path = 'bird_simpletime_european_football_2_badsql/bird_simpletime_badsql_chunk_11.csv'
    gtsqlacm_path = 'bird_simpletime_european_football_2_sql/bird_simpletime_sql_chunk_11.csv'
    
    tpriv_path = 'bird_nlacm_european_football_2_nl/bird_nlacm_nl_chunk_11.csv'
    sqlpriv_path = 'bird_nlacm_european_football_2_badsql/bird_nlacm_badsql_chunk_11.csv'
    gtsqlpriv_path = 'bird_nlacm_european_football_2_sql/bird_nlacm_sql_chunk_11.csv'
    
    # base_outtest = 'amazon_bird_temp_audit_betterl2m'
    # cot_outtest = 'amazon_bird_temp_audit_realcot'
    # fs_outtest = 'amazon_bird_temp_audit_simplefewshot'
    # sc_outtest = 'amazon_bird_temp_audit_simplesc'
    # pers_outtest = 'amazon_bird_temp_audit_simplepersona'
    
    '''
    Database Information
    BIRD: european_football_2
    Spider: orchestra, car_1, employee_hire_evaluation, student_transcripts_tracking
    '''
    
    # db_name = 'european_football_2'
    # diff_obj = TempDiff(tnlacm_path, sqlacm_path, gtsqlacm_path, tpriv_path, sqlpriv_path, gtsqlpriv_path, 'postgres', pg_details, base_outtest)
    #role-view mapping
    # diff_obj.rv_map()
    #temporal privilege subsumption
    # diff_obj.temp_sub_src()
    #permission privilege subsumption
    # diff_obj.priv_sub_src()
    
    #Evaluation
    # temp_full_eval(base_outtest)
    # priv_full_eval(base_outtest)
    
    ######BEGIN RUN VIEWS CODE##############
    # db_name = 'orchestra'
    # nl2sql_tp = 'drspider'
    # spidertest = 'amazon_spider_temp_audit_betterl2m'
    # drspidertest = 'amazon_drs_temp_audit_betterl2m'
    # run_with_views(tnlacm_path, sqlacm_path, gtsqlacm_path, tpriv_path, sqlpriv_path, gtsqlpriv_path, drspidertest, db_name, nl2sql_tp, list(range(10)))
    ######END RUN VIEWS CODE##############
    
    # run_sociome()
    # run_amazon(hier_type='balance')
    run_all_amazon()
