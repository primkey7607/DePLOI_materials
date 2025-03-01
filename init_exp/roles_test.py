import pandas as pd
import os
from sqlalchemy import engine
import psycopg2
import openai
import signal
from ast import literal_eval
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

def read_doc(fname):
    with open(fname, 'r') as fh:
        st = fh.read()
    
    return st

class MyTimeoutException(Exception):
    pass

#register a handler for the timeout
def handler(signum, frame):
    print("Waited long enough!")
    raise MyTimeoutException("STOP")

def get_schema(db_details):
    #for now, let's just hardcode it
    return str(['supplier', 'customer', 'lineitem','region',
            'orders', 'partsupp', 'part', 'nation'])

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_response(doc_text, db_details, db_type, temp_val, outdir, outname, timeout=30):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    
    schema_text = get_schema(db_details)
    st = 'Consider the following access control policy:\n\n'
    st += doc_text + '\n\n'
    st += 'And consider the following ' + db_type + ' database on which the policy will be implemented:\n\n'
    st += 'Tables:\n' + schema_text + '\n\n'
    st += 'Write the postgresql commands to implement such access control, given the database schema.'
    st += ' If you are unsure, make your best guess.'
    # chat = [{'role' : 'system', 'content' : "You are a laconic assistant. You reply with brief, to-the-point answers with no elaboration."}]
    chat = [{'role' : 'system', 'content' : "You are a helpful assistant."}]
    chat.append({'role' : 'user', 'content' : st})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=temp_val
    )
    chat_response = response["choices"][0]["message"]["content"]
    fullchat = chat + [{'role' : 'assistant', 'content' : chat_response}]
    outfile = os.path.join(outdir, outname + '_chat.json')
    with open(outfile, 'r') as fh:
        print(fullchat, file=fh)
    return chat_response

# @retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
# def get_response(doc_name, db_details, db_type, temp_val, timeout=30):
#     doc_text = read_doc(doc_name)
#     schema_text = get_schema(db_details)
    
#     st = 'Consider the following access control policy:\n\n'
#     st += doc_text + '\n\n'
#     st += 'And consider the following ' + db_type + ' database on which the policy will be implemented:\n\n'
#     st += 'Tables:\n' + schema_text + '\n\n'
#     st += 'Write the postgresql commands to implement such access control, given the database schema.'
#     st += ' If you are unsure, make your best guess.'
#     # chat = [{'role' : 'system', 'content' : "You are a laconic assistant. You reply with brief, to-the-point answers with no elaboration."}]
#     chat = [{'role' : 'system', 'content' : "You are a helpful assistant."}]
#     chat.append({'role' : 'user', 'content' : st})
    
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=chat,
#         temperature=temp_val
#     )
#     chat_response = response["choices"][0]["message"]["content"]
#     return chat_response

def parse_queries(chat_file, script_name):
    #chatgpt seems to return the queries in between 3 tildes.
    #it starts with text, then query, then text, then query, etc.
    #we will take advantage of that to parse the queries out.
    with open(chat_file, 'r') as fh:
        ent_lst = literal_eval(fh.read())
    
    query_st = ent_lst[-1]['content']
    query_parts = query_st.split('```')
    queries = [qp for i,qp in enumerate(query_parts) if i % 2 == 1]
    full_script = '\n'.join(queries)
    with open(script_name + '.sql', 'w+') as fh:
        print(full_script, file=fh)

def parse_query(query_st):
    query_parts = query_st.split('```')
    queries = [qp for i,qp in enumerate(query_parts) if i % 2 == 1]
    full_script = '\n'.join(queries)
    return full_script

def query_tofile(script, script_name):
    full_name = script_name
    if script_name.endswith('.sql'):
        full_name = script_name[:-4]
    with open(full_name + '.sql', 'w+') as fh:
        print(script, file=fh)

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def rolled_response(prompt_dir, query_dir, doc_dir, doc_name, db_details, db_type, temp_val, timeout=30):
    doc_text = read_doc(os.path.join(doc_dir, doc_name + '.txt'))
    schema_text = get_schema(db_details)
    
    access_prompt = 'Consider the following access control policy:\n\n'
    access_prompt += doc_text + '\n\n'
    ac_question = 'What parts of this policy seem like access control rules that can be implemented on a database?'
    access_prompt += ac_question
    
    acl_prompt = 'Now consider the following ' + db_type + ' database on which the policy will be implemented:\n\n'
    acl_prompt += 'Tables:\n' + schema_text + '\n\n'
    acl_question = 'For each access control rule you identified above, reformat the rule as: role | table(s) | privileges'
    acl_prompt += acl_question
    
    sql_prompt = 'Write the postgresql commands to implement the access control list you made above. If you are unsure, make your best guess.'
    
    chat = [{'role' : 'system', 'content' : "You are a helpful assistant."}]
    chat.append({'role' : 'user', 'content' : access_prompt})
    
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=temp_val
    )
    ac_rules = response["choices"][0]["message"]["content"]
    
    chat.append({'role' : 'assistant', 'content' : ac_rules})
    chat.append({'role' : 'user', 'content' : acl_prompt})
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=temp_val
    )
    ac_lst = response["choices"][0]["message"]["content"]
    
    chat.append({'role' : 'assistant', 'content' : ac_lst})
    chat.append({'role' : 'user', 'content' : sql_prompt})
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=temp_val
    )
    queries = response["choices"][0]["message"]["content"]
    chat.append({'role' : 'assistant', 'content' : queries})
    full_promptname = os.path.join(prompt_dir, 'fullchat_' + doc_name + '.txt')
    with open(full_promptname, 'w+') as fh:
        print(chat, file=fh)
    
    # write the queries
    query_name = os.path.join(query_dir, 'script_' + doc_name)
    parse_queries(full_promptname, query_name)

def teardown():
    con = psycopg2.connect(user='YOUR_USER', password='YOUR_PASS', host='XXXXXXXX', port='XXXX', database='XXXX')
    cur = con.cursor()
    get_myroles = "select rolname from pg_authid where rolname != 'postgres' and not rolname ilike 'pg' || '%';"
    cur.execute(get_myroles)
    cur_rolelst = [tup[0] for tup in cur.fetchall()]
    drop_st = ''
    for role in cur_rolelst:
        drop_st +=  'drop owned by ' + role + '; '
        drop_st += 'drop role ' + role + '; '
    
    if drop_st != '':
        cur.execute(drop_st)
    
    cur.close()
    con.close()

def eval_syntax(res_dir, subdir='code'):
    out_schema = ['File Name', 'Successful', 'Error Msg']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    code_dir = os.path.join(res_dir, subdir)
    con = psycopg2.connect(user='YOUR_USER', password='YOUR_PASS', host='XXXXXXXX', port='XXXX', database='XXXX')
    cur = con.cursor()
    for i,f in enumerate(os.listdir(code_dir)):
        print("Currently processing: {}".format(f))
        if f.endswith('.sql'):
            teardown()
            fullf = os.path.join(code_dir, f)
            with open(fullf, 'r') as fh:
                sql_st = fh.read()
            err_msg = None
            try:
                cur.execute(sql_st)
                print(f + "--query " + str(i) + ": Successful")
                cur.close()
                con.close()
                con = psycopg2.connect(user='YOUR_USER', password='YOUR_PASS', host='XXXXXXXX', port='XXXX', database='XXXX')
                cur = con.cursor()
            except Exception as err:
                print(f + "--query " + str(i) + ": Error")
                err_msg = str(err)
                cur.close()
                con.close()
                con = psycopg2.connect(user='YOUR_USER', password='YOUR_PASS', host='XXXXXXXX', port='XXXX', database='XXXX')
                cur = con.cursor()
        
        out_dct['File Name'].append(f)
        if err_msg == None:
            out_dct['Successful'].append(True)
        else:
            out_dct['Successful'].append(False)
        
        out_dct['Error Msg'].append(err_msg)
    
    out_df = pd.DataFrame(out_dct)
    out_df.to_csv(res_dir + '_' + subdir + '_syntaxres.csv', index=False)
    cur.close()
    con.close()

def reverse_prompt(sql_st, outdir, temp_val=1.0):
    st = 'Consider the following SQL query:\n\n'
    st += sql_st + '\n\n'
    st += 'I do not know anything about SQL. Explain to me in plain English what this query does.'
    chat = [{'role' : 'system', 'content' : "You are a helpful assistant."}]
    chat.append({'role' : 'user', 'content' : st})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=temp_val
    )
    chat_response = response["choices"][0]["message"]["content"]
    fullchat = chat + [{'role' : 'assistant', 'content' : chat_response}]
    return chat_response
    

def eval_frf(trans_func, res_dir, subdir='code'):
    #assume the forward pass is finished. now, do the reverse
    outdir = os.path.join(res_dir, 'frf')
    revdir = os.path.join(res_dir, 'reverse')
    out_schema = ['File Name', 'Successful', 'Error Msg']
    out_dct = {}
    for o in out_schema:
        out_dct[o] = []
    
    code_dir = os.path.join(res_dir, subdir)
    for i,f in enumerate(os.listdir(code_dir)):
        print("Currently processing: {}".format(f))
        if f.endswith('.sql'):
            teardown()
            fullf = os.path.join(code_dir, f)
            with open(fullf, 'r') as fh:
                sql_st = fh.read()
        
    
            
            

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))    
def fix_query(query_chat, temp_val=0, timeout=30):
    fix_prompt = 'Here is an incorrect query meant to grant access control on a postgres database:\n'
    fix_prompt += query_chat
    fix_prompt += 'Revise this query so it is correct.'
    
    chat = [{'role' : 'system', 'content' : "You are a helpful assistant."}]
    chat.append({'role' : 'user', 'content' : fix_prompt})
    
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=chat,
        temperature=temp_val
    )
    return response
    

def fix_syntax(res_dir, subdir='code', temp_val=0, timeout=30):
    if not os.path.exists(res_dir + '_' + subdir + '_syntaxres.csv'):
        print("Syntax Results not yet Evaluated!")
        eval_syntax(res_dir)
    
    df = pd.read_csv(res_dir + '_' + subdir + '_syntaxres.csv')
    errdf = df[df['Successful'] == False]
    errfiles = errdf['File Name'].tolist()
    
    for f in errfiles:
        out_script = os.path.join(res_dir, 'corrected_code', 'corrected' + f)
        if os.path.exists(out_script):
            continue
        fullf = os.path.join(res_dir, 'code', f)
        with open(fullf, 'r') as fh:
            query_st = fh.read()
        
        query_chat = "```\n" + query_st + "\n```\n"
        response = fix_query(query_chat)
        fixed_query_st = response["choices"][0]["message"]["content"]
        fixed_query = parse_query(fixed_query_st)
        query_tofile(fixed_query, out_script)
        
def test_correctedsyntax(res_dir):
    eval_syntax(res_dir, subdir='corrected_code')
        
    
    

if __name__=='__main__':
    # doc1 = 'hierarchynoview_ex.txt'
    # doc2 = 'siloed_ex.txt'
    # doc3 = 'overlap.txt'
    
    #Run Naive version
    # print(doc1 + ' Response')
    # print(get_response(doc1, {}, 'postgres', 0.0))
    # print(doc2 + ' Response')
    # print(get_response(doc2, {}, 'postgres', 0.0))
    # print(doc3 + ' Response')
    # print(get_response(doc3, {}, 'postgres', 0.0))
    
    #Run Rolled version
    # print(doc1 + ' Response')
    # print(rolled_response(doc1, {}, 'postgres', 0.0))
    # print(doc2 + ' Response')
    # print(rolled_response(doc2, {}, 'postgres', 0.0))
    # print(doc3 + ' Response')
    # print(rolled_response(doc3, {}, 'postgres', 0.0))
    # parse_queries('fullchat_hierarchynoview_ex.txt', 'hierarchynoview_script')
    # parse_queries('fullchat_siloed_ex.txt', 'siloed_script')
    # parse_queries('fullchat_overlap.txt', 'overlap_script')
    
    # testdocnames = [f[:-4] for f in os.listdir('testdocs/docs')]
    # for docname in testdocnames:
    #     rolled_response('testdocs/prompts', 'testdocs/code', 'testdocs/docs', docname, {}, 'postgres', 0.0)
    # eval_syntax('testdocs')
    fix_syntax('testdocs')
    # test_correctedsyntax('testdocs')
    
    
    
    





