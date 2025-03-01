import pandas as pd
import os
import re
from sqlalchemy import engine
import psycopg2
import openai
import pickle
from compare_sentences import smallest_semantic_similarity
from pydantic import BaseModel
from enum import Enum
import signal
import tiktoken
import importlib
from abc import ABC, abstractmethod
from ast import literal_eval
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

class MyTimeoutException(Exception):
    pass

#register a handler for the timeout
def handler(signum, frame):
    print("Waited long enough!")
    raise MyTimeoutException("STOP")

def read_plain_chat(chatfile):
    if not chatfile.endswith('.json'):
        raise Exception("chatfile is wrong format, expected .json: {}".format(chatfile))
    
    with open(chatfile, 'r') as fh:
        ents = literal_eval(fh.read())
    
    explanation = ents[-1]['content']
    return explanation

class ParsedOut:
    def __init__(self, inputs : dict, chat, parse_details : str, explanation, parsed):
        #there can be multiple inputs of various formats, so
        #key: name describing input, value: the python representation of the input
        self.inputs = inputs
        self.chat = chat
        self.parse_details = parse_details
        self.explanation = explanation
        self.parsed = parsed
    

def read_parsed_chat(chatfile):
    #NOTE: we will assume we are given a pickle file, even if the file does not end
    #with the pickle extension
    # if not chatfile.endswith('.pkl'):
    #     raise Exception("chatfile is wrong format, expected .pkl: {}".format(chatfile))
    
    with open(chatfile, 'rb') as fh:
        parse_obj = pickle.load(fh)
    
    #the type of parse_details is a MyOutput with some type of enum.
    #it would be nice to make that explicit in a programmatic way here, but
    #I don't think we can without a lot of effort.
    
    explanation = parse_obj.explanation
    parsed = parse_obj.parsed
    
    return parsed, explanation

#read the chat, but overwrite the parsed version of it
def read_chat_overwrite_parse(chatfile, parse_func):
    with open(chatfile, 'rb') as fh:
        parse_obj = pickle.load(fh)
    
    explanation = parse_obj.explanation
    parsed = parse_func(explanation)
    
    return parsed, explanation

class ResponseHandler(ABC):
    
    def __init__(self, model_name, api_keyfile=None):
        if api_keyfile != None:
            with open(api_keyfile, 'r') as fh:
                api_key = fh.read()
                api_key = api_key.replace('\n', '')
            self.api_key = api_key
        else:
            self.api_key = None
        self.model_name = model_name
    
    @abstractmethod
    def get_response(self, inputs, chat, temp_val, timeout=30, write_dir=None, write_file=None):
        raise Exception("Must be implemented")

class OpenAIHandler(ResponseHandler):
    
    @retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_response(self, chat, temp_val, timeout=30, write_dir=None, write_file=None):
        #first, check if a response already exists
        if write_dir != None and write_file != None:
            if not os.path.exists(write_dir):
                os.mkdir(write_dir)
            
            outname = os.path.join(write_dir, write_file)
            if os.path.exists(outname):
                out_resp = read_plain_chat(outname)
                return out_resp
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                  model=self.model_name,
                  messages=chat,
                  temperature=temp_val
            )
            
            chat_response = response.choices[0].message.content
            
            full_chat = chat + [{'role' : 'assistant', 'content' : chat_response}]
            
            with open(outname, 'w+') as fh:
                print(full_chat, file=fh)
        
        elif write_dir != None or write_file != None:
            raise Exception("Both write_dir and write_file must be specified: {}, {}".format(write_dir, write_file))
        else:
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                  model=self.model_name,
                  messages=chat,
                  temperature=temp_val
            )
            
            chat_response = response.choices[0].message.content
            
            full_chat = chat + [{'role' : 'assistant', 'content' : chat_response}]
            
        return chat_response
    
    def execute_parse(self, parse_func, raw_resp, inputs):
        func_name = parse_func.__name__
        
        if func_name == 'parse_yn' or func_name == 'parse_yn_llm':
            parsed = parse_func(raw_resp)
            return parsed
        elif func_name == 'parse_el_sem':
            neg_code = 'None'
            in_lst = inputs['list']
            parsed = parse_func(raw_resp, in_lst, neg_code)
            
            return parsed
        else:
            raise Exception("Unsupported Function: {}".format(func_name))
            
    
    @retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_parsed_response(self, inputs, chat, temp_val, parse_func, timeout=30, write_dir=None, write_file=None):
        #first, check if a response already exists
        if write_dir != None and write_file != None:
            if not os.path.exists(write_dir):
                os.mkdir(write_dir)
            
            outname = os.path.join(write_dir, write_file)
            if os.path.exists(outname):
                out_obj = read_parsed_chat(outname)
                out_resp = out_obj[1] #first is parsed, second is explanation
                out_parsed = out_obj[0]
                #both need to be returned when the file exists
                return out_parsed, out_resp
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                  model=self.model_name,
                  messages=chat,
                  temperature=temp_val
            )
            
            chat_response = response.choices[0].message.content
            
            full_chat = chat + [{'role' : 'assistant', 'content' : chat_response}]
            
            # with open(outname, 'w+') as fh:
            #     print(full_chat, file=fh)
            # parsed = parse_func(chat_response)
            parsed = self.execute_parse(parse_func, chat_response, inputs)
            parse_details = f"Response parsed using the parsing function: {parse_func.__name__}"
            new_out = ParsedOut(inputs, chat, parse_details, chat_response, parsed)
            with open(outname, 'wb') as fh:
                pickle.dump(new_out, file=fh)
        
        elif write_dir != None or write_file != None:
            raise Exception("Both write_dir and write_file must be specified: {}, {}".format(write_dir, write_file))
        else:
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                  model=self.model_name,
                  messages=chat,
                  temperature=temp_val
            )
            
            chat_response = response.choices[0].message.content
            # parsed = parse_func(chat_response)
            parsed = self.execute_parse(parse_func, chat_response, inputs)
            
            full_chat = chat + [{'role' : 'assistant', 'content' : chat_response}]
            
        return parsed, chat_response
    
    @retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def select_from_lst(self, chat, el, lst, temp_val, none_option=None, timeout=30, write_dir=None, write_file=None):
        #add "None" to the list if needed
        
        #TODO: remove this once debugging is done
        if type(lst) != list:
            raise Exception("We have a real problem: {}".format(lst))

        if none_option != None:
            print("Adding None to List: {}".format(lst))
            full_lst = lst + [none_option]
        else:
            full_lst = lst
        
        #first, check if a response already exists
        if write_dir != None and write_file != None:
            if not os.path.exists(write_dir):
                os.mkdir(write_dir)
            
            outname = os.path.join(write_dir, write_file)
            if os.path.exists(outname):
                print("File {} Exists!!!!".format(outname))
                parsed, explanation = read_parsed_chat(outname)
                return parsed, explanation
            
            LstEnum = Enum('LstEnum', full_lst)
            
            class MyOutput(BaseModel):
                explanation: str
                output: LstEnum

            in_msgs = chat[:2]
            client = openai.OpenAI()
            completion = client.beta.chat.completions.parse(model='gpt-4o', messages=in_msgs, response_format=MyOutput)
            
            parsed_cl = completion.choices[0].message.parsed
            parsed = parsed_cl.output.name
            explanation = parsed_cl.explanation
            
            #now, create the class to store all needed information
            #NOTE: we cannot store the ParseCompletion object because
            parsed_in = {'element' : el, 'list' : lst}
            parsed_out = ParsedOut(parsed_in, chat, str(completion), explanation, parsed)
            print("Storing class: {}".format(vars(parsed_out)))
            with open(outname, 'wb') as fh:
                pickle.dump(parsed_out, file=fh)
            
            file_size_in_bytes = os.path.getsize(outname)
            print("File size:", file_size_in_bytes, "bytes")
            
            return parsed, explanation
        
        elif write_dir != None or write_file != None:
            raise Exception("Both write_dir and write_file must be specified: {}, {}".format(write_dir, write_file))
        else:
            
            LstEnum = Enum('LstEnum', full_lst)
            
            class MyOutput(BaseModel):
                explanation: str
                output: LstEnum
            
            in_msgs = chat[:2]
            client = openai.OpenAI()
            completion = client.beta.chat.completions.parse(
                model='gpt-4o',
                messages=in_msgs,
                response_format=MyOutput
                )
            
            parsed_cl = completion.choices[0].message.parsed
            parsed = parsed_cl.output.name
            explanation = parsed_cl.explanation
            
            #now, create the class to store all needed information
            # parsed_out = ParsedOut(chat, completion, explanation, parsed)
            # with open(outname, 'wb') as fh:
            #     pickle.dump(parsed_out, file=fh)
            
            return parsed, explanation

class AnyScaleHandler(ResponseHandler):
    
    @retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_response(self, chat, temp_val, timeout=30, write_dir=None, write_file=None):
        client = openai.OpenAI(
            base_url = "https://api.endpoints.anyscale.com/v1",
            api_key=self.api_key)
        chat_completion = client.chat.completions.create(
            model=self.model_name,
            messages=chat,
            temperature=temp_val)
        chat_response = chat_completion.choices[0].message.content
        full_chat = chat + [{'role' : 'assistant', 'content' : chat_response}]
        
        if write_dir != None and write_file != None:
            if not os.path.exists(write_dir):
                os.mkdir(write_dir)
            outname = os.path.join(write_dir, write_file)
            with open(outname, 'w+') as fh:
                print(full_chat, file=fh)
        elif write_dir != None or write_file != None:
            raise Exception("Both write_dir and write_file must be specified: {}, {}".format(write_dir, write_file))
        
        return chat_response
    
def auto_init_handler(config_file):
    with open(config_file, 'r') as fh:
        dct = literal_eval(fh.read())
    
    model_name = dct['model_name']
    handler = dct['handler']
    as_key = None
    if handler == 'AnyScaleHandler':
        as_key = dct['keyfile']
    
    # module = importlib.import_module('chat_utils')
    # HandlerClass = getattr(module, handler)
    HandlerClass = globals()[handler]
    if as_key == None:
        new_inst = HandlerClass(model_name)
    else:
        new_inst = HandlerClass(model_name, api_keyfile=as_key)
    
    return new_inst
    
    
@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def fancy_get_response(chat, temp_val, timeout=30, write_dir=None, write_pref=None):
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=chat,
        temperature=temp_val
    )
    chat_response = response["choices"][0]["message"]["content"]
    print("Received response: {}".format(chat_response))
    
    if write_dir != None and write_pref != None:
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
            chat_ind = 0
        else:
            all_fs = os.listdir(write_dir)
            all_inds = [int(f[f.index('chat') + 4 :-5]) for f in all_fs]
            chat_ind = max(all_inds) + 1
        
        full_chat = chat + [{'role' : 'assistant', 'content' : chat_response}]
        
        outname = os.path.join(write_dir, write_pref + '_chat' + str(chat_ind) + '.json')
        with open(outname, 'w+') as fh:
            print(full_chat, file=fh)
    elif write_dir != None or write_pref != None:
        raise Exception("Both write_dir and write_pref must be specified: {}, {}".format(write_dir, write_pref))
        
    return chat_response

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_response(chat, temp_val, timeout=30, write_dir=None, write_file=None):
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=chat,
        temperature=temp_val
    )
    chat_response = response["choices"][0]["message"]["content"]
    print("Received response: {}".format(chat_response))
    
    if write_dir != None and write_file != None:
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        # else:
            #let's keep this simple--we know outdir, we know outpref upstream
            #so we can just write those, and ignore the below three lines.
            # all_fs = os.listdir(write_dir)
            # all_inds = [int(f[f.index('chat') + 4 :-5]) for f in all_fs]
            # chat_ind = max(all_inds) + 1
        
        full_chat = chat + [{'role' : 'assistant', 'content' : chat_response}]
        
        outname = os.path.join(write_dir, write_file)
        with open(outname, 'w+') as fh:
            print(full_chat, file=fh)
    elif write_dir != None or write_file != None:
        raise Exception("Both write_dir and write_file must be specified: {}, {}".format(write_dir, write_file))
        
    return chat_response

@retry(retry=retry_if_exception_type(Exception), wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def response_from_model(chat, temp_val, model_name="gpt-3.5-turbo-0125", timeout=30, write_dir=None, write_file=None):
    
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=chat,
        temperature=temp_val
    )
    chat_response = response["choices"][0]["message"]["content"]
    print("Received response: {}".format(chat_response))
    
    if write_dir != None and write_file != None:
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        # else:
            #let's keep this simple--we know outdir, we know outpref upstream
            #so we can just write those, and ignore the below three lines.
            # all_fs = os.listdir(write_dir)
            # all_inds = [int(f[f.index('chat') + 4 :-5]) for f in all_fs]
            # chat_ind = max(all_inds) + 1
        
        full_chat = chat + [{'role' : 'assistant', 'content' : chat_response}]
        
        outname = os.path.join(write_dir, write_file)
        with open(outname, 'w+') as fh:
            print(full_chat, file=fh)
    elif write_dir != None or write_file != None:
        raise Exception("Both write_dir and write_file must be specified: {}, {}".format(write_dir, write_file))
        
    return chat_response

#this function was written by
# https://stackoverflow.com/questions/75804599/openai-api-how-do-i-count-tokens-before-i-send-an-api-request
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def chunk_response(prompt : str, context_prompt : str, context : list, model_name, temp_val, chunkdir, chunkpref, timeout=30):
    if not os.path.exists(chunkdir):
        os.mkdir(chunkdir)
    
    #we'll need to play with the below constant to see how much perf degrades as we add more context
    max_tokens = 500 
    full_st = ' '.join(context)
    context_size = num_tokens_from_string(full_st, model_name)
    
    if context_size > max_tokens:
        #then, chunk it
        sent_sizes = [num_tokens_from_string(sent, model_name) for sent in context]
        chunks = []
        chunk_sz = 0
        cur_chunk = []
        for i,sent in enumerate(context):
            if sent_sizes[i] > max_tokens:
                raise Exception("There's no reason why a sentence expressing a single privilege should be this long: {}".format(sent))
            
            if chunk_sz + sent_sizes[i] > max_tokens:
                chunks.append(cur_chunk)
                chunk_sz = 0
                cur_chunk = []
                
                chunk_sz = sent_sizes[i]
                cur_chunk.append(sent)
            else:
                chunk_sz += sent_sizes[i]
                cur_chunk.append(sent)
        
        if cur_chunk != []:
            chunks.append(cur_chunk)
        
    else:
        chunks = [[full_st]]
    
    chunk_resps = []
    for i,chunk in enumerate(chunks):
        chunkfile = os.path.join(chunkdir, chunkpref + '_chunk' + str(i) + '.json')
        if os.path.exists(chunkfile):
            with open(chunkfile, 'r') as fh:
                chunk_resps.append(fh.read())
        else:
            chat = [{'role' : 'system', 'content' : 'You are a helpful assistant.'}]
            cur_context = ' '.join(chunk)
            full_prompt = context_prompt + '\n' + cur_context + '\n' + prompt
            chat += [{'role' : 'user', 'content' : full_prompt}]
            cur_resp = get_response(chat, temp_val)
            chunk_resps.append(cur_resp)
            with open(chunkfile, 'w+') as fh:
                print(cur_resp, file=fh)
    
    return chunk_resps
            

def parse_lst(lst_resp, quote_lst=None):
    #chatgpt will delimit code by three apostrophes
    print("Parsing response: {}".format(lst_resp))
    query_st = lst_resp
    if '```' in query_st:
        query_parts = query_st.split('```')
        queries = [qp for i,qp in enumerate(query_parts) if i % 2 == 1]
        out_st = queries[0]
    else:
        out_st = query_st
    
    if '[' in out_st and ']' in out_st:
        out_st = out_st[out_st.index('[') : out_st.index(']') + 1]
        out_st.replace('\n', '')
        out_st.replace('\r', '')
        out_st.replace(' ', '')
        if quote_lst == None:
            out_lst = literal_eval(out_st)
        else:
            full_els = [q for q in quote_lst if q in out_st]
            out_lst = full_els
    else:
        print("WARNING: simply picking out keywords from string")
        full_els = [q for q in quote_lst if q in out_st]
        out_lst = full_els
    return out_lst

def parse_c3(raw_resp : str):
    if '```' in raw_resp:
        query_parts = raw_resp.split('```')
        queries = [qp for i,qp in enumerate(query_parts) if i % 2 == 1]
        print("NOTE: Assuming the 0th element is query: {}".format(queries))
        out_st = queries[0]
    else:
        out_st = raw_resp
    return out_st

def parse_el(raw_resp : str, lst : list, neg_code : str):
    lower_lst = [el.lower() for el in lst]
    l_neg = neg_code.lower()
    l_resp = raw_resp.lower()
    present_els = [lst[i] for i,el in enumerate(lower_lst) if el in l_resp]
    print("present_els: {}".format(present_els))
    absent = (l_neg in l_resp)
    print("absent: {}".format(absent))
    
    if absent and present_els != []:
        if l_resp.startswith(l_neg):
            print("Response started with negative token: {}, {}".format(l_resp, l_neg))
            return neg_code
        print("WARNING: assuming first element found in response is correct: {}, {}".format(present_els[0], raw_resp))
        return present_els[0]
    elif absent and present_els == []:
        return neg_code
    elif not absent and present_els == []:
        print("WARNING: Assuming none, but retry may be needed: {}".format(raw_resp))
        return neg_code
    elif not absent and present_els != []:
        if len(present_els) > 1:
            print("WARNING: Assuming first element found in response is correct: {}, {}".format(present_els, raw_resp))
            return present_els[0]
        
        return present_els[0]
    else:
        raise Exception("Not all cases captured: {}, {}".format(absent, present_els))

def approx_present(el, full_st):
    if len(el) < 5 or len(full_st) < 5:
        return (el in full_st)
    else:
        if el in full_st:
            return True
        
        most_len = int(len(el) * 0.8)
        if el[:most_len] in full_st:
            return True
        
        return False
        

#this is like parse_el, but we want to find strings that are most likely to match some element,
#but maybe not exact matches.
def parse_elv2(raw_resp : str, lst : list, neg_code : str):
    #first, properly process the raw response
    proc_resp = raw_resp.lower()
    proc_resp = proc_resp.replace(' ', '')
    proc_resp = proc_resp.replace('\n', '')
    proc_resp = proc_resp.replace('\t', '')
    
    
    l_lst = [el.lower() for el in lst]
    proc_lst = [el.replace(' ', '') for el in l_lst]
    proc_lst = [el.replace('\n', '') for el in proc_lst]
    proc_lst = [el.replace('\t', '') for el in proc_lst]
    
    l_neg = neg_code.lower()
    present_els = [lst[i] for i,el in enumerate(proc_lst) if approx_present(el, proc_resp)]
    print("present_els: {}".format(present_els))
    absent = (l_neg in proc_resp)
    print("absent: {}".format(absent))
    
    if absent and present_els != []:
        if proc_resp.startswith(l_neg):
            print("Response started with negative token: {}, {}".format(proc_resp, l_neg))
            return neg_code
        print("WARNING: assuming first element found in response is correct: {}, {}".format(present_els[0], raw_resp))
        return present_els[0]
    elif absent and present_els == []:
        return neg_code
    elif not absent and present_els == []:
        print("WARNING: Assuming none, but retry may be needed: {}".format(raw_resp))
        return neg_code
    elif not absent and present_els != []:
        if len(present_els) > 1:
            print("WARNING: Assuming first element found in response is correct: {}, {}".format(present_els, raw_resp))
            return present_els[0]
        
        return present_els[0]
    else:
        raise Exception("Not all cases captured: {}, {}".format(absent, present_els))

def parse_el_sem(raw_resp : str, lst : list, neg_code : str):
    
    present_els = [el for el in lst if el in raw_resp]
    if len(present_els) == 1:
        print("Found Exact Match: {}".format(present_els))
        return present_els[0]
    
    if len(present_els) > 1:
        print("Multiple elements matched, using semantics to choose: {}".format(present_els))
        best_tup = smallest_semantic_similarity(raw_resp, present_els)
        print("Best Match among Multiple: {}".format(best_tup))
        return best_tup[0]
    
    if len(present_els) == 0:
        if neg_code in raw_resp:
            print("Found Negative Indication, returning: {}".format(raw_resp))
            return neg_code
        
        best_tup = smallest_semantic_similarity(raw_resp, lst)
        print("Best Match among All: {}".format(best_tup))
        return best_tup[0]

#this is like parse_el, but we want to find strings that are most likely to match some element,
#but maybe not exact matches.
#TODO: adapt this for roles and users, e.g., ChatGPT only returned the role r, 
#but the full phrase is 'u has role r', so we parse to None incorrectly.
def parse_role(raw_resp : str, lst : list, neg_code : str):
    #first, properly process the raw response
    proc_resp = raw_resp.lower()
    proc_resp = proc_resp.replace(' ', '')
    proc_resp = proc_resp.replace('\n', '')
    proc_resp = proc_resp.replace('\t', '')
    
    
    l_lst = [el.lower() for el in lst]
    proc_lst = [el.replace(' ', '') for el in l_lst]
    proc_lst = [el.replace('\n', '') for el in proc_lst]
    proc_lst = [el.replace('\t', '') for el in proc_lst]
    
    l_neg = neg_code.lower()
    present_els = [lst[i] for i,el in enumerate(proc_lst) if approx_present(el, proc_resp)]
    #search also for cases where the model only gave the role for a user
    
    print("present_els: {}".format(present_els))
    absent = (l_neg in proc_resp)
    print("absent: {}".format(absent))
    
    if absent and present_els != []:
        if proc_resp.startswith(l_neg):
            print("Response started with negative token: {}, {}".format(proc_resp, l_neg))
            return neg_code
        print("WARNING: assuming first element found in response is correct: {}, {}".format(present_els[0], raw_resp))
        return present_els[0]
    elif absent and present_els == []:
        return neg_code
    elif not absent and present_els == []:
        print("WARNING: Assuming none, but retry may be needed: {}".format(raw_resp))
        return neg_code
    elif not absent and present_els != []:
        if len(present_els) > 1:
            print("WARNING: Assuming first element found in response is correct: {}, {}".format(present_els, raw_resp))
            return present_els[0]
        
        return present_els[0]
    else:
        raise Exception("Not all cases captured: {}, {}".format(absent, present_els))

#given a directory of files, clean the results using a bash script
def clean_badchars(rawdir):
    print("Fixing '^M' characters...")
    os.system(f"bash ./fix_newline.sh {rawdir}")
    

def parse_yn(raw_resp : str):
    u_resp = raw_resp.upper()
    
    if 'YES' in u_resp and 'NO' not in u_resp:
        return 'YES'
    elif 'YES' not in u_resp and 'NO' in u_resp:
        return 'NO'
    elif 'YES' not in u_resp and 'NO' not in u_resp:
        print("WARNING: response is unclear, so defaulting to 'no': {}".format(raw_resp))
        return 'NO'
    elif 'YES' in u_resp and 'NO' in u_resp:
        print("WARNING: choosing earlier token: {}".format(raw_resp))
        #choosing the earlier token will help account for the case where
        #the intended response was put at the beginning, and it will exclude
        #the case where either of these tokens are part of the answer.
        y_ind = u_resp.index('YES')
        n_ind = u_resp.index('NO')
        if y_ind < n_ind:
            return 'YES'
        else:
            return 'NO'
    else:
        raise Exception("Not all cases captured: {}".format(raw_resp))

def parse_yn_llm(raw_resp : str, retries=5):
    prompt = f'I asked a yes/no question, and got the following response: \n\n {raw_resp} \n\n'
    prompt += 'Is the final answer of this response Yes, or No? Respond only with either Yes, or No.'
    
    user_prompt = [{'role' : 'user', 'content' : prompt}]
    
    client = openai.OpenAI()
    output = client.chat.completions.create(
        model = 'gpt-4o',
        messages = user_prompt
    )
    
    final_answer = output.choices[0].message.content
    
    if final_answer != 'Yes' and final_answer != 'No':
        cur_chat = user_prompt + [{'role' : 'assistant', 'content' : final_answer}]
        for i in range(retries):
            followup = 'Please follow my instructions. I asked for a Yes or No answer. If you are unsure, default to No.'
            followup += ' Respond only with a Yes or No.'
            
            if i != 0:
                followup = 'I am asking for the ' + str(i) + 'th time. ' + followup
            
            cur_chat += [{'role' : 'user', 'content' : followup}]
            cur_out = client.chat.completions.create(
                model = 'gpt-4o',
                messages = cur_chat
            )
            
            raw_ans = cur_out.choices[0].message.content
            #do just a little cleaning
            cur_ans = raw_ans.replace(' ', '')
            cur_ans = cur_ans.replace('\n', '')
            cur_ans = cur_ans.replace('\t', '')
            
            if cur_ans == 'Yes' or cur_ans == 'No':
                return cur_ans
            else:
                cur_chat += [{'role' : 'assistant', 'content' : raw_ans}]
        
        print(f"WARNING: Attempt to Parse Final Answer was Unsuccessful, defaulting to No:\n\n{final_answer}")
        return 'No'
    
    return final_answer
            
        

def add_and_to_last_ordinal(sentence):
    """
    Adds 'and' between the second-to-last and last ordinal in a sentence
    with the structure:
    "This role can access this view for X (time units) starting on the Y1, Y2, ..., Yn (time units) of every (time unit)."
    """
    # Match the ordinal list in the "starting on the ..." part
    match = re.search(r'starting on the ([\w\s,]+?) of every', sentence)
    
    if match:
        # Extract the list of ordinals
        ordinals_part = match.group(1).strip()
        ordinals = [o.strip() for o in ordinals_part.split(',')]
        
        if len(ordinals) > 1:
            # Insert 'and' before the last ordinal
            ordinals[-2] = f"{ordinals[-2]} and"
            updated_ordinals_part = ', '.join(ordinals)
            
            # Replace the original ordinal part in the sentence
            sentence = sentence.replace(ordinals_part, updated_ordinals_part)
    
    return sentence

def test_add_and():
    # Example usage
    sentence = "This role can access this view for 8 days starting on the fifth, eleventh, twelfth months of every year."
    corrected_sentence = add_and_to_last_ordinal(sentence)
    print(corrected_sentence)

def fix_ending(sentence):
    """
    Suppose we have a sentence that ends with ',' and then some spaces.
    We want to fix it so it ends with a period
    """
    
    comma_ind = sentence.rindex(',')
    new_sent = sentence[:comma_ind] + '.'
    return new_sent

