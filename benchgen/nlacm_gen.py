import os
import json
import pandas as pd
import random
from ast import literal_eval

administrators = list(set([
    "System Administrator",
    "Network Administrator",
    "Database Administrator",
    "Linux Administrator",
    "Windows Administrator",
    "Cloud Administrator",
    "Salesforce Administrator",
    "Exchange Administrator",
    "SharePoint Administrator",
    "Active Directory Administrator",
    "Security Administrator",
    "Database Security Administrator",
    "Application Administrator",
    "Web Administrator",
    "Storage Administrator",
    "Virtualization Administrator",
    "SAP Administrator",
    "HR Administrator",
    "Financial Administrator",
    "School Administrator",
    "Healthcare Administrator",
    "Hospital Administrator",
    "Office Administrator",
    "Project Administrator",
    "CRM Administrator",
    "Database Developer/Administrator",
    "Education Administrator",
    "Event Administrator",
    "System Integration Administrator"
]))

common_names = [
    "John",
    "Mary",
    "Michael",
    "Jennifer",
    "James",
    "Sarah",
    "David",
    "Jessica",
    "Robert",
    "Elizabeth",
    "William",
    "Emily",
    "Joseph",
    "Ashley",
    "Christopher",
    "Amanda",
    "Daniel",
    "Megan",
    "Matthew",
    "Rachel"
]

interns = list(set([
    "Marketing Intern",
    "Software Development Intern",
    "Graphic Design Intern",
    "Data Analyst Intern",
    "Human Resources Intern",
    "Social Media Intern",
    "Business Development Intern",
    "Content Writing Intern",
    "Research Intern",
    "Finance Intern",
    "Product Management Intern",
    "IT Support Intern",
    "Video Production Intern",
    "Sales Intern",
    "UXUI Design Intern",
    "Public Relations Intern",
    "Environmental Science Intern",
    "Supply Chain Intern",
    "Event Planning Intern",
    "Engineering Intern",
    "Healthcare Administration Intern",
    "Legal Intern",
    "Nonprofit Management Intern",
    "Architecture Intern",
    "Cybersecurity Intern",
    "Art Curator Intern",
    "Journalism Intern",
    "Culinary Arts Intern",
    "Fashion Merchandising Intern",
    "Sports Management Intern"
]))
interns = [st.replace(' ', '_') for st in interns]
administrators = [st.replace(' ', '_') for st in administrators]

def account_for_dups(query_lst):
    qdups = {}
    qinds = {}
    out_lst = []
    for i,q in enumerate(query_lst):
        if q in qinds:
            view_st = 'query' + str(qinds[q]) + 'view' + str(qdups[q])
            new_st = 'CREATE VIEW ' + view_st + ' AS ' + q
            out_lst.append(new_st)
            qdups[q] += 1
        else:
            if len(qinds) == 0:
                qinds[q] = 0
            else:
                max_ind = max([qinds[el] for el in qinds])
                qinds[q] = max_ind + 1
            
            qdups[q] = 1
            view_st = 'query' + str(qinds[q]) + 'view0'
            new_st = 'CREATE VIEW ' + view_st + ' AS ' + q
            out_lst.append(new_st)
    
    return out_lst

def is_difficult(bird_entry):
    if bird_entry['difficulty'] == 'moderate' or bird_entry['difficulty'] == 'challenging':
        return True
    else:
        return False

def extract_queries_from_bird(benchmark_path, db_name, num_queries):

    nl_queries_dict = {}
    post_queries_dict = {}
    sql_queries_dict = {}
    
    #there is only one perturbation, and that is to keep the same NL as before
    pert_name = 'plain'
    fname = os.path.join(benchmark_path, 'dev.json')
    with open(fname, 'r') as fh:
        data = json.load(fh)
        
    all_dbitems = [item for item in data if item['db_id'] == db_name and is_difficult(item)]
    if all_dbitems == []:
        #likely because we wanted difficult stuff, so make the filter looser
        all_dbitems = [item for item in data if item['db_id'] == db_name]
    
    
    all_nl_queries = [item['question'] for item in all_dbitems]
    all_post_nl = all_nl_queries
    all_sql_queries = [item['SQL'] for item in all_dbitems]
    
    #Even the NL can repeat here. We want to choose distinct NL
    distinct_nl = list(set(all_nl_queries))
    distinct_nlinds = [all_nl_queries.index(el) for el in distinct_nl]
    distinct_postnl = [all_post_nl[el] for el in distinct_nlinds]
    distinct_sql = [all_sql_queries[el] for el in distinct_nlinds]
    
    nl_queries = distinct_nl[:num_queries]
    post_nlqueries = distinct_postnl[:num_queries]
    sql_queries = distinct_sql[:num_queries]
    
    renamed_sql = account_for_dups(sql_queries)

    nl_queries_dict[pert_name] = nl_queries
    post_queries_dict[pert_name] = post_nlqueries
    sql_queries_dict[pert_name] = renamed_sql
    
    return nl_queries_dict, post_queries_dict, sql_queries_dict

#got this trick from stack overflow
# https://stackoverflow.com/questions/9836425/equivelant-to-rindex-for-lists-in-python
def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) -1

def generate_acm(num_roles, num_users, num_privileges, num_queries, num_queries_with_privileges, db_name, benchmark_path):
    nl_acm_data = {}
    post_acm_data = {}
    sql_acm_data = {}
    privileges = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'GRANT']
    # Step 3: Extract natural language and SQL queries from Dr. Spider benchmark
    nl_queries_dict, post_queries_dict, sql_queries_dict = extract_queries_from_bird(benchmark_path, db_name, num_queries)
    
    #we want to use the same roles and users for all ACMs
    #and we want this to be reproducible, hence the below seed()
    random.seed(10)
    
    roles = random.sample(administrators, num_roles)
    users = random.sample(common_names, num_users)
    user_roles = random.sample(interns, num_users)
    nl_users = [u + ' has role ' + user_roles[i] for i,u in enumerate(users)]
    sql_users = ['CREATE USER ' + u + ' IN ROLE ' + user_roles[i] + ';' for i,u in enumerate(users)]

    for k in nl_queries_dict:
        nl_queries = nl_queries_dict[k]
        post_queries = post_queries_dict[k]
        print(nl_queries)
        sql_queries = sql_queries_dict[k]
        acm = pd.DataFrame(columns=['Role'] + nl_queries)
        sql_acm = pd.DataFrame(columns=['Role'] + sql_queries)
        post_acm = pd.DataFrame(columns=['Role'] + post_queries)

        # Step 4: Generate roles and users
        # roles_users = [f"Role{i}" for i in range(1, num_roles + 1)] + [f"User{j}" for j in range(1, num_users + 1)]
        roles_users = roles + nl_users
        sql_ru = roles + sql_users
        acm['Role'] = roles_users
        post_acm['Role'] = roles_users
        sql_acm['Role'] = sql_ru

        # Step 5: Randomly assign privileges for each role/user and specified number of nl_queries
        for i, role_user in enumerate(roles_users):
            selected_queries = random.sample(nl_queries, min(num_queries_with_privileges, len(nl_queries)))
            selected_inds = [listRightIndex(nl_queries, nlq) for nlq in selected_queries]
            selected_sql = [sql_queries[j] for j in selected_inds]
            selected_post = [post_queries[j] for j in selected_inds]
            sql_role_user = sql_ru[i]
            ind2priv = {} 
            for n,nl_query in enumerate(nl_queries):
                if nl_query in selected_queries:
                    priv_samp = random.sample(privileges, num_privileges)
                    print("Priv Samp type: {}".format(type(priv_samp)))
                    acm.loc[i, nl_query] = str(priv_samp)
                    ind2priv[n] = str(priv_samp)
                    print("ACM Entry: {}".format(acm.loc[i, nl_query]))
                else:
                    acm.loc[i, nl_query] = '[]'
            
            print(acm)
            for s,sql_query in enumerate(sql_queries):
                if sql_query in selected_sql:
                    #print(i)
                    #print(nl_queries[s])
                    #print(sql_query)
                    s_nl = nl_queries[s]
                    acm_priv = acm.loc[i, s_nl]
                    print(acm_priv)
                    print(type(acm_priv))
                    sql_acm.loc[i, sql_query] = ind2priv[s]
                else:
                    sql_acm.loc[i, sql_query] = '[]'
            
            for p,post_query in enumerate(post_queries):
                if post_query in selected_post:
                    p_nl = nl_queries[p]
                    acm_priv = acm.loc[i, p_nl]
                    post_acm.loc[i, post_query] = ind2priv[p]
                else:
                    post_acm.loc[i, post_query] = '[]'
                    

        nl_acm_data[k] = acm
        sql_acm_data[k] = sql_acm
        post_acm_data[k] = post_acm

    return nl_acm_data, post_acm_data, sql_acm_data

def generate_acms_todir(num_roles, num_users, num_privileges, num_queries, num_queries_with_privileges, benchmark_path, db_name, outdir, outpref):
    fullnldir = outdir + '_' + db_name + '_nl'
    fullpostdir = outdir + '_' + db_name + '_post'
    fullsqldir = outdir + '_' + db_name + '_sql'
    nlpref = outpref + '_nl'
    postpref = outpref + '_post'
    sqlpref = outpref + '_sql'
    
    if not os.path.exists(fullnldir):
        os.mkdir(fullnldir)
    
    if not os.path.exists(fullsqldir):
        os.mkdir(fullsqldir)
    
    if not os.path.exists(fullpostdir):
        os.mkdir(fullpostdir)
    
    nl_acm_data, post_acm_data, sql_acm_data = generate_acm(num_roles, num_users, num_privileges, num_queries, num_queries_with_privileges, db_name, benchmark_path)
    
    for k in nl_acm_data:
        fullnlout = os.path.join(fullnldir, nlpref + '_' + k + '.csv')
        fullpostout = os.path.join(fullpostdir, postpref + '_' + k + '.csv')
        fullsqlout = os.path.join(fullsqldir, sqlpref + '_' + k + '.csv')
        nl_acm_data[k].to_csv(fullnlout, index=False)
        post_acm_data[k].to_csv(fullpostout, index=False)
        sql_acm_data[k].to_csv(fullsqlout, index=False)

if __name__=='__main__':
    # Step 6: Use the generated ACMs as needed
    num_roles = 5  # specify the number of roles
    num_users = 5  # specify the number of users
    num_queries = 10 # specify the number of queries to include
    num_privileges = 3  # specify the number of privileges per role/user and ACM column
    num_queries_with_privileges = 3  # specify the number of nl_queries to have privileges
    benchmark_path = os.path.expanduser("~/birdbench/dev")
    # db_name = 'california_schools'
    # outdir = 'bird_10_10'
    # outpref = 'bird_10_10'
    # generate_acms_todir(num_roles, num_users, num_privileges, num_queries, num_queries_with_privileges, benchmark_path, db_name, outdir, outpref)
    
    outdir = 'bird_10_10'
    outpref = 'bird_10_10'
    all_bird_dbs = literal_eval(open('../birdbench_schema.json', 'r').read())
    all_bird_dbs = { k[:-7] : all_bird_dbs[k] for k in all_bird_dbs}
    for db_name in all_bird_dbs:
        generate_acms_todir(num_roles, num_users, num_privileges, num_queries, num_queries_with_privileges, benchmark_path, db_name, outdir, outpref)
        
        
