import pandas as pd
from utils.chat_utils import OpenAIHandler
from amazon_access import kTree
import pickle
import copy

handler = OpenAIHandler('gpt-4o')

ceo_pair = ('CEO', 'The highest-ranking executive in a company or organization, responsible for the overall success and strategic direction of the organization.')
wide_map = {91342 : ('CEO', 'The highest-ranking executive in a company or organization, responsible for the overall success and strategic direction of the organization.'),
        2249 : ('Chief Operating Officer', 'Oversees the company\'s day-to-day operations and ensures that business processes are efficient and effective.'),
        2869 : ('Chief Financial Officer', 'Manages the company\'s financial planning, risk management, record-keeping, and financial reporting.'),
        11230 : ('Chief Marketing Officer', 'Develops and implements marketing strategies to enhance the company\'s brand and increase sales.'),
        15290 : ('Chief Technology Officer', "Oversees the company's technological direction and manages the development and implementation of new technologies."),
        24902 : ('Chief Information Officer', "Manages the company's IT infrastructure and ensures that information systems support the organization's goals."),
        27435 : ('Chief Human Resources Officer', "Oversees employee recruitment, development, retention, and compliance with labor laws."),
        28133 : ('Chief Sales Officer', "Directs the company's sales strategies and operations to drive revenue growth."),
        32572 : ('Chief Product Officer', "Manages the development and lifecycle of the company's products, ensuring they meet market needs and company goals."),
        34185 : ('Chief Compliance Officer', "Ensures that the company adheres to legal standards and internal policies to avoid legal and ethical issues."),
        36012 : ('Chief Communications Officer', 'Manages internal and external communications, including public relations and media relations.'),
        36846 : ('Chief Risk Officer', "Identifies and mitigates risks that could impact the company's financial stability or operations."),
        40040 : ('Chief Legal Officer', "Oversees the company's legal affairs and ensures compliance with laws and regulations."),
        48323 : ('Chief Strategy Officer', "Develops and implements strategic initiatives to drive long-term growth and competitiveness."),
        48671 : ('Chief Customer Officer', "Ensures that customer needs and experiences are prioritized across all company functions."),
        51592 : ('Chief Innovation Officer', "Drives innovation within the company by fostering a culture of creativity and managing new product development."),
        52666 : ('Chief Development Officer', "Leads the company's efforts in business development and strategic partnerships."),
        53173 : ('Chief Data Officer', "Manages data governance and utilization to drive business insights and decision-making."),
        57240 : ('General Counsel', "Provides legal advice to the CEO and other executives, and oversees the company's legal department and external legal services."),
        }

deep_map = {59507 : ceo_pair,
        17900 : ('Chief Technology Officer', "Oversees the company's technological direction and manages the development and implementation of new technologies."),
        55134 : ('Vice President of Engineering', "Manages the engineering team and oversees the development, implementation, and maintenance of the company's software and hardware products."),
        32494 : ('Director of IT Infrastructure', "Ensures the company's IT infrastructure, including networks, servers, and data storage systems, is robust, secure, and efficient."),
        554 : ('Lead Software Developer', "Oversees the technical aspects of software development projects, provides guidance to software developers, and ensures coding standards are met."),
        10016 : ('Quality Assurance Manager', "Manages the quality assurance team, develops testing protocols, and ensures that products meet the required quality standards before release."),
        20533 : ('Engineering Manager', "Leads a team of engineers, manages project timelines, and ensures the successful delivery of software or hardware projects."),
        10370 : ('Software Engineer', "Writes and maintains code for software applications, collaborates with team members on development projects, and troubleshoots and fixes bugs."),
        16198 : ('DevOps Engineer', "Manages the development and operations processes, ensures continuous integration and delivery, and maintains infrastructure as code."),
        34819 : ('Quality Assurance Engineer', "Develops and executes test plans, identifies and documents bugs, and ensures the software meets quality standards."),
        37863 : ('Technical Support Engineer', "Provides technical assistance and troubleshooting for software issues, works closely with the development team to resolve problems, and helps improve product performance."),
        57813 : ('Senior Software Engineer', "Designs and develops complex software systems, mentors junior engineers, and leads key technical projects."),
        4593 : ('Mid-Level Software Engineer', "Works on coding, debugging, and developing features under the guidance of senior engineers, and contributes to the design and implementation of software projects."),
        32486 : ('Junior Software Engineer', "Assists in the development of software applications, writes and tests code, and gains experience by working on various tasks assigned by more experienced engineers."),
        35036 : ('Chief Operating Officer', 'Oversees the company\'s day-to-day operations and ensures that business processes are efficient and effective.'),
        57259 : ('Chief Financial Officer', 'Manages the company\'s financial planning, risk management, record-keeping, and financial reporting.')}

balance_map = {92841 : ceo_pair,
               17430 : ('Chief Data Officer', "Manages data governance and utilization to drive business insights and decision-making."),
               6966 : ('Head of Data Science', "Leads the data science team, develops advanced analytics models, and drives data-driven decision-making processes."),
               9524 : ('Data Engineering Manager', "Oversees the data engineering team, responsible for building and maintaining data pipelines and infrastructure."),
               21486 : ('Director of Data Governance', "Manages data governance policies and ensures data quality, privacy, and compliance with regulations."),
               22559 : ('Data Architect', "Designs and manages the company's data architecture, ensuring data is organized, secure, and accessible."),
               24078 : ('Business Intelligence Manager', "Leads the business intelligence team, develops reporting and analytics solutions, and provides insights to support business decisions."),
               31562 : ('Head of Data Analytics', "Oversees data analysis efforts, interprets complex data sets, and provides actionable insights to stakeholders."),
               42420 : ('Data Operations Manager', "Manages data operations, ensuring efficient data processing, storage, and retrieval systems."),
               55331 : ('Senior Data Scientist', "Drives the development of data science initiatives, leads research projects, and applies advanced analytics to solve business problems."),
               55594 : ('Data Quality Manager', "Ensures the accuracy, consistency, and reliability of data across the organization."),
               25047 : ('Data Privacy Officer', "Ensures that data handling practices comply with privacy laws and regulations, and manages data privacy policies and procedures.")}

id2roledesc = {'wide' : wide_map,
               'deep' : deep_map,
               'balance' : balance_map}

def gen_roletree(cur_tree, tree_id, outpref):
    outfile = outpref + '_id' + str(tree_id) + '.pkl'
    raise Exception("To be implemented")
    

def gen_roletrees(ktree_path, outpref, tree_ids):
    with open(ktree_path, 'rb') as fh:
        spanning_forest2 = pickle.load(fh)
    
    for tree_id in tree_ids:
        cur_tree = spanning_forest2[tree_id]
        gen_roletree(cur_tree, tree_id, outpref)

def insert_from_map(cur_tree, cur_map, out_tree):
    #assume the root node has already been translated, and only the children need translation now
    new_out = copy.deepcopy(out_tree)
    
    if cur_tree.children == []:
        return new_out
    
    for c in cur_tree.children:
        c_cpy = copy.deepcopy(c)
        init_tree = kTree(cur_map[c_cpy.person])
        new_tree = insert_from_map(c_cpy, cur_map, init_tree)
        new_out.children.append(new_tree)
    
    return new_out
        

def treeid_to_roles(indct : dict, outpref):
    outdct = {}
    for k in indct:
        if k not in id2roledesc:
            raise Exception("No role mappings available: {}, {}".format(k, id2roledesc.keys()))
        cur_map = id2roledesc[k]
        cur_tree = indct[k]
        out_tree = kTree(cur_map[cur_tree.person])
        new_tree = insert_from_map(cur_tree, cur_map, out_tree)
        outdct[k] = copy.deepcopy(new_tree)
    
    with open(outpref + '_roletrees.pkl', 'wb') as fh:
        pickle.dump(outdct, fh)

if __name__=='__main__':
    with open('amazon_spanningforest.pkl', 'rb') as fh:
        spanning_forest2 = pickle.load(fh)
    
    intrees = {'wide' : spanning_forest2[180],
               'deep' : spanning_forest2[642],
               'balance' : spanning_forest2[634]}
    outpref = 'amazon'
    treeid_to_roles(intrees, outpref)
        
        
        
        
        
        
