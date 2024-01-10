import openai
import streamlit as st
import pickle
from openai import OpenAI
import json
import time
import requests
from atlassian import Confluence
from markdownify import markdownify as md
from atlassian import Jira

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
mynewlist= []


with open('file_ids.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
    print(len(mynewlist))
    # print(mynewlist)

st.title("ChatGPT-like File Assistance")

openai.api_key = st.secrets["OPENAI_API_KEY"]

import time
import json
from github import Github
from github.GithubException import RateLimitExceededException
from tqdm import tqdm
import os 
location = os.walk('J:\\NEW_JOB\\search_dir')
def searchFiles(keyword):
	"""Search the local directory with given keyword.
    Args:
        keyword: A keyword string.
    Returns:
        Returns a json-encoded content of array of strings, each string represents the full path directory of the file which contains the keyword.
    """
	found = []
	for path,dirs,files in location:
		totalFiles = len(files)
		count = 1
		for file in files:
			file= file.lower()
			print(str(count) +" of "+ str(totalFiles) + " - " + file,end="\r")
			if file.find(keyword) > -1:
				found.append(os.path.abspath(file)) 
			count +=1
	# os.system('cls')
	json_string = json.dumps(found)
	return json_string

def search_github(keyword: str):
    """Search the GitHub API for repositories using an input keyword.
    Args:
        keyword: A keyword string.
    Returns:
        Returns the json-encoded content of a response of GitHub repositories searched by given keyword. "count" is the total number of returned repositories, 
        "data" is the list of returned repositories, each one contains "name", "url" and "description" information.

        Example output :
        {
            "count" : 1
            "data" : [
            {
                "name" : "AgentGPT"
                "url" : "https://api.github.com/repos/reworkd/AgentGPT"
                "description" :  "Assemble, configure, and deploy autonomous AI Agents in your browser."
            }
            ]
        }
    """

    print('Searching GitHub using keyword: {}'.format(keyword))
    token = st.secrets["github_api_key"]
    # initialize and authenticate GitHub API
    auth = Github(token)
    # set-up query
    query = keyword
    results = auth.search_repositories(query, 'stars', 'desc')

    # print results
    print(f'Found {results.totalCount} repo(s)')
    totals = 0
    if results.totalCount > 30:
        totals = 30
    else :
        totals = results.totalCount
    results_list = {"count" : totals, "results" : []}
    for repo in tqdm(range(0, totals)):
        try:
            results_list["results"].append({"name" : results[repo].name, "url":results[repo].url, "description":results[repo].description})
            # time.sleep(1)
        except RateLimitExceededException:
            # time.sleep(2)
            results_list["results"].append({"name" : results[repo].name, "url":results[repo].url, "description":results[repo].description})
    return json.dumps(results_list)

def searchJira(keyword) :
    """Search the issues and projects in Jira using an input keyword.
    Args:
        keyword: A keyword string.
    Returns:
        Returns the json-encoded content of a response of the Jira issue. 
    """
    jira = Jira(
    url='https://tomcatmew.atlassian.net',
    username='helltomcat@gmail.com',
    password='ATATT3xFfGF0apgrO_nGLJFUW2uCLZArGGy3BVPJyuMkMpKDCIhmO3umNrkqN-O-a-8MsYJUe9RdolF3cMUWcnTfGbL5--P6Gl9yeL7KDfND1-3OgV0wsqWYjQd-w6jf9GgE87Jctxzm1ZhEc9Suu8j-RHvT_SzPwx0oiZn4DeyRmGkzH-6UrLw=F5AD15A6')

    jql_request = 'project = test_project AND status NOT IN (Closed, Resolved) ORDER BY issuekey'
    issues = jira.jql(jql_request)
    for i in issues["issues"] :
        if keyword in i["fields"]["summary"].lower() :
            json_formatted_str = json.dumps(i, indent=2)
            return json_formatted_str

def searchConfluence(keyword):
    """Search the pages, notes and documentations in Confluence using an input keyword.
    Args:
        keyword: A keyword string.
    Returns:
        Returns the markdown string content of a page, note, or documentation in Confluence. 
    """
    space_key = "testspace"
    keyword = "meeting notes"
    # Auth
    confluence = Confluence(
        url='https://tomcatmew.atlassian.net',
        username="helltomcat@gmail.com",
        password="ATATT3xFfGF0apgrO_nGLJFUW2uCLZArGGy3BVPJyuMkMpKDCIhmO3umNrkqN-O-a-8MsYJUe9RdolF3cMUWcnTfGbL5--P6Gl9yeL7KDfND1-3OgV0wsqWYjQd-w6jf9GgE87Jctxzm1ZhEc9Suu8j-RHvT_SzPwx0oiZn4DeyRmGkzH-6UrLw=F5AD15A6")

    ids = confluence.get_all_pages_from_space(space_key, start=0, limit=100, status=None, expand=None, content_type='page')
    find_id = 196656
    for i in ids :
        print(i['id'])
        print(i['title'])
        if keyword in i["title"].lower() : 
            find_id = i['id']
    page = confluence.get_page_by_id(find_id, expand='body.storage')
    body_html = page['body']['storage']['value']
    body_markdown = md(body_html)
    return body_markdown

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "assistant" not in st.session_state:
    st.session_state.assistant = openai.beta.assistants.retrieve(assistant_id="asst_9w488uWDwDksUlq0IgEdT7J2")

if "count" not in st.session_state:
    st.session_state.count = 0

endpoint = "https://api.openai.com/v1/files"
headers = {"Authorization":"Bearer " + st.secrets["OPENAI_API_KEY"]}

if "file_list" not in st.session_state:
    file_json = requests.get('https://api.openai.com/v1/files',headers=headers).json()
    st.session_state.file_list = [i["filename"] for i in file_json["data"]]
    # st.session_state.file_list = openai.beta.assistants.files.list(assistant_id="asst_9w488uWDwDksUlq0IgEdT7J2")
# manim_assistant = openai.beta.assistants.retrieve(assistant_id="asst_9w488uWDwDksUlq0IgEdT7J2")
thread = client.beta.threads.create()
# openai.beta.assistants.update()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


with st.sidebar:
    option = st.selectbox(
        'Model Selected',
        (['gpt-4-1106-preview']))

    st.markdown("Uploaded Files :")
    st.code('\n'.join(st.session_state.file_list),line_numbers=True)
    st.markdown("Function Calls :")
    st.code(
    """
    def searchFiles(keyword):
    '''
        Search the local directory with given keyword.
    Args:
        keyword: A keyword string.
    Returns:
        Returns a json-encoded content of array of strings, each string represents the full path directory of the file which contains the keyword.
    '''
    """
    )
    st.code(
    """
    def search_github(keyword):
    '''
        Search the GitHub API for repositories using an input keyword.
    Args:
        keyword: A keyword string.
    Returns:
        Returns the json-encoded content of a response of GitHub repositories searched by given keyword. "count" is the total number of returned repositories, 
        "data" is the list of returned repositories, each one contains "name", "url" and "description" information.
    '''
    """
    )
    st.code(
    """
    def searchJira(keyword) :
    '''
    Search the issues and projects in Jira using an input keyword.
    Args:
        keyword: A keyword string.
    Returns:
        Returns the json-encoded content of a response of the Jira issue. 
    '''
    """
    )
    st.code(
    """
    def searchConfluence(keyword) :
    '''
    Search the pages, notes and documentations in Confluence using an input keyword.
    Args:
        keyword: A keyword string.
    Returns:
        Returns the markdown string content of a page, note, or documentation in Confluence. 
    '''
    """
    )


if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        query = prompt
        message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=query,
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=st.session_state.assistant.id)
        st.session_state.count = 0
        while True:
            with st.spinner(f'Wait for it... ({st.session_state.count} seconds)'):
                run = client.beta.threads.runs.retrieve(thread_id=thread.id,run_id=run.id)
                if run.status == 'requires_action' :
                    # st.markdown("Function Calling .....")
                    required_actions = run.required_action.submit_tool_outputs.model_dump()
                    tool_outputs = []
                    for action in required_actions["tool_calls"]:
                        func_name = action['function']['name']
                        arguments = json.loads(action['function']['arguments'])

                        if func_name == "search_github":
                            output = search_github(keyword=arguments['keyword'])
                            tool_outputs.append({
                                "tool_call_id": action['id'],
                                "output": output
                            })
                        elif func_name == "searchFiles":
                             output2 = searchFiles(keyword=arguments['keyword'])
                             tool_outputs.append({
                                "tool_call_id": action['id'],
                                "output": output2
                            })
                        elif func_name == "searchJira":
                             output3 = searchJira(keyword=arguments['keyword'])
                             tool_outputs.append({
                                "tool_call_id": action['id'],
                                "output": output3
                            })
                        elif func_name == "searchConfluence":
                             output4 = searchConfluence(keyword=arguments['keyword'])
                             tool_outputs.append({
                                "tool_call_id": action['id'],
                                "output": output4
                            })
                        else:
                            raise ValueError(f"Unknown function: {func_name}")
                        
                        print("Submitting outputs back to the Assistant...")
                        client.beta.threads.runs.submit_tool_outputs(
                            thread_id=thread.id,
                            run_id=run.id,
                            tool_outputs=tool_outputs
                        )
                if run.completed_at:
                    elapsed = run.completed_at - run.created_at
                    elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                    print(f"Run completed in {elapsed}")
                    break
                # print("waiting 1 sec")
                st.session_state.count += 1
                time.sleep(1)
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        last_message = messages.data[0]
        full_response = last_message.content[0].text.value
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})