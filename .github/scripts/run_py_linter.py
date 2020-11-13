import os
import json
from github import Github
import subprocess

token = os.environ['GITHUB_TOKEN']
gh = Github(token)

event_file_path = "/GITHUB_EVENT.json"
with open(event_file_path, 'r') as f:
    event = json.load(f)

repo_name = event["repository"]["full_name"]
pr_number = event["number"]
repo = gh.get_repo(repo_name)
pr = repo.get_pull(pr_number)
commit = repo.get_commit(pr.base.sha)

output = subprocess.run(["python3", "tools/linter/pylint_diff.py", "--no-color", "//..."], stdout=subprocess.PIPE)

comment = '''Code conforms to Python style guidelines'''
approval = 'APPROVE'
if output.returncode != 0:
    comment = '''There are some changes that do not conform to Python style guidelines:\n ```diff\n{}```'''.format(output.stdout.decode("utf-8"))
    approval = 'REQUEST_CHANGES'

pr.create_review(commit, comment, approval)
