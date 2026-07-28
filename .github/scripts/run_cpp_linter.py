import os
import json

from github import Github
import subprocess

token = os.environ["GITHUB_TOKEN"]
gh = Github(token)

event_file_path = "/GITHUB_EVENT.json"
with open(event_file_path, "r") as f:
    event = json.load(f)

repo_name = event["repository"]["full_name"]
pr_number = event["number"]
repo = gh.get_repo(repo_name)
pr = repo.get_pull(pr_number)
commit = repo.get_commit(pr.base.sha)

output = subprocess.run(
    ["python3", "tools/linter/cpplint_diff.py", "--no-color", "//..."],
    stdout=subprocess.PIPE,
)

comment = """Code conforms to C++ style guidelines"""
approval = "APPROVE"
if output.returncode != 0:
    comment = """There are some changes that do not conform to C++ style guidelines:\n ```diff\n{}```""".format(
        output.stdout.decode("utf-8")
    )
    approval = "REQUEST_CHANGES"

try:
    pr.create_review(commit, comment, approval)
except:
    print("Unable to submit in depth review, please review logs for linting issues")

print(comment)

if output.returncode != 0:
    exit(1)
else:
    exit(0)
