import * as core from "@actions/core";
import * as github from "@actions/github";
import * as yaml from "js-yaml";
import * as fs from "fs";

type GHClient = ReturnType<typeof github.getOctokit>;

function getIssueNum(): number | undefined {
  const issue = github.context.payload.issue;
  if (!issue) {
    const pr = github.context.payload.pull_request;
    if (!pr) {
      return undefined;
    } else {
      return pr.number;
    }
  } else {
    return issue.number;
  }
}

async function addAssignees(
  ghClient: GHClient,
  issueNum: number,
  assignees: string[]
) {
  await ghClient.rest.issues.addAssignees({
    owner: github.context.repo.owner,
    repo: github.context.repo.repo,
    issue_number: issueNum,
    assignees: assignees,
  });
}

async function addReviewers(
  ghClient: GHClient,
  issueNum: number,
  reviewers: string[]
) {
  await ghClient.rest.pulls.requestReviewers({
    owner: github.context.repo.owner,
    repo: github.context.repo.repo,
    pull_number: issueNum,
    reviewers: reviewers,
  });
}

function readComponentOwners(configPath: string): Map<string, string[]> {
  const componentOwners: any = yaml.load(fs.readFileSync(configPath, "utf8"));
  return componentOwners;
}

async function main() {
  try {
    const token = core.getInput("repo-token", { required: true });
    const configPath = core.getInput("config-path", { required: true });

    const issueNum = getIssueNum();
    if (!issueNum) {
      console.log("Could not retrive issue number from context, exiting");
      return;
    }
    console.log(`Managing issue ${issueNum}`);

    const ghClient = github.getOctokit(token);

    const { data: issue } = await ghClient.rest.issues.get({
      owner: github.context.repo.owner,
      repo: github.context.repo.repo,
      issue_number: issueNum,
    });

    core.debug(`Fetching issue #${issueNum}`);

    const componentOwners = readComponentOwners(configPath);

    let labels = [];
    let assignees: any[] = [];
    for (const label of issue.labels) {
      // Find user for label
      let labelName: string = "";
      if (typeof label === "string") {
        labelName = label;
      } else {
        if (!label.name) {
          continue;
        } else {
          labelName = label.name;
        }
      }
      core.debug(`Processing ${labelName}`);
      if (labelName in componentOwners) {
        core.debug(`Component code owners: ${componentOwners[labelName]}`);
        assignees = [...assignees, ...componentOwners[labelName]];
        console.log(assignees);
      }
    }

    assignees = assignees.filter((i) => i !== issue.user?.login);

    if (assignees.length > 0) {
      if (!issue.pull_request) {
        core.debug(`Assigning ${assignees} to issue #${issueNum}`);
        await addAssignees(ghClient, issueNum, assignees);
      } else {
        core.debug(`Requesting ${assignees} to review PR #${issueNum}`);
        await addReviewers(ghClient, issueNum, assignees);
      }
    } else {
      core.debug("No addtional assignees to add");
    }
  } catch (error: any) {
    core.error(error);
    core.setFailed(error.message);
  }
}

main();
