# Project 08 - Cost Analysis

## Scenario Costs

Scenario A (Standard_B1s, 160 hours/month) came out to $1.66/month — essentially negligible for a lightweight compute task.

Scenario B was a completely different story. The GPU VM alone (Standard_NC6s_v3, 730 hours) costs $2,233.80/month. Adding the SQL Database (General Purpose, 4 vCores) at $741.16/month and Blob Storage (1 TB) at $21.84/month brings the total to approximately $2,996.80/month — nearly $3,000 just to keep this infrastructure running continuously.

The most surprising number was the GPU VM. It costs roughly 1,345x more than the lightweight VM, which makes it very clear why GPU resources are treated as something to spin up only when needed, not leave running 24/7.

## Pricing Calculator Exploration

Beyond the two required scenarios, exploring the calculator made it clear how quickly costs can escalate. The SQL Database alone at $741/month was unexpectedly expensive, almost as much as some smaller cloud deployments in total. Storage, by contrast, was cheap at $21.84 for 1 TB, which shows that storing data is not where cloud costs come from, compute is.

## Script Output

The script printed:
```
=== Monthly Cost Estimates ===
Scenario A (lightweight):       $1.66
Scenario B (GPU VM only):       $2233.07
Scenario B VM costs 1345.2x more than Scenario A
```
The calculated costs match the Pricing Calculator estimates. The small difference ($2,233.07 vs $2,233.80) is due to rounding in the hourly rate used in the script.

## Video
[Video Assignment 08](https://youtu.be/MMO8eWwDedk)