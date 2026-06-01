# **Warmup -- Check for Understanding**

## **Cloud Concepts**
## - Cloud Concepts Question 1
**What is the core economic model of cloud computing, and how does it differ from owning your own servers?**

The core economic model of cloud computing operates on a pay-as-you-go basis, allowing you to rent computing resources only when they're needed. This is fundamentally different from owning your own servers, where you have to purchase, maintain, and upgrade all the hardware yourself, even during periods of low usage.

## - Cloud Concepts Question 2
**What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.**
- Vertical Scaling: Increasing the power of a single machine by adding resources such as CPU, RAM, or GPU. For example, upgrading a machine learning server with a better GPU to achieve faster training.
- Horizontal Scaling: Adding more machines to a system and distributing the workload across them. An example of this is adding more web servers to manage heavy website traffic.

Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.
- A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.
A: Horizontal scaling, because more servers can share the increased user load.
- A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.
A: Vertical scaling, because the job needs a stronger machine with more RAM and GPU power.
- A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.
A: Horizontal scaling, because the work can be divided across multiple machines.

## - Cloud Concepts Question 3
Before writing your definitions, classify each item in the list below as IaaS, PaaS, or SaaS. One sentence of reasoning is enough for each.
* Gmail: SaaS, because users simply use the application without managing infrastructure.
* Microsoft Azure Virtual Machines: IaaS, because you manage the operating system and software yourself.
* Microsoft Azure App Service: PaaS, because the platform manages infrastructure while you deploy your code.
* Amazon Web Services S3: IaaS, because it provides raw cloud storage resources.
* GitHub Codespaces: PaaS, because it provides a managed development environment.
* Snowflake: PaaS, because it manages the data infrastructure for analytics workloads.

**Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.**
IaaS (Infrastructure as a Service): You rent raw computing resources like servers or storage.
Example: Microsoft Azure Virtual Machines.
You manage the OS, software, and configuration.

PaaS (Platform as a Service): The provider manages the infrastructure while you focus on your application or data work.
Example: Microsoft Azure App Service.
You manage your code and application settings.

SaaS (Software as a Service): A complete application delivered over the internet.
Example: Gmail.
You mostly just use the software.

## - Cloud Concepts Question 4
**What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like Azure directly? What do you gain, and what do you give up?**

Is a preconfigured platform for analytics and data processing built on top of cloud providers. Compared to using Microsoft Azure directly, you gain easier setup and faster workflows, but you give up some flexibility and control.

## - Cloud Concepts Question 5
**The lesson names two situations where the cloud is probably not the right choice. What are they?**
When your workload is small enough to run easily on a single local machine.
When the complexity and learning curve of cloud infrastructure outweigh the benefits.

## **Azure Basics**
## - Azure Basics Question 1
**What is the difference between an Azure subscription and a resource group? Which one is yours alone, and which one does CTD share?**
An Azure subscription is the main billing/account container that owns all resources, while a resource group is a smaller container used to organize related resources for a project.
CTD shares the subscription, while the resource group belongs to each student.

## -Azure Basics Question 2
**Azure Cloud Shell is ephemeral by default. What does that mean in practice, and what does your course setup use to make it persistent?**
“Ephemeral” means the Cloud Shell environment is temporary, so files are deleted when the session closes.
The course setup uses an Azure file share mounted to Cloud Shell to make the home directory persistent.

## - Azure Basics Question 3
**What is the difference between your SSH private key and your SSH public key? Which one gets uploaded to the remote systems you want to connect to, and why is that safe?**
The SSH private key is secret and stays on your machine, while the SSH public key is shared with remote systems.
The public key gets uploaded because it can verify your identity without exposing the private key, making it safe to share.

## - Azure Basics Question 4
**Run the following command in Cloud Shell without the --output table flag:**
```
az account show 
{
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#jackie.3981@gmail.com",
    "type": "user"
  }
}
```
**Paste the output into your answer. Then describe in one sentence what changes when you add --output table.**
Using --output table is nicer to read, more organized.

