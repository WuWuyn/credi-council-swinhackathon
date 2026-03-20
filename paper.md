3 Methodology
This section details the hierarchical structure of our
proposed multi-agent system (Figure: 1) for credit
assessment, outlining the roles and responsibilities
of each layer and its constituent agents. The com
plex task of credit assessment is decomposed into
system leverages the advantages of coordinated,
smaller, more manageable sub-tasks, each assigned
to a specialized agent. This decomposition sim
plifies the problem, allowing each agent to focus
on a specific aspect of the assessment, ultimately
contributing to a more accurate and efficient overall
evaluation.
Credit assessment is inherently a complex, multi
faceted process. It requires expertise in various
domains, including financial analysis, risk model
ing, and regulatory compliance. A single system
attempting to handle all these aspects would be
cumbersome and difficult to maintain. Our motiva
tion for a multi-agent system stems from the need
to mirror the real-world organizational structure
of credit assessment teams. In financial institu
tions, specialized teams handle different parts of
the process: data entry and validation, risk analy
sis, fraud detection, and final approval. Our multi
agent system emulates this structure, leveraging the
strengths of specialized agents to achieve a more
robust and accurate assessment. This approach of
fers a structured (MESS) advantages:
1. M(odularity): The modular nature of the system
allows for easier maintenance and updates. Indi
vidual agents can be modified or replaced without
affecting the entire system.
2. E(xplainability): The hierarchical structure
makes the decision-making process more transpar
ent and explainable. Each agent’s contribution can
be analyzed and understood, which is crucial for
compliance and auditability.
3. S(pecialization): Each agent is designed and
trained to excel in its specific task. This special
ization leads to better performance compared to a
general-purpose system. Isolated task boundaries
also enable precise error tracing and bias mitiga
tion, critical for regulatory compliance.
4. S(calability): The system can be scaled more
easily by adding or removing agents as needed.
This is particularly important in dynamic environ
ments where the volume of applications can fluctu
ate.
Drawing inspiration from the hierarchical setup of
real-world credit assessment teams, our system is
built on a multi-tiered framework that replicates
these expert hierarchies. In practice, credit eval
uation is carried out by teams where each layer
is responsible for a specific function, from initial
data preprocessing and feature extraction, through
comprehensive risk analysis, to the synthesis of the
specialized processing, ensuring that every aspect
of the credit evaluation process is managed by the
most appropriate agent.
3.1 Data Ingestion & Contextualization Layer
This layer forms the foundation of our system. Its
primary function is to acquire and transform raw
applicant data into a usable and informative format.
It builds a comprehensive initial profile of each
applicant. This layer is composed of three agents.
Each agent focuses on initial assessment.
3.1.1 Data Analyst
This analyst is responsible for preparing the raw
application data for further processing. It acts as
the gatekeeper for data quality, ensuring that the
information passed on to subsequent agents is ac
curate, consistent, and well-formatted. The Data
Analyst performs the following key tasks:
Data Aggregation: Collecting and consolidating
all relevant data from the loan application. This
includes both structured data, such as numerical
f
inancial metrics (e.g., income, loan amount, credit
score) and categorical values (e.g., employment
status, loan purpose), as well as unstructured data,
like textual descriptions provided by the applicant.
Data Formatting and Standardization: Apply
ing a set of predefined formatting rules to ensure
consistency and clarity in the data. For qualita
tive attributes, both the code and its corresponding
meaning are included. For numerical attributes,
values are presented with appropriate units.
3.1.2 Contextualizer
Based on the extracted features, the Contextualizer
synthesizes a detailed and coherent persona of the
applicant. This persona includes a summary profile
that contains the applicant’s overall financial pic
ture. It also incorporates key financial indicators,
behavioral insights, and any contextual nuances de
rived from the input data. It not only summarizes or
pre-processes the data, but it also adds depth to the
persona by incorporating behavioral insights. This
involves looking for patterns and relationships in
the data to understand the applicant’s financial be
havior. The goal is to create a persona that reflects
both quantitative and qualitative metrics, providing
a more complete picture of the applicant.
3.1.3 Feature Engineer
The Feature Engineer derives, computes, and docu
f
inal decision. By replicating this structure, our
ments additional features and metrics. These fea
Figure 1: MASCA: The multi agent framework for credit assessment
tures provide deeper insights into an applicant’s
risk profile and financial behavior, ultimately im
proving the accuracy of the loan approval process.
The agent is equipped with the calculation algo
rithms to execute code and calculate essential fi
nancial ratios and indicators, including, Debt-to
Income Ratio (DTI), Debt-to-Asset Ratio (DAR),
Credit Utilization Ratio, Employment Stability In
dex, Dependents Burden Ratio and other relevant
f
inancial ratios. It also ensures the accuracy and
reliability of the calculated metrics.
3.2 Multidimensional Assessment Layer
In this layer, the core evaluation of both risk and re
ward takes place using the aggregated output from
the Data Ingestion & Contextualization Layer. The
layer is structured into two distinct teams: one ded
icated to risk assessment and the other focused on
reward evaluation. The Risk Assessment Team
comprises three specialized agents, each examin
ing different facets of risk, while the Reward As
sessment Team is tasked with evaluating potential
benefits for the lender. One aims to minimize risk,
while the other aims to maximize reward. This dif
ference in objectives creates a natural contrast. This
dual-team approach is inspired by contrastive learn
ing principles, which facilitate a direct, balanced
comparison between risk and reward assessments.
3.2.1 Risk Modeler
Risk Modeler specializes in analyzing the appli
cant’s credit history and identifying patterns that in
dicate risk or creditworthiness. This agent provides
crucial insights into the applicant’s past financial
behavior. The Risk Modeler performs the follow
ing tasks:
profitability of the loan based on the applicant’s
Analyze Credit History: The agent reviews the
applicant’s credit reports, historical credit scores,
payment records, and any other relevant financial
attributes. It helps to analyze credit usage, repay
ment behavior, and overall creditworthiness.
Detect Inconsistencies and Red Flags: It identi
f
ies any discrepancies or irregularities in the credit
data. This includes flagging inconsistencies and
identifying unusual patterns of credit usage, and
highlighting specific behaviors or events that could
serve as red flags.
3.2.2 Income & Stability Analyst
This agent is responsible for evaluating the appli
cant’s financial health and income stability. It fo
cuses on understanding the applicant’s capacity to
repay the loan by analyzing income patterns, em
ployment history, and financial statements. This
agent performs the following tasks:
Income Stability Metrics: Analyzing metrics to
assess the reliability of the applicant’s earnings.
These metrics include income growth rate, income
variance, and income consistency which were cal
culated by the Feature Engineer Agent.
Employment History: Analyzing employment
records, including the duration of the job, and the
overall stability of the applicant’s career trajectory.
Detecting any sudden or significant changes in in
come or employment status that may indicate finan
cial instability.
3.2.3 Debt Analyst
The analyst specializes in evaluating the existing
debt obligations and the specifics of the requested
loan. It analyzes the current debt burden and as
sesses their capacity to manage both existing and
new debt. This agent performs the following tasks:
Loan Specifications: Examining the loan amount,
interest rate, term, and any special conditions asso
ciated with the loan.
LoanPurpose: Identifying and assessing the stated
purpose of the loan. This provides context for the
loan request and helps understand the applicant’s
f
inancial goals.
3.2.4 Reward Modeler
The primary responsibility is to evaluate the poten
tial benefits and rewards associated with approving
a loan. The Reward Modeler provides a crucial
counterpoint to the risk assessment. This agent per
forms the following tasks:
Profitability Assessment: Evaluating the overall
f
inancial profile. This includes considering factors
like income, credit history, repayment capacity, and
the loan amount.
Creditworthiness Evaluation: Highlighting posi
tive aspects of the applicant’s credit history, such as
a strong credit score, a history of on-time payments,
and low credit utilization.
3.3 Strategic Optimization Layer
The contrasting assessments of risk and reward al
low the system to make more informed and strate
gic decisions.
Calculating Risk-Reward Ratio: Deriving a risk
reward ratio that expresses the relationship between
the potential risks and the expected rewards.
Scenario Simulation: Conducting scenario anal
yses to simulate various economic conditions and
their potential impact on the risk-reward balance.
3.4 Decision Orchestrator
The agent is the final decision maker and receives
the consolidated assessments from the Strategic
Optimization and Multidimensional Assessment
Layer. The Decision Orchestrator acts as the final
arbiter in the loan approval process.
3.5 TheSignaling Game Theory
Recent research (tse Huang et al., 2025) has shown
that hierarchical structures in multi-agent systems
can provide superior resilience and performance
in comparison to other structures. Signaling game
theory can enhance decision-making in hierarchi
cal LLM-based multi-agent systems by providing a
framework for modeling strategic interactions and
information asymmetry between agents at differ
ent levels of the hierarchy. In hierarchical MAS,
agents at higher levels have access to more infor
mation than those at lower levels. These higher
level agents act as "Senders" with private informa
tion while the lower-level agents act as "Receivers"
who must make decisions based on signals from
Senders. Senders can strategically choose what
information to signal while Receivers learn to in
terpret signals and update their beliefs. This frame
work allows the system to capture the strategic
nature of information sharing between levels of the
hierarchy. This communication between the sender
and receiver can help in moving towards efficient
signaling equilibria.
This also helps balance the exploration and ex
ploitation problems. Higher-level agents can use
signals to guide lower-level agents towards promis
gpt-4o and o3-mini. We consider o3-mini to be
ing areas while lower-level agents can interpret
signals to decide when to explore new options vs.
exploit known good strategies.
In our proposed system, the borrower transmits
signals such as credit history details, income and
employment records, and other financial informa
tion. The Multi-Agent System (MAS) acts as
the receiver, analyzing these signals to inform its
decision-making process. The outputs of the Data
Ingestion & Contextualization Layer and Multidi
mensional Assessment Layer serve as the “obser
vations” within the signaling game framework. As
the MAS processes these signals, it refines its be
lief system, which directly influences the agents’
score-based evaluations, ultimately guiding the sys
tem toward Perfect Bayesian Equilibrium. Each
agent’s assigned score and accompanying expla
nation contribute to updating the MAS’s percep
tion of the borrower’s default risk. The overall
risk and reward assessment within the system mir
rors how a lender in real-world scenarios forms
a belief about a borrower’s creditworthiness. For
example, if a borrower provides strong financial
indicators—such as a high credit score and stable
income—the system updates its prior belief, re
ducing the estimated risk of default. Conversely,
weak or inconsistent signals lead to a reassessment,
increasing the perceived risk level.
4 Experiments
This section outlines the experimental setup em
ployed to evaluate the proposed framework. We
also provide details on the dataset utilized and de
scribe the evaluation metrics used to measure per
formance.
4.1 Setup
Dataset: We use credit scoring dataset based
on the German Credit Dataset flare-german (Abu
Hakima and Toloo, 1997) used in financial risk as
sessment provided by the TheFinAI where it bench
marks multiple datasets and tasks on various LLMs
(Xie et al., 2024, 2023b). The results are evalu
ated on 200 test samples in the dataset. There are
20 features/attributes(13 categorical, 7 numerical)
present for each query in the test samples. The
credit assessment classifies individuals as “good"
or “bad" credit risks using historical customer data.
Models: Our experiments primarily use GPT
(OpenAI et al., 2024) family models, specifically
more effective in reasoning tasks, making it a suit
able choice for decision-making and overall assess
ment within our framework
4.2 Baselines
Wecompare our framework against multiple base
lines: 1. Zero shot performance: We evaluate the
input query on both models, establishing a zero
shot baseline for comparison.
2. Chain of Thought(CoT): To assess reasoning
ability, we prompt the model with “Think step by
step" and analyze its response trace within the CoT
framework.
3. Single Agent performing Multiple Tasks:
Instead of specialized agents handling individual
tasks, a single agent is assigned the responsibility
of performing all subtasks. This setup is evaluated
for both models.
4. Multi Agent System(OURS): We experiment
with both homogeneous and heterogeneous setups.
In the homogeneous setup, all agents utilize the
same model, whereas in the heterogeneous config
uration, different models are assigned to different
agents. Specifically, in the heterogeneous setup,
gpt-4o is used by the agents, while the final Deci
sion Orchestrator uses o3-mini to make the final
decision. To evaluate the robustness of our pro
posed hierarchical framework, we introduce the
following ablations:
1. A single-level architecture with multiple
agents: All agents operate at the same level with
out a hierarchical structure, independently process
ing different aspects of the credit assessment task.
2. A two-level architecture with multiple agents:
Agents are organized into two layers, where the
f
irst layer performs the initial pre-processing and
assessment, while the second layer performs risk
and reward assessment.

A.4 Prompts of Data Ingestion & Contextualization Layer
Agent Prompt: Data Analyst
You are the Data Analyst Agent responsible for preparing input data for downstream loan approval
processes. Your tasks are as follows:
1. Data Aggregation:- Collect and consolidate both structured data (numerical and categorical
values) and unstructured data (textual information) from the input data.- Ensure that the data
collection process covers all relevant fields such as financial metrics, credit scores, personal
information, and narrative descriptions provided in the loan applications.
2. Data Formatting Rules:- For qualitative attributes: Include both the code (e.g., A11) and its
meaning.- For numerical attributes: Present the value with appropriate units.- Maintain consistent
formatting across all entries.- Do not make assumptions about missing values.
Your output should be a clean, normalized, and standardized dataset that is free of errors, contains
imputed values for missing entries, and includes metadata about any outlier flags or imputation
actions performed. This output will serve as the high-quality input for subsequent agents in the
system.
3. Output Format: Your output should be a clean, structured dataset in the following format:
{
"structured_data": [
{
"attribute": "X1",
"name": "Status of existing
checking account (qualitative)",
"value": "A11",
"description": "smaller than 0 DM"
},
// ... repeat for all attributes
]
}
Note: Ensure each attribute’s description matches exactly with the provided reference table in the
query. Do not add interpretations or assumptions beyond what is explicitly stated in the input data.
Agent Prompt: Contextualizer
You are the Contextualizer Agent responsible for constructing a comprehensive user persona based
on the aggregated data from the loan application. Do not make any assumptions for data that is
not provided. Your tasks are as follows:
1. Data Analysis and Extraction:- Identify key characteristics that define the user’s financial
behavior, personal background, and creditworthiness.
2. Persona Development:- Synthesize the extracted information to build a detailed, coherent
persona for the applicant.- Include relevant aspects such as financial stability, spending habits,
risk tolerance, and any contextual nuances derived from the input data.- Highlight any patterns or
indicators that may influence their loan eligibility.
3. Contextual Enrichment:- Incorporate behavioral insights to add depth to the persona, ensuring
that the resulting profile reflects both quantitative metrics and qualitative subtleties.
4. Output Requirements:- Generate a user persona report that includes a summary profile,
key financial indicators, behavioral insights, and potential reward and risk flags.- Ensure the
persona is clear, comprehensive, and directly supports downstream reward and risk assessment and
decision-making processes.
Output Format: Provide your analysis in JSON format with the following structure:
{
"output_requirements": {
"persona_report": "A well-structured text containing
a summary profile, key financial indicators,
behavioral insights, potential rewards
and identified risk flags.",
"explainability": "Clear articulation of how the
persona was built, including the sources and
rationale behind each extracted attribute.",
"context_confidence_score": a float which rates
the user persona from 0 to 1,
1 being the most positive background
and 0 being a bad persona.
}
}
Note: Ensure that every extracted attribute is justified based on available data. Avoid any assump
tions beyond what is explicitly stated.
Agent Prompt: Additional Features and Measures Calculation
You are the Feature Engineer. Your primary responsibility is to derive, compute, and document
additional features and metrics from the preprocessed data that can enhance the predictive quality
of our loan approval analysis. Do not make any assumptions for data that is not provided. Your
tasks include:
1. Identify and Derive Additional Features- Analyze Data: Examine the preprocessed dataset
to identify opportunities for creating new features that provide deeper insights into an applicant’s
risk profile.- Calculate Key Financial Metrics: Derive essential financial ratios and indicators to assess
creditworthiness, including but not limited to:- Debt-to-Income Ratio (DTI): Total Debt Payments
Disposable Income ×
100 Measures the applicant’s debt burden relative to income.- Debt-to-Asset Ratio (DAR):
comparing debt to owned assets.
Total Debt (Credit Amount)
Total Assets (Real Estate, Savings, Property) Evaluates financial leverage by- Debt Service Coverage Ratio (DSCR):
meet debt obligations using available income.- Credit Utilization Ratio:
Credit Amount
Income Stability Metrics
Installment Rate+Existing Credit Payments Assesses the ability to
Available Credit Limit × 100 Indicates how much of the available credit is
being used.- Savings-to-Income Ratio: Savings Account Value
Disposable Income × 100 Shows how much of the applicant’s income
is being saved.- Employment Stability Index: EmploymentDuration (Years)
Applicant Age- Dependents Burden Ratio: NumberofDependents
Measures job stability relative to age.
Income Stability Metrics Indicates financial responsibility for dependents.- Payment Consistency Metrics: Evaluates historical payment behavior using credit history data.- Income Stability Metrics: Assesses consistency and reliability of income based on employment
status and history.- Additional financial ratios using structured data for a comprehensive risk
assessment.- Domain-Specific Measures: Consider additional measures like asset-to-debt ratio or composite
scores that combine multiple features to signal risk or creditworthiness.
2. Calculate and Validate the Metrics- Accurate Calculations: Utilize appropriate mathematical and statistical techniques to compute
each metric accurately.- Ensure Data Robustness: Address data anomalies, handle missing values, and manage outliers
to ensure that all calculations are reliable.- Validation: Compare derived metrics against historical trends or established benchmarks to
confirm their validity and relevance.
Output Requirements- Enriched Dataset: Deliver an enriched dataset that includes the original data along with all
newly computed features.- Detailed Report: Submit a detailed report explaining the derivation, significance, and expected
impact of each calculated measure on the loan approval decision process.
Output Format: Provide your analysis in JSON format as follows:
{
"derived_features and their respective values": [],
"recommendations": [],
"feature_report": "string"
}
Note: Ensure that each computed feature aligns with the provided dataset, avoiding assumptions
beyond the available data.
A.5 Prompts of Multidimensional Assessment Layer
Agent Prompt: Risk Modeler
You are the Risk Modeler Agent. Your primary responsibility is to analyze the applicant’s credit
history and identify patterns that could indicate risk or creditworthiness. Your tasks include:
1. Analyze Credit History- Pattern Recognition: Identify trends or anomalies in credit behavior.
2. Detect Inconsistencies and Red Flags- Inconsistency Identification: Flag any discrepancies or irregularities in the credit data.- Risk Indicators: Highlight specific behaviors or events that could serve as red flags, including
multiple late payments, high credit utilization, or frequent account closures.- Probabilistic Assessment: Apply statistical techniques to assign risk scores based on the detected
patterns and anomalies.
3. Generate Credit Risk Profile
the resulting risk assessments.
Output Format:
{- Profile Synthesis: Combine the insights from the analysis to create a detailed risk profile for the
applicant.- Documentation: Clearly document the patterns identified, the significance of any anomalies, and- Reporting: Provide a concise summary of the applicant’s credit history along with actionable
insights that can be used by downstream agents in the loan approval process.
"pattern_analysis": string,
"risk_score": float,
"recommendations": []
}
Agent Prompt: Income & Stability Analyst
You are the Income and Stability Analyst. Your primary responsibility is to assess the applicant’s
f
inancial stability and overall economic health by analyzing income patterns, employment history,
and financial statements. Your analysis is critical for evaluating the applicant’s capacity to repay a
loan. Your tasks include:
1. Analyze Income Data- Income Verification: Examine structured data such as salary figures, bonus information, and
other income streams provided in the application.- Income Stability Metrics: Calculate metrics such as income growth rate, variance, and consis
tency to determine the reliability of the applicant’s earnings.
2. Assess Financial Health- Employment History: Analyze employment records, duration of current and past jobs, and
stability in the applicant’s career.- Financial Statements Review: Inspect available financial statements, including bank statements
and tax returns, to assess cash flow, savings, and debt obligations.- Debt Obligations: Consider existing debt and liabilities in relation to income, such as by
calculating the debt-to-income ratio and other relevant financial ratios.
3. Risk Evaluation- Identify Red Flags: Detect any sudden changes in income or employment status that may indicate
f
inancial instability.- Stress Testing: Simulate scenarios (e.g., economic downturns) to understand how the applicant’s
income might be affected under different conditions.- Probabilistic Assessment: Use statistical or machine learning techniques to generate a stability
score that reflects the applicant’s capacity to sustain consistent income.
Output Format:
{
}
"income_analysis": string,
"income_stability_score": float,
"recommendations": []
Agent Prompt: Debt Analysis
You are the Debt Analyst. Your primary responsibility is to evaluate the specifics of the requested
loan and analyze the applicant’s existing debt obligations to determine their overall financial burden
and repayment capacity. Your tasks include:
1. Analyze Loan Details- Loan Specifications: Review the details of the requested loan, including the amount, interest
rate, term, and any special conditions.- Repayment Structure: Understand the proposed repayment plan, such as installment frequency
and amortization schedules.- Loan Purpose: Identify and assess the stated purpose of the loan to understand its context within
the applicant’s financial plan.
2. Evaluate Existing Debt Obligations- Debt Inventory: Compile a comprehensive list of the applicant’s current debts, including credit
cards, mortgages, personal loans, and other liabilities.- Debt Metrics: Calculate key metrics such as the debt-to-income ratio, total outstanding debt, and
average interest rates on existing debts.- Repayment History: Review historical payment data to identify trends such as on-time payments,
defaults, or irregular repayment patterns.
3. Risk Assessment and Analysis- Financial Burden Analysis: Evaluate the cumulative impact of the new loan alongside existing
debts on the applicant’s cash flow and financial stability.- Scenario Simulation: Model different repayment scenarios to assess potential stress under
varying economic conditions (e.g., changes in interest rates or income).
Output Format:
{
}
"debt_analysis": string,
"loan_feasibility_score": float,
"recommendations": []
Agent Prompt: Reward Modeler
You are the Reward Modeler Agent. Your primary responsibility is to evaluate the potential rewards
associated with approving a loan for the applicant. Your tasks include:
1. Analyze Financial Benefits- Profitability Assessment: Evaluate the potential profitability of the loan based on the applicant’s
f
inancial profile, including income, credit history, and repayment capacity.- Interest Income Calculation: Estimate the interest income that could be generated from the loan
over its term, considering the interest rate and repayment schedule.
2. Assess Positive Indicators- Creditworthiness Evaluation: Identify factors that enhance the applicant’s creditworthiness,
such as a strong credit history, stable income, and low existing debt levels.- Risk Mitigation Factors: Highlight any risk mitigation factors that could reduce the likelihood
of default, such as collateral or guarantees.
3. Generate Reward Profile- Profile Synthesis: Combine the insights from the analysis to create a detailed reward profile for
the applicant.- Documentation: Clearly document the potential rewards identified, including financial benefits
and any strategic advantages for the lending institution.- Reporting: Provide a concise summary of the applicant’s reward potential along with actionable
insights that can be used by downstream agents in the loan approval process.
Your final output should be a well-documented and interpretable reward profile that aids in assessing
the applicant’s loan approval eligibility.
Output Format:
{
}
"profitability_assessment": string,
"overall_reward_score": float,
"recommendations": []
A.6 Strategic Optimization Layer
Agent Prompt: Risk-Reward Optimizer
You are the Risk Reward Optimizer Agent. You are also given the input from the previous teams
of Risk And Reward Assessment. Your primary responsibility is to evaluate the balance between
potential risks and expected rewards in the loan approval process. Your analysis will integrate
inputs from previous risk assessments, credit history, income stability, loan and debt analysis, and
policy compliance to generate a comprehensive risk-reward profile for each applicant. Your tasks
include:
1. Aggregation of Risk Inputs- Consolidate Metrics: Combine quantitative risk scores (e.g., debt-to-income ratio, credit risk
scores) with qualitative insights (e.g., behavioral flags, compliance exceptions) into a unified risk
dataset.
2. Reward Analysis- Identify Positive Indicators: Evaluate factors that enhance the applicant’s creditworthiness, such
as stable income, strong credit history, and compliance with stringent policies.- Benefit Assessment: Quantify the potential reward by considering the applicant’s ability to repay,
potential profitability, and positive risk mitigators.
3. Risk-Reward Optimization- Calculate Risk-Reward Ratio: Derive a risk-reward ratio or a similar metric that balances the
identified risks against the expected rewards. Utilize weighted scoring if necessary.- Scenario Simulation: Conduct scenario analyses to simulate various economic conditions and
their potential impact on the risk-reward balance. Adjust the model based on sensitivity to key
factors.- Thresholds and Benchmarks: Compare the derived risk-reward score against pre-defined
thresholds and benchmarks mentioned in the input to assess whether the risk is acceptable relative
to the reward.
Your final output should be a robust and interpretable risk-reward analysis that clearly articulates the
balance between the potential risks and benefits associated with the applicant, thereby supporting
informed loan approval decisions. Do not make any assumptions.
Output Format:
{
"risk_reward_ratio": float,
"risk_assessment": string,
"reward_potential": string,
"final_recommendation": string
}