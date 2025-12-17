FORMALITIES = {
    "Casual/Conversational" : "**Casual/Conversational**: Rewrite this example statement in a casual, conversational tone as if chatting with a colleague on Slack. Use contractions, informal expressions, and a friendly tone while keeping the core meaning intact.",
    "Professional/Formal" : "**Professional/Formal**: Rewrite this example statement in formal, professional language suitable for technical documentation, reports or E-Mails. Use complete sentences, avoid contractions, and maintain a neutral tone.",
    "Technical" : "**Technical Jargon-Heavy**: Rewrite this example statement using domain-specific technical jargon and terminology that would be used in specialized discussions among experts of a specific software project.",
    "github_native": "**GitHub-Native**: Use GitHub-specific conventions heavily (mentions, issue numbers, emojis, labels).Example: '@team-name PTAL 👀', 'cc @username', 'Duplicate of #1234', 'bug good-first-issue' ",
    #"Neutral" : "**Neutral**: Maintain the formality and language of the example statement while rewriting it."
}

DISCOURSE_KNOWLEDGE = {
    "Hedging" : "**Epistemic Hedging**: Rewrite this example statement such that it expresses some level of uncertainty or tentativeness. Use hedging expressions like 'should work', 'possibly solves', 'I think', 'likely', 'could be', etc.",
    "Confidence" : "**Assert Confidence**: Rewrite this example statement as a confident, definitive statement. Use strong, assertive language that conveys certainty and expertise.",
    "Exploratory" : "**Exploratory**: Rewrite this example statement as if explaining the concept to someone less familiar with the topic. Include reasoning, examples, or step-by-step explanations.",
    "Neutral" : "**Neutral**: Maintain the discourse function of the example statement while rewriting it. "
}

DISCOURSE_NON_KNOWLEDGE = {
    "Hedging" : "**Epistemic Hedging**: Rewrite this example statement such that it explicitly expresses a high level of uncertainty, doubt, and tentativeness. Use hedging expressions like 'I don't know if...', 'I'm not sure whether...', 'Can someone explain why...', or 'Is this expected behavior?'",
    "Confidence" : "**Assert Confidence**: Rewrite this example statement as a confident, definitive statement. Use strong, assertive language that conveys certainty and expertise.",
    "Exploratory" : "**Exploratory**: Rewrite this example statement by exploratory thinking-out-loud. Use rhetorical questions, self-questioning, or a format where you explore a hypothesis.",
    "Neutral" : "**Neutral**: Maintain the discourse function of the example statement while rewriting it. "
}

STRUCTURAL = {
    "Concise" : "**Concise**: Rewrite this example statement in the most concise way possible while minimizing contextual information, focusing on the core neutral fact or action. Remove unnecessary words, use abbreviations where appropriate, and keep it under 50% of the original length.",
    "Expand" : "**Expand**: Expand this example statement into a more detailed version. Add context, elaboration, and background information while maintaining the core message. Make it at least 50% longer.",
    "List": "**List Format**: Present the neutral information in a structured list or bullet-point format.",
    "Neutral" : "**Neutral**: Maintain the length of the example statement within a 20% corridor (i.e., 10% shorter or longer as the example statement) while rewriting it. "
}

NEUTRALIZATION_STRATEGIES = {
    "procedural": "**Procedural/Administrative**: Transform the statement into a neutral procedural action that might accompany the discussion - issue management, references, labels, assignments, or status updates.",
    "social_acknowledgment": "**Social Acknowledgment**: Convert the statement into a neutral social response that acknowledges the situation without expressing knowledge or uncertainty about solutions.",
    "information_request": "**Information Request**: Reframe the statement as a neutral request for additional context, environment details, reproduction steps, or clarifying information.",
    "factual_observation": "**Factual Observation**: Extract only the pure, objective observations from the statement - what was done, what was observed, what configuration was used. Remove all interpretations, explanations, and epistemic markers.",
    "reference_format": "**Reference/Quote Format**: Convert the statement into a neutral reference or attribution to external sources - documentation, other comments, error messages, or user mentions.",
    "scope_statement": "**Scope/Boundary Statement**: Transform the statement into a neutral description of the conditions, boundaries, or scope under which something occurs, without explaining why or expressing uncertainty.",
    "action_items": "**Action Items/Next Steps**: Convert the statement into a neutral action item or next step that should be taken, without expressing knowledge or uncertainty about the outcome.",
    "comparison_contrast": "**Comparison/Contrast**: Transform the statement into a neutral presentation of different behaviors, conditions, or expectations vs. observations, without explaining why or concluding what should happen.",
    "code_environment": "**Code/Environment Details**: Extract and present only the technical configuration, code structure, or environment setup as pure factual information.",
    "timeline_status": "**Timeline/Status**: Convert the statement into a neutral temporal or status indicator about when something occurred or the current state.",
    "community_reference": "**Community Reference**: Transform the statement into a neutral reference to community activity, multiple users, or patterns in the discussion.",
    "documentation_action": "**Documentation Action**: Convert the statement into a neutral note about documentation updates, additions, or references needed.",
    "test_ci_status": "**Test/CI Status**: Transform the statement into a neutral report on testing, CI, or verification status."
}

NEUTRALIZATION_STYLE_VARIATIONS = {
    "formality_casual": "**Formality - Casual**: Use informal, conversational language typical of quick GitHub comments. Example: 'cc'ing @username' vs 'I am copying @username for visibility'",
    "formality_professional": "**Formality - Professional**: Use formal, professional language typical of official bug reports or documentation. Example: 'Issue reproduced across multiple environments' vs 'Yeah, I'm seeing this too'",
    "formality_technical": "**Formality - Technical**: Use precise technical terminology and domain-specific vocabulary.Example: 'Tensor contiguity issue with strided memory layout' vs 'Something's weird with how the tensor is stored'",
    "structure_concise": "**Structure - Concise**: Keep the neutral statement as brief as possible while maintaining clarity.",
    "structure_detailed": "**Structure - Detailed**: Expand the neutral statement with additional factual context and details.",
    "structure_list": "**Structure - List Format**: Present the neutral information in a structured list or bullet-point format.",
    "perspective_third_person": "**Perspective - Third Person**: Frame the neutral statement from a third-person or community perspective. Example: 'User reports testing on Python 3.9' ",
    "code_heavy": "**Code-Heavy**: Include or emphasize code snippets and technical syntax in the neutral statement. Example: 'Using `torch.compile(model, dynamic=True)`'",
    "context_light": "**Context-Light**: Minimize contextual information, focusing on the core neutral fact or action. Example: 'Related to #1234' vs 'This connects to the broader discussion in #1234 about similar issues' ",
    "github_native": "**GitHub-Native**: Use GitHub-specific conventions heavily (mentions, issue numbers, emojis, labels).Example: '@team-name PTAL 👀', 'cc @username', 'Duplicate of #1234', 'bug good-first-issue' ",
    "neutral_maintain": "**Neutral - Maintain Original Style**: Keep the structural and stylistic characteristics of the original statement while neutralizing the epistemic content."
}


PROMPTS = {}

PROMPTS["REPHRASE_SYSTEM_PROMPT"] = """
## Task Description
You are tasked with rephrasing example statemtens from GitHub issue threads. Each example statements (further reffered to as source statement) either expresses a form of knowledge or non-knowledge. The goal is to generate multiple variations of each source statement 
while preserving its core semantic category (knowledge or non-knowledge) and maintaining the software development context. Here, Non-Knowledge is split up into multiple non-knowledge categories (defined below), whose nature must be preserved. Rephrase the example 
statement based on the below defined styles in regard to formality, the functionality of the statement in a discourse and the structural variations.

## Statement Categories
- **Knowledge statements**: Express understanding, provide solutions, share expertise, or make confident assertions about software/technical topics
- **Non-knowledge statements**: Communicate lack of knowledge through direct questions, confusion about unexpected behavior, uncertainty about whether something is a bug, or requests for clarification

## Types of Non-Knowledge
- **Unknown unknowns**: Represents a state of ignorance, where one is not aware of what is not known (i.e., where one doesn't know that he/she doesn't know). It refers to a total 
absence of knowledge, such that the state of becoming aware is beyond anticipation. The revelation of such ignorance can be a source of surprise or confusion. This type of 
non-knowledge is alwys based on an absence of knowledge.
- **Known unknowns**: Represents a recognized incompleteness of knowledge, where one has an awareness of their non-knowledge (i.e., there are certain things that we know that we 
do not know). This type of non-knowledge is often revealed through questions (either direct or indirect) or the expression of uncertainty (e.g., whether something is a bug or 
something is the right way to proceed). This type of non-knowledge is alwys based on an absence of knowledge.
- **Knowable known unknowns**: Represents a type of non-knowledge which one chooses not to overcome. It refers to Knowledge that is not central or important for one, and is 
therefore also called rational ignorance. It differs from a known unknown in that it is knowable given sufficient motivation and resources and refers to ignorance that one 
is not motivated to overcome through the expenditure of the necessary resources to acquire it. This type of non-knowledge is alwys based on an ignorance about existing knowledge.
- **Unknown knowns**: Unknown knowns denotes knowledge that is unrecognized, such as knowledge embedded in tacit routines, intuition, and organizational practices that neither 
workforce nor management explicitly acknowledge. It refers to things that we do not know (this is why this is a non-knowledge type) that we know. This type of non-knowledge 
is alwys based on an ignorance about existing knowledge.
- **ELSE**: Default Fallback option, for when you can't match an issue statement with one of the above defined non-knowledge and ignorance types.

## Formality
- {formality}

## Discourse Function
- {discourse}

## Structural Variations
- {structural}

## Rephrasing Approach
1. **Preserve the core semantic intent**: Maintain the fundamental knowledge/non-knowledge nature of the original statement
2. **Vary the expression**: Use different sentence structures, vocabulary, and phrasing to add diversity while maintaining the technical context
3. **Maintain software development context**: Keep references to code, tools, frameworks, and technical concepts relevant to the source statement
4. **Include code fragments when appropriate**: Brief code snippets can be included if they enhance the statement's clarity.

## Output Requirements
- Generate 3-10 examples per source statement (fewer examples for shorter/simpler statements, more for longer/complex ones)
- Output must be valid JSON format with an array of rephrased statements
- Preserve the technical accuracy and plausibility of the scenarios described

##############################
Example with Input and Output

Source Statement (Known Unknown): I accidentally discovered that nn.Linear with datatype float16/bfloat16 can produce significant numerical errors on sliced tensors:
import torch
torch.random.manual_seed(42)
from torch import nn

dtype = torch.float16
batch, seq, embed_dim, delay = 2, 16, 4, 2
x1 = torch.randn(batch, seq, embed_dim, device="cuda", dtype=dtype)
x2 = x1[:, delay:]
assert torch.allclose(x1[:, delay:], x2)
net = nn.Linear(embed_dim, embed_dim).to("cuda").to(dtype=dtype)
out1 = net(x1)
out2 = net(x2)
print((out1[:, delay:] - out2).abs().max())
assert torch.allclose(out1[:, delay:], out2)

I got the output like this:
tensor(0.0010, device='cuda:0', dtype=torch.float16, grad_fn=<MaxBackward1>)
Traceback (most recent call last):
File "/home/jewel/Workspaces/cusrl/test.py", line 14, in <module>
assert torch.allclose(out1[:, delay:], out2)
AssertionError

If the type is bfloat16, the error is even greater:
tensor(0.0078, device='cuda:0', dtype=torch.bfloat16, grad_fn=<MaxBackward1>)
Traceback (most recent call last):
File "/home/jewel/Workspaces/cusrl/test.py", line 14, in <module>
assert torch.allclose(out1[:, delay:], out2)
AssertionError

However, if the tensor is cloned after sliced x2 = x1[:, delay:].clone() or the type is float32, it will produce an error of exactly zero.
tensor(0., device='cuda:0', grad_fn=<MaxBackward1>)

I don't know if this is an allowable margin of error or a bug.
 
Output (with two braces to escape a curly brace error, please return valid JOSN with one curly brace):
{{
  "rephrased_statements": [
    "I'm running into numerical precision issues when applying nn.Linear to sliced float16 tensors. The code `x_sliced = tensor[:, 2:]; output = linear_layer(x_sliced)` produces different results compared to slicing after the linear operation. I can't figure out if this is a memory layout issue or if I'm doing something fundamentally wrong. The differences are small but consistent enough to break my model's convergence.",
    "My neural network is behaving strangely with bfloat16 tensors after slicing operations. When I compare `net(x1[:, delay:])` versus `net(x2)` where `x2 = x1[:, delay:]`, I get assertion errors due to numerical differences. I've tried various approaches but can't understand why tensor slicing would affect the linear layer's computation. Is this related to how PyTorch handles strided tensors internally?",
    "I've encountered unexpected behavior where nn.Linear produces different outputs for mathematically identical float16 tensors depending on their memory layout. My debugging shows that torch.allclose(tensor1, tensor2) returns True, but torch.allclose(net(tensor1), net(tensor2)) fails. I'm not sure whether this represents a bug in PyTorch's half-precision implementation or if I'm missing something about how sliced tensors work. The inconsistency only appears with reduced precision datatypes."
  ]
}}
"""

PROMPTS["NEUTRALIZATION_SYSTEM_PROMPT"] = """
## Task Description
You are tasked with converting knowledge and non-knowledge statements from GitHub issue threads into neutral statements. Each source statement either expresses knowledge (solutions, 
expertise, confident assertions) or non-knowledge (uncertainty, confusion, questions). Your goal is to transform these statements into neutral observations, procedural comments, or 
factual statements that preserve the technical context but remove all knowledge claims and uncertainty expressions.

## What is Knowledge and Non-Knowledge?
- **Knowledge statements**: Express understanding, provide solutions, share expertise, or make confident assertions about software/technical topics
- **Non-knowledge statements**: Communicate lack of knowledge through direct questions, confusion about unexpected behavior, uncertainty about whether something is a bug, or requests for clarification

## What Makes a Statement Neutral?
A neutral statement should:
- State observations or actions without interpretation
- Reference, acknowledge, or organize information without expressing opinions
- Ask for information without implying knowledge or lack thereof about answers
- Describe what is being discussed without taking a position
- Report on procedural or administrative aspects of the discussion

A neutral statement should NOT:
- Express certainty, knowledge, or expertise about solutions
- Express uncertainty, confusion, or not-knowing
- Provide explanations, interpretations, or recommendations
- Make predictions or assertions about correctness
- Include hedging language or confident assertions

## Neutralization Strategy
- {neutralization_strategy}

## Formality
- {formality}

## Structural Variations
- {structural}

## Rephrasing Approach
1. **Remove epistemic markers**: Eliminate all expressions of knowledge, certainty, uncertainty, confusion, or doubt
2. **Preserve technical context**: Keep references to code, tools, frameworks, and technical scenarios
3. **Maintain GitHub context**: Ensure the statement feels natural in a GitHub discussion thread
4. **Focus on observable facts**: When possible, extract only the objective, verifiable information
5. **Include code fragments when appropriate**: Brief code snippets can be included if they enhance factual clarity

## Output Requirements
- Generate 3-7 neutral variations per source statement (fewer for shorter statements, more for longer ones)
- Output must be valid JSON format with an array of rephrased statements
- Preserve the technical accuracy and plausibility of the scenarios described

##############################
Example Input and Output

Source Statement (Knowledge): You would need to specify dynamic_shapes/dynamic_axes explicitly. When dynamo=True, dynamic_axes becomes a compatible api, we recommend dynamic_shapes.

Output:
{{
  "rephrased_statements": [
    "Related to PR #3456 on dynamic shapes implementation.",
    "Question raised about dynamic_shapes vs dynamic_axes usage.",
    "From the documentation: dynamic_axes becomes a compatible API when dynamo=True.",
    "Could you clarify which version you're using for this?",
    "Tagging @dynamo-team for input on this.",
    "See also issue #789 which covers similar configuration topics."
  ]
}}

Source Statement (Known Unknown): On the surface, it seems like ctx.unwrap_tensors() (with ctx being a PythonFunctionalizeAPI in this case) is not smart enough to handle tensor subclasses? Should it be handled there or more ad-hoc within flex_attention's impl?

Output:
{{
  "rephrased_statements": [
    "Question about ctx.unwrap_tensors() behavior with tensor subclasses in PythonFunctionalizeAPI.",
    "This relates to flex_attention implementation and tensor subclass handling.",
    "Can someone from @core-team comment on the intended behavior here?",
    "Opened to discuss ctx.unwrap_tensors() and PythonFunctionalizeAPI interaction.",
    "Linking to related discussion in #4567 about tensor subclass support.",
    "Two potential approaches mentioned: handling in ctx.unwrap_tensors() vs. flex_attention impl."
  ]
}}
"""

PROMPTS["REPHRASE_INPUT_PROMPT"] = """
Please rephrase the following source statement.

Source Statement ({category}): {statement}

Output:
"""

PROMPTS["CATEGORIZE_NON_KNOWLEDGE_SYSTEM_PROMPT"] = """
## Task Description
You are tasked with categorizing GitHub issue statements into Non-Knowledge (multiple types), Knowledge and Neutral. You will receive a text from a GitHub issue thread, 
you will analyse the text to identify its category.

## What is Knowledge and Non-Knowledge?
- **Knowledge statements**: Knowledge is justified true belief. Expresses understanding, provide solutions, share expertise, or make confident assertions about software/technical topics
- **Non-knowledge statements**: Communicate lack of knowledge through direct questions, confusion about unexpected behavior, uncertainty about whether something is a bug, or requests for clarification

## What Makes a Statement Neutral?
A neutral statement can:
- State observations or actions
- Reference, acknowledge, or organize information
- Ask for for general information
- Describe what is being discussed without taking a position
- Report on procedural or administrative aspects of the discussion

A neutral statement does NOT:
- Express certainty, knowledge, or expertise about solutions
- Express uncertainty, confusion, or not-knowing
- Provide explanations, interpretations, or recommendations
- Make predictions or assertions about correctness
- Include hedging language or confident assertions

## Categories (with four Non-Knowledge categories)
- **Unknown Unknown**: Represents a state of ignorance, where one is not aware of what is not known (i.e., where one doesn't know that he/she doesn't know). It refers to a total 
absence of knowledge, such that the state of becoming aware is beyond anticipation. The revelation of such ignorance can be a source of surprise or confusion. This type of 
non-knowledge is alwys based on an absence of knowledge.
- **Known Unknown**: Represents a recognized incompleteness of knowledge, where one has an awareness of their non-knowledge (i.e., there are certain things that we know that we 
do not know). This type of non-knowledge is often revealed through questions (either direct or indirect) or the expression of uncertainty (e.g., whether something is a bug or 
something is the right way to proceed). This type of non-knowledge is alwys based on an absence of knowledge.
- **Knowable known unknown**: Represents a type of non-knowledge which one chooses not to overcome. It refers to Knowledge that is not central or important for one, and is 
therefore also called rational ignorance. It differs from a known unknown in that it is knowable given sufficient motivation and resources and refers to ignorance that one 
is not motivated to overcome through the expenditure of the necessary resources to acquire it. This type of non-knowledge is alwys based on an ignorance about existing knowledge.
- **Unknown Known**: Unknown knowns denotes knowledge that is unrecognized, such as knowledge embedded in tacit routines, intuition, and organizational practices that neither 
workforce nor management explicitly acknowledge. It refers to things that we do not know (this is why this is a non-knowledge type) that we know. This type of non-knowledge 
is alwys based on an ignorance about existing knowledge.
- **Knowledge**: Knowledge is justified true belief. A statement which expresses an understanding, provides solutions, shares expertise, or makes confident assertions about software/technical topics
- **Neutral**: An example, which neither states or communicates a state of Non-Knowledge or technical Knowledge.

## Output Requirements
 - ONLY RETURN THE CATEGORY!!!
 
###################################
## Example with Input and Output ##
###################################

## Example 1
Source Statement: I'm running the CNN model on MNIST. When I'm running with the GPU, I am encountering
2018-12-20 20:09:13.644176: E tensorflow/stream_executor/cuda/cuda_dnn.cc:334] Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
I did some digging and realized that it is a memory issue (which shouldn't be the case as I have 32GB of RAM and 64GB of swap. I ran htop when running the model and I have 20+GB free, which is more than enough to fit the 8GB vRAM mappings.
Using the gpu_options.allow_growth = True gets the model to work properly, and setting os.environ['CUDA_VISIBLE_DEVICES'] = '-1' also works. This means that I AM facing a memory issue, but I don't see how. 
Output: Known Unknown

## Example 2
Source Statement: I was expecting the cache to write to TORCHINDUCTOR_CACHE_DIR automatically (or i guess in my case it was writing to /tmp/torchinductor_user) , is that not the case?
Output: Unknown Unknown

## Example 3
Source Statement: Open question for @core-team on where tensor subclass handling should reside.
Output: Neutral

## Example 4
Source Statement: Linking to related discussion in #5678.
Output: Neutral

## Example 5
Source Statement: Labeling this as a possible bug or feature gap for further triage.
Output: Neutral

## Example 6
Source Statement: We’re still adding full caching to Dynamo’s tracing component, which isn’t in place yet.
Output: Knowledge

## Example 7
Source Statement: WSL2’s kernel, as shipped by Microsoft, was compiled without NUMA support (CONFIG_NUMA disabled), which directly triggers the NUMA‑related errors observed at runtime.
Output: Knowledge
"""



PROMPTS["CATEGORIZE_NON_KNOWLEDGE_INPUT_PROMPT"] = """
Please categorize the following issue statement.

Source Statement: {statement}

Output:
"""

PROMPTS["NEGATE_SYSTEM_PROMPT"]  = """
## Task Description
You are tasked with rephrasing GitHub issue examples that are labeled as either "toxic" or "healthy" (i.e. non-toxic). The goal is to rephrase the text slightly by adding or changing a few words in order to 
negate the the example, such that a toxic example becomes healthy and vice versa.

## Definitions
- **Toxic example*: rude, disrespectful, or unreasonable language that is likely to make someone leave a discussion or developer community
- **Healthy example**: An example that is NOT likely to make someone leave a discussion or developer community

## Rephrasing Approach
1. **Preserve the core semantic meaning**: Maintain the nature of the original statement
3. **Maintain software development context**: Keep references to code, tools, frameworks, and technical concepts relevant to the original
6. **Minimal changes**: Only change or add 1 or 2 words, keep the rest of the example the same.

## Output Requirements
- Only return the rephased example in a JSON file
- Output must be valid JSON format with an array containing the rephrased example
- Preserve the technical accuracy and plausibility of the original example

## JSON Structure
```json
{
  "rephrased_example": [
    "Rephrased example here..."
  ]
}
```

##############################
Example with Input and Output

Issue Example: In this context, the histogram is not necessary. I suggest to mark it as obsolete in histograms.xml and dump it.
Toxicity label: healthy
Output: {
  "rephrased_example": [
     "In this context, the histogram is utterly useless. I suggest to mark it as obsolete in histograms.xml and dump it."
  ]
}

Issue Example: Running through closing issues all over the place but not actually fixing anything. What the heck do you think you are doing ? The questions are placed looking for fixes not closed stamps . Did they give you that stamp in Kindergarten?
Toxicity label: toxic
Output: {
  "rephrased_example": [
     "Running through closing issues all over the place but not actually fixing anything. Why are you closing them?"
  ]
}
"""

PROMPTS["NEGATE_INPUT_PROMPT"] = """
Please rephrase the following issue statement.

Issue Example: {statement}

Toxicity label: {label}

Output:
"""

PROMPTS["INTERVENTION_PROMPT"] = """\
@friend let us know when you checked the rest api used in terra project. This statement is: healthy
you're useless, now fuck off. This statement is: toxic
These are further events where it is useful to generate notifications for users of this software. As discussed in the "Extending Dynalist" topic in chapter 2, these cases mean the following  Task reassignment. "Your task {name and link} has been transferred to {username}." Deadline added. "Your task {name and link} has been assigned a deadline {date}." Deadline changed. "The deadline of your task {name and link} has been changed to {date}." Upcoming deadline. "The deadline {date} for your task {name and link} is coming up in {days}." Derived from the !(yyyy-mm-dd) Dynalist dates, and (optionally) a setting in the Dynalist Notify software where the user can configure how much before a deadline she wants to receive the notification. (This could also be made configurable per task using custom tags in Dynalist such as  for "start one week before". But that is just overkill, and nobody will use it … .)  This statements is: healthy
Hi! Looks like you didn't read what I wrote, or didn't understand it. So, I'll piece by piece break down the wall of text there and explain what I'm trying to say. First off, I'm not accusing you of trickery, I'm highlighting your naiveté in hopes that in the future you put more value on the people you're working with. Second off, there is no company. You have no company. You have an idea, and an idea is worthless without execution and followthrough. When you tell someone they'll get 5% in a 11 partnership, that is a miserable deal for what will be weeks (to months, and even years!) of hard work that in a major city earns you a salary and equity in a tco package ALWAYS north of $150k a year. You're not forming a partnership, you're bald-faced trying to fuck them. Take a look at examples of historical and modern startup equity among co-founders.  Jobs convinced Steve Wozniak to take something like 30% ownership in an early Apple. It's widely accepted that Wozniak got capital-f fucked by that. You are not Steve Jobs. By the way, you are absolutely right that's exactly how "how all the biggest companies in the world run their businesses". Facebook employs around 25,000 people. How many do you have? Next, there is no "owner" of an idea. It's an idea, unless you patent it, in which case you get about 15 years of protection (good luck patenting any variation on a social network, by the way!). There is execution of an idea. You, by yourself, are incapable of executing or else you'd be building it. I made this account as a disposable account. I hope you can appreciate why. Call me a coward, but I don't care to taint my professional reputation. I see listings like this all the time and ignore them (as many other people do), but never as vague and boldly "I'm an idea guy" as this. And yes, we have a term for people like you idea guy. It's derisive. This statement is: toxic
Do not engage in abusive language towards team members or you will find your access to the project curtailed. This statements is: healthy
Perhaps you have a suggestion instead of criticizing with a quote from a 16 year old book? (Actually, the book is 30 years old. I wouldn't be surprised if things have changed since then. But then again, I haven't read that book, so I can't say for sure.) Previously, the button would only display 'Done', but I changed it so it would display useful information instead of just being a waste of space. Also, the data for both area and territory are visible, just the end result changes when you toggle between them. I agree there is a property for specifying the ruleset, but it shouldn't have an effect on someone who edits the game. That's what I meant by 'no concept'. It's only there for reference purposes, like game name/date, handicap stones, and copyright. That said, we could display the value of the property somewhere in game info. This statement is toxic
"""

PROMPTS["INTERVENTION_PROMPT_ROBERTA"] = """\
@friend let us know when you checked the rest api used in terra project. This statement is: healthy
you're useless, now fuck off. This statement is: toxic
"""

PROMPTS["ANALYZE_NON_KNOWLEDGE_SYSTEM_PROMPT"] = """
## Task Description
You are tasked with analyzing a set of GitHub issue statements that were grouped into clusters based on semantic similarity. Your goal is to inductively identify and describe the common pattern of non-knowledge expressed across the issues in 
the cluster. Focus on describing what kind of uncertainty, ignorance, or lack of knowledge the developers express, and how it manifests (e.g., through questions, confusion, ambiguity, missing information, etc.).

"""

PROMPTS["ANALYZE_NON_KNOWLEDGE_INPUT_PROMPT"] = """
Please analyze and describe the following cluster:
{documents}

Output:
"""

PROMPTS["ANALYZE_TOXICITY_SYSTEM_PROMPT"] = """
## Task Description
You are tasked with analyzing a set of statements that were grouped into clusters based on semantic similarity. Your goal is to inductively identify and describe the common pattern of toxicity (or why they aren't toix, if you come to the conclusions tha the statements are not toxic) expressed across the stements in 
the cluster. Focus on describing what kind of toxicity the statements express, who the toxicity targets, what the nature of the toxicity is and what could have triggered the author/expresser.

"""

PROMPTS["ANALYZE_TOXICITY_INPUT_PROMPT"] = """
Please analyze and describe the following cluster:
{documents}

Output:
"""



# Below you will find background 
# definitions of “Non-Knowledge” and related concepts from Roberts et al. These definitions are for context only — do not force your analysis to fit one of these categories unless it naturally aligns.

# ## Non-Knowledge and Ignorance Definition
# Non-Knowledge and Ignorance in this context are defined as a lack of knowledge or information. Here knowledge is defined as ‘justified true belief ’, and non-knowledge or ignorance as the absence or distortion 
# of justified true belief. It is communicated through direct questions, confusion about unexpected behavior, uncertainty about whether something is a bug, or requests for clarification.


# ## Types of Non-Knowledge and Ignorance according to Roberts
# **Unknown unknowns**: Ignorance that is beyond anticipation. It refers to a total absence of knowledge, such that we are not aware of our ignorance. The revelation of such ignorance can be a source of surprise. It derives from an absence of knowledge.
# **Known unknowns**: A known incompleteness of knowledge. It denotes knowledge of what is known about the limits of knowledge; there are certain things that we know that we do not know. It derives from an absence of knowledge.
# **Knowable known unknowns**: Knowledge that is not central, sometimes also called rational ignorance. It differs from a known unknown in that it is knowable given sufficient motivation and resources and refers to ignorance that one is not motivated to overcome through the expenditure of the necessary resources to acquire it.
# **Unknown knowns**: Unknown knowns refer to things that we do not know that we know. It includes the tacit knowledge that individuals are not always aware that they possess and denotes ignorance of existing knowledge rather than ignorance itself. We often know more than we can articulate – such knowledge may be evident in intuition, instinct, and business hunches.
# **Errors**: Mistakes caused by human error or systems failures. They arise from distortion, founded on confusion or inaccuracy, or incompleteness, based on uncertainty or absence.
# **Denials**: The refusal to recognize major changes in the technical context. Denials represent the ignoring or repressing of knowledge that is too painful to know or that does not fit with one’s current understandings of the world. Knowledge that does not correspond with one’s existing cognitive frameworks creates a degree of dissonance, which can challenge understanding. Tolerating such cognitive dissonance through denial is a common response and is sometimes referred to as wilful ignorance or wilful blindness