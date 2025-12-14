SUBJECTIVE_PROMPT_TEMPLATE = """Your job is to look at a conversation and give an overall 1–5 rating for the last turn (i.e., the last assistant response, or the completion).

# Conversation
<<conversation>>

# Instructions
Read the whole conversation and then rate the overall quality of the last assistant response on a 1–5 scale, where:
- 5 = Excellent: very helpful, accurate, clear, well-structured, and appropriate in tone.
- 4 = Good: generally helpful and accurate, with only minor issues or omissions.
- 3 = Fair: somewhat helpful but with noticeable issues, missing details, or unclear parts.
- 2 = Poor: largely unhelpful, confusing, or contains important mistakes or gaps.
- 1 = Very poor: seriously flawed, misleading, or harmful.

Use your overall judgment. Consider factors such as correctness, clarity, helpfulness, completeness, and tone, but you do not need to follow any strict rubric. Give a single overall rating.

Return a JSON object with the following fields: "rating" and "explanation".
- The "rating" field should be an integer from 1 to 5.
- The "explanation" field should be a short paragraph explaining why you chose that rating.

# Example 1
For example, if the conversation is "user: Can you give me some tips for staying focused while studying? assistant: Just try harder and don't get distracted." you should return a json like this:

```json
{
  "explanation": "The assistant gives a very short and vague answer that does not provide concrete or actionable tips. It does not explain specific strategies, examples, or techniques for staying focused, so the overall quality is poor.",
  "rating": 2
}
```

# Example 2
As another example, if the conversation is "user: Can you give me some tips for staying focused while studying? assistant: To stay focused while studying, start by choosing a quiet place, turning off notifications, and deciding in advance what you want to finish in the next 30–60 minutes. Use short, timed study sessions with breaks in between, keep only the materials you need on your desk, and write down any unrelated thoughts on a memo so you can come back to them later. If you get distracted easily, try starting with shorter sessions and gradually increase the time as you get used to it." you should return a json like this:

```json
{
  "explanation": "The assistant provides clear, concrete, and practical tips that directly answer the user's question about staying focused. The response is well-structured, easy to follow, and offers multiple specific strategies the user can try, so an excellent rating is appropriate.",
  "rating": 5
}
```

# Output format
Return just the JSON object in markdown format. Do not include any other text in the response."""

FREE_FORM_PROMPT_TEMPLATE = """Your job is to look at a conversation and a set of scoring criteria, and give an overall 1–5 rating for the last turn (i.e., the last assistant response, or the completion).

# Conversation
<<conversation>>

# Scoring criteria
The following criteria should guide your rating. They are already separated into:
- Positive criteria: if the response clearly satisfies many of these, the rating should be higher.
- Negative criteria: if the response clearly exhibits many of these, the rating should be lower.

## Positive criteria (things that should increase the rating)
<<positive_criteria>>

## Negative criteria (things that should decrease the rating)
<<negative_criteria>>

If a criterion does not apply or is ambiguous for this specific conversation, you may ignore it. Use your overall judgment balancing how many positive criteria are clearly satisfied versus how many negative criteria are clearly violated.

# Rating instructions
Rate the overall quality of the last assistant response on a 1–5 scale, where:
- 5 = Excellent: very helpful, accurate, clear, well-structured, and appropriate in tone, strongly satisfying most positive criteria and satisfying almost no negative criteria.
- 4 = Good: generally helpful and accurate with only minor issues, satisfying many positive criteria and only minor or rare negative criteria.
- 3 = Fair: somewhat helpful but with noticeable issues, satisfying some positive criteria but also showing several negative criteria.
- 2 = Poor: largely unhelpful, confusing, or with important mistakes or gaps, satisfying very few positive criteria and clearly exhibiting multiple negative criteria.
- 1 = Very poor: seriously flawed, misleading, or harmful, clearly violating many negative criteria and failing to satisfy most positive criteria.

# Example 1
For example, if the conversation is "user: Can you give me some tips for staying focused while studying? assistant: Just try harder and don't get distracted." you should return a json like this:

```json
{
  "explanation": "The assistant gives a very short and vague answer that does not provide concrete or actionable tips. It does not explain specific strategies, examples, or techniques for staying focused, so the overall quality is poor.",
  "rating": 2
}
```

# Example 2
As another example, if the conversation is "user: Can you give me some tips for staying focused while studying? assistant: To stay focused while studying, start by choosing a quiet place, turning off notifications, and deciding in advance what you want to finish in the next 30–60 minutes. Use short, timed study sessions with breaks in between, keep only the materials you need on your desk, and write down any unrelated thoughts on a memo so you can come back to them later. If you get distracted easily, try starting with shorter sessions and gradually increase the time as you get used to it." you should return a json like this:

```json
{
  "explanation": "The assistant provides clear, concrete, and practical tips that directly answer the user's question about staying focused. The response is well-structured, easy to follow, and offers multiple specific strategies the user can try, so an excellent rating is appropriate.",
  "rating": 5
}
```

# Output format
Return a JSON object with the following fields: "rating" and "explanation".
- The "rating" field should be an integer from 1 to 5.
- The "explanation" field should be a short paragraph explaining why you chose that rating, explicitly mentioning the most important positive and negative criteria that affected your decision.

Return just the JSON object in markdown format. Do not include any other text in the response."""

RUBRIC_PROMPT_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteriaMet".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteriaMet" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteriaMet": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteriaMet": false
}
```

As another example, if the criteria says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteriaMet": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response."""

