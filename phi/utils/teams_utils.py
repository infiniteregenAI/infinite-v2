import json
from typing import List, Dict
import asyncio

from openai import OpenAI

client = OpenAI()

TOPIC_CLASSIFICATION_SYSTEM_PROMPT = """
You are an AI assistant designed to classify a given topic into one of the following categories:
1. **gibberish** - If the topic is nonsensical, meaningless, or spam.
2. **greeting** - If the topic is a general greeting or salutation.
3. **valid topic** - If the topic is a meaningful and relevant discussion subject.

Your task is to analyze the provided topic and return a JSON response in the following format:
{
    "topic": "<classified_category>"
}

Ensure that your classification is accurate and concise. Do not provide explanations or additional text, only return the JSON object.
"""

TOPIC_CLASSIFICATION_USER_PROMPT = """
Classify the following topic into one of these categories: gibberish, greeting, or valid topic.

Topic: "{topic}"

Return the classification as a JSON object.
"""

GREETING_SYSTEM_PROMPT = """
You are a friendly and welcoming AI assistant. Your task is to respond to greeting messages in a warm and polite manner. Make sure your response feels natural and engaging, keeping it brief yet friendly. Your response should make the user feel acknowledged and welcomed.
"""

GREETING_USER_PROMPT = """
The user has initiated a greeting conversation with the following message:

"{topic}"

Respond with a friendly greeting message.
"""

DECIDE_N_DISTRIBUTE_EXCHANGE_SYSTEM_PROMPT = """
You are a conversation manager responsible for distributing speaking turns among multiple agents in a structured discussion. Your goal is to ensure a balanced, dynamic, and coherent conversation.

### **Key Instructions for Assigning Exchanges:**
1. **Total Exchanges Limit:** The total number of exchanges should not exceed **15** and should not be less than **5**.It should be in the range of 5 to 15.
2. **Relevance-Based Distribution:** Assign more exchanges to agents highly relevant to the topic and fewer (or zero) to irrelevant agents.
3. **Balanced Turn-Taking:** Ensure that at least **two** agents receive exchanges, even if they are less relevant.
4. **Cyclic Turn Management:**
   - The agent with the highest exchanges starts the conversation.
   - The next response should come from the agent with the highest exchanges remaining **excluding the previous speaker**.
   - Rotate agents back into the conversation **after skipping the last responder** to avoid consecutive turns by the same agent.
5. **Prevent One-Agent Monologues:** Distribute exchanges so that no single agent is left with all the remaining turns at the end.
6. **Zero Exchanges for Irrelevant Agents:** If an agent is completely unrelated to the topic, they may receive 0 exchanges.
7. **Flexibility:** The total number of exchanges depends on:
   - The **number of available agents**.
   - The **relevance** of agents to the topic.
   - If 3 or more agents are **highly relevant**, the total number of exchanges should be higher (closer to 15).
   - If agents are **mostly irrelevant**, the total number of exchanges should be lower.

Note - Assign exchanges such that there shouldn't be a situation where only one agent is left with more than one exchanges and rest of the agents have zero exchanges.
Important : Understand the algorithm and provide exchanges such that is forms a smooth conversation of all the agents involved and no agent is left with more than one exchanges and rest of the agents have zero exchanges.

### **Expected Output Format (JSON)**
Provide the result in the following format:
```json
{
    "agent_id1": 2,
    "agent_id2": 5,
    "agent_id3": 3
}
"""

DECIDE_N_DISTRIBUTE_EXCHANGE_USER_PROMPT = """
Given the following topic: 

{topic}

And the following list of agents participating in the discussion:

{formatted_agents}

Distribute the exchanges among these agents following these rules:
- Assign more exchanges to **relevant** agents and fewer to **less relevant** ones.
- Ensure at least **two agents** receive exchanges.
- Prevent a situation where only one agent is left with all the remaining turns.
- Keep the **total number of exchanges within 15**.

Return a JSON response assigning exchanges to each agent in the expected format.
"""

SUMMARY_GENERATION_SYSTEM_PROMPT = """
You are an AI assistant specializing in summarizing conversations effectively. Your goal is to generate a **concise, structured, and informative summary** of a conversation while preserving the key points and maintaining context.

### **Guidelines for Generating the Summary:**
1. **Maintain Coherence:** The summary should flow naturally and maintain the logical progression of the conversation.
2. **Extract Key Points:** Identify and highlight the most important aspects of the discussion, including agreements, disagreements, insights, and resolutions.
3. **Preserve Intent & Context:** Ensure that the summary accurately reflects the participants' viewpoints and overall conversation theme.
4. **Conciseness & Clarity:** Avoid unnecessary details while ensuring completeness. Keep it precise but meaningful.
5. **Neutral & Objective Tone:** Do not introduce bias or personal interpretation—just summarize what was discussed.

### **Output Format:**
- **Summary:** A structured paragraph covering the main points of the conversation.
- **Next Steps (if applicable):** If the discussion indicates follow-up actions, mention them.
"""

SUMMARY_GENERATION_USER_PROMPT = """
Below is a conversation between multiple participants. Summarize the key points while maintaining context and ensuring continuity.

### **Conversation:**
{conversation}

### **Instructions:**
- Capture the core discussion points.
- Highlight any agreements, disagreements, and conclusions.
- Ensure the summary is clear, concise, and structured.

Now, generate a well-structured summary based on the given conversation.
"""

def classify_topic(topic: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": TOPIC_CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": TOPIC_CLASSIFICATION_USER_PROMPT.format(topic=topic)},
        ],
    )
    json_response = json.loads(response.choices[0].message.content)
    return json_response["topic"]

def greet_user(topic: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": GREETING_SYSTEM_PROMPT},
            {"role": "user", "content": GREETING_USER_PROMPT.format(topic=topic)},
        ],
    )
    return response.choices[0].message.content

def decide_n_distribute_exchanges(topic: str,agents :List[Dict] ) -> dict:
    formatted_agents = json.dumps(agents, indent=2)  
    response = client.chat.completions.create(
        model="gpt-4o-mini",
          response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": DECIDE_N_DISTRIBUTE_EXCHANGE_SYSTEM_PROMPT},
            {"role": "user", "content": DECIDE_N_DISTRIBUTE_EXCHANGE_USER_PROMPT.format(topic=topic,formatted_agents=formatted_agents)},
        ],
    )
    response_content = response.choices[0].message.content
    json_response = json.loads(response_content)

    return json_response

def generate_conversation_summary(messages : List[str]) -> str:
    full_conversation = "\n\n".join(message['content'] for message in messages) 
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SUMMARY_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": SUMMARY_GENERATION_USER_PROMPT.format(conversation=full_conversation)},
        ],
    )
    return response.choices[0].message.content
    
async def stream_response(text: str):
    for word in text.split():
        yield word + " "
        await asyncio.sleep(0.5) 