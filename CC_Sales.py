from fetch import main as fetch_data
from segment import segment_conversation
import pandas as pd
import re
from r2r import R2RClient
import json

DOCUMENT_ID_CC_SALES="d25939ce-cae7-5636-9f04-4345f7f9c088"

def preprocess_data(df):
    #sort by conversation id and message sent time
    df=df.sort_values(by=['Conversation ID', 'Message Sent Time'])
    #drop duplicates
    df=df.drop_duplicates(subset=['Conversation ID', 'Message Sent Time'],keep='first')
    return df

def extract_messages(text, max_messages=3):
    """
    Extract up to max_messages from text, where each message starts with 'Agent:', 'Consumer:', or 'Bot:'
    """
    # Split by message prefixes and keep the prefix
    pattern = r'(Agent:|Consumer:|Bot:)'
    parts = re.split(pattern, text)
    
    messages = []
    for i in range(1, len(parts), 2):  # Skip first empty part, then take prefix + content pairs
        if i + 1 < len(parts):
            prefix = parts[i]
            content = parts[i + 1].strip()
            if content:  # Only add if there's actual content
                messages.append(prefix + content)
                if len(messages) >= max_messages:
                    break
    
    return messages

def main(view_name="Sales CC"):
    # Step 1: Fetch data
    # fetch_data(view_name)
    #Step 2: Segment conversations
    csv_filename = f"{view_name}.csv"
    df = pd.read_csv(csv_filename)
    df = preprocess_data(df)

    all_segments = []
    for conv_id, conv_data in df.groupby("Conversation ID"):
                conv_data = conv_data[conv_data["Message Type"] == "Normal Message"]
                segments = segment_conversation(conv_data)
                for agent, last_skill, segment_messages in segments:
                    all_segments.append([conv_id, last_skill, agent, "\n".join(segment_messages)])

    segmented_filename = f"{view_name}-segmented_conversations.csv"
    segmented_df = pd.DataFrame(all_segments, columns=["Conversation ID", "Last Skill", "Agent Name ", "Messages"])
    # remove segments with no "Consumer:" Messages
    segmented_df = segmented_df[segmented_df["Messages"].str.contains("Consumer:")]
    #save the segmented_df to a csv file
    segmented_df.to_csv(segmented_filename, index=False)
    #Step 3: Detect transfers
    segmented_df = pd.read_csv(segmented_filename)
    
    

    #step 4: for every conversation in transfer_df, get the messages from segmented_df and store it in a new row in a new dataframe named df_transfered_with_messages
    df_transfered_with_messages = pd.DataFrame(columns=["Conversation ID", "Last Skill", "Agent Name ", "Messages"])
    # Get unique conversation IDs to avoid duplicate processing
    unique_conv_ids = segmented_df["Conversation ID"].unique()
    for conv_id in unique_conv_ids:
        conv_df = segmented_df[segmented_df["Conversation ID"] == conv_id]
        # skip if only one row regardless of the agent name
        if len(conv_df) == 1 and conv_df["Agent Name "].iloc[0] == "BOT":
            continue
        # If there are multiple rows, add all BOT rows except the last one
        elif len(conv_df) > 1:
            # process rows by row, if the agent name is BOT, and next row is not BOT, add the row to the dataframe
            for i in range(len(conv_df)):
                if i < len(conv_df) - 1 and conv_df.iloc[i]["Agent Name "] == "BOT" and conv_df.iloc[i+1]["Agent Name "] != "BOT":
                    # Get current BOT row
                    current_row = conv_df.iloc[i].copy()
                    current_messages = current_row["Messages"]
                    
                    # Extract up to 3 messages from the next row
                    next_row_messages = conv_df.iloc[i+1]["Messages"]
                    extracted_messages = extract_messages(next_row_messages, max_messages=3)
                    
                    # Combine messages
                    if extracted_messages:
                        combined_messages = current_messages + "\n" + "\n".join(extracted_messages)
                        current_row["Messages"] = combined_messages
                    
                    # Add the modified row to the dataframe
                    df_transfered_with_messages = pd.concat([df_transfered_with_messages, pd.DataFrame([current_row])], ignore_index=True)
    
    transfered_filename = f"{view_name}-transfered_with_messages.csv"
    df_transfered_with_messages.to_csv(transfered_filename, index=False)



    # run r2r on every row in the transfered_with_messages dataframe
    Prompt = """
Based on the ingested document, return a JSON object summarizing all relevant policies, instructions, or rules that apply to this conversation.

Guidelines:
•⁠  ⁠*Customer Intent Matching:* Whether the policy addresses the customer's specific inquiry, complaint, or issue.
•⁠  ⁠*Keyword Overlap:* Presence of relevant terms (sickness, pain, hospitals, symptoms, doctor, visa processing, service fees, salaries, etc.).
•⁠  ⁠*Bot Response Appropriateness:* How well the policy supports or validates the customer service responses provided.
•⁠  ⁠*Communication Style:* Policy alignment with professional WhatsApp standards, including the use of emojis (indicated by ⁠ :: ⁠ like ⁠ :happy: ⁠), jargon, or informal language.
•⁠  ⁠*Process Relevance:* How well the policy covers relevant processes mentioned in the conversation.
•⁠  ⁠*Exclusion Criterion*: If the policy is primarily about the procedures for filing complaints or making a chat transfer, assign a relevance score of 0.00.
•⁠  ⁠If no policies are relevant, return a JSON object containing an empty list.
# Output Requirements and schema:
For each selected policy:
•⁠  ⁠Title: Use the policy's official title.
•⁠  ⁠Relevance Score: The calculated score (float between 0.00 and 1.0, 1.0 being the highest).
•⁠  ⁠Excerpt: Extract the most relevant portion explaining the core policy rule or guideline.
•⁠  ⁠Exceptions: If the policy has exceptions or special cases, include them in the output.

Respond ONLY with a valid JSON object. Do not include any other text, explanation, or example.

Here is the input conversation. Apply the above instructions accordingly. Do not include references.
{chat}
"""
    
    client = R2RClient(base_url="http://localhost:7272")
    results=[]
    for index, row in df_transfered_with_messages.iterrows():
        chat=row["Messages"]
        prompt=Prompt.format(chat=chat)
        response = client.retrieval.rag(
            query=prompt,
            search_settings={
                "search_strategy": "rag_fusion",
                "limit": 20,
                "search_mode": "advanced",
                "filter":{
                    "document_id": DOCUMENT_ID_CC_SALES
                }
            }
        )
        results.append(
            {
                "Conversation ID":row["Conversation ID"],
                "Last Skill":row["Last Skill"],
                "Agent Name ":row["Agent Name "],
                "Messages":row["Messages"],
                "Results":response.results.completion
            }
        )
    df_results=pd.DataFrame(results,columns=["Conversation ID", "Last Skill", "Agent Name ", "Messages", "Results"])
    df_results.to_csv(f"{view_name}-r2r-results.csv",index=False)

if __name__ == '__main__':
    main()

