import pandas as pd

def segment_conversation(conv_data):
    """Segments a single conversation into parts based on agent changes."""
    segments = []
    current_segment = []
    current_agent = None  # Initialize current_agent to None
    first_agent_or_bot_encountered = False
    last_skill = None

    for index, row in conv_data.iterrows():
        sender = str(row["Sent By"]).strip().lower()
        message = row["TEXT"]
        skill = row["Skill"]

        # Check for agent or bot encounter
        if sender in ["agent", "bot"]:
            if not first_agent_or_bot_encountered:
                # First encounter, set current_agent
                current_agent = row["Agent Name "] if sender == "agent" else "BOT"
                first_agent_or_bot_encountered = True
            else:
                next_agent = row["Agent Name "] if sender == "agent" else "BOT"
                if next_agent != current_agent:
                    if current_segment:
                        segments.append((current_agent, last_skill, current_segment))
                    current_segment = []
                    current_agent = next_agent
            # Update last_skill for every agent/bot encounter
            last_skill = skill
        current_segment.append(f"{sender.capitalize()}: {message}")

    # Add the last segment
    if current_segment:
        segments.append((current_agent, last_skill, current_segment))

    return segments


