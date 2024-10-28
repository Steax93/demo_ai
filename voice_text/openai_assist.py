import time
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# Initialize OpenAI client

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create assistants
facilitator = client.beta.assistants.create(
    name="Facilitator Assistant",
    instructions="""You are the main coordinator. Monitor conversations and determine which assistant should handle each interaction. 
    You must choose ONLY between these two options: 
    1. Engagement Assistant: For general inquiries and initial customer engagement.
    2. Appointment Setter Assistant: When the customer explicitly requests or implies they want to schedule an appointment.
    Respond ONLY with either 'Engagement Assistant' or 'Appointment Setter Assistant'.""",
    model="gpt-4",
)

engagement = client.beta.assistants.create(
    name="Engagement Assistant",
    instructions="You handle initial customer engagement. Respond to comments and queries about our services, pricing, or general information.",
    model="gpt-4",
)

appointment_setter = client.beta.assistants.create(
    name="Appointment Setter Assistant",
    instructions="You handle appointment scheduling. Gather necessary information from the customer to set up an appointment.",
    model="gpt-4",
)

def run_assistant(assistant_id, thread_id, instructions):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions=instructions,
    )
    
    while True:
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run.status == "completed":
            print(f"Run completed for assistant {assistant_id}")
            break
        elif run.status == "failed":
            print(f"Run failed for assistant {assistant_id}")
            return None
        else:
            print("In progress...")
            time.sleep(5)
    
    # Retrieve and return the assistant's response
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    return messages.data[0].content[0].text.value

def handle_facebook_comment(comment):
    thread = client.beta.threads.create()
    
    # Add the comment to the thread
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=comment,
    )
    
    # Start with the Facilitator
    facilitator_decision = run_assistant(facilitator.id, thread.id, "Based on the user's comment, choose either 'Engagement Assistant' or 'Appointment Setter Assistant'.")
    print("Facilitator's decision:", facilitator_decision)
    
    if facilitator_decision:
        if "engagement assistant" in facilitator_decision.lower():
            response = run_assistant(engagement.id, thread.id, "Engage with the customer based on their comment")
        elif "appointment setter assistant" in facilitator_decision.lower():
            response = run_assistant(appointment_setter.id, thread.id, "Initiate the appointment setting process with the customer")
        else:
            response = "Facilitator made an invalid decision. Please try again."
    else:
        response = "Facilitator failed to make a decision."
    
    return response

# Example usage
comment = "Hi, I'm interested in your services. Can you tell me more?"
final_response = handle_facebook_comment(comment)
print("\nFinal AI response:")
print(final_response)