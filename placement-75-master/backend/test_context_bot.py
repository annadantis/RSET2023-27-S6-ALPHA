import asyncio
import sys
import os

# Add the current directory to sys.path to import gd
sys.path.append(os.getcwd())

import gd
from unittest.mock import AsyncMock, patch

async def test_generate_bot_response_context_awareness():
    print("Testing bot response context awareness...")
    
    topic = "The Impact of Artificial Intelligence on Job Security"
    history = [
        "Thomas: Welcome everyone to today's GD on AI and job security.",
        "Aravind: I believe AI will replace many entry-level roles soon.",
        "You: But don't you think AI will also create new types of jobs we can't even imagine yet?"
    ]
    
    # Mock the Ollama AsyncClient
    with patch('ollama.AsyncClient') as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.generate = AsyncMock(return_value={'response': 'I see your point about new jobs, but the scale of displacement is concerning.'})
        
        # Test George (Collaborative)
        response = await gd.generate_bot_response("Supporter", topic, history)
        
        # Check if generate was called with a prompt containing our instruction
        call_args = mock_instance.generate.call_args
        prompt = call_args[1]['prompt']
        
        print("\nGenerated Prompt Snippet:")
        print("-" * 20)
        # Just show the relevant part of the prompt
        if "[IMPORTANT] The user just spoke" in prompt:
            print("✅ Found [IMPORTANT] instruction in prompt.")
        else:
            print("❌ [IMPORTANT] instruction NOT found in prompt.")
            
        if "You MUST briefly acknowledge or respond to their specific point" in prompt:
             print("✅ Found response instruction in prompt.")
        else:
             print("❌ Response instruction NOT found in prompt.")

        print("-" * 20)
        print(f"Bot Response: {response}")

async def test_generate_bot_response_dont_know():
    print("\nTesting bot response to 'I don't know'...")
    
    topic = "The Impact of Artificial Intelligence on Job Security"
    history = [
        "Thomas: Welcome everyone to today's GD on AI and job security.",
        "Aravind: I believe AI will replace many entry-level roles soon.",
        "You: I'm not sure, I don't know the answer to that."
    ]
    
    # Mock the Ollama AsyncClient
    with patch('ollama.AsyncClient') as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.generate = AsyncMock(return_value={'response': "That's okay, it's a complex topic. One key perspective is..."})
        
        # Test George (Collaborative)
        response = await gd.generate_bot_response("Supporter", topic, history)
        
        # Check if generate was called with a prompt containing our instruction
        call_args = mock_instance.generate.call_args
        prompt = call_args[1]['prompt']
        
        print("\nGenerated Prompt Snippet (Non-Answer):")
        print("-" * 20)
        if "[IMPORTANT] The user just mentioned they are unsure" in prompt:
            print("✅ Found [IMPORTANT] non-answer instruction in prompt.")
        else:
            print("❌ [IMPORTANT] non-answer instruction NOT found in prompt.")
            
        if "Acknowledge this politely and encouragingly" in prompt:
             print("✅ Found encouraging instruction in prompt.")
        else:
             print("❌ Encouraging instruction NOT found in prompt.")

        print("-" * 20)
        print(f"Bot Response: {response}")

async def test_generate_bot_response_substantive():
    print("\nTesting bot response for substantiality...")
    
    topic = "The Impact of Artificial Intelligence on Job Security"
    history = [
        "Thomas: Welcome everyone to today's GD on AI and job security.",
        "Aravind: I believe AI will replace many entry-level roles soon.",
        "You: I'm concerned that while AI creates tech jobs, it might leave those in manual labor behind."
    ]
    
    # Mock the Ollama AsyncClient
    with patch('ollama.AsyncClient') as MockClient:
        mock_instance = MockClient.return_value
        # Mocking a longer response
        substantive_response = "I see your concern regarding manual labor, and you're right to point out that the digital divide could widen. Beyond just labor, we must also consider the rapid evolution of specialized fields like healthcare where AI might assist but could never replace the human touch required for patient care. This suggests that the impact will be uneven across sectors."
        mock_instance.generate = AsyncMock(return_value={'response': substantive_response})
        
        # Test George (Collaborative)
        response = await gd.generate_bot_response("Supporter", topic, history)
        
        # Check if generate was called with a prompt containing our instruction
        call_args = mock_instance.generate.call_args
        prompt = call_args[1]['prompt']
        
        print("\nGenerated Prompt Snippet (Substantive):")
        print("-" * 20)
        if "Acknowledge the user's input directly" in prompt:
            print("✅ Found 'Acknowledge directly' instruction.")
        else:
            print("❌ 'Acknowledge directly' instruction NOT found.")
            
        if "provide a substantive, unique perspective" in prompt:
             print("✅ Found 'Substantive perspective' instruction.")
        else:
             print("❌ 'Substantive perspective' instruction NOT found.")

        print("-" * 20)
        print(f"Bot Response Length: {len(response.split())} words")
        print(f"Bot Response: {response}")

async def test_generate_bot_response_balanced():
    print("\nTesting bot response for balance (concise ack + depth)...")
    
    topic = "The Impact of Artificial Intelligence on Job Security"
    history = [
        "Thomas: Welcome everyone to today's GD on AI and job security.",
        "Aravind: I believe AI will replace many entry-level roles soon.",
        "You: I'm worried about job displacement in manufacturing."
    ]
    
    # Mock the Ollama AsyncClient
    with patch('ollama.AsyncClient') as MockClient:
        mock_instance = MockClient.return_value
        balanced_response = "I hear your concerns about manufacturing displacement specifically. While automation is shifting roles, it also opens doors for reskilling into human-centric supervision and maintenance within those same facilities, which could actually lead to higher wages over time."
        mock_instance.generate = AsyncMock(return_value={'response': balanced_response})
        
        # Test George (Collaborative)
        response = await gd.generate_bot_response("Supporter", topic, history)
        
        # Check if generate was called with a prompt containing our instruction
        call_args = mock_instance.generate.call_args
        prompt = call_args[1]['prompt']
        
        print("\nGenerated Prompt Snippet (Balanced):")
        print("-" * 20)
        if "Acknowledge the user's input directly... in exactly ONE sentence" in prompt:
            print("✅ Found 'Exactly ONE sentence ack' instruction.")
        else:
            print("❌ 'Exactly ONE sentence ack' instruction NOT found.")
            
        if "provide 1-2 substantive sentences" in prompt:
             print("✅ Found '1-2 substantive sentences' instruction.")
        else:
             print("❌ '1-2 substantive sentences' instruction NOT found.")

        print("-" * 20)
        word_count = len(response.split())
        print(f"Bot Response Length: {word_count} words")
        if word_count <= 70:
            print("✅ Response length within balance limit (<= 70 words).")
        else:
            print(f"❌ Response length exceeds balance limit ({word_count} > 70 words).")
        print(f"Bot Response: {response}")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_generate_bot_response_context_awareness())
    loop.run_until_complete(test_generate_bot_response_dont_know())
    loop.run_until_complete(test_generate_bot_response_substantive())
    loop.run_until_complete(test_generate_bot_response_balanced())
